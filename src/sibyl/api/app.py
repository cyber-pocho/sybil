"""
FastAPI application exposing the diagnosis pipeline over HTTP.

This is the single entry point for the service: `sibyl.api.app:create_app`,
built as a factory so uvicorn is launched with --factory and tests can build a
fresh, isolated app instance per test.

Only what the codebase can actually back today is exposed. /diagnose runs inline
because a diagnosis is milliseconds; /forecast does not, because fitting Prophet
is seconds and holding an HTTP connection open for it is how a service falls over
under load. So /forecast writes a job, hands back an id, and returns 202.

The agent has no endpoint. It takes minutes, needs provider credentials in the
worker, and its reliability depends on which model you point it at — none of
which belongs behind an unqualified POST yet.
"""

import uuid
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from diagnosis.pipeline import run_full_diagnosis
from diagnosis.schemas import FullDiagnosisReport
from forecasting.registry import MODELS
from sibyl.db.engine import session_scope
from sibyl.models.job import Job, JobStatus
from sibyl.tasks.forecast import forecast_task


class DiagnoseRequest(BaseModel):
    # Row-oriented records, i.e. what DataFrame.to_dict("records") produces —
    # the shape a caller gets for free from pandas, JSON or a CSV reader.
    data: list[dict[str, Any]] = Field(..., min_length=1)
    target_column: str | None = None   # defaults to the first numeric column


class DiagnoseResponse(BaseModel):
    report: FullDiagnosisReport
    summary: str   # the 6-line digest, pre-rendered for LLM context


class ForecastRequest(BaseModel):
    # `model_name` collides with Pydantic's protected `model_` namespace, which
    # would otherwise warn on every import. The field name matches the registry
    # and the service signature, so widen the namespace rather than rename it.
    model_config = ConfigDict(protected_namespaces=())

    data: list[dict[str, Any]] = Field(..., min_length=1)
    target_column: str | None = None
    model_name: str = "prophet"
    horizon: int = Field(30, gt=0, le=1000)
    conformal: bool = False

    def to_params(self) -> dict[str, Any]:
        """Keyword arguments for services.forecasting.run_forecast.

        The wire calls the payload `data`; the service calls it `records`. Doing
        the rename here, once, keeps the stored params directly callable — the
        worker does `run_forecast(**job.params)` and nothing has to remember.
        """
        return {
            "records": self.data,
            "target_column": self.target_column,
            "model_name": self.model_name,
            "horizon": self.horizon,
            "conformal": self.conformal,
        }


class JobResponse(BaseModel):
    """A job as a caller sees it: what it is doing, and the result once there is one."""

    id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sibyl",
        version="0.1.0",
        description="Time-series diagnosis: profiling, stationarity, seasonality, anomalies.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/diagnose", response_model=DiagnoseResponse)
    def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
        df = pd.DataFrame(request.data)

        # run_full_diagnosis raises ValueError for input it cannot diagnose —
        # no numeric columns, or a target_column that isn't there. Both are the
        # caller's mistake, so surface them as 422 rather than a 500.
        try:
            report = run_full_diagnosis(df, target_column=request.target_column)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return DiagnoseResponse(report=report, summary=report.to_summary())

    @app.post("/forecast", response_model=JobResponse, status_code=202)
    def forecast(request: ForecastRequest) -> JobResponse:
        # Validate the model name here rather than in the worker. An unknown name
        # is the caller's mistake and they are still on the phone — telling them
        # now beats writing a job that only fails once they poll for it.
        if request.model_name not in MODELS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model '{request.model_name}'. Available: {', '.join(MODELS)}.",
            )

        job_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(Job(
                id=job_id,
                status=JobStatus.pending,
                params=request.to_params(),
            ))

        # Dispatched after the transaction commits, not inside it. A worker can
        # pick the message up before a slow commit lands, and then it would look
        # up a job that does not exist yet.
        forecast_task.delay(job_id)

        return JobResponse(id=job_id, status=JobStatus.pending)

    @app.get("/forecast/{job_id}", response_model=JobResponse)
    def forecast_status(job_id: str) -> JobResponse:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"No job {job_id}")
            return JobResponse(
                id=job.id, status=JobStatus(job.status), result=job.result, error=job.error
            )

    return app
