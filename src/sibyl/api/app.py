"""
FastAPI application exposing the diagnosis pipeline over HTTP.

This is the single entry point for the service: `sibyl.api.app:create_app`,
built as a factory so uvicorn is launched with --factory and tests can build a
fresh, isolated app instance per test.

Only what the codebase can actually back today is exposed — /health and
/diagnose. Forecasting, agents, persistence and background tasks get endpoints
when those layers exist, not before.
"""

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diagnosis.pipeline import run_full_diagnosis
from diagnosis.schemas import FullDiagnosisReport


class DiagnoseRequest(BaseModel):
    # Row-oriented records, i.e. what DataFrame.to_dict("records") produces —
    # the shape a caller gets for free from pandas, JSON or a CSV reader.
    data: list[dict[str, Any]] = Field(..., min_length=1)
    target_column: str | None = None   # defaults to the first numeric column


class DiagnoseResponse(BaseModel):
    report: FullDiagnosisReport
    summary: str   # the 6-line digest, pre-rendered for LLM context


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

    return app
