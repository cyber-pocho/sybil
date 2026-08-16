"""
The worker side of a forecast job.

The queue carries a job id and nothing else. Everything the work needs is already
in the row, which keeps the message small and means a retry reads current state
rather than a snapshot taken when the message was written.

`run_forecast_job` is a plain function and the Celery task is a thin wrapper over
it. That is what lets the whole worker path be tested without a broker: the tests
call the function, and the only thing they are not exercising is Celery's own
delivery, which is not ours to test.
"""

from datetime import UTC, datetime

from celery.utils.log import get_task_logger

from sibyl.db.engine import session_scope
from sibyl.models.job import Job, JobStatus
from sibyl.services.forecasting import run_forecast
from sibyl.tasks.celery_app import celery_app

logger = get_task_logger(__name__)


def run_forecast_job(job_id: str) -> str:
    """Run the job with this id to completion and record what happened.

    Returns the terminal status. Never raises for a failed forecast: a job that
    fails is a recorded outcome a client can read back, not a worker crash. Only
    a missing row raises, because that means the queue and the database disagree
    and retrying would not help.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise LookupError(f"No job {job_id}")
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        params = dict(job.params)

    # Deliberately outside the transaction above. A Prophet fit takes seconds, and
    # holding a database transaction open across it would pin a connection for the
    # whole fit for no benefit.
    try:
        result, error = run_forecast(**params), None
    except Exception as exc:
        # Broad on purpose. Anything a forecaster raises — a ValueError we wrote, a
        # Stan failure from deep inside Prophet — is a result the caller is owed,
        # and letting it escape would leave the row stuck in `running` forever.
        result, error = None, f"{type(exc).__name__}: {exc}"
        logger.warning("job %s failed: %s", job_id, error)

    with session_scope() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.done if error is None else JobStatus.failed
        job.result = result
        job.error = error
        job.finished_at = datetime.now(UTC)
        return job.status


@celery_app.task(name="sibyl.forecast")
def forecast_task(job_id: str) -> str:
    return run_forecast_job(job_id)
