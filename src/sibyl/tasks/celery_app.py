"""
The Celery application.

Its own module so that `celery -A sibyl.tasks.celery_app worker` has something
small to import, and so the app object does not go round in a circle with the
tasks that register against it.

`include` is what makes the worker find the tasks: without it the worker starts
happily, registers nothing, and every dispatched job sits in the queue forever
with no error anywhere. That failure is silent and slow to diagnose, which is
exactly the kind this repository has been bitten by before.
"""

import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "sibyl",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["sibyl.tasks.forecast"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # The job row in Postgres is the source of truth, not Celery's result backend.
    # Keeping results a day is enough for debugging and stops Redis growing without
    # bound; anything a caller needs is in the jobs table.
    result_expires=86400,
    # A forecast is not idempotent to re-run halfway through, and a lost job is
    # cheaper than a duplicated one here, so acknowledge up front.
    task_acks_late=False,
)
