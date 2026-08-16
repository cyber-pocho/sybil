"""
Alembic environment.

Two things worth knowing. The URL comes from `sibyl.db.engine.database_url()`,
not from alembic.ini, so migrations and the application can never be pointed at
different databases by editing one file and forgetting the other. And every model
module has to be imported here for `--autogenerate` to see it: Base.metadata only
knows about classes that have actually been imported, so a model nobody imports
produces an empty migration and no warning at all.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from sibyl.db.base import Base
from sibyl.db.engine import database_url
from sibyl.models import job  # noqa: F401  (registers Job on Base.metadata)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", database_url())
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place. Batch mode rewrites the
            # table instead, which is what makes the default dev database
            # migratable at all; it is a no-op on Postgres.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
