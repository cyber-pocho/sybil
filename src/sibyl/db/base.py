"""
The declarative base every ORM model inherits from.

Its own module, and deliberately almost empty: Alembic's env.py imports it to
find `Base.metadata`, and models import it to register themselves. If the two
shared a module with the engine, importing a model would open a connection.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
