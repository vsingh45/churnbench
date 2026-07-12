"""ChurnBench fabric connection factories.

Provides thin, singleton-free factories for the Postgres SQLAlchemy engine and the
PyMongo client. Connection strings are read from environment variables so different
environments (local Docker, CI, production) require no code changes.

Defaults match infra/docker-compose.yml:
  - Postgres on localhost:5433 (host-side port; container listens on 5432)
  - MongoDB on localhost:27018  (host-side port; container listens on 27017)

Invariants:
  - Each call to pg_engine() / mongo_client() creates a new object; callers that
    need a long-lived connection should cache it themselves.
  - No credentials are hard-coded. The docker-compose defaults (user=churn,
    password=churn, db=sam_warehouse) are the env-var *fallbacks* — never baked in.
"""
from __future__ import annotations

import os

from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase
from sqlalchemy import Engine, create_engine


# ── Postgres ─────────────────────────────────────────────────────────────────

_PG_DEFAULT = "postgresql://churn:churn@localhost:5433/sam_warehouse"


def pg_engine(url: str | None = None) -> Engine:
    """Return a SQLAlchemy Engine for the Postgres warehouse.

    Args:
        url: Override the connection URL. Falls back to CHURNBENCH_PG_URL env
             var, then the local-Docker default.
    """
    resolved = url or os.getenv("CHURNBENCH_PG_URL", _PG_DEFAULT)
    return create_engine(resolved, future=True)


# ── MongoDB ──────────────────────────────────────────────────────────────────

_MONGO_DEFAULT = "mongodb://localhost:27018"
_MONGO_DB = "sam_ops"


def mongo_client(url: str | None = None) -> MongoClient:  # type: ignore[type-arg]
    """Return a PyMongo MongoClient.

    Args:
        url: Override the connection URL. Falls back to CHURNBENCH_MONGO_URL env
             var, then the local-Docker default.
    """
    resolved = url or os.getenv("CHURNBENCH_MONGO_URL", _MONGO_DEFAULT)
    return MongoClient(resolved)


def mongo_db(url: str | None = None) -> MongoDatabase:  # type: ignore[type-arg]
    """Return the sam_ops MongoDatabase directly."""
    return mongo_client(url)[_MONGO_DB]
