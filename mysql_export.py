"""
mysql_export.py
───────────────
MySQL integration for YouTube engagement metrics.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Generator, Optional

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import OperationalError, SQLAlchemyError

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

METRICS_COLUMNS: list[str] = [
    "content_id", "platform", "views", "likes",
    "comments", "shares", "saves", "engagement_rate",
]

_INT_COLUMNS: frozenset[str] = frozenset(["views", "likes", "comments", "shares", "saves"])

# Only allow safe, identifier-style table names (letters, digits, underscores).
# This prevents SQL injection via the table_name config value.
_SAFE_IDENTIFIER_RE = re.compile(r"^\w{1,64}$")


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)          # frozen → hashable → safe to cache on
class DBConfig:
    url: str
    table_name: str         = "engagement_metrics"
    chunk_size: int         = 500
    pool_size: int          = 5
    max_overflow: int       = 10
    pool_pre_ping: bool     = True

    def __post_init__(self) -> None:
        if not _SAFE_IDENTIFIER_RE.match(self.table_name):
            raise ValueError(
                f"table_name '{self.table_name}' contains unsafe characters. "
                "Use only letters, digits, and underscores (max 64 chars)."
            )
        if not self.url.startswith(("mysql+", "mysql:")):
            raise ValueError(
                f"DATABASE_URL does not look like a MySQL URL: '{self.url[:40]}…'"
            )
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")


def load_db_config(
    table_name: str = "engagement_metrics",
    chunk_size: int = 500,
) -> DBConfig:
    """Build a DBConfig from environment variables."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise EnvironmentError(
            "DATABASE_URL is not set. "
            "Export it before running: export DATABASE_URL='mysql+pymysql://...'"
        )
    return DBConfig(url=url, table_name=table_name, chunk_size=chunk_size)


# ── Engine factory (cached) ───────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _get_engine(cfg: DBConfig) -> Engine:
    """
    Return a cached SQLAlchemy engine for *cfg*.

    One engine per unique DBConfig is kept alive for the process lifetime so the
    connection pool is reused across multiple calls to export_to_database().
    Disposing the engine (e.g. in tests) invalidates the cache entry automatically
    because DBConfig is frozen/hashable.
    """
    log.debug("Creating new SQLAlchemy engine for host in DATABASE_URL.")
    return create_engine(
        cfg.url,
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_pre_ping=cfg.pool_pre_ping,
        future=True,
    )


@contextmanager
def get_connection(engine: Engine) -> Generator[Connection, None, None]:
    """
    Yield a transactional connection.

    The connection is returned to the pool (not disposed) after use so that
    subsequent calls benefit from pooling.  engine.dispose() should only be
    called explicitly when you want to tear down the whole pool (e.g. in tests).
    """
    try:
        with engine.begin() as conn:
            yield conn
    except OperationalError as exc:
        log.error("Database connection error: %s", exc)
        raise


# ── Schema validation & prep ──────────────────────────────────────────────────

class SchemaValidationError(ValueError):
    """Raised when the input DataFrame is missing required columns."""


def _validate_schema(df: pd.DataFrame) -> None:
    missing = set(METRICS_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Missing required column(s): {sorted(missing)}. "
            f"DataFrame has: {sorted(df.columns.tolist())}"
        )


def _prepare_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select, cast, and sanitise the metrics columns."""
    out = df[METRICS_COLUMNS].copy()

    for col in _INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # Strip timezone info — MySQL DATETIME columns don't store tz offsets.
    for col in out.select_dtypes(include=["datetimetz"]).columns:
        out[col] = out[col].dt.tz_localize(None)

    return out.reset_index(drop=True)


# ── MySQL upsert ──────────────────────────────────────────────────────────────

def _build_upsert_statement(cols: list[str], table_name: str) -> text:
    """
    Pre-compile the upsert statement for a given column list.

    Table name is validated by DBConfig.__post_init__ before reaching here,
    so the f-string interpolation is safe.
    """
    placeholders   = ", ".join(f":{c}" for c in cols)
    col_list       = ", ".join(cols)
    conflict_keys  = {"content_id", "platform"}
    update_clause  = ", ".join(
        f"{c} = VALUES({c})" for c in cols if c not in conflict_keys
    )
    return text(f"""
        INSERT INTO {table_name} ({col_list})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """)


def _upsert_chunk(
    conn: Connection,
    chunk: pd.DataFrame,
    stmt: text,          # pre-compiled — avoids rebuilding the statement every chunk
) -> int:
    result = conn.execute(stmt, chunk.to_dict(orient="records"))
# We return the length of the chunk as the logical row count, 
# as every row was successfully evaluated by the DB without throwing an error.
    return len(chunk)


# ── Public API ────────────────────────────────────────────────────────────────

def export_to_database(
    df: pd.DataFrame,
    cfg: Optional[DBConfig] = None,
) -> int:
    """
    Upsert *df* into the configured MySQL table.

    Args:
        df:  DataFrame containing at minimum the METRICS_COLUMNS.
        cfg: Database configuration; loads from environment when omitted.

    Returns:
        Number of logical rows inserted or updated.

    Raises:
        SchemaValidationError: If required columns are absent.
        EnvironmentError:      If DATABASE_URL is not set and cfg is None.
        SQLAlchemyError:       On any database-level failure.
    """
    if df.empty:
        log.warning("DataFrame is empty — nothing to export.")
        return 0

    cfg = cfg or load_db_config()
    _validate_schema(df)
    metrics_df = _prepare_metrics_df(df)

    engine     = _get_engine(cfg)          # reuses cached engine / pool
    stmt       = _build_upsert_statement(list(metrics_df.columns), cfg.table_name)
    total_rows = 0
    n_chunks   = (len(metrics_df) - 1) // cfg.chunk_size + 1

    log.info(
        "Exporting %d row(s) to `%s` in %d chunk(s) of %d…",
        len(metrics_df), cfg.table_name, n_chunks, cfg.chunk_size,
    )

    try:
        with get_connection(engine) as conn:
            for i, start in enumerate(range(0, len(metrics_df), cfg.chunk_size), 1):
                chunk      = metrics_df.iloc[start : start + cfg.chunk_size]
                rows       = _upsert_chunk(conn, chunk, stmt)
                total_rows += rows
                log.debug("Chunk %d/%d → %d row(s) affected.", i, n_chunks, rows)

    except SQLAlchemyError as exc:
        log.error("Database export failed after %d row(s): %s", total_rows, exc)
        raise

    log.info("Export complete — %d logical row(s) upserted.", total_rows)
    return total_rows