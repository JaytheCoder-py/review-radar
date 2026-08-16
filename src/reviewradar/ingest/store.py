"""The append-only event log.

Nothing here updates or deletes (D-001). A corrected extraction is a new row carrying
`supersedes`; the row it replaces stays, because what the system *said* is itself a
record and a downstream consumer may have acted on it.

Idempotency comes from `CorporateActionEvent.event_id`, a content hash of the conclusion
plus the run that reached it. Re-running the same run over the same filing produces the
same id and inserts once; a genuinely new run produces a different id and inserts
alongside. No `SELECT` before every insert, and no unique-constraint gymnastics.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from reviewradar.extract.schema import CorporateActionEvent
from reviewradar.types import Accession

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id            VARCHAR PRIMARY KEY,
    accession           VARCHAR NOT NULL,
    cik                 VARCHAR NOT NULL,
    company_name        VARCHAR,
    filed_at            DATE    NOT NULL,
    event_type          VARCHAR NOT NULL,
    event_type_source   VARCHAR NOT NULL,
    treatment           VARCHAR NOT NULL,
    confidence          DOUBLE  NOT NULL,
    ex_date             DATE,
    ratio               VARCHAR,
    counterparty        VARCHAR,
    affected_securities VARCHAR,
    dropped_fields      VARCHAR,
    items               VARCHAR,
    rationale           VARCHAR,
    spans               VARCHAR,
    run_id              VARCHAR NOT NULL,
    supersedes          VARCHAR,
    input_tokens        BIGINT  DEFAULT 0,
    output_tokens       BIGINT  DEFAULT 0,
    latency_ms          DOUBLE  DEFAULT 0.0,
    recorded_at         TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS failures (
    accession   VARCHAR NOT NULL,
    stage       VARCHAR NOT NULL,
    error       VARCHAR NOT NULL,
    run_id      VARCHAR NOT NULL,
    recorded_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS ingested (
    on_date    DATE PRIMARY KEY,
    run_id     VARCHAR NOT NULL,
    n_filings  BIGINT  NOT NULL,
    completed_at TIMESTAMP DEFAULT current_timestamp
);

-- Forward-verification verdicts. A second append-only table, never a column on
-- `events`: the extraction is what the system said, the verdict is what the market
-- later said about it, and writing the second into the first would rewrite the record
-- D-001 exists to protect. Keyed on `event_id` rather than accession, because a filing
-- may carry several extractions and a verdict belongs to the row it tested (D-008).
CREATE SEQUENCE IF NOT EXISTS verification_seq;

CREATE TABLE IF NOT EXISTS verifications (
    verification_id VARCHAR PRIMARY KEY,
    -- Insertion order, not the wall clock. "The latest verdict" has to mean the one
    -- appended last: two verdicts recorded in the same second would otherwise order
    -- arbitrarily, and a clock that steps backwards would reorder history.
    seq             BIGINT  DEFAULT nextval('verification_seq'),
    event_id        VARCHAR NOT NULL,
    accession       VARCHAR NOT NULL,
    ticker          VARCHAR,
    verdict         VARCHAR NOT NULL,
    reason          VARCHAR,
    ex_date         DATE,
    session         DATE,
    prior_session   DATE,
    prior_close     DOUBLE,
    post_close      DOUBLE,
    expected_step   VARCHAR,
    observed_step   DOUBLE,
    relative_error  DOUBLE,
    tolerance       DOUBLE,
    price_source    VARCHAR NOT NULL,
    run_id          VARCHAR NOT NULL,
    checked_at      TIMESTAMP DEFAULT current_timestamp
);
"""


class EventStore:
    """DuckDB-backed, append-only."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute(_SCHEMA)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ----------------------------------------------------------------

    def append(self, events: Sequence[CorporateActionEvent]) -> int:
        """Insert events, skipping ids already present. Returns the number inserted."""
        if not events:
            return 0
        rows = [e.to_row() for e in events]
        columns = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in columns)
        # ON CONFLICT DO NOTHING, not DO UPDATE. An id collision means this exact
        # conclusion from this exact run is already recorded; there is nothing to change.
        sql = (
            f"INSERT INTO events ({', '.join(columns)}) VALUES ({placeholders}) "
            "ON CONFLICT (event_id) DO NOTHING"
        )
        before = self._count("events")
        self._con.executemany(sql, [[row[c] for c in columns] for row in rows])
        return self._count("events") - before

    def record_failure(
        self, accession: Accession | str, stage: str, error: str, run_id: str
    ) -> None:
        """A filing that could not be processed. Never skip silently.

        An absent row is indistinguishable from "this filing carried no event", and that
        ambiguity is how a missed split reaches production.
        """
        self._con.execute(
            "INSERT INTO failures (accession, stage, error, run_id) VALUES (?, ?, ?, ?)",
            [str(accession), stage, error[:2000], run_id],
        )

    def append_verifications(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Insert forward-verification verdicts. Returns the number inserted.

        Rows rather than a domain object, so that `ingest` keeps knowing nothing about
        `evals`: persistence is this module's job, the verdict type is
        `evals.forward.Verification`'s, and the dependency does not need to point
        upwards for both to be true.

        Idempotent on `verification_id`, which hashes the verdict and its evidence. A
        weekly re-check that reaches the same answer from the same session collides and
        inserts once; a verdict that has genuinely changed inserts alongside the old one.
        """
        if not rows:
            return 0
        columns = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in columns)
        sql = (
            f"INSERT INTO verifications ({', '.join(columns)}) VALUES ({placeholders}) "
            "ON CONFLICT (verification_id) DO NOTHING"
        )
        before = self._count("verifications")
        self._con.executemany(sql, [[row[c] for c in columns] for row in rows])
        return self._count("verifications") - before

    def mark_ingested(self, on: dt.date, run_id: str, n_filings: int) -> None:
        self._con.execute(
            "INSERT INTO ingested (on_date, run_id, n_filings) VALUES (?, ?, ?) "
            "ON CONFLICT (on_date) DO NOTHING",
            [on, run_id, n_filings],
        )

    # -- reads -----------------------------------------------------------------

    def already_ingested(self, on: dt.date) -> bool:
        row = self._con.execute("SELECT 1 FROM ingested WHERE on_date = ?", [on]).fetchone()
        return row is not None

    def events(self, since: dt.date | None = None, *, current_only: bool = False) -> pd.DataFrame:
        """All events, newest filing first.

        `current_only` hides rows that a later row supersedes. The superseded rows are
        still there - this is a view, not a delete.
        """
        sql = "SELECT * FROM events"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE filed_at >= ?"
            params.append(since)
        if current_only:
            sql += (" AND " if since is not None else " WHERE ") + (
                "event_id NOT IN (SELECT supersedes FROM events WHERE supersedes IS NOT NULL)"
            )
        sql += " ORDER BY filed_at DESC, accession"
        return self._con.execute(sql, params).df()

    def latest_per_accession(self) -> list[dict[str, Any]]:
        """The newest non-superseded row for each accession, as plain dicts.

        The forward verifier's input. Rows rather than a DataFrame on purpose: pandas
        renders a missing date as `NaT` and a missing string as `nan`, and a verifier
        whose whole job is to distinguish "no ex-date was extracted" from "the ex-date
        has not passed" cannot afford a `None` that is three different objects. DuckDB
        hands back `None`.
        """
        cursor = self._con.execute(
            """
            SELECT * FROM events
            WHERE event_id NOT IN (
                SELECT supersedes FROM events WHERE supersedes IS NOT NULL
            )
            QUALIFY row_number() OVER (
                PARTITION BY accession ORDER BY recorded_at DESC, event_id
            ) = 1
            ORDER BY accession
            """
        )
        columns = [description[0] for description in cursor.description or []]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def verifications(self, *, latest_only: bool = True) -> pd.DataFrame:
        """Forward-verification verdicts, newest first.

        `latest_only` shows one verdict per extraction. The earlier verdicts are still
        there - a claim that went from `unverifiable` to `verified` when its ex-date
        passed is a history worth keeping, and this is a view, not a delete.
        """
        sql = "SELECT * FROM verifications"
        if latest_only:
            sql += " QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY seq DESC) = 1"
        return self._con.execute(sql + " ORDER BY seq DESC").df()

    def verification_counts(self) -> dict[str, int]:
        """Latest verdict per extraction, counted.

        Computed here rather than in the dashboard: a published figure has exactly one
        source, and the service layer renders.
        """
        rows = self._con.execute(
            """
            SELECT verdict, count(*) FROM (
                SELECT * FROM verifications
                QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY seq DESC) = 1
            ) GROUP BY verdict
            """
        ).fetchall()
        return {str(verdict): int(count) for verdict, count in rows}

    def failures(self, since: dt.date | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM failures"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE recorded_at >= ?"
            params.append(since)
        return self._con.execute(sql + " ORDER BY recorded_at DESC", params).df()

    def runs(self) -> pd.DataFrame:
        return self._con.execute("SELECT * FROM ingested ORDER BY on_date DESC").df()

    def _count(self, table: str) -> int:
        row = self._con.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
