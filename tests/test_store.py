"""The append-only event log."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import make_event
from reviewradar.ingest.store import EventStore
from reviewradar.types import EventType


def test_a_failure_is_recorded_not_swallowed(tmp_path: Path) -> None:
    # An absent row is indistinguishable from "this filing carried no event".
    with EventStore(tmp_path / "log.duckdb") as store:
        store.record_failure("0000320193-26-000001", "parse", "truncated submission", "r1")
        failures = store.failures()
        assert len(failures) == 1
        assert "truncated" in failures.iloc[0]["error"]
        assert failures.iloc[0]["stage"] == "parse"


def test_marking_a_date_ingested_makes_it_idempotent(tmp_path: Path) -> None:
    with EventStore(tmp_path / "log.duckdb") as store:
        assert not store.already_ingested(dt.date(2026, 8, 12))
        store.mark_ingested(dt.date(2026, 8, 12), run_id="r1", n_filings=417)
        assert store.already_ingested(dt.date(2026, 8, 12))
        # A second mark for the same date must not create a second row.
        store.mark_ingested(dt.date(2026, 8, 12), run_id="r2", n_filings=1)
        assert len(store.runs()) == 1


def test_appending_the_same_event_twice_stores_it_once(tmp_path: Path) -> None:
    with EventStore(tmp_path / "log.duckdb") as store:
        event = make_event()
        assert store.append([event]) == 1
        assert store.append([event]) == 0
        assert len(store.events()) == 1


def test_a_superseding_event_does_not_delete_its_predecessor(tmp_path: Path) -> None:
    # D-001. What the system said yesterday is itself a record: a downstream consumer
    # may have acted on it, and rewriting it makes that decision unauditable.
    with EventStore(tmp_path / "log.duckdb") as store:
        first = make_event(run_id="r1")
        store.append([first])
        corrected = make_event(run_id="r2", supersedes=first.event_id)
        store.append([corrected])

        rows = store.events()
        assert len(rows) == 2
        assert set(rows["run_id"]) == {"r1", "r2"}

        current = store.events(current_only=True)
        assert len(current) == 1
        assert current.iloc[0]["run_id"] == "r2"


def test_a_different_conclusion_from_the_same_run_is_a_different_row(tmp_path: Path) -> None:
    with EventStore(tmp_path / "log.duckdb") as store:
        store.append([make_event(event_type=EventType.SPLIT_FORWARD)])
        store.append([make_event(event_type=EventType.SPLIT_REVERSE)])
        assert len(store.events()) == 2


def test_appending_nothing_is_not_an_error(tmp_path: Path) -> None:
    with EventStore(tmp_path / "log.duckdb") as store:
        assert store.append([]) == 0


def test_events_can_be_filtered_by_filing_date(tmp_path: Path) -> None:
    with EventStore(tmp_path / "log.duckdb") as store:
        store.append([make_event()])
        assert len(store.events(since=dt.date(2025, 1, 1))) == 1
        assert len(store.events(since=dt.date(2027, 1, 1))) == 0


@settings(deadline=None, max_examples=10)
@given(st.integers(min_value=1, max_value=5))
def test_reappending_n_times_is_a_no_op(n: int) -> None:
    # Idempotency is a property of the event id, not of a SELECT before every insert.
    # A fresh directory per example, because hypothesis reuses the function body and a
    # pytest tmp_path fixture would be shared across examples.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp, EventStore(Path(tmp) / "log.duckdb") as store:
        events = [make_event()]
        for _ in range(n):
            store.append(events)
        assert len(store.events()) == 1
