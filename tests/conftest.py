"""Shared fixtures.

Every fixture is a **real SEC filing**, committed verbatim. None is synthesised, because
the defects worth testing against - leaked SGML tags, drifting item wording, empty header
fields, splits announced only in an exhibit - are things nobody would think to invent.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from reviewradar.extract.schema import CorporateActionEvent, Extracted, Span
from reviewradar.ingest.edgar import EdgarClient, Submission
from reviewradar.types import EventType, Treatment, parse_accession, parse_cik

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


def _load(name: str) -> Submission:
    raw = (FIXTURES / name).read_text(encoding="utf-8")
    return EdgarClient.parse_submission(raw, ref=EdgarClient.ref_from_header(raw))


@pytest.fixture
def daily_index_raw() -> str:
    return (FIXTURES / "master.20250415.idx").read_text(encoding="latin-1")


@pytest.fixture
def eliminated() -> Submission:
    """Item 2.02 alone. An earnings release - correctly eliminated."""
    return _load("eliminated_8k.txt")


@pytest.fixture
def false_elimination() -> Submission:
    """A real 25% stock split, announced under Item 7.01 alone, and therefore eliminated.

    IntegraMed America, 2007. The baseline discards it. Kept as a fixture because it is
    the structural limit of item-code routing, stated as a test rather than a caveat.
    """
    return _load("false_elimination_8k.txt")


@pytest.fixture
def diagnostic() -> Submission:
    """Item 3.01 alone. A delisting notice: the item names the event type outright."""
    return _load("diagnostic_8k.txt")


@pytest.fixture
def ambiguous() -> Submission:
    """Item 5.03 alone. A charter amendment, which may or may not be a reverse split."""
    return _load("ambiguous_8k.txt")


@pytest.fixture
def mixed() -> Submission:
    """Items 3.01 and 9.01 together - the D-004 case.

    9.01 carries no index consequence on its own. Eliminating on its presence would drop
    the delisting notice filed beside it.
    """
    return _load("mixed_8k.txt")


@pytest.fixture
def split() -> Submission:
    """A real three-for-one split, announced only in the EX-99.1 press release.

    Warwick Valley Telephone, 2003. The primary document says almost nothing; reading it
    alone finds no event. This is the fixture that justifies following exhibits.
    """
    return _load("split_8k.txt")


def make_event(
    *,
    accession: str = "0000320193-26-000001",
    run_id: str = "r1",
    event_type: EventType = EventType.SPLIT_FORWARD,
    confidence: float = 0.9,
    supersedes: str | None = None,
    ex_date: dt.date | None = None,
) -> CorporateActionEvent:
    """A minimal event, for storage tests that do not care how it was extracted."""
    return CorporateActionEvent(
        accession=parse_accession(accession),
        cik=parse_cik("320193"),
        company_name="TEST CO",
        filed_at=dt.date(2026, 1, 5),
        event_type=Extracted(value=event_type, span=None, source="model"),
        treatment=Treatment.DIVISOR_ADJUST,
        confidence=confidence,
        run_id=run_id,
        ex_date=(
            Extracted(
                value=ex_date,
                span=Span(doc_id=f"{accession}:1", start=0, end=4, text="test"),
                source="model",
            )
            if ex_date
            else None
        ),
        supersedes=supersedes,
    )
