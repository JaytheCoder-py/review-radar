"""The dashboard, and the boundary it is not allowed to cross."""

from __future__ import annotations

import pathlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import make_event
from reviewradar.ingest.store import EventStore
from reviewradar.service.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    db = tmp_path / "log.duckdb"
    with EventStore(db) as store:
        store.append([make_event()])
    return TestClient(create_app(db))


def test_healthz_reports_the_event_count(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "events": 1}


def test_the_events_page_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "0000320193-26-000001" in response.text


def test_a_missing_filing_is_a_page_not_a_crash(client: TestClient) -> None:
    assert client.get("/filing/0000000000-00-000000").status_code == 200


def test_the_failures_page_renders_when_empty(client: TestClient) -> None:
    assert client.get("/failures").status_code == 200


def test_company_names_are_escaped(tmp_path: Path) -> None:
    # Company names come from SEC filings, which are third-party text. Rendering them
    # unescaped would be an injection with a very respectable source.
    db = tmp_path / "log.duckdb"
    event = make_event()
    object.__setattr__(event, "company_name", "<script>alert(1)</script>")
    with EventStore(db) as store:
        store.append([event])
    body = TestClient(create_app(db)).get("/").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_the_service_layer_recomputes_no_stored_figure() -> None:
    """The dashboard renders figures; it does not derive them.

    A published number must have exactly one source. A dashboard that recomputes a rate
    from raw rows is a second source, and it will eventually disagree with the first -
    at which point nobody can say which is the published value.

    Same discipline as the miniftse ops desk, enforced the same way: by grepping for the
    arithmetic that would create one.
    """
    banned = re.compile(
        r"\*\s*100\b|/\s*100\b|\*\s*252\b|\*\*\s*0\.5|\bsqrt\(|\bmean\(\)|\bsum\(\)\s*/"
    )
    offenders = []
    for path in pathlib.Path("src/reviewradar/service").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line) and "# allow:" not in line:
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "the service layer re-derives a figure:\n" + "\n".join(offenders)
