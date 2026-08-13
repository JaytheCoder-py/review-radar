"""EDGAR ingestion, against real committed filings."""

from __future__ import annotations

import datetime as dt

import pytest

from reviewradar.ingest.edgar import EdgarClient, EdgarError, Submission, normalise

# ----------------------------------------------------------------------------------
# The client refuses to be impolite


def test_client_refuses_to_construct_without_a_contact() -> None:
    # The SEC blocks IPs that omit a contact User-Agent. Making the client
    # unconstructible is cheaper than discovering that in production.
    with pytest.raises(ValueError, match="contact"):
        EdgarClient(contact="")


def test_client_refuses_a_contact_that_is_not_an_address() -> None:
    with pytest.raises(ValueError):
        EdgarClient(contact="jason")


def test_client_declares_its_contact_in_the_user_agent() -> None:
    client = EdgarClient(contact="jason@example.com")
    assert "jason@example.com" in client.user_agent


def test_client_refuses_a_rate_above_the_published_limit() -> None:
    with pytest.raises(ValueError, match="10"):
        EdgarClient(contact="jason@example.com", requests_per_second=25)


def test_rate_limiter_spaces_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    now = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        slept.append(seconds)

    monkeypatch.setattr("time.sleep", fake_sleep)
    client = EdgarClient(contact="jason@example.com", requests_per_second=4.0)
    for _ in range(3):
        client._throttle()
    assert slept and all(s >= 0.25 for s in slept), slept


# ----------------------------------------------------------------------------------
# The daily index


def test_daily_index_parses_the_committed_fixture(daily_index_raw: str) -> None:
    refs = EdgarClient.parse_daily_index(daily_index_raw, on=dt.date(2025, 4, 15))
    assert refs
    assert all(r.form_type == "8-K" for r in refs)
    assert all(len(r.cik) == 10 for r in refs)
    assert all(r.filed_date == dt.date(2025, 4, 15) for r in refs)
    assert all(r.submission_url.startswith("https://www.sec.gov/Archives/") for r in refs)


def test_daily_index_filters_by_form_type(daily_index_raw: str) -> None:
    # The fixture deliberately contains 10-Q rows as well.
    eight_k = EdgarClient.parse_daily_index(daily_index_raw, on=dt.date(2025, 4, 15))
    ten_q = EdgarClient.parse_daily_index(
        daily_index_raw, on=dt.date(2025, 4, 15), form_type="10-Q"
    )
    assert eight_k and ten_q
    assert not {r.accession for r in eight_k} & {r.accession for r in ten_q}


def test_daily_index_without_a_header_rule_raises() -> None:
    with pytest.raises(EdgarError, match="header rule"):
        EdgarClient.parse_daily_index("nothing useful here", on=dt.date(2025, 4, 15))


# ----------------------------------------------------------------------------------
# Submissions


def test_submission_separates_primary_from_exhibits(split: Submission) -> None:
    assert not split.primary().is_exhibit
    assert any(d.doc_type.startswith("EX-") for d in split.exhibits())
    assert split.primary() not in split.exhibits()


def test_every_document_has_a_unique_doc_id(split: Submission) -> None:
    ids = [d.doc_id for d in split.documents]
    assert len(ids) == len(set(ids))


def test_exhibits_are_parsed_and_carry_the_announcement(split: Submission) -> None:
    # Warwick Valley Telephone, 2003. The body refers to the split and defers the
    # detail to the press release - "the press release is included as Exhibit 99.1".
    # An extractor that stops at the primary document gets the event but not the
    # record date, which is the field an index calculator actually needs.
    exhibits = split.exhibits()
    assert exhibits, "the EX-99.1 press release was not parsed"
    assert any("three-for-one stock split" in d.text.lower() for d in exhibits)
    assert any("record date" in d.text.lower() for d in exhibits)


def test_a_truncated_submission_raises_rather_than_returning_empty() -> None:
    # An empty document list is indistinguishable from "this filing had no documents",
    # and that ambiguity is how a missed split reaches production.
    with pytest.raises(EdgarError):
        EdgarClient.parse_submission("<SEC-DOCUMENT>truncated")


def test_xbrl_and_spreadsheet_documents_are_skipped(eliminated: Submission) -> None:
    assert all(
        not d.doc_type.upper().startswith(("EX-101", "XML", "EXCEL", "GRAPHIC"))
        for d in eliminated.documents
    )


# ----------------------------------------------------------------------------------
# A filing is self-describing


def test_ref_is_reconstructed_from_the_submission_header(split: Submission) -> None:
    assert split.ref.accession == "0000950152-03-008360"
    assert split.ref.cik == "0000104777"
    assert split.ref.filed_date == dt.date(2003, 9, 19)
    assert "WARWICK" in split.ref.company_name.upper()


def test_a_header_without_an_accession_raises() -> None:
    with pytest.raises(EdgarError, match="FilingRef"):
        EdgarClient.ref_from_header("CONFORMED SUBMISSION TYPE:\t8-K\n<DOCUMENT>")


def test_header_items_does_not_run_past_the_end_of_its_line() -> None:
    # Older filings leave ITEM INFORMATION empty. A `\s*` capture crosses the newline
    # and reports the next header line as an 8-K item; 2003-2004 filings surfaced
    # "FILED AS OF DATE" as an unrecognised item because of exactly this.
    header = "ITEM INFORMATION:\nFILED AS OF DATE:\t\t20040722\n"
    assert EdgarClient.header_items(header) == ()


def test_header_items_returns_descriptions_not_numbers(diagnostic: Submission) -> None:
    # The finding that shaped the classifier: the header carries the item *description*.
    items = EdgarClient.header_items(diagnostic.header)
    assert items
    assert not any(i.strip()[:1].isdigit() for i in items)


# ----------------------------------------------------------------------------------
# Normalisation, on which every span depends


def test_normalise_strips_tags_and_decodes_entities_once() -> None:
    assert normalise("<p>three-for-one &amp; done</p>") == "three-for-one & done"
    # Decoding twice would turn a filing's literal "&amp;lt;" into a bracket.
    assert normalise("<p>&amp;lt;</p>") == "&lt;"


def test_normalise_folds_typographic_characters() -> None:
    # A smart quote inside a cited span turns a correct citation into a substring
    # match that silently fails.
    smart = "<p>WVT" + chr(0x2019) + "s board " + chr(0x2014) + " today</p>"
    assert normalise(smart) == "WVT's board - today"


def test_normalise_drops_script_and_style_bodies() -> None:
    assert "hidden" not in normalise("<style>.x{content:'hidden'}</style><p>shown</p>")


def test_normalise_is_idempotent(split: Submission) -> None:
    # Spans recorded tonight must resolve to the same characters next month.
    once = split.primary().text
    assert normalise(once) == once
