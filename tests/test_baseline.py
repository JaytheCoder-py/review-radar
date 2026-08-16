"""The deterministic first stage, against real filings."""

from __future__ import annotations

import pytest

from reviewradar.extract.baseline import (
    AMBIGUOUS_ITEMS,
    DIAGNOSTIC_ITEMS,
    ITEM_ALIASES,
    ITEM_DESCRIPTIONS,
    NO_CONSEQUENCE_ITEMS,
    classify,
    parse_items,
)
from reviewradar.ingest.edgar import Submission
from reviewradar.types import EventType

# ----------------------------------------------------------------------------------
# Routing


def test_an_earnings_release_terminates_at_the_baseline(eliminated: Submission) -> None:
    # Item 2.02 alone. Correctly eliminated: an earnings release that refers back to a
    # split already effected announces nothing an index calculator must act on.
    result = classify(eliminated)
    assert result.items == {"2.02"}
    assert result.event_type is EventType.NO_INDEX_ACTION
    assert result.needs_model is False


def test_item_codes_alone_would_discard_a_real_split(false_elimination: Submission) -> None:
    """The two rungs of the baseline, on the filing that establishes why there are two.

    IntegraMed announced a 25% stock split effected as a stock dividend, by press
    release, under Item 7.01 alone - Regulation FD. Every item present carries no index
    consequence, so the *first* rung eliminates it, and a filing eliminated there is
    never seen again: not by the model, not by a queue, not by a person. The gold set
    puts a number on that: 6 of 28 index-relevant filings, 21.4%.

    That number was read for a while as the argument for the second stage. Measurement
    says otherwise. The words "25% stock split" are in the document, and a keyword screen
    over the body recovers all six - so 21.4% was an argument for *reading the body*,
    which is a regex problem (D-007, `test_keyword_screen.py`). Hence the assertion here:
    the item set still says discard, and the filing is routed to the model anyway.

    What the model is for is the field table on the scoreboard, where the baseline scores
    NaN on ex_date, ratio, counterparty and affected_securities because it never extracts
    a field - a screen can say "look at this", it cannot say "1-for-4, ex 13 April" with
    a citation. Detection is cheap; the fields are not.
    """
    result = classify(false_elimination)
    assert result.items == {"7.01"}
    assert result.items <= NO_CONSEQUENCE_ITEMS, "item codes alone would discard it"
    assert result.event_type is EventType.UNRESOLVED
    assert result.needs_model is True, "the screen rescued it; the item codes did not"
    assert "25% stock split" in false_elimination.full_text().lower()


def test_a_delisting_notice_is_typed_but_still_needs_its_fields(diagnostic: Submission) -> None:
    result = classify(diagnostic)
    assert result.event_type is EventType.DELISTING
    assert result.needs_model is True, "the effective date still has to come from somewhere"


def test_a_charter_amendment_is_unresolved(ambiguous: Submission) -> None:
    # Item 5.03 covers a fiscal-year change and a reverse split alike. No rule
    # separates them, which is the case the model exists for.
    result = classify(ambiguous)
    assert result.items == {"5.03"}
    assert result.event_type is EventType.UNRESOLVED
    assert result.needs_model is True


def test_an_other_events_filing_is_never_guessed_at(split: Submission) -> None:
    # Item 8.01 is the dumping ground: a real three-for-one split arrives under the
    # same code as a press release about a new hire.
    result = classify(split)
    assert "8.01" in result.items
    assert result.event_type is EventType.UNRESOLVED


def test_a_harmless_item_never_eliminates_the_event_beside_it(mixed: Submission) -> None:
    # D-004. This filing carries 3.01 (delisting) and 9.01 (exhibits). Eliminating on
    # the presence of 9.01 would drop the delisting, and the filing would simply never
    # appear in the queue - a failure nobody would notice.
    result = classify(mixed)
    assert result.items == {"3.01", "9.01"}
    assert "9.01" in NO_CONSEQUENCE_ITEMS
    assert result.event_type is EventType.DELISTING
    assert result.needs_model is True


def test_every_result_carries_a_rationale(eliminated: Submission, split: Submission) -> None:
    # The rationale is what a reviewer reads first; a result without one is unusable
    # in a queue.
    assert classify(eliminated).rationale.strip()
    assert classify(split).rationale.strip()


# ----------------------------------------------------------------------------------
# The item tables


def test_no_consequence_and_diagnostic_sets_are_disjoint() -> None:
    assert not (NO_CONSEQUENCE_ITEMS & DIAGNOSTIC_ITEMS.keys())


def test_no_consequence_and_ambiguous_sets_are_disjoint() -> None:
    assert not (NO_CONSEQUENCE_ITEMS & AMBIGUOUS_ITEMS)


def test_diagnostic_and_ambiguous_sets_are_disjoint() -> None:
    assert not (DIAGNOSTIC_ITEMS.keys() & AMBIGUOUS_ITEMS)


def test_every_routed_item_is_a_known_item() -> None:
    known = set(ITEM_DESCRIPTIONS)
    assert known >= NO_CONSEQUENCE_ITEMS
    assert DIAGNOSTIC_ITEMS.keys() <= known
    assert known >= AMBIGUOUS_ITEMS


def test_every_known_item_is_routed_somewhere() -> None:
    # An item in no table falls through to UNRESOLVED silently, which is the most
    # expensive kind of miss: it looks like a model decision when it was an omission.
    routed = NO_CONSEQUENCE_ITEMS | DIAGNOSTIC_ITEMS.keys() | AMBIGUOUS_ITEMS
    assert set(ITEM_DESCRIPTIONS) - routed == set()


def test_every_alias_points_at_a_real_item() -> None:
    assert set(ITEM_ALIASES.values()) <= set(ITEM_DESCRIPTIONS)


# ----------------------------------------------------------------------------------
# Description parsing


def test_a_leaked_sgml_tag_still_yields_its_item_number(
    monkeypatch: pytest.MonkeyPatch, eliminated: Submission
) -> None:
    # One filer agent emits `ITEM INFORMATION:  <ITEMS>1.05`. Malformed, but the
    # number is right there and refusing to read it helps nobody.
    monkeypatch.setattr(
        "reviewradar.extract.baseline.EdgarClient.header_items",
        staticmethod(lambda header: ("<ITEMS>1.05",)),
    )
    assert parse_items(eliminated).items == {"1.05"}


def test_a_reworded_description_is_matched_by_similarity(
    monkeypatch: pytest.MonkeyPatch, eliminated: Submission
) -> None:
    monkeypatch.setattr(
        "reviewradar.extract.baseline.EdgarClient.header_items",
        staticmethod(lambda header: ("Results of Operations and Financial Conditions",)),
    )
    parsed = parse_items(eliminated)
    assert parsed.items == {"2.02"}
    assert parsed.fuzzy, "a similarity match must be recorded, not absorbed silently"


def test_an_unrecognisable_description_is_reported_not_guessed(
    monkeypatch: pytest.MonkeyPatch, eliminated: Submission
) -> None:
    monkeypatch.setattr(
        "reviewradar.extract.baseline.EdgarClient.header_items",
        staticmethod(lambda header: ("Entirely Novel Regulatory Disclosure",)),
    )
    parsed = parse_items(eliminated)
    assert parsed.unmapped == ("Entirely Novel Regulatory Disclosure",)
    assert not parsed.items


def test_an_unmapped_description_forces_the_filing_to_the_model(
    monkeypatch: pytest.MonkeyPatch, eliminated: Submission
) -> None:
    # Degrading quietly is the failure mode that never gets noticed. An item the
    # tables do not recognise must not be treated as an item that does not matter.
    monkeypatch.setattr(
        "reviewradar.extract.baseline.EdgarClient.header_items",
        staticmethod(lambda header: ("Regulation FD Disclosure", "Something New")),
    )
    result = classify(eliminated)
    assert result.event_type is EventType.UNRESOLVED
    assert result.needs_model is True
    assert result.unmapped_descriptions == ("Something New",)
