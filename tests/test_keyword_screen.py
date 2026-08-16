"""The keyword rescue stage: the rung between item codes and the model.

These tests are the contract for a stage that does not exist yet; they fail at import
until it does.

Measured on the committed corpus (399 filings, 167 eliminated by item codes alone):

    keyword screen over the body, all patterns   rescues 6/6, eliminates  2.0%
    same, minus a naive `trading symbol` pattern rescues 5/6, eliminates 28.6%
    item codes alone                             rescues 0/6, eliminates 41.9%

Two findings fall out of that, and both belong in the design rather than a caveat:

1. A plain screen recovers five of the six known false eliminations - every one that
   would move a divisor, GE Vernova included. So "item codes miss 21.4%" was never on
   its own an argument for a model; it was an argument for reading the body text.
2. It costs about a third of the elimination rate, and one careless pattern costs all
   of it: `trading symbol` matches the cover-page table that every modern 8-K carries,
   firing on 80.2% of eliminated filings.

What the screen cannot do is the subject of `test_a_completed_split_is_not_rescued` -
the test that decides whether stage two is justified.

Contract surface, matching the item tables (D-003: data, not branches, so a reviewer
who does not read Python can check it):

    CORPORATE_ACTION_PATTERNS: dict[str, str]   # class name -> regex
    def screen(text: str) -> frozenset[str]     # which classes the text mentions

with `classify` consulting the screen before it eliminates. The behaviour these tests
assert is the contract, not the spelling.
"""

from __future__ import annotations

import re

import pytest

from reviewradar.extract.baseline import (  # type: ignore[attr-defined]
    CORPORATE_ACTION_PATTERNS,
    classify,
    screen,
)
from reviewradar.ingest.edgar import Submission
from reviewradar.types import EventType

# ----------------------------------------------------------------------------------
# The rescue


def test_a_split_announced_under_regulation_fd_is_rescued(
    false_elimination: Submission,
) -> None:
    """IntegraMed's 25% stock split, filed under Item 7.01 alone.

    `test_baseline.py::test_item_codes_alone_would_discard_a_real_split` is the other
    half of this: it asserts that the item set alone still says discard. The words "25%
    stock split" are in the document, so the cheap stage catches it and the model is
    never asked to. Detection here was never the model's job; the fields are.
    """
    result = classify(false_elimination)
    assert result.needs_model is True, "the body says 'stock split'; do not discard it"
    assert result.event_type is not EventType.NO_INDEX_ACTION


def test_the_rescue_reads_exhibits_not_just_the_primary_document(
    split: Submission,
) -> None:
    """Warwick Valley Telephone, 2003 - a three-for-one split announced only in EX-99.1.

    The primary document says almost nothing. A screen that reads `documents[0]` and
    stops finds no event here, and the failure is invisible.
    """
    assert "three-for-one" in split.full_text().lower()
    assert screen(split.full_text()), "the exhibit carries the announcement"


# ----------------------------------------------------------------------------------
# The cost, which is the part that decides whether the stage is worth having


def test_a_completed_split_is_not_rescued(eliminated: Submission) -> None:
    """The deciding test: an earnings release that mentions a split already effected.

    Item 2.02 alone, correctly eliminated today - and the body contains the words
    "stock split", because the release refers back to one that has already happened.
    Nothing here obliges a calculator to do anything.

    A screen that matches on the phrase alone rescues this filing and every earnings
    release like it. On the corpus the `split` pattern fires on 32 of the 167 eliminated
    filings while rescuing 3 real events, and this is why.

    The distinction is tense: *effected in March* against *will be effected on 3 June*.
    Requiring forward-looking phrasing near the match is the obvious fix and it is worth
    trying - `compute_confidence` already treats an ex-date before the filing date as
    evidence of a bad extraction, which is the same idea one layer down.

    If this cannot be made to pass without either dropping real rescues or hand-tuning
    against this fixture, that failure is itself the finding - it goes in DECISIONS.md,
    measured. A screen that cannot tell a completed action from a prospective one is the
    argument for stage two that the 21.4% was never able to make on its own.
    """
    result = classify(eliminated)
    assert result.items == {"2.02"}
    assert "stock split" in eliminated.full_text().lower()
    assert result.needs_model is False, "the split already happened; there is nothing to do"


def test_the_cover_page_symbol_table_does_not_trigger_the_screen() -> None:
    """Every 8-K since 2019 carries this table. Matching it screens the entire corpus.

    Kept as a unit test on the text rather than a fixture, because the point is the
    pattern, not any one filing.
    """
    cover_page = (
        "Securities registered pursuant to Section 12(b) of the Act:\n"
        "Title of each class | Trading Symbol(s) | Name of each exchange on which registered\n"
        "Common Stock, $0.01 par value | ACME | The Nasdaq Stock Market LLC"
    )
    assert not screen(cover_page), "a listed ticker is not a ticker change"


def test_a_genuine_symbol_change_still_screens_positive() -> None:
    announcement = (
        "Effective at the open of trading on June 3, 2026, the Company's common stock "
        "will begin trading under the new ticker symbol 'ACME'."
    )
    assert screen(announcement), "the tightened pattern must still catch the real thing"


# ----------------------------------------------------------------------------------
# Reviewability


def test_the_patterns_are_data_a_reviewer_can_read() -> None:
    """Same argument as the item tables. A screen buried in branches cannot be audited."""
    assert isinstance(CORPORATE_ACTION_PATTERNS, dict)
    assert CORPORATE_ACTION_PATTERNS, "an empty table is not a screen"
    for name, pattern in CORPORATE_ACTION_PATTERNS.items():
        assert isinstance(name, str) and name
        re.compile(pattern)  # raises on a malformed pattern, at import time in CI


def test_a_rescued_filing_says_which_pattern_rescued_it(
    false_elimination: Submission,
) -> None:
    """A rescue with no stated reason is indistinguishable from a routing bug."""
    rationale = classify(false_elimination).rationale.lower()
    fired = screen(false_elimination.full_text())
    assert any(name.lower() in rationale for name in fired), rationale


@pytest.mark.parametrize("fixture", ["diagnostic", "mixed"])
def test_the_screen_never_downgrades_a_filing_already_bound_for_the_model(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    """The screen may only ever rescue. It must not be able to eliminate."""
    submission: Submission = request.getfixturevalue(fixture)
    assert classify(submission).needs_model is True
