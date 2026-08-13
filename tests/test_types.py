"""Domain primitives.

The point of this module is to make a class of bug unrepresentable rather than
merely tested-for. Read each test as a claim about the domain, not as a claim
about Python.
"""

import pytest

from reviewradar.types import (
    Accession,
    Cik,
    EventType,
    Ratio,
    Treatment,
    parse_accession,
    parse_cik,
)


def test_ratio_is_exact_where_float_is_not() -> None:
    # This is the whole reason Ratio is a Fraction. A 1-for-3 reverse split
    # applied and then reversed must return exactly to par. Held as a float
    # it does not, and the error lands in a divisor.
    assert Ratio(1, 3) * 3 == 1
    assert Ratio(1, 10) + Ratio(2, 10) == Ratio(3, 10)
    assert 0.1 + 0.2 != 0.3  # for contrast


def test_cik_is_zero_padded_to_ten_digits() -> None:
    # EDGAR gives CIKs unpadded in the daily index and padded in the
    # submissions API. One canonical form, established at the boundary.
    assert parse_cik("320193") == Cik("0000320193")
    assert parse_cik("0000320193") == Cik("0000320193")


@pytest.mark.parametrize("bad", ["", "AAPL", "32019X", "12345678901", " 320193"])
def test_cik_rejects_anything_that_is_not_a_ten_digit_number(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_cik(bad)


def test_accession_normalises_the_dashless_form() -> None:
    assert parse_accession("000032019324000123") == Accession("0000320193-24-000123")
    assert parse_accession("0000320193-24-000123") == Accession("0000320193-24-000123")


@pytest.mark.parametrize("bad", ["", "0000320193-24", "not-an-accession", "0000320193240001234"])
def test_accession_rejects_malformed_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_accession(bad)


def test_the_error_message_names_the_offending_input() -> None:
    # A stack trace that says "invalid CIK" and not which one costs an hour
    # when it fires at 4am against 400 filings.
    with pytest.raises(ValueError, match="AAPL"):
        parse_cik("AAPL")


def test_unresolved_is_distinct_from_no_index_action() -> None:
    # "I looked and there is nothing to do" and "I could not tell" are
    # different outcomes. Collapsing them hides every classifier failure.
    assert EventType.UNRESOLVED is not EventType.NO_INDEX_ACTION


def test_manual_review_is_a_treatment_not_an_error() -> None:
    # Abstention is a first-class outcome, not an exception path.
    assert Treatment.MANUAL_REVIEW in Treatment
