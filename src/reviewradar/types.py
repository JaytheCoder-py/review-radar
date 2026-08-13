"""Domain primitives.

The job of this module is to make a class of bug unrepresentable rather than merely
tested for.

Two rules hold across the codebase and are not enforced by the type system:

1. **Nothing outside this module constructs a `Cik` or an `Accession` directly.**
   `NewType` does nothing at runtime - `Cik("banana")` typechecks and runs - so the
   validation has to live at the boundary, in `parse_cik` and `parse_accession`, and
   everything downstream has to be able to assume it happened.

2. **A `Ratio` is never built from a float.** `Fraction(0.1)` is not `Fraction(1, 10)`;
   it is the exact binary double, 3602879701896397/36028797018963968. A 1-for-3 reverse
   split held as a float does not invert cleanly, and the residue lands in a divisor.
"""

from __future__ import annotations

import re
from enum import StrEnum
from fractions import Fraction
from typing import NewType

Cik = NewType("Cik", str)
"""SEC Central Index Key, canonically ten digits, zero-padded.

EDGAR is inconsistent about this: the daily index gives it unpadded, the submissions API
gives it padded, and the SGML header pads it. One canonical form, established here."""

Accession = NewType("Accession", str)
"""EDGAR accession number, canonically ``0000000000-00-000000``."""

Ratio = Fraction
"""A corporate-action ratio. Exact, always. See the module docstring.

A plain alias rather than ``type Ratio = Fraction``: the 3.12 keyword form builds a
``TypeAliasType``, which is not callable, and ``Ratio(1, 3)`` has to work."""


class EventType(StrEnum):
    """What happened.

    ``NO_INDEX_ACTION`` and ``UNRESOLVED`` are deliberately distinct. "I looked and there
    is nothing for an index calculator to do" and "I could not tell" are different
    outcomes, and collapsing them hides every classifier failure behind a reassuring
    label.
    """

    SPLIT_FORWARD = "split_forward"
    SPLIT_REVERSE = "split_reverse"
    SPECIAL_DIVIDEND = "special_dividend"
    MERGER_COMPLETED = "merger_completed"
    SPINOFF = "spinoff"
    RIGHTS_ISSUE = "rights_issue"
    DELISTING = "delisting"
    BANKRUPTCY = "bankruptcy"
    TICKER_CHANGE = "ticker_change"
    NAME_CHANGE = "name_change"
    NO_INDEX_ACTION = "no_index_action"
    UNRESOLVED = "unresolved"


#: Event types that imply a ratio. Used to validate extractions and gold labels: a
#: ratio attached to a bankruptcy is a parse error, not a fact.
RATIO_BEARING: frozenset[EventType] = frozenset(
    {
        EventType.SPLIT_FORWARD,
        EventType.SPLIT_REVERSE,
        EventType.RIGHTS_ISSUE,
        EventType.SPINOFF,
    }
)

#: Event types an index calculator must act on. Everything else is noise for our
#: purposes, however material it is to the company.
INDEX_RELEVANT: frozenset[EventType] = frozenset(
    {
        EventType.SPLIT_FORWARD,
        EventType.SPLIT_REVERSE,
        EventType.SPECIAL_DIVIDEND,
        EventType.MERGER_COMPLETED,
        EventType.SPINOFF,
        EventType.RIGHTS_ISSUE,
        EventType.DELISTING,
        EventType.BANKRUPTCY,
        EventType.TICKER_CHANGE,
        EventType.NAME_CHANGE,
    }
)


class Treatment(StrEnum):
    """What an index calculator must do about it.

    ``MANUAL_REVIEW`` is a first-class outcome, not an exception path. A system that
    always answers is a system that guesses, and on a corporate action a guess is a
    wrong divisor.
    """

    DIVISOR_ADJUST = "divisor_adjust"
    SHARES_UPDATE = "shares_update"
    PRICE_ADJUST = "price_adjust"
    REMOVE_CONSTITUENT = "remove_constituent"
    NO_ACTION = "no_action"
    MANUAL_REVIEW = "manual_review"


_CIK_DIGITS = re.compile(r"\A[0-9]{1,10}\Z")
_ACCESSION_DASHED = re.compile(r"\A([0-9]{10})-([0-9]{2})-([0-9]{6})\Z")
_ACCESSION_BARE = re.compile(r"\A([0-9]{10})([0-9]{2})([0-9]{6})\Z")


def parse_cik(raw: str) -> Cik:
    """Normalise a CIK to ten zero-padded digits.

    Raises ``ValueError`` naming the offending input. That naming is not politeness: at
    4am against 400 filings, "invalid CIK" without the value costs an hour.
    """
    if not _CIK_DIGITS.match(raw):
        raise ValueError(f"not a CIK: {raw!r} (expected 1-10 digits, no whitespace)")
    return Cik(raw.zfill(10))


def parse_accession(raw: str) -> Accession:
    """Normalise an accession number to ``0000000000-00-000000``.

    Accepts the dashed and bare forms, which EDGAR uses interchangeably - the daily
    index carries the dashed form in the path and the bare form in the directory name.
    """
    if _ACCESSION_DASHED.match(raw):
        return Accession(raw)
    if bare := _ACCESSION_BARE.match(raw):
        return Accession(f"{bare.group(1)}-{bare.group(2)}-{bare.group(3)}")
    raise ValueError(
        f"not an accession number: {raw!r} "
        "(expected 0000000000-00-000000 or the same 18 digits undashed)"
    )
