"""The deterministic first stage.

Its job is **cheap elimination, not classification**. Most 8-K traffic carries no index
consequence at all, and every filing this stage terminates is a filing the model is never
paid to read. What survives is the residual, and the residual is what the scoreboard
measures the model against.

Three tables, all data rather than branches, so they can be reviewed by someone who does
not read Python:

* ``NO_CONSEQUENCE_ITEMS`` - items that can never oblige an index calculator to act.
* ``DIAGNOSTIC_ITEMS``     - items that identify the event type outright. Fields still
                             have to be extracted, so these are not terminal.
* ``AMBIGUOUS_ITEMS``      - items that may or may not carry an event. The model decides.

Item numbers are recovered from the header's ``ITEM INFORMATION:`` description strings,
which is the only reliable source - see D-003 and
``memos/W2_what_the_8k_header_actually_contains.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from reviewradar.ingest.edgar import EdgarClient, Submission
from reviewradar.types import EventType

# ----------------------------------------------------------------------------------
# Item number <- description string
#
# EDGAR generates these strings from a fixed vocabulary, so the mapping is exact rather
# than heuristic. Matching is done on a normalised form (lowercased, punctuation
# stripped, whitespace collapsed) because filer agents vary in punctuation and the SEC
# has revised the wording of several items over time.

ITEM_DESCRIPTIONS: Final[dict[str, str]] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": (
        "Creation of a Direct Financial Obligation or an Obligation under an "
        "Off-Balance Sheet Arrangement of a Registrant"
    ),
    "2.04": (
        "Triggering Events That Accelerate or Increase a Direct Financial Obligation or "
        "an Obligation under an Off-Balance Sheet Arrangement"
    ),
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": (
        "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; "
        "Transfer of Listing"
    ),
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": (
        "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report "
        "or Completed Interim Review"
    ),
    "5.01": "Changes in Control of Registrant",
    "5.02": (
        "Departure of Directors or Certain Officers; Election of Directors; Appointment "
        "of Certain Officers: Compensatory Arrangements of Certain Officers"
    ),
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": ("Temporary Suspension of Trading Under Registrant's Employee Benefit Plans"),
    "5.05": (
        "Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics"
    ),
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or Other External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "6.06": "Static Pool",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

#: Wordings observed in real filings that differ from the SEC's current text. Kept
#: explicit rather than fuzzy-matched, because each one was found by a filing failing
#: to map and being looked at. Singular/plural drift and inserted articles dominate.
ITEM_ALIASES: Final[dict[str, str]] = {
    "Material Modifications to Rights of Security Holders": "3.03",
    "Shareholder Nominations Pursuant to Exchange Act Rule 14a-11": "5.08",
    "Amendments to the Registrant's Code of Ethics, or Waiver of a Provision of the "
    "Code of Ethics": "5.05",
    "Cost Associated with Exit or Disposal Activities": "2.05",
    "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; "
    "Transfer of Listing.": "3.01",
    # Pre-2004 wordings. EDGAR full-text search reaches back to 2001, so a corpus
    # sampled by phrase pulls in filings written against the older item vocabulary.
    "Departure of Directors or Principal Officers; Election of Directors; Appointment "
    "of Principal Officers": "5.02",
    "Acquisition or disposition of assets": "2.01",
    "Financial statements and exhibits": "9.01",
    "Other events": "8.01",
    "Regulation FD Disclosure.": "7.01",
}

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")
#: One filer agent leaks the raw SGML tag into the description field, producing
#: `ITEM INFORMATION:  <ITEMS>1.05`. Malformed, but real, and the number is right there.
_LEAKED_TAG = re.compile(r"<ITEMS>\s*([0-9]\.[0-9]{2})")

#: Similarity above which a near-miss is accepted rather than reported as unmapped.
#: Set high on purpose: a wrong item number routes a real event into the eliminated
#: pile, where nothing will ever look at it again.
_FUZZY_FLOOR: Final[float] = 0.85


def _norm(text: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


#: normalised description -> item number
_BY_DESCRIPTION: Final[dict[str, str]] = {
    **{_norm(desc): item for item, desc in ITEM_DESCRIPTIONS.items()},
    **{_norm(desc): item for desc, item in ITEM_ALIASES.items()},
}


def _similarity(a: str, b: str) -> float:
    """Jaccard over token sets. Cheap, symmetric, and explainable to a reviewer."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ----------------------------------------------------------------------------------
# Routing tables

#: Items that can never oblige an index calculator to act. A filing carrying only these
#: terminates here and never reaches the model.
NO_CONSEQUENCE_ITEMS: Final[frozenset[str]] = frozenset(
    {
        "1.04",  # mine safety
        "1.05",  # cybersecurity incident
        "2.02",  # results of operations
        "2.03",  # direct financial obligation
        "2.04",  # triggering events on an obligation
        "2.05",  # exit or disposal costs
        "2.06",  # material impairments
        "4.01",  # change of auditor
        "4.02",  # non-reliance on prior financials
        "5.02",  # directors and officers
        "5.04",  # blackout under benefit plans
        "5.05",  # code of ethics
        "5.07",  # shareholder vote - the outcome that matters gets its own filing
        "5.08",  # director nominations
        "6.01",  # ABS items: not equities
        "6.02",
        "6.03",
        "6.04",
        "6.05",
        "6.06",
        "7.01",  # Regulation FD
        "9.01",  # financial statements and exhibits - never an event by itself
    }
)

#: Items that identify the event type outright. Not terminal: the dates, ratios and
#: counterparties still have to come from somewhere.
DIAGNOSTIC_ITEMS: Final[dict[str, EventType]] = {
    "1.03": EventType.BANKRUPTCY,
    "2.01": EventType.MERGER_COMPLETED,
    "3.01": EventType.DELISTING,
}

#: Items that may or may not carry an index-relevant event. The model decides.
#:
#: 8.01 is the one that matters. Splits and special dividends overwhelmingly arrive
#: under "Other Events", which is a dumping ground carrying everything from a buyback
#: authorisation to a press release about a new hire. No rule resolves it; that is
#: precisely where the model earns its cost.
AMBIGUOUS_ITEMS: Final[frozenset[str]] = frozenset(
    {
        "1.01",  # material definitive agreement - merger agreements land here
        "1.02",  # termination of same
        "3.02",  # unregistered sales - changes shares outstanding
        "3.03",  # modification to rights of security holders
        "5.01",  # change in control
        "5.03",  # charter amendment - reverse splits need one
        "5.06",  # shell company status
        "8.01",  # other events
    }
)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """What the deterministic stage concluded, and why."""

    event_type: EventType
    items: frozenset[str]
    unmapped_descriptions: tuple[str, ...]
    rationale: str
    needs_model: bool


@dataclass(frozen=True, slots=True)
class ItemParse:
    """Item numbers recovered from a header, and how each one was recovered."""

    items: frozenset[str]
    unmapped: tuple[str, ...]
    #: (description, item, similarity) for descriptions matched by similarity rather
    #: than exactly. Carried so that wording drift is visible in monitoring instead of
    #: being absorbed silently.
    fuzzy: tuple[tuple[str, str, float], ...]


def parse_items(submission: Submission) -> ItemParse:
    """Item numbers from the header's description strings.

    Four steps, in descending order of confidence: the leaked-tag shortcut, exact match
    on the normalised description, the explicit alias table, then token similarity above
    `_FUZZY_FLOOR`. Anything below that is reported as unmapped rather than guessed.

    Unmapped descriptions are surfaced, not dropped. The SEC revises item wording, and a
    silently unrecognised label would degrade classification without anything failing -
    which is the failure mode that never gets noticed.
    """
    items: set[str] = set()
    unmapped: list[str] = []
    fuzzy: list[tuple[str, str, float]] = []

    for description in EdgarClient.header_items(submission.header):
        if leaked := _LEAKED_TAG.search(description):
            items.add(leaked.group(1))
            continue
        key = _norm(description)
        if key in _BY_DESCRIPTION:
            items.add(_BY_DESCRIPTION[key])
            continue
        best_item, best_score = "", 0.0
        for norm, item in _BY_DESCRIPTION.items():
            score = _similarity(key, norm)
            if score > best_score:
                best_item, best_score = item, score
        if best_score >= _FUZZY_FLOOR:
            items.add(best_item)
            fuzzy.append((description, best_item, round(best_score, 3)))
        else:
            unmapped.append(description)

    return ItemParse(frozenset(items), tuple(unmapped), tuple(fuzzy))


def classify(submission: Submission) -> BaselineResult:
    """Route a filing: terminate it, type it, or hand it to the model.

    The elimination rule is over the **whole item set** (D-004). A filing carrying both
    2.02 (results) and 2.01 (completed acquisition) is not eliminated, because companies
    routinely file both at once and eliminating on the harmless one drops the event
    sitting beside it.
    """
    parsed = parse_items(submission)
    items, unmapped = parsed.items, parsed.unmapped

    if not items and not unmapped:
        return BaselineResult(
            event_type=EventType.UNRESOLVED,
            items=items,
            unmapped_descriptions=unmapped,
            rationale="no ITEM INFORMATION in the header; cannot route on items alone",
            needs_model=True,
        )

    if unmapped:
        return BaselineResult(
            event_type=EventType.UNRESOLVED,
            items=items,
            unmapped_descriptions=unmapped,
            rationale=f"unrecognised item description(s): {'; '.join(unmapped)}",
            needs_model=True,
        )

    if items <= NO_CONSEQUENCE_ITEMS:
        return BaselineResult(
            event_type=EventType.NO_INDEX_ACTION,
            items=items,
            unmapped_descriptions=(),
            rationale=(f"every item carries no index consequence: {', '.join(sorted(items))}"),
            needs_model=False,
        )

    diagnostic = sorted(items & DIAGNOSTIC_ITEMS.keys())
    if len(diagnostic) == 1 and not (items & AMBIGUOUS_ITEMS):
        item = diagnostic[0]
        return BaselineResult(
            event_type=DIAGNOSTIC_ITEMS[item],
            items=items,
            unmapped_descriptions=(),
            rationale=f"item {item} identifies the event type; fields still required",
            needs_model=True,
        )

    ambiguous = sorted(items - NO_CONSEQUENCE_ITEMS)
    return BaselineResult(
        event_type=EventType.UNRESOLVED,
        items=items,
        unmapped_descriptions=(),
        rationale=f"item(s) {', '.join(ambiguous)} may carry an event; no rule resolves them",
        needs_model=True,
    )
