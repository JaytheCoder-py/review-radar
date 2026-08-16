"""The deterministic first stage.

Its job is **cheap elimination, not classification**. Most 8-K traffic carries no index
consequence at all, and every filing this stage terminates is a filing the model is never
paid to read. What survives is the residual, and the residual is what the scoreboard
measures the model against.

Two rungs, in order. The first routes on item codes:

* ``NO_CONSEQUENCE_ITEMS`` - items that can never oblige an index calculator to act.
* ``DIAGNOSTIC_ITEMS``     - items that identify the event type outright. Fields still
                             have to be extracted, so these are not terminal.
* ``AMBIGUOUS_ITEMS``      - items that may or may not carry an event. The model decides.

The second rung is ``CORPORATE_ACTION_PATTERNS``, a keyword screen over the whole body
that runs **before** an elimination is allowed to stand. Item codes alone discarded 6 of
28 index-relevant filings in the stratified gold set - 21.4%, GE Vernova included -
because a split or a spin-off is routinely announced under 7.01, 9.01 or 2.02 with the
event confined to the body or an attached press release (D-007).

Every table here is data rather than branches, so it can be reviewed by someone who does
not read Python.

Item numbers are recovered from the header's ``ITEM INFORMATION:`` description strings,
which is the only reliable source - see D-003 and
``memos/what_the_8k_header_actually_contains.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
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
_FUZZY_FLOOR: Final[float] = 0.90


def _norm(text: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


#: normalised description -> item number
_BY_DESCRIPTION: Final[dict[str, str]] = {
    **{_norm(desc): item for item, desc in ITEM_DESCRIPTIONS.items()},
    **{_norm(desc): item for desc, item in ITEM_ALIASES.items()},
}


def _similarity(a: str, b: str) -> float:
    """Character-level similarity, symmetric and explainable to a reviewer.

    Not Jaccard over tokens, which was the first attempt: it scores "Financial
    Condition" against "Financial Conditions" at 0.71, because a plural is a wholly
    different token. Wording drift in the SEC's item list is overwhelmingly singular
    versus plural and inserted articles, which is exactly what character-level
    matching sees and token matching cannot.
    """
    return SequenceMatcher(None, a, b).ratio()


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


# ----------------------------------------------------------------------------------
# The keyword rescue
#
# A screen over the full body text - primary document plus exhibits - consulted before
# an elimination is allowed to stand. It may only ever move a filing **back** to the
# model; it cannot eliminate, cannot type, and cannot lower a filing's routing. That
# asymmetry is the entire safety argument: a false positive costs one model call, a
# false negative costs an index print (D-007).
#
# Scope is deliberately narrow. The screen exists for the event classes item codes
# systematically miss - splits, spin-offs, special and stock dividends, ticker changes,
# rights offerings. Mergers are absent on purpose: a merger agreement is Item 1.01 and a
# completed acquisition is Item 2.01, neither of which is ever eliminated, so a merger
# pattern buys no recall and fires on every earnings release that mentions a pending
# deal.

#: Class name -> regex over the normalised body text. Matched case-insensitively.
#:
#: `ticker change` is the pattern that has to be written carefully. Every 8-K filed
#: since 2019 carries a "Trading Symbol(s)" cover-page table, so a bare `trading symbol`
#: fires on 141 of the 167 filings item codes eliminate - 84.4% - and the screen stops
#: screening. The pattern therefore requires a *change* - "new", "changed to",
#: "symbol ... from" - which a listing table never contains.
CORPORATE_ACTION_PATTERNS: Final[dict[str, str]] = {
    "stock split": (
        r"\b(?:stock|share)\s+splits?\b"
        r"|\bsplit\s+of\s+(?:its|the)\s+(?:issued\s+and\s+outstanding\s+)?common\s+(?:stock|shares)\b"
    ),
    "reverse split": r"\breverse\s+(?:stock\s+|share\s+)?splits?\b",
    # Ratios are written in words as often as in digits, and the words go up to twenty.
    "split ratio": (
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|\d{1,3})"
        r"[-\s]for[-\s]"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|\d{1,3})\b"
    ),
    "spin-off": r"\bspin[-\s]?offs?\b|\bspun[-\s]off\b",
    "special dividend": r"\b(?:special|extraordinary|one[-\s]time)\s+(?:cash\s+|stock\s+)?dividends?\b",
    "stock dividend": r"\bstock\s+dividends?\b",
    "rights offering": r"\brights\s+(?:offering|issue|issuance|distribution)\b",
    "ticker change": (
        r"\b(?:new|chang\w*|renam\w*)\s+(?:\S+\s+){0,3}?(?:ticker|trading|stock)\s+symbols?\b"
        r"|\b(?:ticker|trading|stock)\s+symbols?\s+(?:chang\w*|from)\b"
    ),
}

#: A match only counts when the passage around it also *announces* something. Without
#: this the table fires on 53 of the 167 filings item codes eliminate rather than 31,
#: `stock split` alone on 33 rather than 18, because every earnings release restates
#: prior periods "to reflect the three-for-one stock split in 2004". All six known
#: false eliminations survive the requirement.
#:
#: The discriminator is not the verb's tense but whether the sentence is a disclosure
#: act - a declaration, an approval, a dated step, or a future-tense consequence.
#: "Completed the previously announced separation" is past and still an announcement;
#: "adjusted to reflect the split in 2004" is neither. Same idea as `compute_confidence`
#: treating an ex-date before the filing date as evidence of a bad extraction.
ANNOUNCEMENT_CONTEXT: Final[str] = (
    r"\bannounc\w+\b"  # announced / announces / announcing / announcement
    r"|\bdeclar\w+\b"  # the board declared ...
    r"|\bauthoriz\w+\b|\bapprov\w+\b"
    r"|\bwill\b|\bshall\b|\bexpects?\s+to\b|\bintends?\s+to\b"
    r"|\bto\s+be\s+effective\b|\beffective\s+(?:on|as\s+of|at\s+the)\b"
    r"|\b(?:record|payable|payment|distribution|ex|ex-dividend|effective)\s+dates?\b"
    r"|\bboard\s+of\s+directors\b"
)

#: How far either side of a match the announcement context may be looked for. Capped so
#: that a numeric table, which can run for pages without a sentence-ending stop, cannot
#: reach into unrelated prose and validate itself.
_CONTEXT_CHARS: Final[int] = 200

#: Sentence-ish boundaries. A full stop before whitespace, a blank line, or a rule of
#: dashes - the last because pre-2005 filings are plain text and separate blocks with
#: one, not with punctuation.
_BOUNDARY = re.compile(r"[.!?](?=\s)|\n[ \t]*\n|-{3,}")

_COMPILED_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    name: re.compile(pattern, re.I) for name, pattern in CORPORATE_ACTION_PATTERNS.items()
}
_ANNOUNCEMENT = re.compile(ANNOUNCEMENT_CONTEXT, re.I)


def _announcement_near(text: str, start: int, end: int) -> bool:
    """Is the match at `[start, end)` inside a passage that announces something?

    The passage is the match's own sentence, clipped to `_CONTEXT_CHARS` either side.
    Sentence-scoped rather than window-scoped because the two readings of "stock split"
    are often one sentence apart: an earnings release says the split happened in 2004,
    then announces its quarterly figures in the very next breath.
    """
    before = text[max(0, start - _CONTEXT_CHARS) : start]
    after = text[end : end + _CONTEXT_CHARS]
    if opening := [m.end() for m in _BOUNDARY.finditer(before)]:
        before = before[opening[-1] :]
    if closing := _BOUNDARY.search(after):
        after = after[: closing.start()]
    return _ANNOUNCEMENT.search(before + text[start:end] + after) is not None


def screen(text: str) -> frozenset[str]:
    """Which corporate-action classes `text` actually announces.

    Empty is the common case and means only "no pattern fired" - never "no event".
    Nothing in this module may eliminate on that basis.
    """
    return frozenset(
        name
        for name, pattern in _COMPILED_PATTERNS.items()
        if any(_announcement_near(text, m.start(), m.end()) for m in pattern.finditer(text))
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

    Elimination is then checked a second time against the body text (D-007). The screen
    is consulted only on the branch that would terminate the filing, because it is only
    allowed to rescue - reaching it from any other branch could only make routing worse.
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
        if fired := screen(submission.full_text()):
            return BaselineResult(
                event_type=EventType.UNRESOLVED,
                items=items,
                unmapped_descriptions=(),
                rationale=(
                    f"item(s) {', '.join(sorted(items))} carry no index consequence, but the "
                    f"body announces {', '.join(sorted(fired))}; not eliminated"
                ),
                needs_model=True,
            )
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
