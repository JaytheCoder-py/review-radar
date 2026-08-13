"""Routing, span validation, confidence, and the event that comes out.

The span rule (D-003) is the load-bearing part. A model told to cite will cite
*something*; the only question worth asking is whether the value it reported is actually
in the words it pointed at. A field that fails that test is **dropped**, not flagged -
flagging leaves a plausible wrong number in the record, and someone downstream will use it.

**On offsets.** Models are poor at character arithmetic and good at quotation. So the
citation is verified by *relocating* `span.text` verbatim in the named document, and the
offsets are corrected to wherever it was actually found. If the quoted text is not in the
document at all, the field is dropped. This keeps the citation load-bearing while not
throwing away correct extractions over off-by-forty offsets.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Final

from reviewradar.extract.baseline import BaselineResult, classify
from reviewradar.extract.llm import LlmClient, LlmResponse
from reviewradar.extract.schema import EVENT_SCHEMA, CorporateActionEvent, Extracted, Span
from reviewradar.ingest.edgar import Submission, SubmissionDocument
from reviewradar.treatment.rules import DEFAULT_THRESHOLD, decide
from reviewradar.types import RATIO_BEARING, EventType, Ratio

#: Ratios are written in words at least as often as in digits, and "three-for-one" has
#: to satisfy a check on 3/1. Twenty is the practical ceiling for split ratios.
NUMBER_WORDS: Final[dict[int, tuple[str, ...]]] = {
    1: ("one",),
    2: ("two",),
    3: ("three",),
    4: ("four",),
    5: ("five",),
    6: ("six",),
    7: ("seven",),
    8: ("eight",),
    9: ("nine",),
    10: ("ten",),
    11: ("eleven",),
    12: ("twelve",),
    13: ("thirteen",),
    14: ("fourteen",),
    15: ("fifteen",),
    16: ("sixteen",),
    17: ("seventeen",),
    18: ("eighteen",),
    19: ("nineteen",),
    20: ("twenty",),
    25: ("twenty-five", "twenty five"),
    30: ("thirty",),
    40: ("forty",),
    50: ("fifty",),
    100: ("hundred", "one hundred"),
}

MONTHS: Final[tuple[str, ...]] = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

#: Fields whose absence for a given event type is itself evidence of a bad extraction.
REQUIRED_FIELDS: Final[dict[EventType, tuple[str, ...]]] = {
    EventType.SPLIT_FORWARD: ("ratio",),
    EventType.SPLIT_REVERSE: ("ratio",),
    EventType.SPINOFF: ("ex_date",),
    EventType.SPECIAL_DIVIDEND: ("ex_date",),
    EventType.MERGER_COMPLETED: ("counterparty",),
}

_WS = re.compile(r"\s+")


def _flat(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


# ----------------------------------------------------------------------------------
# Span validation


def _relocate(span_text: str, doc: SubmissionDocument) -> tuple[int, int] | None:
    """Find `span_text` verbatim in `doc`, tolerating whitespace differences."""
    if not span_text.strip():
        return None
    idx = doc.text.find(span_text)
    if idx >= 0:
        return idx, idx + len(span_text)
    # Whitespace-insensitive fallback: build a regex that treats any run of
    # whitespace in the quotation as any run of whitespace in the document.
    pattern = r"\s+".join(re.escape(tok) for tok in span_text.split())
    match = re.search(pattern, doc.text, re.I)
    return (match.start(), match.end()) if match else None


def _value_supported(field_name: str, value: Any, quoted: str) -> bool:
    """Is `value` actually present in the words that were cited?"""
    flat = _flat(quoted)
    if field_name == "event_type":
        # A category is a judgement about the passage, not a quotation from it. The
        # span still has to be a real quotation - that is checked in validate_spans -
        # but requiring the literal string "split_forward" in a filing is nonsense.
        return True
    if field_name == "ex_date":
        assert isinstance(value, dt.date)
        if str(value.year) not in flat:
            return False
        month_ok = MONTHS[value.month - 1][:3] in flat or f"{value.month}/" in flat
        day_ok = re.search(rf"\b0?{value.day}\b", flat) is not None
        return month_ok and day_ok
    if field_name == "ratio":
        assert isinstance(value, Fraction)
        return all(
            re.search(rf"\b{n}\b", flat) is not None
            or any(w in flat for w in NUMBER_WORDS.get(n, ()))
            for n in (value.numerator, value.denominator)
        )
    if field_name == "affected_securities":
        assert isinstance(value, tuple)
        return all(str(t).lower() in flat for t in value)
    return _flat(str(value)) in flat


def _coerce(field_name: str, raw: Any) -> Any:
    """Parse a schema-valid string into its domain type. Raises on anything else."""
    if field_name == "event_type":
        return EventType(raw)
    if field_name == "ex_date":
        return dt.date.fromisoformat(str(raw))
    if field_name == "ratio":
        num, _, den = str(raw).partition("/")
        ratio = Ratio(int(num), int(den))
        if ratio <= 0:
            raise ValueError(f"non-positive ratio {raw!r}")
        return ratio
    if field_name == "affected_securities":
        return tuple(str(t).strip().upper() for t in raw if str(t).strip())
    return str(raw).strip()


def validate_spans(
    payload: Mapping[str, Any], documents: Sequence[SubmissionDocument]
) -> tuple[dict[str, Extracted[Any]], list[str]]:
    """Keep the fields whose citations hold up. Drop the rest.

    Returns `(kept, dropped_field_names)`. Never raises: a pathological payload produces
    an empty result and a full drop list, because one bad model response must not take
    down a nightly job.
    """
    by_id = {d.doc_id: d for d in documents}
    kept: dict[str, Extracted[Any]] = {}
    dropped: list[str] = []

    for name in ("event_type", "ex_date", "ratio", "counterparty", "affected_securities"):
        entry = payload.get(name)
        if entry is None:
            continue
        if not isinstance(entry, Mapping) or "value" not in entry:
            dropped.append(name)
            continue
        try:
            value = _coerce(name, entry["value"])
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            dropped.append(name)
            continue

        raw_span = entry.get("span")
        # Abstention has nothing to quote, and demanding a citation for "nothing
        # happened" would push every clean filing into the review queue.
        if name == "event_type" and value in (EventType.NO_INDEX_ACTION, EventType.UNRESOLVED):
            kept[name] = Extracted(value=value, span=None, source="model")
            continue
        if not isinstance(raw_span, Mapping):
            dropped.append(name)
            continue

        doc = by_id.get(str(raw_span.get("doc_id")))
        quoted = str(raw_span.get("text") or "")
        if doc is None:
            dropped.append(name)
            continue
        located = _relocate(quoted, doc)
        if located is None:
            dropped.append(name)
            continue
        start, end = located
        actual = doc.text[start:end]
        if not _value_supported(name, value, actual):
            dropped.append(name)
            continue
        kept[name] = Extracted(
            value=value,
            span=Span(doc_id=doc.doc_id, start=start, end=end, text=actual),
            source="model",
        )

    return kept, dropped


# ----------------------------------------------------------------------------------
# Confidence


def compute_confidence(
    *,
    baseline: BaselineResult,
    kept: Mapping[str, Extracted[Any]],
    dropped: Sequence[str],
    filed_at: dt.date,
) -> float:
    """Confidence from observable facts, never from the model's self-report.

    A model's stated confidence is not a calibrated probability and must not be treated
    as one (D-004). Four deductions, each of which is a thing that can be checked:

    * a field was dropped because its citation did not hold up
    * the model disagreed with a diagnostic item code
    * a field the event type requires is missing entirely
    * the ex-date precedes the filing - a split cannot go ex before it is announced
    """
    event = kept.get("event_type")
    if event is None:
        return 0.0
    event_type: EventType = event.value

    if event_type is EventType.NO_INDEX_ACTION:
        # Nothing to be wrong about, and a low score here would flood the queue with
        # exactly the filings the pipeline exists to clear.
        return 0.95 if not dropped else 0.85

    score = 1.0
    score -= 0.25 * len(dropped)

    disagreed = baseline.event_type is not event_type
    if disagreed and baseline.event_type not in (EventType.UNRESOLVED, EventType.NO_INDEX_ACTION):
        score -= 0.30

    for required in REQUIRED_FIELDS.get(event_type, ()):
        if required not in kept:
            score -= 0.20

    if event_type in RATIO_BEARING and "ratio" not in kept:
        score -= 0.10

    ex = kept.get("ex_date")
    if ex is not None and isinstance(ex.value, dt.date):
        if ex.value < filed_at:
            score -= 0.40
        elif (ex.value - filed_at).days > 365:
            score -= 0.20

    if event.span is None:
        score -= 0.30

    return max(0.0, min(1.0, round(score, 3)))


# ----------------------------------------------------------------------------------
# Orchestration

MAX_PROMPT_CHARS: Final = 60_000


def build_prompt(submission: Submission, baseline: BaselineResult) -> str:
    """The filing, as plain text, with its documents labelled by the ids spans must use."""
    parts = [
        f"Filing: {submission.ref.accession} ({submission.ref.company_name})",
        f"Filed: {submission.ref.filed_date.isoformat()}",
        f"8-K items present: {', '.join(sorted(baseline.items)) or 'none stated'}",
        "",
        "Documents follow. Cite spans using the doc_id shown in each header.",
    ]
    budget = MAX_PROMPT_CHARS
    for doc in submission.documents:
        if budget <= 0:
            break
        body = doc.text[:budget]
        budget -= len(body)
        parts.append(f"\n--- doc_id: {doc.doc_id} | type: {doc.doc_type} ---\n{body}")
    return "\n".join(parts)


def extract(
    submission: Submission,
    *,
    client: LlmClient,
    run_id: str,
    threshold: float = DEFAULT_THRESHOLD,
    baseline_only: bool = False,
) -> CorporateActionEvent:
    """Baseline, then the model on the residual, then treatment.

    `baseline_only` runs the deterministic stage alone. That is not a debug switch: it is
    how the scoreboard measures what the model is worth, so it has to produce a real
    event rather than a stub.
    """
    ref = submission.ref
    baseline = classify(submission)

    def build(
        event_type: Extracted[EventType],
        kept: Mapping[str, Extracted[Any]],
        dropped: Sequence[str],
        confidence: float,
        response: LlmResponse | None,
    ) -> CorporateActionEvent:
        return CorporateActionEvent(
            accession=ref.accession,
            cik=ref.cik,
            company_name=ref.company_name,
            filed_at=ref.filed_date,
            event_type=event_type,
            treatment=decide(event_type.value, confidence=confidence, threshold=threshold),
            confidence=confidence,
            run_id=run_id,
            ex_date=kept.get("ex_date"),
            ratio=kept.get("ratio"),
            counterparty=kept.get("counterparty"),
            affected_securities=kept.get("affected_securities"),
            dropped_fields=tuple(dropped),
            items=tuple(sorted(baseline.items)),
            rationale=baseline.rationale,
            input_tokens=response.input_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            latency_ms=response.latency_ms if response else 0.0,
        )

    if not baseline.needs_model:
        typed = Extracted(value=baseline.event_type, span=None, source="baseline")
        return build(typed, {}, (), 0.95, None)

    if baseline_only:
        # The honest baseline-only answer for a filing it cannot resolve. Reporting
        # UNRESOLVED here rather than guessing is what makes the delta meaningful.
        typed = Extracted(value=baseline.event_type, span=None, source="baseline")
        confidence = 0.95 if baseline.event_type is EventType.NO_INDEX_ACTION else 0.30
        return build(typed, {}, (), confidence, None)

    response = client.extract(build_prompt(submission, baseline), EVENT_SCHEMA)
    kept, dropped = validate_spans(response.payload, submission.documents)

    if "event_type" not in kept:
        # The model produced nothing usable about the one field that is required.
        # Fall back to the baseline's view and let the confidence say so.
        fallback = Extracted(value=baseline.event_type, span=None, source="baseline")
        return build(fallback, kept, [*dropped, "event_type"], 0.20, response)

    confidence = compute_confidence(
        baseline=baseline, kept=kept, dropped=dropped, filed_at=ref.filed_date
    )
    return build(kept["event_type"], kept, dropped, confidence, response)
