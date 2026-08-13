"""The extracted event, and the JSON Schema the model is held to.

Every model-derived field carries a `Span` into the normalised text of a specific
document, and the pipeline **drops** any field whose value does not appear inside its own
cited span (D-003). Dropping rather than flagging is deliberate: it forces the citation to
be load-bearing. A model told to cite will cite something; the only question that matters
is whether the value is actually in there.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Final, Literal

from reviewradar.types import Accession, Cik, EventType, Ratio, Treatment

Source = Literal["baseline", "model"]


@dataclass(frozen=True, slots=True)
class Span:
    """A character range in the normalised text of one document.

    `doc_id` is `"{accession}:{sequence}"`. Offsets index the normalised text produced by
    `ingest.edgar.normalise`, never the raw markup - see that module's docstring.
    """

    doc_id: str
    start: int
    end: int
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True, slots=True)
class Extracted[T]:
    """A value, where it came from, and the words it came from."""

    value: T
    span: Span | None
    source: Source

    def to_json(self) -> dict[str, Any]:
        value: Any = self.value
        if isinstance(value, Fraction):
            value = f"{value.numerator}/{value.denominator}"
        elif isinstance(value, dt.date):
            value = value.isoformat()
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, EventType):
            value = value.value
        return {
            "value": value,
            "span": self.span.to_json() if self.span else None,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CorporateActionEvent:
    """One filing's worth of conclusion."""

    accession: Accession
    cik: Cik
    company_name: str
    filed_at: dt.date
    event_type: Extracted[EventType]
    treatment: Treatment
    confidence: float
    run_id: str
    ex_date: Extracted[dt.date] | None = None
    ratio: Extracted[Ratio] | None = None
    counterparty: Extracted[str] | None = None
    affected_securities: Extracted[tuple[str, ...]] | None = None
    dropped_fields: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    rationale: str = ""
    supersedes: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    _event_id: str = field(default="", repr=False)

    @property
    def event_id(self) -> str:
        """Content hash of the conclusion, plus the run that reached it.

        Deterministic rather than a UUID, so re-running the same run over the same filing
        collides and inserts once, while a genuinely new run inserts a new row alongside
        the old one. That is what makes the log append-only *and* idempotent (D-001).
        """
        if self._event_id:
            return self._event_id
        payload = json.dumps(
            {
                "accession": self.accession,
                "run_id": self.run_id,
                "event_type": self.event_type.value.value,
                "treatment": self.treatment.value,
                "ex_date": self.ex_date.value.isoformat() if self.ex_date else None,
                "ratio": str(self.ratio.value) if self.ratio else None,
                "counterparty": self.counterparty.value if self.counterparty else None,
                "affected": (
                    list(self.affected_securities.value) if self.affected_securities else None
                ),
                "dropped": list(self.dropped_fields),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def to_row(self) -> dict[str, Any]:
        """Flat representation for the event log."""

        def unwrap(x: Extracted[Any] | None) -> Any:
            return x.to_json() if x else None

        return {
            "event_id": self.event_id,
            "accession": self.accession,
            "cik": self.cik,
            "company_name": self.company_name,
            "filed_at": self.filed_at,
            "event_type": self.event_type.value.value,
            "event_type_source": self.event_type.source,
            "treatment": self.treatment.value,
            "confidence": self.confidence,
            "ex_date": self.ex_date.value if self.ex_date else None,
            "ratio": str(self.ratio.value) if self.ratio else None,
            "counterparty": self.counterparty.value if self.counterparty else None,
            "affected_securities": (
                ",".join(self.affected_securities.value) if self.affected_securities else None
            ),
            "dropped_fields": ",".join(self.dropped_fields),
            "items": ",".join(self.items),
            "rationale": self.rationale,
            "spans": json.dumps(
                {
                    name: unwrap(getattr(self, name))
                    for name in (
                        "event_type",
                        "ex_date",
                        "ratio",
                        "counterparty",
                        "affected_securities",
                    )
                }
            ),
            "run_id": self.run_id,
            "supersedes": self.supersedes,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }


# ----------------------------------------------------------------------------------
# The contract handed to the model.

_SPAN_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "description": (
        "Where in the document this value was read. Offsets index the plain text you "
        "were given, and `text` must be the exact substring at those offsets."
    ),
    "properties": {
        "doc_id": {"type": "string"},
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 0},
        "text": {"type": "string"},
    },
    "required": ["doc_id", "start", "end", "text"],
    "additionalProperties": False,
}


def _cited(value_schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {"value": value_schema, "span": _SPAN_SCHEMA},
        # `span` is required. If the citation can be omitted, it will be.
        "required": ["value", "span"],
        "additionalProperties": False,
    }


EVENT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "event_type": _cited(
            {"type": "string", "enum": [e.value for e in EventType]},
            "The corporate action this filing announces, or no_index_action if none.",
        ),
        "ex_date": _cited(
            {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}"},
            "The ex-date or distribution date, ISO format. Omit if not stated.",
        ),
        "ratio": _cited(
            {"type": "string", "pattern": r"\d+/\d+"},
            (
                "The ratio as an exact fraction of new shares to old. A three-for-one "
                "forward split is 3/1; a one-for-eight reverse split is 1/8. Omit if the "
                "event carries no ratio."
            ),
        ),
        "counterparty": _cited(
            {"type": "string"},
            "The acquirer, acquiree or spun-off entity. Omit if not applicable.",
        ),
        "affected_securities": _cited(
            {"type": "array", "items": {"type": "string"}},
            "Ticker symbols affected, as written in the filing. Omit if none are named.",
        ),
    },
    "required": ["event_type"],
    "additionalProperties": False,
}
