"""Tooling for hand-labelling the gold set.

The labels themselves are the owner's; a gold set labelled by the model under test is not
gold, and nothing in this module reads an extraction. What it does is make 134 more
filings a tractable evening: pick what to read next, parse what a human types into a
`GoldLabel`, append it the moment it is accepted, and measure the labeller against
themselves a week later.

Three properties are load-bearing, and each is a pure function so it can be tested
without a terminal:

* **The order is fixed by the seed, not by what is left.** The whole stratum is shuffled
  once from the seed and already-labelled accessions are filtered *out of that order*.
  Shuffling the unlabelled pool instead would reshuffle every time a label lands, so
  quitting and resuming would re-order the queue and the sample would stop being the
  reproducible one the seed promises.
* **A label is appended the instant it is accepted**, one JSONL line, never rewritten.
  The alternative - accumulate and save at the end - loses an evening's reading to a
  closed laptop, and the resume path is then untested code nobody runs until it matters.
* **A relabel never shows the old label.** Self-agreement measured against a label you
  can see is a measurement of your memory.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from reviewradar.evals.gold import GOLD_DIR, GoldLabel, Stratum, load_gold
from reviewradar.types import Accession, EventType, Ratio, parse_accession

#: The seed the queue was first drawn with. Recorded rather than defaulted silently: a
#: sample nobody can reproduce is a sample nobody can check.
DEFAULT_SEED: Final[int] = 20260817

#: The event types a human may assign. `UNRESOLVED` is absent because it is a classifier
#: outcome, never a label - `GoldLabel.__post_init__` refuses it, and offering it in a
#: menu would invite the refusal.
LABELLABLE_TYPES: Final[tuple[EventType, ...]] = tuple(
    event_type for event_type in EventType if event_type is not EventType.UNRESOLVED
)

#: Fields compared when measuring a labeller against themselves. `notes` is excluded:
#: it is free prose, and disagreement in it is not disagreement about the filing.
AGREEMENT_FIELDS: Final[tuple[str, ...]] = (
    "event_type",
    "ex_date",
    "ratio",
    "counterparty",
    "affected_securities",
)

_RATIO_FOR = re.compile(r"\A(\d+)\s*(?:-|\s)\s*for\s*(?:-|\s)\s*(\d+)\Z", re.I)
_RATIO_QUOTIENT = re.compile(r"\A(\d+)\s*/\s*(\d+)\Z")
_RATIO_WHOLE = re.compile(r"\A(\d+)\Z")


# ----------------------------------------------------------------------------------
# What to read next


def labelling_order(
    manifest: Mapping[str, Mapping[str, Any]], stratum: Stratum, *, seed: int
) -> list[Accession]:
    """Every corpus filing in `stratum`, in a fixed pseudo-random order.

    Sorted before shuffling so the permutation depends on the seed alone and not on the
    order the manifest happened to be written in.
    """
    members = sorted(
        accession for accession, meta in manifest.items() if str(meta.get("stratum")) == stratum
    )
    rng = random.Random(seed)
    return [parse_accession(a) for a in rng.sample(members, len(members))]


def unlabelled(order: Sequence[Accession], labelled: Iterable[str]) -> list[Accession]:
    """The queue: `order` with the already-labelled accessions removed.

    A filter over a fixed order, never a fresh shuffle - see the module docstring. The
    result is always a subsequence of `order`, which is the property the resume path
    depends on.
    """
    done = {str(a) for a in labelled}
    return [accession for accession in order if str(accession) not in done]


def labelled_accessions(path: Path | str) -> set[str]:
    """Accessions already labelled in a stratum's JSONL. Empty if the file is absent."""
    target = Path(path)
    if not target.exists():
        return set()
    return {str(label.accession) for label in load_gold(target)}


def sample_for_relabel(labelled: Sequence[str], n: int, *, seed: int) -> list[str]:
    """`n` already-labelled accessions, drawn reproducibly from the seed."""
    ordered = sorted(str(a) for a in labelled)
    return random.Random(seed).sample(ordered, min(n, len(ordered)))


def relabel_path(stratum: Stratum, seed: int, gold_dir: Path | str = GOLD_DIR) -> Path:
    """Where a blind relabelling run writes.

    A sidecar, never the stratum file. The second reading is evidence about the labeller,
    not a correction to the gold set, and merging the two would quietly overwrite the
    labels every measured number in this repository was computed against.
    """
    return Path(gold_dir) / f"{stratum}.relabel.{seed}.jsonl"


# ----------------------------------------------------------------------------------
# What a human types, into a label


def parse_event_type(raw: str) -> EventType:
    """An event type from a menu index, a full value, or an unambiguous prefix.

    Three ways in because this gets typed 134 times. An ambiguous prefix raises rather
    than picking one - `split_` is not a decision.
    """
    text = raw.strip().lower()
    if not text:
        raise ValueError("an event type is required")
    if text.isdigit():
        index = int(text)
        if not 1 <= index <= len(LABELLABLE_TYPES):
            raise ValueError(f"no event type numbered {index}; expected 1-{len(LABELLABLE_TYPES)}")
        return LABELLABLE_TYPES[index - 1]
    matches = [t for t in LABELLABLE_TYPES if t.value == text]
    if not matches:
        matches = [t for t in LABELLABLE_TYPES if t.value.startswith(text)]
    if not matches:
        raise ValueError(f"not an event type: {raw!r}")
    if len(matches) > 1:
        raise ValueError(
            f"{raw!r} matches {', '.join(t.value for t in matches)}; type enough to disambiguate"
        )
    return matches[0]


def parse_ex_date(raw: str) -> dt.date | None:
    """An ISO date, or `None` for blank.

    ISO only. "3/5/26" is 5 March in London and 3 May in New York, and a gold set is not
    the place to find out which one the labeller meant.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"not an ISO date: {text!r} (expected YYYY-MM-DD, or blank)") from exc


def parse_ratio(raw: str) -> Ratio | None:
    """A ratio as new shares per old, or `None` for blank.

    Accepts `5/4`, `5-for-4`, `5 for 4` and a bare `3`, because that is the range of ways
    a filing writes one and retyping it into a single form is where transcription errors
    come from. The orientation is the schema's: a three-for-one forward split is `3/1`, a
    one-for-eight reverse split is `1/8`.

    A decimal is refused outright. `Fraction(1.25)` is not five quarters, and a gold label
    that went through a float is exactly the defect D-005 exists to prevent - so the
    labeller is asked for the quotient the filing actually used.
    """
    text = raw.strip()
    if not text:
        return None
    if "." in text:
        raise ValueError(
            f"{text!r} is a decimal; write the quotient the filing uses, e.g. 5/4 for a 25% "
            "stock split. Ratios are exact here and never pass through a float."
        )
    for pattern in (_RATIO_QUOTIENT, _RATIO_FOR):
        if match := pattern.match(text):
            numerator, denominator = int(match.group(1)), int(match.group(2))
            break
    else:
        if whole := _RATIO_WHOLE.match(text):
            numerator, denominator = int(whole.group(1)), 1
        else:
            raise ValueError(f"not a ratio: {text!r} (expected 5/4, 5-for-4, or 3)")
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"a ratio must be positive; got {text!r}")
    return Ratio(numerator, denominator)


def parse_securities(raw: str) -> tuple[str, ...]:
    """Comma-separated tickers, uppercased, order preserved, duplicates dropped."""
    seen: dict[str, None] = {}
    for part in raw.split(","):
        if ticker := part.strip().upper():
            seen.setdefault(ticker, None)
    return tuple(seen)


def parse_label(
    accession: str,
    stratum: Stratum,
    answers: Mapping[str, str],
) -> GoldLabel:
    """Build a `GoldLabel` from what was typed.

    Validation is `GoldLabel.__post_init__` and nothing else. Duplicating its rules here
    would produce a second definition of a valid label, and the two would drift.
    """
    return GoldLabel(
        accession=parse_accession(accession),
        stratum=stratum,
        event_type=parse_event_type(answers.get("event_type", "")),
        ex_date=parse_ex_date(answers.get("ex_date", "")),
        ratio=parse_ratio(answers.get("ratio", "")),
        counterparty=(answers.get("counterparty", "").strip() or None),
        affected_securities=parse_securities(answers.get("affected_securities", "")),
        notes=answers.get("notes", "").strip(),
    )


def append_label(path: Path | str, label: GoldLabel) -> None:
    """Append one label as one JSONL line, immediately.

    Opened in append mode per label rather than held open, so an interrupted session
    leaves a complete file rather than a truncated one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.to_json(), sort_keys=True) + "\n")


# ----------------------------------------------------------------------------------
# Reading the filing


def preview(text: str, lines: int = 40) -> str:
    """The head of a filing's text, for deciding whether the editor is needed at all."""
    return "\n".join(text.splitlines()[:lines])


def write_body(accession: str, text: str, directory: Path | str) -> Path:
    """Drop the normalised body text where an editor can open it."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{accession}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------
# The labeller against themselves


@dataclass(frozen=True, slots=True)
class FieldAgreement:
    """How often two readings of the same filing said the same thing."""

    field: str
    agreed: int
    compared: int

    @property
    def rate(self) -> float | None:
        """`None` rather than 1.0 when nothing was compared, as in `score.FieldScore`."""
        return self.agreed / self.compared if self.compared else None


def _comparable(label: GoldLabel, field: str) -> object:
    if field == "counterparty":
        return (label.counterparty or "").strip().lower() or None
    if field == "affected_securities":
        return frozenset(label.affected_securities)
    return getattr(label, field)


def agreement(
    original: Sequence[GoldLabel], relabelled: Sequence[GoldLabel]
) -> list[FieldAgreement]:
    """Per-field agreement between a first and a second reading.

    Only accessions present in both are compared. Two absences count as agreement: "the
    filing states no ex-date" is a reading of the filing, and scoring it as an
    uncomparable blank would quietly restrict the measurement to the filings where
    something happened - which is the half where a labeller is most consistent, so the
    number would come out flattering.
    """
    by_accession = {str(label.accession): label for label in original}
    pairs = [
        (by_accession[str(second.accession)], second)
        for second in relabelled
        if str(second.accession) in by_accession
    ]
    return [
        FieldAgreement(
            field=field,
            agreed=sum(
                1
                for first, second in pairs
                if _comparable(first, field) == _comparable(second, field)
            ),
            compared=len(pairs),
        )
        for field in AGREEMENT_FIELDS
    ]
