"""The hand-labelled gold set.

**Two strata, never pooled.** They answer different questions and a pooled figure is
neither:

* **random** - a uniform sample of 8-K traffic across a set of trading days. Gives
  *population rates*: what fraction of filings carry an index consequence, how much the
  baseline eliminates correctly, what the true manual-review rate would be. The business
  case is computed from this stratum and only from this stratum.
* **stratified** - found by full-text search for the phrases that accompany
  index-relevant events, so that per-class precision and recall have enough observations
  to mean anything. Rates computed here are **not** population rates and must never be
  quoted as such.

Pooling them would produce a headline accuracy that is neither a population rate nor a
per-class rate, and being able to explain why is worth more than the number.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from reviewradar.ingest.edgar import FilingRef
from reviewradar.types import RATIO_BEARING, Accession, EventType, Ratio, parse_accession

Stratum = Literal["random", "stratified"]

GOLD_DIR = Path("data/gold")


@dataclass(frozen=True, slots=True)
class GoldLabel:
    """One filing, as a human read it."""

    accession: Accession
    stratum: Stratum
    event_type: EventType
    ex_date: dt.date | None = None
    ratio: Ratio | None = None
    counterparty: str | None = None
    affected_securities: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.ratio is not None and self.event_type not in RATIO_BEARING:
            raise ValueError(
                f"{self.accession}: a ratio on a {self.event_type.value} is a labelling "
                "error, not a fact"
            )
        if self.event_type is EventType.UNRESOLVED:
            raise ValueError(
                f"{self.accession}: UNRESOLVED is a classifier outcome, never a label. "
                "A human who cannot tell should record the ambiguity in `notes` and pick "
                "the reading the filing best supports."
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "stratum": self.stratum,
            "event_type": self.event_type.value,
            "ex_date": self.ex_date.isoformat() if self.ex_date else None,
            "ratio": f"{self.ratio.numerator}/{self.ratio.denominator}" if self.ratio else None,
            "counterparty": self.counterparty,
            "affected_securities": list(self.affected_securities),
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> GoldLabel:
        ratio_raw = raw.get("ratio")
        ratio = None
        if ratio_raw:
            num, _, den = str(ratio_raw).partition("/")
            ratio = Ratio(int(num), int(den))
        ex_raw = raw.get("ex_date")
        return cls(
            accession=parse_accession(raw["accession"]),
            stratum=raw["stratum"],
            event_type=EventType(raw["event_type"]),
            ex_date=dt.date.fromisoformat(ex_raw) if ex_raw else None,
            ratio=ratio,
            counterparty=raw.get("counterparty") or None,
            affected_securities=tuple(raw.get("affected_securities") or ()),
            notes=raw.get("notes", ""),
        )


def load_gold(path: Path | str) -> list[GoldLabel]:
    """Read a JSONL gold file. Raises on a malformed line rather than skipping it."""
    labels: list[GoldLabel] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            labels.append(GoldLabel.from_json(json.loads(line)))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return labels


def save_gold(labels: Sequence[GoldLabel], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\n".join(json.dumps(label.to_json(), sort_keys=True) for label in labels) + "\n",
        encoding="utf-8",
    )


def load_all(gold_dir: Path | str = GOLD_DIR) -> dict[Stratum, list[GoldLabel]]:
    """Both strata, kept apart. There is deliberately no function that merges them."""
    directory = Path(gold_dir)
    out: dict[Stratum, list[GoldLabel]] = {"random": [], "stratified": []}
    strata: tuple[Stratum, ...] = ("random", "stratified")
    for stratum in strata:
        path = directory / f"{stratum}.jsonl"
        if path.exists():
            out[stratum] = load_gold(path)
    return out


def sample_random(refs: Sequence[FilingRef], n: int, *, seed: int) -> list[FilingRef]:
    """Uniform sample, reproducible from the seed."""
    rng = random.Random(seed)
    ordered = sorted(refs, key=lambda r: r.accession)
    return rng.sample(ordered, min(n, len(ordered)))


def stratum_counts(labels: Sequence[GoldLabel]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.event_type.value] = counts.get(label.event_type.value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def index_relevant_rate(labels: Sequence[GoldLabel]) -> float:
    """Fraction of filings carrying an index consequence.

    Meaningful on the random stratum. Meaningless on the stratified stratum, which was
    constructed to be full of them.
    """
    if not labels:
        return 0.0
    hits = sum(1 for lab in labels if lab.event_type is not EventType.NO_INDEX_ACTION)
    return hits / len(labels)


def as_fraction(text: str) -> Fraction:
    """Parse "3/1" without going through a float on the way."""
    num, _, den = text.partition("/")
    return Fraction(int(num), int(den))
