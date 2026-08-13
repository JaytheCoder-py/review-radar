"""The scoreboard.

The number that matters most here is not accuracy. It is the **false-elimination rate**:
of the filings that genuinely carry an index consequence, what fraction did the
deterministic stage terminate as `NO_INDEX_ACTION`?

That number is the cost of the baseline, and it has to be measured rather than assumed,
because a false elimination is invisible by construction. A filing the baseline discards
never reaches a queue, never reaches the model, and never reaches a person. Nobody finds
out until the index prints wrong.

Strata are scored separately and never pooled (see `gold`). A rate from the stratified
stratum is not a population rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from reviewradar.evals.gold import GoldLabel, Stratum
from reviewradar.extract.schema import CorporateActionEvent
from reviewradar.types import INDEX_RELEVANT, EventType, Treatment


@dataclass(frozen=True, slots=True)
class FieldScore:
    field: str
    precision: float | None
    recall: float | None
    f1: float | None
    n_gold: int
    n_predicted: int

    def as_row(self) -> dict[str, object]:
        def pct(x: float | None) -> str:
            return "-" if x is None else f"{x:.1%}"

        return {
            "field": self.field,
            "precision": pct(self.precision),
            "recall": pct(self.recall),
            "f1": pct(self.f1),
            "n_gold": self.n_gold,
            "n_pred": self.n_predicted,
        }


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _gold_value(label: GoldLabel, field: str) -> object:
    if field == "event_type":
        return label.event_type
    if field == "ex_date":
        return label.ex_date
    if field == "ratio":
        return label.ratio
    if field == "counterparty":
        return (label.counterparty or "").lower() or None
    if field == "affected_securities":
        return tuple(sorted(label.affected_securities)) or None
    raise KeyError(field)


def _pred_value(event: CorporateActionEvent, field: str) -> object:
    if field == "event_type":
        return event.event_type.value
    extracted = getattr(event, field, None)
    if extracted is None:
        return None
    value = extracted.value
    if field == "counterparty":
        return str(value).lower()
    if field == "affected_securities":
        return tuple(sorted(value))
    return value


def score_field(
    gold: Sequence[GoldLabel],
    predicted: Mapping[str, CorporateActionEvent],
    field: str,
) -> FieldScore:
    """Precision, recall and F1 for one field.

    Only filings where gold carries a value for the field count towards recall; only
    filings where a prediction exists count towards precision. A system that abstains on
    everything therefore gets recall 0 and precision **undefined**, not 1.0 - which is
    the honest reading, and the reason `precision` is `float | None`.
    """
    tp = fp = fn = 0
    n_gold = n_pred = 0
    for label in gold:
        event = predicted.get(label.accession)
        want = _gold_value(label, field)
        got = _pred_value(event, field) if event is not None else None
        if want is not None:
            n_gold += 1
        if got is not None:
            n_pred += 1
        if want is None and got is None:
            continue
        if want is not None and got is None:
            fn += 1
        elif want is None and got is not None:
            fp += 1
        elif want == got:
            tp += 1
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return FieldScore(field, precision, recall, _f1(precision, recall), n_gold, n_pred)


@dataclass(frozen=True, slots=True)
class EliminationReport:
    """What the cheap stage costs.

    `false_eliminations` are the filings that genuinely carried an index consequence and
    were terminated as `NO_INDEX_ACTION`. They are the reason the elimination rate alone
    is not a result.
    """

    stratum: Stratum
    n: int
    n_index_relevant: int
    n_eliminated: int
    false_eliminations: tuple[str, ...]

    @property
    def elimination_rate(self) -> float:
        return self.n_eliminated / self.n if self.n else 0.0

    @property
    def false_elimination_rate(self) -> float:
        """Of the genuinely index-relevant filings, the fraction silently discarded."""
        if not self.n_index_relevant:
            return 0.0
        return len(self.false_eliminations) / self.n_index_relevant


def elimination_report(
    gold: Sequence[GoldLabel],
    predicted: Mapping[str, CorporateActionEvent],
    stratum: Stratum,
) -> EliminationReport:
    relevant = [lab for lab in gold if lab.event_type in INDEX_RELEVANT]
    eliminated = 0
    missed: list[str] = []
    for label in gold:
        event = predicted.get(label.accession)
        if event is None:
            continue
        if event.event_type.value is EventType.NO_INDEX_ACTION:
            eliminated += 1
            if label.event_type in INDEX_RELEVANT:
                missed.append(label.accession)
    return EliminationReport(
        stratum=stratum,
        n=len(gold),
        n_index_relevant=len(relevant),
        n_eliminated=eliminated,
        false_eliminations=tuple(sorted(missed)),
    )


def review_rate(predicted: Mapping[str, CorporateActionEvent]) -> float:
    """Fraction of filings a human would have to look at."""
    if not predicted:
        return 0.0
    queued = sum(1 for e in predicted.values() if e.treatment is Treatment.MANUAL_REVIEW)
    return queued / len(predicted)


def cost_report(predicted: Mapping[str, CorporateActionEvent]) -> dict[str, float]:
    """Token and latency accounting, per filing processed."""
    events = list(predicted.values())
    if not events:
        return {"filings": 0.0, "billed": 0.0, "input_tokens": 0.0, "output_tokens": 0.0}
    billed = [e for e in events if e.input_tokens]
    latencies = sorted(e.latency_ms for e in billed) or [0.0]
    return {
        "filings": float(len(events)),
        "billed": float(len(billed)),
        "input_tokens": sum(e.input_tokens for e in events) / max(1, len(billed)),
        "output_tokens": sum(e.output_tokens for e in events) / max(1, len(billed)),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
    }


def field_table(
    gold: Sequence[GoldLabel],
    baseline: Mapping[str, CorporateActionEvent],
    with_model: Mapping[str, CorporateActionEvent] | None = None,
) -> pd.DataFrame:
    """Per-field scores, baseline against baseline-plus-model."""
    fields = ("event_type", "ex_date", "ratio", "counterparty", "affected_securities")
    rows: list[dict[str, object]] = []
    for field in fields:
        base = score_field(gold, baseline, field)
        row: dict[str, object] = {
            "field": field,
            "n_gold": base.n_gold,
            "baseline_f1": base.f1,
        }
        if with_model is not None:
            model = score_field(gold, with_model, field)
            row["model_f1"] = model.f1
            row["delta"] = (
                None if base.f1 is None or model.f1 is None else round(model.f1 - base.f1, 4)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_threshold(
    gold: Sequence[GoldLabel],
    predicted: Mapping[str, CorporateActionEvent],
    *,
    target_precision: float,
) -> float:
    """The lowest confidence threshold that hits `target_precision` on auto-accepted events.

    Calibrated against the random stratum, not chosen in advance (D-004). Raises if no
    threshold reaches the target: silently returning 1.0 would look like a calibration
    when it is a failure to find one.
    """
    candidates = sorted({round(e.confidence, 2) for e in predicted.values()} | {0.0, 1.0})
    by_accession: dict[str, GoldLabel] = {str(lab.accession): lab for lab in gold}
    for threshold in candidates:
        accepted = [
            (e, by_accession[a])
            for a, e in predicted.items()
            if a in by_accession
            and e.confidence >= threshold
            and e.event_type.value is not EventType.NO_INDEX_ACTION
        ]
        if not accepted:
            continue
        correct = sum(1 for e, lab in accepted if e.event_type.value is lab.event_type)
        if correct / len(accepted) >= target_precision:
            return threshold
    raise ValueError(
        f"no threshold reaches {target_precision:.0%} precision on {len(predicted)} "
        "predictions; the extractor is not good enough to auto-accept anything"
    )
