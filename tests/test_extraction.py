"""Span grounding, confidence, treatment, and the pipeline end to end.

The span tests are the heart of the project. A model told to cite will cite *something*;
the only question worth asking is whether the value it reported is in the words it
pointed at.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reviewradar.extract.baseline import classify
from reviewradar.extract.llm import OfflineLlm
from reviewradar.extract.pipeline import (
    build_prompt,
    compute_confidence,
    extract,
    validate_spans,
)
from reviewradar.ingest.edgar import Submission, SubmissionDocument
from reviewradar.treatment.rules import decide
from reviewradar.types import EventType, Ratio, Treatment

DOC = SubmissionDocument(
    doc_id="a:1",
    doc_type="8-K",
    filename="f.htm",
    text=(
        "Warwick Valley Telephone Company announced a three-for-one stock split. "
        "The record date is September 15, 2003 and the shares trade as WWVY."
    ),
)


def cite(value: Any, quote: str, doc_id: str = "a:1") -> dict[str, Any]:
    start = DOC.text.find(quote)
    return {
        "value": value,
        "span": {"doc_id": doc_id, "start": start, "end": start + len(quote), "text": quote},
    }


# ----------------------------------------------------------------------------------
# Span validation


def test_a_correctly_quoted_field_survives() -> None:
    kept, dropped = validate_spans({"ratio": cite("3/1", "three-for-one stock split")}, [DOC])
    assert dropped == []
    assert kept["ratio"].value == Ratio(3, 1)
    assert kept["ratio"].span is not None


def test_a_value_not_in_its_own_span_is_dropped_not_flagged() -> None:
    # D-003. Flagging leaves a plausible wrong number in the record, and someone
    # downstream will use it.
    kept, dropped = validate_spans({"ratio": cite("5/1", "three-for-one stock split")}, [DOC])
    assert "ratio" not in kept
    assert dropped == ["ratio"]


def test_a_span_naming_a_document_that_does_not_exist_is_dropped() -> None:
    payload = {"ex_date": cite("2003-09-15", "September 15, 2003", doc_id="a:99")}
    _, dropped = validate_spans(payload, [DOC])
    assert dropped == ["ex_date"]


def test_a_quotation_absent_from_the_document_is_dropped() -> None:
    payload = {
        "ex_date": {
            "value": "2003-09-15",
            "span": {"doc_id": "a:1", "start": 0, "end": 20, "text": "a sentence never written"},
        }
    }
    _, dropped = validate_spans(payload, [DOC])
    assert dropped == ["ex_date"]


def test_wrong_offsets_with_a_correct_quotation_are_repaired() -> None:
    # Models are poor at character arithmetic and good at quotation. The citation is
    # what must hold up; the offsets are a convenience worth repairing.
    payload = {
        "ratio": {
            "value": "3/1",
            "span": {
                "doc_id": "a:1",
                "start": 9999,
                "end": 10042,
                "text": "three-for-one stock split",
            },
        }
    }
    kept, dropped = validate_spans(payload, [DOC])
    assert dropped == []
    span = kept["ratio"].span
    assert span is not None
    assert DOC.text[span.start : span.end] == "three-for-one stock split"


def test_a_ratio_written_in_words_satisfies_its_digits() -> None:
    # "three-for-one" has to satisfy a check on 3/1, or every real split is dropped.
    kept, _ = validate_spans({"ratio": cite("3/1", "three-for-one stock split")}, [DOC])
    assert kept["ratio"].value == Ratio(3, 1)


def test_a_date_must_have_its_day_month_and_year_in_the_quotation() -> None:
    good, dropped = validate_spans({"ex_date": cite("2003-09-15", "September 15, 2003")}, [DOC])
    assert dropped == [] and good["ex_date"].value == dt.date(2003, 9, 15)
    _, dropped_wrong = validate_spans({"ex_date": cite("2003-09-16", "September 15, 2003")}, [DOC])
    assert dropped_wrong == ["ex_date"]


def test_a_ticker_must_appear_in_its_quotation() -> None:
    kept, _ = validate_spans({"affected_securities": cite(["WWVY"], "as WWVY")}, [DOC])
    assert kept["affected_securities"].value == ("WWVY",)
    _, dropped = validate_spans({"affected_securities": cite(["AAPL"], "as WWVY")}, [DOC])
    assert dropped == ["affected_securities"]


def test_abstention_needs_no_citation() -> None:
    # Demanding a quotation for "nothing happened" would push every clean filing
    # into the review queue.
    kept, dropped = validate_spans(
        {"event_type": {"value": "no_index_action", "span": None}}, [DOC]
    )
    assert dropped == []
    assert kept["event_type"].value is EventType.NO_INDEX_ACTION
    assert kept["event_type"].span is None


def test_a_positive_classification_does_need_a_citation() -> None:
    kept, dropped = validate_spans({"event_type": {"value": "split_forward", "span": None}}, [DOC])
    assert dropped == ["event_type"] and "event_type" not in kept


def test_an_unparseable_value_is_dropped_not_half_read() -> None:
    for bad in ("3/0", "not-a-ratio", "3//1", ""):
        _, dropped = validate_spans({"ratio": cite(bad, "three-for-one stock split")}, [DOC])
        assert dropped == ["ratio"], bad


@settings(max_examples=60, deadline=None)
@given(
    st.text(min_size=0, max_size=120),
    st.integers(min_value=-10_000, max_value=10_000),
    st.integers(min_value=-10_000, max_value=10_000),
)
def test_validate_spans_never_raises_on_arbitrary_input(quote: str, start: int, end: int) -> None:
    # One pathological model response must not take down a nightly job.
    payload = {
        "ex_date": {
            "value": "2003-09-15",
            "span": {"doc_id": "a:1", "start": start, "end": end, "text": quote},
        }
    }
    validate_spans(payload, [DOC])


# ----------------------------------------------------------------------------------
# Confidence


def test_confidence_falls_when_a_field_is_dropped(split: Submission) -> None:
    baseline = classify(split)
    kept, _ = validate_spans(
        {
            "event_type": cite("split_forward", "three-for-one stock split"),
            "ratio": cite("3/1", "three-for-one stock split"),
        },
        [DOC],
    )
    filed = dt.date(2003, 9, 19)
    high = compute_confidence(baseline=baseline, kept=kept, dropped=[], filed_at=filed)
    low = compute_confidence(baseline=baseline, kept=kept, dropped=["ex_date"], filed_at=filed)
    assert low < high


def test_an_ex_date_before_the_filing_lowers_confidence(split: Submission) -> None:
    # A split cannot go ex before it is announced. This is arithmetic, not a model call.
    baseline = classify(split)
    kept, _ = validate_spans(
        {
            "event_type": cite("split_forward", "three-for-one stock split"),
            "ratio": cite("3/1", "three-for-one stock split"),
            "ex_date": cite("2003-09-15", "September 15, 2003"),
        },
        [DOC],
    )
    sane = compute_confidence(
        baseline=baseline, kept=kept, dropped=[], filed_at=dt.date(2003, 9, 1)
    )
    impossible = compute_confidence(
        baseline=baseline, kept=kept, dropped=[], filed_at=dt.date(2003, 9, 19)
    )
    assert impossible < sane


def test_disagreeing_with_a_diagnostic_item_lowers_confidence(diagnostic: Submission) -> None:
    baseline = classify(diagnostic)
    assert baseline.event_type is EventType.DELISTING
    agree, _ = validate_spans({"event_type": cite("delisting", "stock split")}, [DOC])
    disagree, _ = validate_spans({"event_type": cite("spinoff", "stock split")}, [DOC])
    filed = dt.date(2007, 5, 24)
    assert compute_confidence(
        baseline=baseline, kept=disagree, dropped=[], filed_at=filed
    ) < compute_confidence(baseline=baseline, kept=agree, dropped=[], filed_at=filed)


def test_a_missing_required_field_lowers_confidence(split: Submission) -> None:
    baseline = classify(split)
    with_ratio, _ = validate_spans(
        {
            "event_type": cite("split_forward", "three-for-one stock split"),
            "ratio": cite("3/1", "three-for-one stock split"),
        },
        [DOC],
    )
    without, _ = validate_spans(
        {"event_type": cite("split_forward", "three-for-one stock split")}, [DOC]
    )
    filed = dt.date(2003, 9, 19)
    assert compute_confidence(
        baseline=baseline, kept=without, dropped=[], filed_at=filed
    ) < compute_confidence(baseline=baseline, kept=with_ratio, dropped=[], filed_at=filed)


def test_confidence_is_bounded(split: Submission) -> None:
    baseline = classify(split)
    kept, _ = validate_spans(
        {"event_type": cite("split_forward", "three-for-one stock split")}, [DOC]
    )
    score = compute_confidence(
        baseline=baseline,
        kept=kept,
        dropped=["a", "b", "c", "d", "e"],
        filed_at=dt.date(2003, 9, 19),
    )
    assert 0.0 <= score <= 1.0


def test_no_event_type_at_all_is_zero_confidence(split: Submission) -> None:
    assert (
        compute_confidence(
            baseline=classify(split), kept={}, dropped=[], filed_at=dt.date(2003, 9, 19)
        )
        == 0.0
    )


# ----------------------------------------------------------------------------------
# Treatment


def test_a_confident_split_adjusts_the_divisor() -> None:
    assert decide(EventType.SPLIT_FORWARD, confidence=0.95) is Treatment.DIVISOR_ADJUST


def test_the_same_split_at_low_confidence_goes_to_review() -> None:
    # Acting on a low-confidence split applies a divisor change before anyone notices,
    # and unwinding it is a recalculation event.
    assert decide(EventType.SPLIT_FORWARD, confidence=0.30) is Treatment.MANUAL_REVIEW


def test_no_index_action_is_exempt_from_the_confidence_floor() -> None:
    # Routing low-confidence "nothing happened" to review would drown the queue in
    # exactly the filings the baseline exists to clear.
    assert decide(EventType.NO_INDEX_ACTION, confidence=0.10) is Treatment.NO_ACTION


def test_unresolved_always_goes_to_review() -> None:
    assert decide(EventType.UNRESOLVED, confidence=1.0) is Treatment.MANUAL_REVIEW


def test_an_out_of_range_confidence_raises() -> None:
    with pytest.raises(ValueError):
        decide(EventType.SPLIT_FORWARD, confidence=1.5)


def test_every_event_type_has_a_treatment() -> None:
    for event_type in EventType:
        assert decide(event_type, confidence=0.99) in Treatment


# ----------------------------------------------------------------------------------
# The pipeline, end to end


def test_an_eliminated_filing_never_reaches_the_model(eliminated: Submission) -> None:
    client = OfflineLlm()
    event = extract(eliminated, client=client, run_id="r1")
    assert client.calls == 0, "the baseline terminated this filing; the model was billed anyway"
    assert event.event_type.value is EventType.NO_INDEX_ACTION
    assert event.event_type.source == "baseline"
    assert event.treatment is Treatment.NO_ACTION


def test_a_residual_filing_does_reach_the_model(split: Submission) -> None:
    client = OfflineLlm()
    extract(split, client=client, run_id="r1")
    assert client.calls == 1


def test_the_model_extracts_what_it_can_cite(split: Submission) -> None:
    baseline = classify(split)
    prompt = build_prompt(split, baseline)
    exhibit = next(d for d in split.exhibits() if "three-for-one" in d.text.lower())
    quote = "three-for-one stock split"
    start = exhibit.text.find(quote)
    client = OfflineLlm()
    client.register(
        prompt,
        {
            "event_type": {
                "value": "split_forward",
                "span": {
                    "doc_id": exhibit.doc_id,
                    "start": start,
                    "end": start + len(quote),
                    "text": quote,
                },
            },
            "ratio": {
                "value": "3/1",
                "span": {
                    "doc_id": exhibit.doc_id,
                    "start": start,
                    "end": start + len(quote),
                    "text": quote,
                },
            },
        },
    )
    event = extract(split, client=client, run_id="r1")
    assert event.event_type.value is EventType.SPLIT_FORWARD
    assert event.ratio is not None and event.ratio.value == Ratio(3, 1)
    assert event.treatment is Treatment.DIVISOR_ADJUST
    assert event.dropped_fields == ()


def test_an_uncitable_extraction_is_dropped_and_lands_in_review(split: Submission) -> None:
    baseline = classify(split)
    prompt = build_prompt(split, baseline)
    client = OfflineLlm()
    client.register(
        prompt,
        {
            "event_type": {
                "value": "split_forward",
                "span": {"doc_id": "nope:1", "start": 0, "end": 5, "text": "absent"},
            }
        },
    )
    event = extract(split, client=client, run_id="r1")
    assert "event_type" in event.dropped_fields
    assert event.treatment is Treatment.MANUAL_REVIEW


def test_baseline_only_mode_still_produces_a_real_event(split: Submission) -> None:
    # This is how the scoreboard measures the model's contribution, so it has to
    # produce a comparable event rather than a stub.
    client = OfflineLlm()
    event = extract(split, client=client, run_id="r1", baseline_only=True)
    assert client.calls == 0
    assert event.event_type.value is EventType.UNRESOLVED
    assert event.event_type.source == "baseline"
    assert event.treatment is Treatment.MANUAL_REVIEW


def test_the_prompt_labels_documents_with_the_ids_spans_must_use(split: Submission) -> None:
    prompt = build_prompt(split, classify(split))
    for doc in split.documents:
        assert f"doc_id: {doc.doc_id}" in prompt
