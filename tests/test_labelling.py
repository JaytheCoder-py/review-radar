"""The labelling CLI's pure half.

Nothing here touches a terminal. The interactive loop in `cli.py` is a shell over these
functions precisely so that the rules - what to read next, what a typed answer means,
whether two readings agree - can be tested without one.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from reviewradar.evals.gold import GoldLabel, load_gold
from reviewradar.evals.labelling import (
    LABELLABLE_TYPES,
    agreement,
    append_label,
    labelled_accessions,
    labelling_order,
    parse_event_type,
    parse_ex_date,
    parse_label,
    parse_ratio,
    parse_securities,
    preview,
    relabel_path,
    sample_for_relabel,
    unlabelled,
    write_body,
)
from reviewradar.types import EventType, Ratio, parse_accession

MANIFEST: dict[str, dict[str, Any]] = {
    f"00000000{i:02d}-25-000001": {"stratum": "random" if i % 2 else "stratified"}
    for i in range(1, 21)
}


def label(accession: str, **overrides: Any) -> GoldLabel:
    fields: dict[str, Any] = {
        "accession": parse_accession(accession),
        "stratum": "random",
        "event_type": EventType.NO_INDEX_ACTION,
    }
    fields.update(overrides)
    return GoldLabel(**fields)


# ----------------------------------------------------------------------------------
# What to read next


def test_the_order_is_fixed_by_the_seed() -> None:
    assert labelling_order(MANIFEST, "random", seed=7) == labelling_order(
        MANIFEST, "random", seed=7
    )
    assert labelling_order(MANIFEST, "random", seed=7) != labelling_order(
        MANIFEST, "random", seed=8
    )


def test_the_order_covers_its_stratum_and_only_its_stratum() -> None:
    order = labelling_order(MANIFEST, "random", seed=1)
    assert len(order) == 10
    assert all(MANIFEST[str(a)]["stratum"] == "random" for a in order)


def test_labelling_some_leaves_exactly_the_tail() -> None:
    """The resume property, and the reason the whole stratum is shuffled rather than the
    remainder.

    Shuffling the unlabelled pool instead would re-order the queue every time a label
    landed, so quitting and re-running would hand back a different sample than the seed
    promised - and the seeded order would be a comment rather than a guarantee.
    """
    order = labelling_order(MANIFEST, "random", seed=3)
    done = [str(a) for a in order[:4]]
    assert unlabelled(order, done) == order[4:]


def test_an_unknown_accession_in_the_labelled_set_is_harmless() -> None:
    order = labelling_order(MANIFEST, "random", seed=3)
    assert unlabelled(order, ["9999999999-99-999999"]) == order


def test_the_relabel_sample_is_reproducible_and_capped() -> None:
    pool = [f"acc-{i}" for i in range(50)]
    assert sample_for_relabel(pool, 20, seed=5) == sample_for_relabel(pool, 20, seed=5)
    assert len(sample_for_relabel(pool, 20, seed=5)) == 20
    assert len(sample_for_relabel(pool[:3], 20, seed=5)) == 3


def test_a_relabelling_run_writes_to_a_sidecar(tmp_path: Path) -> None:
    # Never the stratum file. The second reading is evidence about the labeller, not a
    # correction to the labels every measured number in the repo was computed against.
    path = relabel_path("stratified", 42, tmp_path)
    assert path.name == "stratified.relabel.42.jsonl"
    assert path != tmp_path / "stratified.jsonl"


# ----------------------------------------------------------------------------------
# What a human types, into a label


def test_unresolved_is_not_on_the_menu() -> None:
    # It is a classifier outcome, and `GoldLabel.__post_init__` refuses it. Offering it
    # would invite the refusal.
    assert EventType.UNRESOLVED not in LABELLABLE_TYPES
    assert len(LABELLABLE_TYPES) == len(EventType) - 1
    with pytest.raises(ValueError):
        parse_event_type("unresolved")


def test_an_event_type_can_be_typed_three_ways() -> None:
    assert parse_event_type("1") is LABELLABLE_TYPES[0]
    assert parse_event_type("split_reverse") is EventType.SPLIT_REVERSE
    assert parse_event_type("spin") is EventType.SPINOFF


def test_an_ambiguous_prefix_is_refused_rather_than_guessed() -> None:
    # `split_` is not a decision.
    with pytest.raises(ValueError, match="disambiguate"):
        parse_event_type("split")
    for bad in ("", "  ", "banana", "0", "99"):
        with pytest.raises(ValueError):
            parse_event_type(bad)


def test_a_date_must_be_iso_or_blank() -> None:
    assert parse_ex_date("2026-03-05") == dt.date(2026, 3, 5)
    assert parse_ex_date("  ") is None
    # 3/5/26 is 5 March in London and 3 May in New York. A gold set is not the place to
    # find out which one was meant. (`20260305` is the ISO basic form and is accepted,
    # because `date.fromisoformat` reads it and it is not ambiguous.)
    for bad in ("3/5/26", "5 March 2026", "March 5"):
        with pytest.raises(ValueError):
            parse_ex_date(bad)


def test_a_ratio_may_be_written_the_way_the_filing_writes_it() -> None:
    assert parse_ratio("5/4") == Ratio(5, 4)
    assert parse_ratio("5-for-4") == Ratio(5, 4)
    assert parse_ratio("1 for 8") == Ratio(1, 8)
    assert parse_ratio("3") == Ratio(3, 1)
    assert parse_ratio("") is None


def test_a_decimal_ratio_is_refused() -> None:
    # D-005. `Fraction(1.25)` is not five quarters, and a gold label that went through a
    # float is the exact defect the type rule exists to prevent.
    with pytest.raises(ValueError, match="decimal"):
        parse_ratio("1.25")


def test_a_malformed_ratio_is_refused() -> None:
    for bad in ("three-for-one", "5/0", "0/4", "5//4", "-3"):
        with pytest.raises(ValueError):
            parse_ratio(bad)


def test_tickers_are_uppercased_and_deduplicated() -> None:
    assert parse_securities(" mlhr , MLKN , mlhr ") == ("MLHR", "MLKN")
    assert parse_securities("") == ()


def test_a_label_is_validated_by_gold_label_itself() -> None:
    # One definition of a valid label. Re-stating its rules in the CLI would produce a
    # second, and the two would drift.
    with pytest.raises(ValueError, match="labelling error"):
        parse_label(
            "0000320193-26-000001",
            "random",
            {"event_type": "bankruptcy", "ratio": "3/1"},
        )


def test_a_complete_label_round_trips(tmp_path: Path) -> None:
    built = parse_label(
        "0000885988-07-000009",
        "stratified",
        {
            "event_type": "split_forward",
            "ex_date": "2007-05-15",
            "ratio": "5-for-4",
            "counterparty": "",
            "affected_securities": "inmd",
            "notes": "25% stock split declared 2007-05-01.",
        },
    )
    path = tmp_path / "stratified.jsonl"
    append_label(path, built)
    assert load_gold(path) == [built]
    assert built.ratio == Ratio(5, 4)
    assert built.counterparty is None
    assert built.affected_securities == ("INMD",)


# ----------------------------------------------------------------------------------
# Appending, and resuming


def test_each_label_is_on_disk_before_the_next_is_asked_for(tmp_path: Path) -> None:
    # A closed laptop must not cost an evening's reading.
    path = tmp_path / "random.jsonl"
    append_label(path, label("0000000001-25-000001"))
    assert labelled_accessions(path) == {"0000000001-25-000001"}
    append_label(path, label("0000000002-25-000001"))
    assert len(load_gold(path)) == 2


def test_an_absent_file_is_an_empty_start_not_an_error(tmp_path: Path) -> None:
    assert labelled_accessions(tmp_path / "nothing.jsonl") == set()


def test_a_rerun_continues_where_it_stopped(tmp_path: Path) -> None:
    path = tmp_path / "random.jsonl"
    order = labelling_order(MANIFEST, "random", seed=11)
    for accession in order[:3]:
        append_label(path, label(str(accession)))
    assert unlabelled(order, labelled_accessions(path)) == order[3:]


def test_the_body_is_written_where_an_editor_can_open_it(tmp_path: Path) -> None:
    path = write_body("0000000001-25-000001", "line one\nline two\n", tmp_path / "bodies")
    assert path.read_text(encoding="utf-8").startswith("line one")
    assert preview("a\nb\nc\nd", lines=2) == "a\nb"


# ----------------------------------------------------------------------------------
# The labeller against themselves


def test_two_identical_readings_agree_everywhere() -> None:
    first = [label("0000000001-25-000001"), label("0000000002-25-000001")]
    report = agreement(first, list(first))
    assert all(field.rate == 1.0 for field in report)
    assert {field.field for field in report} == {
        "event_type",
        "ex_date",
        "ratio",
        "counterparty",
        "affected_securities",
    }


def test_a_changed_event_type_shows_up_as_disagreement() -> None:
    original = [
        label("0000000001-25-000001", event_type=EventType.SPLIT_FORWARD, ratio=Ratio(3, 1)),
        label("0000000002-25-000001"),
    ]
    second = [
        label("0000000001-25-000001", event_type=EventType.SPINOFF, ratio=Ratio(3, 1)),
        label("0000000002-25-000001"),
    ]
    by_field = {field.field: field for field in agreement(original, second)}
    assert by_field["event_type"].agreed == 1
    assert by_field["event_type"].compared == 2
    assert by_field["event_type"].rate == 0.5
    # The ratio was unchanged, so it still agrees.
    assert by_field["ratio"].rate == 1.0


def test_two_absences_are_an_agreement() -> None:
    # "The filing states no ex-date" is a reading of the filing. Treating it as an
    # uncomparable blank would restrict the measurement to the filings where something
    # happened, which is the half a labeller is most consistent on.
    report = {
        f.field: f
        for f in agreement([label("0000000001-25-000001")], [label("0000000001-25-000001")])
    }
    assert report["ex_date"].compared == 1
    assert report["ex_date"].agreed == 1


def test_only_filings_read_twice_are_compared() -> None:
    original = [label("0000000001-25-000001"), label("0000000002-25-000001")]
    second = [label("0000000002-25-000001")]
    report = {f.field: f for f in agreement(original, second)}
    assert report["event_type"].compared == 1


def test_nothing_compared_is_an_undefined_rate_not_a_perfect_one() -> None:
    # Same honesty as `score.FieldScore.precision`: a system that measured nothing did
    # not score 100%.
    assert all(field.rate is None for field in agreement([], []))


def test_counterparty_agreement_ignores_case_and_spacing() -> None:
    first = [label("0000000001-25-000001", counterparty="GE Vernova Inc.")]
    second = [label("0000000001-25-000001", counterparty="  ge vernova inc. ")]
    report = {f.field: f for f in agreement(first, second)}
    assert report["counterparty"].rate == 1.0


def test_ticker_agreement_ignores_order() -> None:
    first = [label("0000000001-25-000001", affected_securities=("MLHR", "MLKN"))]
    second = [label("0000000001-25-000001", affected_securities=("MLKN", "MLHR"))]
    report = {f.field: f for f in agreement(first, second)}
    assert report["affected_securities"].rate == 1.0
