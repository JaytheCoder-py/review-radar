"""Forward verification against the tape.

The two tests that matter most are the least exciting: a reverse split has to multiply
the price where a forward split divides it, and an ex-date that has not passed has to
come out `unverifiable` no matter what the price series says. Getting the first wrong
inverts every verdict; getting the second wrong turns the scoreboard into an accusation
machine pointed at events nothing has happened to yet.
"""

from __future__ import annotations

import ast
import datetime as dt
import pathlib
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reviewradar.evals.forward import (
    TOLERANCE,
    Claim,
    FixturePrices,
    Unverifiable,
    Verdict,
    claims_from_rows,
    expected_step,
    resolve_ticker,
    verdict_counts,
    verify,
    verify_all,
)
from reviewradar.ingest.store import EventStore
from reviewradar.types import Accession, EventType, Ratio, parse_accession

PRICES = Path(__file__).parent / "fixtures" / "prices"

#: Well after every fixture ex-date, so "the ex-date has passed" is never in question
#: except where a test puts it there.
TODAY = dt.date(2026, 4, 1)
EX = dt.date(2026, 3, 5)
THREE_FOR_ONE = Ratio(3, 1)


@pytest.fixture
def prices() -> FixturePrices:
    return FixturePrices.from_dir(PRICES)


def make_claim(
    *,
    event_type: EventType = EventType.SPLIT_FORWARD,
    ex_date: dt.date | None = EX,
    ratio: Ratio | None = THREE_FOR_ONE,
    tickers: tuple[str, ...] = ("FSPL",),
    event_id: str = "e1",
    accession: str = "0000320193-26-000001",
) -> Claim:
    return Claim(
        accession=parse_accession(accession),
        event_id=event_id,
        event_type=event_type,
        ex_date=ex_date,
        ratio=ratio,
        tickers=tickers,
    )


# ----------------------------------------------------------------------------------
# Ratio orientation


def test_the_expected_step_is_the_reciprocal_of_the_ratio() -> None:
    # The schema states the ratio as new shares per old, so the price moves by 1/r.
    # A forward split divides the as-traded price; a reverse split multiplies it.
    assert expected_step(Ratio(3, 1)) == Ratio(1, 3)
    assert expected_step(Ratio(1, 8)) == Ratio(8, 1)
    # IntegraMed's 25% stock split, as `data/gold/stratified.jsonl` records it: 5/4 in
    # shares is four fifths in price, a 20% fall.
    assert expected_step(Ratio(5, 4)) == Ratio(4, 5)


def test_a_forward_split_that_stepped_is_verified(prices: FixturePrices) -> None:
    result = verify(make_claim(), source=prices, today=TODAY, run_id="t")
    assert result.verdict is Verdict.VERIFIED
    assert result.expected_step == Ratio(1, 3)
    assert result.relative_error == Fraction(1, 200)


def test_a_reverse_split_that_stepped_the_other_way_is_verified(prices: FixturePrices) -> None:
    # The direction test. A 1/8 reverse split multiplies the price by eight; a verifier
    # that divided would contradict every real one.
    result = verify(
        make_claim(event_type=EventType.SPLIT_REVERSE, ratio=Ratio(1, 8), tickers=("RSPL",)),
        source=prices,
        today=TODAY,
        run_id="t",
    )
    assert result.verdict is Verdict.VERIFIED
    assert result.expected_step == Ratio(8, 1)
    assert result.relative_error == Fraction(1, 56)


def test_a_split_with_no_price_step_is_contradicted(prices: FixturePrices) -> None:
    result = verify(make_claim(tickers=("FLAT",)), source=prices, today=TODAY, run_id="t")
    assert result.verdict is Verdict.CONTRADICTED
    # 300.00 to 298.00 on a day that should have thirded the price: 1.98, an ordinary
    # day's drift below the 2.0 identity below.
    assert result.relative_error == Fraction(99, 50)


@pytest.mark.parametrize("ratio", [Ratio(3, 1), Ratio(5, 4), Ratio(13, 10), Ratio(1, 8)])
def test_a_step_that_never_happened_scores_exactly_ratio_minus_one(ratio: Ratio) -> None:
    """The identity the tolerance band in D-008 is set against.

    An unchanged close scores `|ratio - 1|`, so the weakest signal this system will ever
    be asked to read is the smallest ratio it accepts: a 5/4 stock split, at 0.25. The
    band has to sit well under that and well over an ordinary day's move.
    """
    source = FixturePrices()
    source.add("SAME", [("2026-03-04", 42.00), ("2026-03-05", 42.00)])
    result = verify(
        make_claim(ratio=ratio, tickers=("SAME",)), source=source, today=TODAY, run_id="t"
    )
    assert result.relative_error == abs(ratio - 1)
    assert result.verdict is Verdict.CONTRADICTED


# ----------------------------------------------------------------------------------
# The calendar


def test_an_ex_date_on_a_non_trading_day_uses_the_next_session(prices: FixturePrices) -> None:
    # Ex-date Saturday 7 March; the step lands on Monday the 9th and the prior close is
    # Friday's. Requiring a close *on* the ex-date would contradict every split whose
    # ex-date fell on a weekend or a holiday.
    result = verify(
        make_claim(ex_date=dt.date(2026, 3, 7), tickers=("HOLI",)),
        source=prices,
        today=TODAY,
        run_id="t",
    )
    assert result.verdict is Verdict.VERIFIED
    assert result.session == dt.date(2026, 3, 9)
    assert result.prior_session == dt.date(2026, 3, 6)


def test_a_future_ex_date_is_unverifiable_and_costs_no_fetch(prices: FixturePrices) -> None:
    result = verify(
        make_claim(ex_date=dt.date(2026, 6, 1), tickers=("FLAT",)),
        source=prices,
        today=TODAY,
        run_id="t",
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.reason is Unverifiable.EX_DATE_IN_FUTURE
    # Checked before the source is consulted, so the asymmetry is structural rather than
    # a side effect of an empty series.
    assert prices.calls == 0


@settings(max_examples=50, deadline=None)
@given(st.dates(min_value=dt.date(2026, 4, 2), max_value=dt.date(2035, 1, 1)))
def test_no_future_ex_date_is_ever_contradicted(ex_date: dt.date) -> None:
    # The one-way rule, as a property. `FLAT` is a series that contradicts if consulted.
    source = FixturePrices.from_dir(PRICES)
    result = verify(
        make_claim(ex_date=ex_date, tickers=("FLAT",)), source=source, today=TODAY, run_id="t"
    )
    assert result.verdict is Verdict.UNVERIFIABLE


def test_a_passed_ex_date_with_no_session_yet_is_unverifiable(prices: FixturePrices) -> None:
    # The tape stops before the ex-date: the exchange has not printed the close this
    # test needs. Not a contradiction - there is nothing to contradict.
    result = verify(make_claim(tickers=("SOON",)), source=prices, today=TODAY, run_id="t")
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.reason is Unverifiable.NO_SESSION_YET


def test_a_ticker_the_source_does_not_know_is_unverifiable(prices: FixturePrices) -> None:
    result = verify(make_claim(tickers=("ZZZZ",)), source=prices, today=TODAY, run_id="t")
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.reason is Unverifiable.NO_PRICE_DATA


# ----------------------------------------------------------------------------------
# The tolerance band


def test_the_band_is_inclusive_at_its_edge() -> None:
    source = FixturePrices()
    # 108.00 against an expected 100.00: a relative error of exactly 0.08.
    source.add("EDGE", [("2026-03-04", 300.00), ("2026-03-05", 108.00)])
    result = verify(make_claim(tickers=("EDGE",)), source=source, today=TODAY, run_id="t")
    assert result.relative_error == TOLERANCE
    assert result.verdict is Verdict.VERIFIED


def test_a_cent_past_the_edge_contradicts() -> None:
    source = FixturePrices()
    source.add("EDGE", [("2026-03-04", 300.00), ("2026-03-05", 108.01)])
    result = verify(make_claim(tickers=("EDGE",)), source=source, today=TODAY, run_id="t")
    assert result.relative_error > TOLERANCE
    assert result.verdict is Verdict.CONTRADICTED


def test_the_band_separates_a_real_step_from_a_missing_one(prices: FixturePrices) -> None:
    near = verify(make_claim(tickers=("NEAR",)), source=prices, today=TODAY, run_id="t")
    wide = verify(make_claim(tickers=("WIDE",)), source=prices, today=TODAY, run_id="t")
    assert near.verdict is Verdict.VERIFIED and near.relative_error == Fraction(3, 40)
    assert wide.verdict is Verdict.CONTRADICTED and wide.relative_error == Fraction(3, 25)


# ----------------------------------------------------------------------------------
# Scope: what gets no verdict, and why


def test_a_spinoff_ratio_does_not_license_a_price_test(prices: FixturePrices) -> None:
    # GE Vernova is 1 share per 4 GE shares held. The parent's price step is the market
    # value of the child, which that ratio does not give you.
    result = verify(
        make_claim(event_type=EventType.SPINOFF, ratio=Ratio(1, 4), tickers=("FSPL",)),
        source=prices,
        today=TODAY,
        run_id="t",
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.reason is Unverifiable.EVENT_TYPE_HAS_NO_PRICE_TEST


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"event_type": EventType.MERGER_COMPLETED}, Unverifiable.EVENT_TYPE_HAS_NO_PRICE_TEST),
        ({"event_type": EventType.DELISTING}, Unverifiable.EVENT_TYPE_HAS_NO_PRICE_TEST),
        ({"ex_date": None}, Unverifiable.NO_EX_DATE),
        ({"ratio": None}, Unverifiable.NO_RATIO),
        ({"tickers": ()}, Unverifiable.NO_TICKER),
        ({"tickers": ("COMMON STOCK",)}, Unverifiable.NO_TICKER),
    ],
)
def test_everything_unverifiable_says_why(
    prices: FixturePrices, kwargs: dict[str, object], reason: Unverifiable
) -> None:
    # A bare "unverifiable" is indistinguishable from a bug in the verifier.
    result = verify(make_claim(**kwargs), source=prices, today=TODAY, run_id="t")  # type: ignore[arg-type]
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.reason is reason


def test_a_security_name_is_not_a_ticker() -> None:
    assert resolve_ticker(make_claim(tickers=("Common Stock",))) is None
    assert resolve_ticker(make_claim(tickers=("MLHR", "MLKN"))) == "MLHR"
    assert resolve_ticker(make_claim(tickers=("BRK.B",))) == "BRK.B"


# ----------------------------------------------------------------------------------
# Lifting claims out of the log


def event_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "e1",
        "accession": "0000320193-26-000001",
        "event_type": "split_forward",
        "ex_date": dt.date(2026, 3, 5),
        "ratio": "3",
        "affected_securities": "FSPL",
    }
    row.update(overrides)
    return row


def test_a_whole_number_ratio_survives_the_round_trip() -> None:
    # `CorporateActionEvent.to_row` writes `str(Fraction(3, 1))`, which is "3" and not
    # "3/1". A parser that split on a slash would drop every whole-number forward split
    # in the log, which is most of them.
    claims = claims_from_rows([event_row()])
    assert claims[0].ratio == Ratio(3, 1)
    assert claims_from_rows([event_row(ratio="1/8")])[0].ratio == Ratio(1, 8)


def test_only_index_relevant_rows_become_claims() -> None:
    # A `no_index_action` conclusion asserts nothing about a price. Counting it as
    # unverifiable would pad the denominator with 8-K traffic.
    rows = [
        event_row(),
        event_row(event_type="no_index_action"),
        event_row(event_type="unresolved"),
    ]
    assert [c.event_type for c in claims_from_rows(rows)] == [EventType.SPLIT_FORWARD]


@pytest.mark.parametrize(
    "overrides",
    [
        {"ratio": "not-a-ratio"},
        {"ratio": "3/0"},
        {"ratio": None},
        {"ex_date": "the fifteenth"},
        {"ex_date": None},
        {"affected_securities": None},
        {"event_type": "banana"},
        {"accession": "nonsense"},
    ],
)
def test_log_content_never_takes_the_verifier_down(overrides: dict[str, object]) -> None:
    # The log is append-only and may hold whatever an older run wrote.
    claims_from_rows([event_row(**overrides)])


def test_a_stored_date_may_arrive_as_a_datetime_or_a_string() -> None:
    assert claims_from_rows([event_row(ex_date=dt.datetime(2026, 3, 5, 13, 30))])[0].ex_date == EX
    assert claims_from_rows([event_row(ex_date="2026-03-05")])[0].ex_date == EX


def test_verify_all_reports_every_verdict_even_at_zero(prices: FixturePrices) -> None:
    # A missing key reads as "not measured"; an explicit zero is a measurement.
    results = verify_all([make_claim()], source=prices, today=TODAY, run_id="t")
    assert verdict_counts(results) == {"verified": 1, "contradicted": 0, "unverifiable": 0}


# ----------------------------------------------------------------------------------
# The verdict log


def test_the_same_verdict_from_a_later_run_is_not_a_new_row(
    tmp_path: Path, prices: FixturePrices
) -> None:
    # `verification_id` excludes the run: a verdict is a statement about market data, and
    # re-reaching it next week is not a new fact.
    first = verify(make_claim(), source=prices, today=TODAY, run_id="verify-20260401")
    later = verify(make_claim(), source=prices, today=TODAY, run_id="verify-20260408")
    assert first.verification_id == later.verification_id
    with EventStore(tmp_path / "log.duckdb") as store:
        assert store.append_verifications([first.to_row()]) == 1
        assert store.append_verifications([later.to_row()]) == 0


def test_a_changed_verdict_appends_beside_its_predecessor(
    tmp_path: Path, prices: FixturePrices
) -> None:
    # D-001, applied to verdicts. What the verifier said last week stays said.
    before = verify(
        make_claim(ex_date=dt.date(2026, 6, 1)), source=prices, today=TODAY, run_id="r1"
    )
    after = verify(make_claim(), source=prices, today=TODAY, run_id="r2")
    with EventStore(tmp_path / "log.duckdb") as store:
        store.append_verifications([before.to_row()])
        store.append_verifications([after.to_row()])
        assert len(store.verifications(latest_only=False)) == 2
        latest = store.verifications(latest_only=True)
        assert len(latest) == 1
        assert latest.iloc[0]["verdict"] == "verified"
        assert store.verification_counts() == {"verified": 1}


def test_the_verifier_reads_the_latest_extraction_per_accession(tmp_path: Path) -> None:
    from conftest import make_event

    with EventStore(tmp_path / "log.duckdb") as store:
        first = make_event(run_id="r1", ex_date=dt.date(2026, 3, 5))
        store.append([first])
        store.append(
            [make_event(run_id="r2", ex_date=dt.date(2026, 4, 9), supersedes=first.event_id)]
        )
        rows = store.latest_per_accession()
    assert len(rows) == 1
    assert claims_from_rows(rows)[0].ex_date == dt.date(2026, 4, 9)


# ----------------------------------------------------------------------------------
# The trap this module exists to avoid


def test_no_price_source_reads_an_adjusted_series() -> None:
    """Adjusted prices erase the step being tested.

    A back-adjusted close has the split divided out of every price before the ex-date, so
    a 3-for-1 split leaves no step at all in the adjusted series. A verifier fed one
    contradicts every real split and verifies nothing, quietly - which is the failure
    mode nobody finds, so it is banned rather than documented.

    Over the AST rather than the source text, because this module has to be able to
    *write about* `adjclose` at length in order to explain why it is forbidden. A grep
    that cannot tell prose from code would either fire on the explanation or be watered
    down until it caught nothing.
    """
    offenders: list[str] = []
    for path in sorted(pathlib.Path("src/reviewradar").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            found = None
            if isinstance(node, ast.Constant) and node.value == "adjclose":
                found = 'the literal key "adjclose"'
            elif isinstance(node, ast.Attribute) and node.attr == "adjclose":
                found = "an .adjclose attribute"
            elif (
                isinstance(node, ast.keyword)
                and node.arg == "auto_adjust"
                and not (isinstance(node.value, ast.Constant) and node.value.value is False)
            ):
                found = "auto_adjust, which is how the convenience wrappers adjust by default"
            elif isinstance(node, ast.Import | ast.ImportFrom) and "yfinance" in ast.dump(node):
                found = "yfinance, which back-adjusts unless told not to"
            if found:
                offenders.append(f"{path}:{getattr(node, 'lineno', 0)}: {found}")
    assert not offenders, "a price series is being adjusted:\n" + "\n".join(offenders)


def test_the_fixtures_declare_that_they_are_constructed() -> None:
    # Unlike the EDGAR fixtures, these could not be recorded without a market-data fetch.
    # Every one says so in its own file, so nobody reads them as a tape.
    import json

    for file in sorted(PRICES.glob("*.json")):
        note = str(json.loads(file.read_text(encoding="utf-8"))["note"])
        assert "not recorded" in note, file


def test_every_committed_series_loads(prices: FixturePrices) -> None:
    assert set(prices.series) == {"FSPL", "RSPL", "FLAT", "HOLI", "NEAR", "WIDE", "SOON"}
    for ticker, series in prices.series.items():
        assert series, ticker
        assert list(series) == sorted(series, key=lambda row: row.session), ticker


def test_an_accession_is_still_an_accession(prices: FixturePrices) -> None:
    result = verify(make_claim(), source=prices, today=TODAY, run_id="t")
    assert isinstance(result.accession, str)
    assert result.accession == Accession("0000320193-26-000001")
