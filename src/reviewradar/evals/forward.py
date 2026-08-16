"""Forward verification: did the market do what the extraction said it would?

An extraction is a claim about the future. "3-for-1, ex 5 March" predicts that the
as-traded price steps to about a third of the prior close on the first session on or
after that date. This module checks the claim against the tape and produces a verdict.
No hand-labelling is involved and none is needed: the market is the label, and this half
of the scoreboard grows on its own every night the job runs.

**Adjusted prices lie, and they lie about exactly this.** A back-adjusted close series
has the split divided out of every price *before* the ex-date, so a 3-for-1 split leaves
no step at all in the adjusted series - the signal being tested is precisely the thing
the adjustment removes. A verifier fed adjusted closes contradicts every real split it
sees and verifies nothing, silently, while looking like it works. Every source behind
`PriceSource` must therefore serve **unadjusted, as-traded** closes;
`YahooChartPrices` reads `indicators.quote[0].close` and never `indicators.adjclose`.
That requirement is the reason this is a Protocol with a written contract instead of a
convenient library call, because the convenient library calls adjust by default (D-008).

**Scope.** A verdict is produced only where the ratio *determines* the price step, which
is splits and nothing else. A spin-off ratio says how many child shares are distributed
per parent share; the parent's step is the market value of the child, which no ratio
gives you. A rights issue's step turns on the subscription price, which this schema does
not extract. Both are `unverifiable` with a stated reason, as is every event with no
ex-date, no ratio or no resolvable ticker. A guessed verdict sitting next to a measured
one is worse than no verdict.

**An unpassed ex-date is never a contradiction.** That asymmetry is checked before any
price is fetched, so it is structural rather than a consequence of empty data: absence of
evidence about a future date is not evidence against the claim.

Nothing here writes. `verify_all` returns verdicts and the caller persists them, which
keeps the spec's dependency direction - `evals` reads the log and never writes to it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from reviewradar.types import INDEX_RELEVANT, Accession, EventType, Ratio, parse_accession

if TYPE_CHECKING:
    # Type-only, mirroring `extract.llm`: nothing in the verification path imports a
    # network library until someone deliberately constructs the live source.
    import requests


class Verdict(StrEnum):
    """What the tape said about the claim.

    Three outcomes, never two. Collapsing `unverifiable` into `contradicted` would turn
    every event we cannot price - a spin-off, a future ex-date, a private company - into
    an accusation against the extractor, and the queue would fill with noise until
    nobody read it.
    """

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Unverifiable(StrEnum):
    """Why no verdict was reachable. Always recorded; never left blank.

    A bare "unverifiable" is indistinguishable from a bug in the verifier, which is the
    same argument that makes `ingest.store.record_failure` mandatory.
    """

    EVENT_TYPE_HAS_NO_PRICE_TEST = "event_type_has_no_price_test"
    NO_EX_DATE = "no_ex_date"
    NO_RATIO = "no_ratio"
    NO_TICKER = "no_ticker"
    EX_DATE_IN_FUTURE = "ex_date_in_future"
    NO_SESSION_YET = "no_session_yet"
    NO_PRIOR_CLOSE = "no_prior_close"
    NO_PRICE_DATA = "no_price_data"
    UNUSABLE_PRICE = "unusable_price"


#: One line per reason, for the dashboard and the CLI. Presentation text, not a figure:
#: the reason code in the row is the record, this is how it reads to a human.
REASON_TEXT: Final[dict[Unverifiable, str]] = {
    Unverifiable.EVENT_TYPE_HAS_NO_PRICE_TEST: (
        "the ratio does not determine a price step for this event type"
    ),
    Unverifiable.NO_EX_DATE: "no ex-date was extracted",
    Unverifiable.NO_RATIO: "no ratio was extracted",
    Unverifiable.NO_TICKER: "no ticker could be resolved from the extraction",
    Unverifiable.EX_DATE_IN_FUTURE: "the ex-date has not passed",
    Unverifiable.NO_SESSION_YET: "no session has closed on or after the ex-date",
    Unverifiable.NO_PRIOR_CLOSE: "no session closed before the ex-date within the window",
    Unverifiable.NO_PRICE_DATA: "the price source returned nothing for this ticker",
    Unverifiable.UNUSABLE_PRICE: "a close in the window was zero or negative",
}


#: Event types whose ratio fixes the expected price step. A strict subset of
#: `RATIO_BEARING`: a spin-off and a rights issue carry a ratio that says nothing about
#: how far the price moves, so they are `unverifiable` rather than tested against a
#: number that does not apply to them.
VERIFIABLE_TYPES: Final[frozenset[EventType]] = frozenset(
    {EventType.SPLIT_FORWARD, EventType.SPLIT_REVERSE}
)

#: Relative error at which an observed step stops counting as the expected one. See
#: D-008 for the arithmetic: 0.08 sits about four standard deviations above an ordinary
#: single-stock daily move and about three times below the weakest step this system will
#: ever be asked to confirm (a 5/4 split, whose "nothing happened" signal is 0.25).
TOLERANCE: Final[Fraction] = Fraction("0.08")

#: Calendar days fetched either side of the ex-date. The longest gap in the US session
#: calendar is a holiday weekend; 14 days clears that and a short halt besides. A stock
#: suspended for longer produces `NO_PRIOR_CLOSE`, which is the honest answer.
PRICE_WINDOW_DAYS: Final[int] = 14

#: A ticker as a filing writes it. Deliberately narrow: the model is asked for symbols
#: "as written in the filing" and sometimes returns a security *name*, which would send
#: the price source looking for a company called "Common Stock".
_TICKER = re.compile(r"\A[A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z0-9]{1,3})?\Z")


def _exact(price: float) -> Fraction:
    """A quoted close as an exact rational.

    `Fraction(str(x))`, never `Fraction(x)`: D-005 says nothing in this codebase builds a
    rational out of a float's binary expansion. At an 8% band the residue could not
    change a verdict - the rule is kept anyway, because the harmless exception is how the
    rule dies.
    """
    return Fraction(str(price))


# ----------------------------------------------------------------------------------
# The price source


@dataclass(frozen=True, slots=True)
class DailyClose:
    """One session's **unadjusted** closing price. See the module docstring."""

    session: dt.date
    close: float


@runtime_checkable
class PriceSource(Protocol):
    """The only surface the verifier knows about.

    The contract is one sentence and it is the whole design: `closes` returns as-traded
    closes, in session order, with nothing divided out. An implementation that returns an
    adjusted series satisfies the type and destroys the measurement.
    """

    name: str

    def closes(self, ticker: str, start: dt.date, end: dt.date) -> Sequence[DailyClose]: ...


@dataclass
class FixturePrices:
    """Deterministic double. No network, no keys, no clock.

    The same role `OfflineLlm` plays for the model (D-006): the entire test suite and CI
    run against this, and the live source is constructed only when somebody asks for it
    by name. Series are committed as JSON under `tests/fixtures/prices/`.

    Unlike the EDGAR fixtures, these are **constructed rather than recorded** - building
    the repository must not require a market-data fetch, so there is no honest way to
    commit a real tape here. Each file says so in its own `note`. What they test is the
    arithmetic and the calendar handling; what they cannot test is whether a live source
    returns adjusted prices, which is why that contract is written down instead.
    """

    series: dict[str, tuple[DailyClose, ...]] = field(default_factory=dict)
    name: str = "fixtures"
    calls: int = 0

    @classmethod
    def from_dir(cls, path: Path | str) -> FixturePrices:
        """Load every `*.json` series in a directory."""
        loaded: dict[str, tuple[DailyClose, ...]] = {}
        for file in sorted(Path(path).glob("*.json")):
            raw = json.loads(file.read_text(encoding="utf-8"))
            ticker = str(raw["ticker"]).upper()
            loaded[ticker] = tuple(
                DailyClose(session=dt.date.fromisoformat(day), close=float(close))
                for day, close in raw["closes"]
            )
        return cls(series=loaded)

    def add(self, ticker: str, closes: Iterable[tuple[str, float]]) -> None:
        self.series[ticker.upper()] = tuple(
            DailyClose(session=dt.date.fromisoformat(day), close=close) for day, close in closes
        )

    def closes(self, ticker: str, start: dt.date, end: dt.date) -> tuple[DailyClose, ...]:
        self.calls += 1
        return tuple(
            row for row in self.series.get(ticker.upper(), ()) if start <= row.session <= end
        )


class YahooChartPrices:
    """As-traded closes from Yahoo's v8 chart endpoint. No key, no account.

    **`quote[0].close`, never `adjclose`.** The same response carries both series and
    they differ by exactly the corporate actions this module exists to check. Taking the
    adjusted one would make every real split look like a non-event, and the failure would
    be silent - the verifier would report `contradicted` on correct extractions and
    nobody would have a reason to doubt it. `yfinance` and most convenience wrappers
    adjust by default, which is why this reads the endpoint directly (D-008).

    Constructed only by `reviewradar verify --source yahoo`. `requests` is imported here
    rather than at module scope so that importing `evals.forward` - which every test
    does - pulls in no HTTP machinery at all, the same arrangement as `VertexLlm`.
    """

    ENDPOINT: Final = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, *, user_agent: str = "reviewradar-research/0.1", timeout: float = 20.0):
        import requests

        self.name = "yahoo-v8-unadjusted"
        self.timeout = timeout
        self._session: requests.Session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def closes(self, ticker: str, start: dt.date, end: dt.date) -> tuple[DailyClose, ...]:
        """Daily closes over `[start, end]`, inclusive. Empty on any failure.

        Empty rather than raising: one delisted ticker must not take down a weekly run,
        and the caller turns an empty series into `NO_PRICE_DATA`, which is a recorded
        outcome rather than a swallowed one.
        """
        params: dict[str, int | str] = {
            "period1": int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp()),
            # +1 day: the endpoint's period2 is exclusive of the session that opens on it.
            "period2": int(
                dt.datetime.combine(end + dt.timedelta(days=1), dt.time(), dt.UTC).timestamp()
            ),
            "interval": "1d",
        }
        try:
            response = self._session.get(
                f"{self.ENDPOINT}/{ticker}", params=params, timeout=self.timeout
            )
            if response.status_code != 200:
                return ()
            payload: Any = response.json()
            result = payload["chart"]["result"][0]
            stamps: list[int] = result["timestamp"]
            # `indicators.quote[0].close` is the as-traded series. `indicators.adjclose`
            # is in the same payload and must not be read here.
            values: list[float | None] = result["indicators"]["quote"][0]["close"]
            offset = int(result.get("meta", {}).get("gmtoffset", 0))
        except Exception:
            return ()
        rows: list[DailyClose] = []
        for stamp, value in zip(stamps, values, strict=False):
            if value is None:
                continue  # a halted session: Yahoo returns a null rather than dropping it
            session = dt.datetime.fromtimestamp(stamp + offset, tz=dt.UTC).date()
            rows.append(DailyClose(session=session, close=float(value)))
        return tuple(rows)


# ----------------------------------------------------------------------------------
# The claim, lifted out of the log


@dataclass(frozen=True, slots=True)
class Claim:
    """One stored extraction's testable content.

    Deliberately thin. The verifier reads the log, not the filings, so everything it can
    know about a filing is in this object - which is also why the only ticker available
    is the one the model extracted and cited.
    """

    accession: Accession
    event_id: str
    event_type: EventType
    ex_date: dt.date | None = None
    ratio: Ratio | None = None
    tickers: tuple[str, ...] = ()


def _as_date(value: object) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _as_ratio(value: object) -> Ratio | None:
    """Parse the stored ratio column.

    `Fraction(text)` rather than a split on "/": the column is written by
    `CorporateActionEvent.to_row` as `str(Fraction)`, and `str(Fraction(3, 1))` is
    **"3"**, not "3/1". A parser that insisted on a slash would drop every whole-number
    forward split in the log, which is most of them.
    """
    if isinstance(value, Fraction):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        ratio = Fraction(value.strip())
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 else None


def _as_tickers(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def claims_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[Claim]:
    """Lift the event-log rows that claim something about the tape.

    Only `INDEX_RELEVANT` rows become claims. A `no_index_action` conclusion asserts
    nothing about a price, and counting it as `unverifiable` would pad the denominator
    with hundreds of filings nobody made a claim about - the verified fraction would then
    move with 8-K traffic instead of with extraction quality.

    Never raises on log content. The log is append-only and may hold anything an older
    run wrote; a row that will not parse becomes a claim with the offending field missing,
    which the verifier reports as `unverifiable` with a reason.
    """
    claims: list[Claim] = []
    for row in rows:
        try:
            event_type = EventType(str(row.get("event_type")))
        except ValueError:
            continue
        if event_type not in INDEX_RELEVANT:
            continue
        try:
            accession = parse_accession(str(row.get("accession")))
        except ValueError:
            continue
        claims.append(
            Claim(
                accession=accession,
                event_id=str(row.get("event_id") or ""),
                event_type=event_type,
                ex_date=_as_date(row.get("ex_date")),
                ratio=_as_ratio(row.get("ratio")),
                tickers=_as_tickers(row.get("affected_securities")),
            )
        )
    return claims


def resolve_ticker(claim: Claim) -> str | None:
    """The symbol to price, or `None`.

    The only candidate is `affected_securities`, which the model extracted and which the
    pipeline already made earn its place: `_value_supported` drops the field unless every
    symbol appears inside its own cited span, so a ticker that reaches the log was quoted
    from the filing.

    The 8-K cover page's "Trading Symbol(s)" table is the other place a symbol lives, and
    it is out of reach here on purpose - this module reads the log, not the archive, and
    the cover page names the *registrant*, which on a spin-off or a merger is routinely
    not the security whose price moves.

    Where several symbols were extracted, the first is used and recorded. Returning
    `None` is a normal outcome, not an error.
    """
    for candidate in claim.tickers:
        if _TICKER.match(candidate):
            return candidate
    return None


# ----------------------------------------------------------------------------------
# The verdict


@dataclass(frozen=True, slots=True)
class Verification:
    """One verdict about one stored extraction.

    Keyed on `event_id`, not on the accession: the log is append-only, so a filing can
    carry several extractions and a verdict belongs to the exact row it tested.
    """

    accession: Accession
    event_id: str
    verdict: Verdict
    price_source: str
    run_id: str
    ticker: str | None = None
    reason: Unverifiable | None = None
    ex_date: dt.date | None = None
    session: dt.date | None = None
    prior_session: dt.date | None = None
    prior_close: float | None = None
    post_close: float | None = None
    expected_step: Ratio | None = None
    observed_step: Ratio | None = None
    relative_error: Fraction | None = None
    tolerance: Fraction = TOLERANCE

    @property
    def verification_id(self) -> str:
        """Content hash of the verdict and the evidence for it.

        `run_id` is deliberately **not** in the hash, which is the one place this differs
        from `CorporateActionEvent.event_id`. A verdict is a statement about market data,
        not about a run: re-checking the same session next week and reaching the same
        answer is not a new fact, and recording it as one would bury the verdict changes
        that matter under a weekly heartbeat. A verdict that *does* change appends
        alongside the old one, unchanged (D-001).
        """
        payload = json.dumps(
            {
                "event_id": self.event_id,
                "verdict": self.verdict.value,
                "reason": self.reason.value if self.reason else None,
                "ticker": self.ticker,
                "session": self.session.isoformat() if self.session else None,
                "prior_close": self.prior_close,
                "post_close": self.post_close,
                "source": self.price_source,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def to_row(self) -> dict[str, Any]:
        """Flat representation for the verdict log."""
        return {
            "verification_id": self.verification_id,
            "event_id": self.event_id,
            "accession": self.accession,
            "ticker": self.ticker,
            "verdict": self.verdict.value,
            "reason": self.reason.value if self.reason else None,
            "ex_date": self.ex_date,
            "session": self.session,
            "prior_session": self.prior_session,
            "prior_close": self.prior_close,
            "post_close": self.post_close,
            "expected_step": (
                f"{self.expected_step.numerator}/{self.expected_step.denominator}"
                if self.expected_step is not None
                else None
            ),
            "observed_step": (
                float(self.observed_step) if self.observed_step is not None else None
            ),
            "relative_error": (
                float(self.relative_error) if self.relative_error is not None else None
            ),
            "tolerance": float(self.tolerance),
            "price_source": self.price_source,
            "run_id": self.run_id,
        }


def expected_step(ratio: Ratio) -> Ratio:
    """The factor the as-traded price is expected to move by.

    Value is conserved across a split: `old_price * old_shares == new_price * new_shares`.
    The schema states the ratio as **new shares per old** - a three-for-one forward split
    is `3/1`, a one-for-eight reverse split is `1/8`, and IntegraMed's 25% stock split is
    `5/4` (`extract/schema.py`, and the labels in `data/gold/`). So the price moves by the
    reciprocal: a forward split divides it, a reverse split multiplies it, and 5/4 takes
    it to four fifths - a 20% fall for a 25% increase in shares.
    """
    return Ratio(1, 1) / ratio


def verify(
    claim: Claim,
    *,
    source: PriceSource,
    today: dt.date,
    run_id: str,
    tolerance: Fraction = TOLERANCE,
) -> Verification:
    """One claim against the tape.

    The order of the checks matters. The future-ex-date test runs before any fetch, so
    "not yet" can never be reached by way of an empty price series and can never come out
    as `contradicted`.
    """

    def unresolved(reason: Unverifiable, **extra: Any) -> Verification:
        return Verification(
            accession=claim.accession,
            event_id=claim.event_id,
            verdict=Verdict.UNVERIFIABLE,
            reason=reason,
            price_source=source.name,
            run_id=run_id,
            ex_date=claim.ex_date,
            tolerance=tolerance,
            **extra,
        )

    if claim.event_type not in VERIFIABLE_TYPES:
        return unresolved(Unverifiable.EVENT_TYPE_HAS_NO_PRICE_TEST)
    if claim.ex_date is None:
        return unresolved(Unverifiable.NO_EX_DATE)
    if claim.ratio is None:
        return unresolved(Unverifiable.NO_RATIO)
    ticker = resolve_ticker(claim)
    if ticker is None:
        return unresolved(Unverifiable.NO_TICKER)
    if claim.ex_date > today:
        # Before any price is fetched. An event whose ex-date has not passed is
        # unverifiable and can never be contradicted.
        return unresolved(Unverifiable.EX_DATE_IN_FUTURE, ticker=ticker)

    window = sorted(
        source.closes(
            ticker,
            claim.ex_date - dt.timedelta(days=PRICE_WINDOW_DAYS),
            claim.ex_date + dt.timedelta(days=PRICE_WINDOW_DAYS),
        ),
        key=lambda row: row.session,
    )
    if not window:
        return unresolved(Unverifiable.NO_PRICE_DATA, ticker=ticker)

    # The first session that *closed* on or after the ex-date. An ex-date on a Saturday,
    # a holiday, or today-before-the-bell has no close of its own; the step shows up in
    # the next session that traded, and until that session exists there is nothing to
    # contradict.
    post_index = next((i for i, row in enumerate(window) if row.session >= claim.ex_date), None)
    if post_index is None:
        return unresolved(Unverifiable.NO_SESSION_YET, ticker=ticker)
    if post_index == 0:
        return unresolved(Unverifiable.NO_PRIOR_CLOSE, ticker=ticker)

    post, prior = window[post_index], window[post_index - 1]
    if prior.close <= 0 or post.close <= 0:
        return unresolved(
            Unverifiable.UNUSABLE_PRICE,
            ticker=ticker,
            session=post.session,
            prior_session=prior.session,
        )

    expected = expected_step(claim.ratio)
    observed = _exact(post.close) / _exact(prior.close)
    # Relative error of the observed step against the expected one. Relative rather than
    # absolute because the same absolute miss means different things at 3/1 and at 5/4 -
    # and note that a step that never happened scores exactly `ratio - 1`, which is what
    # D-008's band is set against.
    relative_error = abs(observed / expected - 1)

    return Verification(
        accession=claim.accession,
        event_id=claim.event_id,
        verdict=Verdict.VERIFIED if relative_error <= tolerance else Verdict.CONTRADICTED,
        price_source=source.name,
        run_id=run_id,
        ticker=ticker,
        ex_date=claim.ex_date,
        session=post.session,
        prior_session=prior.session,
        prior_close=prior.close,
        post_close=post.close,
        expected_step=expected,
        observed_step=observed,
        relative_error=relative_error,
        tolerance=tolerance,
    )


def verify_all(
    claims: Sequence[Claim],
    *,
    source: PriceSource,
    today: dt.date,
    run_id: str,
    tolerance: Fraction = TOLERANCE,
) -> list[Verification]:
    """Every claim, in accession order. Returns; does not persist."""
    return [
        verify(claim, source=source, today=today, run_id=run_id, tolerance=tolerance)
        for claim in sorted(claims, key=lambda c: (c.accession, c.event_id))
    ]


def verdict_counts(verifications: Iterable[Verification]) -> dict[str, int]:
    """Counts by verdict, with every verdict present even at zero.

    Present at zero on purpose: a missing key reads as "not measured", and the whole
    point of this section of the scoreboard is that an empty verdict log is an honest
    statement about a pipeline that has not run yet, not a gap in the reporting.
    """
    counts = {verdict.value: 0 for verdict in Verdict}
    for verification in verifications:
        counts[verification.verdict.value] += 1
    return counts
