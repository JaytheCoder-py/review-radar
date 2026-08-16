# Review Radar

Corporate-action extraction and index-treatment classification over SEC 8-K filings.

A nightly job reads new 8-K filings from EDGAR, extracts corporate-action events into a
typed schema where **every field is grounded in a character span of the source document**,
classifies what an index calculator must do about it, and appends the result to an
immutable log. A read-only dashboard renders the log and computes nothing of its own.

```bash
uv sync
uv run pytest                              # 178 tests, no network, no credentials
uv run reviewradar replay --baseline-only  # the pipeline over 399 real filings
uv run reviewradar score                   # the scoreboard
uv run reviewradar label --stratum random  # hand-label into the gold set
uv run reviewradar serve                   # the dashboard on :8080

# Past-dated extractions against the tape. `--source yahoo` for live unadjusted closes;
# the committed fixtures are constructed, and exist to exercise the arithmetic offline.
uv run reviewradar verify --prices tests/fixtures/prices
```

Design and plan: [`docs/specs/`](docs/specs/) · [`docs/plans/`](docs/plans/) · judgement
calls: [`DECISIONS.md`](DECISIONS.md) · what broke: [`docs/what_the_build_found.md`](docs/what_the_build_found.md)

---

## Why

Corporate actions are the largest single source of index calculation error and the largest
manual-review burden in index Data Operations. Every split, merger, spin-off, rights issue
and special dividend must be detected, classified, dated and applied to the divisor
**before the ex-date**, or the index prints wrong and the error becomes a recalculation
event.

Commercial feeds supply this, but they arrive late relative to the filing, disagree with
each other, and drop detail — so providers run exception queues staffed by analysts. This
reads the **primary source** and emits a typed event with a confidence and a citation.

## Two stages, and why

A deterministic baseline runs on **every** filing and eliminates the ones that carry no
index consequence. Only the residual reaches the model.

That split exists so the scoreboard can state what the model is *worth* against a baseline
that runs for free. A system that cannot answer "what would this cost you if you deleted
the LLM?" has not been engineered, it has been assembled.

The baseline itself has two rungs: **item codes**, then a **keyword screen over the body
text** that may only ever put a filing back. The second rung exists because the first one
was measured and found to discard real events — see below.

## Measured on 399 real filings

| | |
|---|---|
| Filings parsed | 399 / 399, **0 failures** |
| Unrecognised 8-K item descriptions | **0** |
| Eliminated by item codes alone | 41.9% |
| Eliminated after the keyword screen, no model call | **34.1%** |
| Residual sent to the model | 65.9% |
| Tests | **178**, no network, no credentials |
| `ruff`, `mypy --strict` | clean |

### The finding that matters

The elimination rate is not a result on its own. Against the hand-labelled gold set,
routing on item codes alone:

| Stratum | n | index-relevant | eliminated | **false eliminations** |
|---|---:|---:|---:|---:|
| random | 30 | 1 | 63.3% | 0 / 1 |
| stratified | 36 | 28 | 30.6% | **6 / 28 = 21.4%** |

A fifth of genuinely index-relevant filings were **silently discarded** by item-code
routing — including GE's April 2024 spin-off of GE Vernova, the largest US corporate
action of that year. All six were disclosed under Items 7.01, 9.01 or 2.02, which carry no
index consequence on their own, with the announcement in an attached press release.

A false elimination is invisible by construction: the filing never reaches a queue, never
reaches the model, never reaches a person. Nobody finds out until the index prints wrong.
So the CI gate is on the false-elimination rate, not on accuracy.

### What the 21.4% actually argued for

Not the model. The words "25% stock split" are in the IntegraMed document; "Spin-Off" is in
GE's. Item-code routing missed them because it never read the body. That is a regex
problem, and the fix is a **keyword rescue stage** (D-007): a screen over the primary
document *and its exhibits*, consulted before any elimination is allowed to stand, that can
only ever move a filing back to the model.

| | eliminated | false eliminations (stratified) |
|---|---:|---:|
| item codes alone | 41.9% | 6 / 28 = **21.4%** |
| item codes + keyword screen | **34.1%** | 0 / 28 = **0.0%** |

All six are recovered, for 7.8 points of elimination rate — 31 of the 167 filings item
codes would have discarded now go to the model. The stratified stratum's elimination rate
falls from 30.6% to 5.6%, which is the point: that stratum is 28 real events out of 36.

Two things make the screen affordable rather than useless:

- **The cover-page table.** Every 8-K filed since 2019 carries a "Trading Symbol(s)" block,
  so a bare `trading symbol` pattern fires on **84.4%** of the filings item codes eliminate
  and the screen stops screening. The pattern requires a *change* — "new", "changed to",
  "symbol … from" — which a listing table never contains.
- **Announcement context.** A match only counts when its own sentence also announces
  something: a declaration, an approval, a dated step, a future-tense consequence. Without
  it the screen fires on 31.7% of eliminated filings instead of 18.6% (elimination 28.6%
  rather than 34.1%), because every earnings release restates prior periods "to reflect the
  three-for-one stock split in 2004". `stock split` alone drops from 33 hits to 18.

**So what is the model for?** The field table `uv run reviewradar score` prints: the
baseline scores `NaN` on `ex_date`, `ratio`, `counterparty` and `affected_securities`,
because it never extracts a field. A regex can say *look at this filing*; it cannot say
*1-for-4, ex 13 April, cited here*. Detection is cheap. The fields, with provenance, are
not — and the fields are what moves a divisor.

The screen's own cost is visible rather than hidden: stratified `event_type` F1 falls from
0.33 to 0.25 and the manual-review rate rises from 58.1% to 65.9%, because a rescued filing
is reported `UNRESOLVED` and, with the model absent, lands in the queue. That is the
intended direction — the alternative is a confident wrong answer.

**The model column is unmeasured.** The Vertex client is written and the pipeline runs
against it end to end, but no run has been made — that needs the owner's own GCP project.
Every model figure here is absent rather than estimated.

## Forward verification, and the trap in it

An extraction is a claim about the future. "3-for-1, ex 5 March" predicts that the price
steps to about a third on the first session on or after that date. `reviewradar verify`
checks the claim against the tape and appends a verdict, which is the one half of the
scoreboard that needs no hand-labelling and grows on its own every night the job runs.

**Split-adjusted prices erase exactly the step being tested.** A back-adjusted close has
the split divided out of every price *before* the ex-date, so a 3-for-1 split leaves **no
step at all** in the adjusted series. A verifier fed one contradicts every correct
extraction and verifies nothing — while looking like it works, because the output is full
of confident verdicts. This is the default behaviour of the obvious library
(`yfinance`'s `auto_adjust=True`), so the source reads Yahoo's v8 chart endpoint directly
and takes `indicators.quote[0].close`, never `indicators.adjclose`. The ban is a test over
the AST of `src/`, not a comment.

The verdict is one of three, never two:

| verdict | meaning |
|---|---|
| `verified` | the unadjusted close stepped by the ratio, within the band |
| `contradicted` | the session came and went and the price did not step |
| `unverifiable` | no test was possible — **including every ex-date that has not passed** |

An unpassed ex-date is never a contradiction, and that is checked before any price is
fetched rather than falling out of an empty series. The band is a relative error of
**0.08**, which sits about four standard deviations above an ordinary daily move and about
three times below the weakest signal it must catch — an unchanged close on a 5/4 stock
split scores 0.25. What it buys and what it costs is worked through in
[D-008](DECISIONS.md), including the case it gets wrong: it cannot tell a 5/4 from a 13/10,
because those differ by 4% in price step. A `verified` verdict says *a step of about the
right size happened*, not *the ratio is exactly right*.

Only splits get a verdict. A spin-off ratio — one GE Vernova share per four GE shares — says
nothing about how far the parent's price falls; that step is the market value of the child.
Testing it against the ratio anyway would measure the verifier's model of the event rather
than the extraction.

**The production count is zero, and the reason is worth reading.** Replaying the 399-filing
corpus with the baseline alone puts 29 index-relevant events in the log — 26 delistings and
3 completed mergers, every one of them typed by a diagnostic item code. Not one is a split,
and none carries a ratio or an ex-date, because the baseline extracts no fields at all. So
`reviewradar verify` reports **0 verified, 0 contradicted, 29 unverifiable**, all of them
`event_type_has_no_price_test`:

```
399 stored extractions, 29 index-relevant claims
  verified          0
  contradicted      0
  unverifiable     29
```

That is the machinery working correctly on an empty case, and it is the model-shaped hole
in this repository showing up in a third place. The `/scoreboard` page says as much on the
page rather than presenting a blank table as a result.

What *is* measured is the arithmetic, against committed price fixtures — a forward split, a
reverse split, a split that never happened, an ex-date on a Saturday, and both sides of the
tolerance boundary. Those fixtures are **constructed rather than recorded**, because
building this repository must not require a market-data fetch and no free source permits
redistributing its tape. Every fixture file says so in its own `note`, and a test asserts
that it does.

## Design decisions worth knowing

**The event log is append-only.** A corrected extraction is a new row carrying
`supersedes`. What the system said yesterday is itself a record; a consumer may have acted
on it. Same logic as an index provider's recalculation policy.

**Span grounding, and dropping rather than flagging.** Every model-derived field carries a
character span, and a field whose value does not appear inside its own citation is
*dropped*. Flagging leaves a plausible wrong number in the record and someone downstream
will use it. Because models are poor at character arithmetic and good at quotation, the
citation is verified by *relocating* the quoted text and correcting the offsets — the
quotation must hold up, the offsets are a convenience.

**Confidence is computed, not self-reported.** A model's stated confidence is not a
calibrated probability. Confidence here is a function of three checkable facts: fields
dropped, agreement with a diagnostic item code, and whether the ex-date is possible given
the filing date.

**Ratios are `Fraction`, never `float`.** A 1-for-3 reverse split held as a float does not
invert cleanly, and the residue lands in a divisor.

**The model sits behind a Protocol with an offline double.** The full test suite and all of
CI run with no credentials and no spend.

## Known limits

- **The 8-K is not the canonical source for every corporate action.** Splits are declared
  by the board and disseminated through the exchange and DTC; some events generate no 8-K.
  Recall is measured against filings, not against the universe of corporate actions.
- **The gold set is 66 filings, not the 200 planned.** Per-class counts are thin. Read the
  21.4% as "roughly a fifth", not a point estimate — and read the 0 / 28 after the screen
  as "no known miss", not as recall of 1.0. `reviewradar label` is the tool for the rest;
  the labels are hand-applied, because a gold set labelled by the model under test is not
  gold.
- **The stratified stratum was built by phrase search**, so it over-represents filings
  whose language is explicit. That flatters the keyword screen specifically: a stratum
  selected by phrase is a stratum a phrase pattern was always going to find. On obliquely
  worded events both rungs of the baseline do worse than these numbers, not better, and
  that residue is the model's to catch.
- **Nothing is deployed yet.** [`deploy/README.md`](deploy/README.md) is the Cloud Run
  runbook; running it needs the owner's GCP account.

## Layout

```
src/reviewradar/
  types.py            Cik, Accession, Ratio, EventType, Treatment
  ingest/edgar.py     daily index, submissions, exhibits, normalisation
  ingest/store.py     append-only DuckDB event log
  extract/baseline.py item-code routing, then a rescue-only keyword screen
  extract/schema.py   the event, and the JSON Schema the model is held to
  extract/llm.py      LlmClient Protocol, offline double, Vertex client
  extract/pipeline.py span validation, confidence, orchestration
  treatment/rules.py  event -> what an index calculator must do
  evals/gold.py       the two strata, kept apart
  evals/score.py      false-elimination rate, per-field F1, cost
  evals/forward.py    extractions against the tape. Unadjusted closes only
  evals/labelling.py  the hand-labelling queue, parsers and self-agreement
  service/app.py      the dashboard. Renders; computes nothing
```

Reproduce the corpus: `uv run python scripts/fetch_corpus.py --contact you@example.com`,
then `scripts/fetch_stratified.py` and `scripts/build_manifest.py`. The raw filings
(~400 MB) are not committed; the manifest and the gold labels are.
