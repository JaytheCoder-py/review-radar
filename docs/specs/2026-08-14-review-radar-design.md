# Review Radar — design

**Status:** approved 2026-08-14 · **Owner:** Jason Chung · **Build mode:** agent-built
(Claude implements against this spec; the owner directs, reviews and owns every decision)

---

## 1. What this is

A nightly service that reads new SEC 8-K filings, extracts corporate-action events into a typed,
span-grounded schema, applies deterministic index-treatment rules, and publishes both the events
and a running accuracy scoreboard.

It runs as a scheduled Cloud Run job writing to an append-only event log, with a small read-only
FastAPI dashboard over the log.

It is deliberately small — a target of 3–4k lines — so the owner can read, question and defend
every line and every measured number in it.

## 2. Why it has business value

Corporate actions are the largest single source of index calculation error and the largest
manual-review burden in index Data Operations. Every split, merger, spin-off, rights issue and
special dividend must be detected, classified, dated and applied to the divisor **before the
ex-date**, or the index prints wrong and the error becomes a recalculation event.

Commercial feeds supply this, but they arrive late relative to the filing, they disagree with each
other, and they drop detail. Index providers therefore run reconciliation and exception queues
staffed by analysts.

A tool that reads the **primary source** — the filing itself — and emits a typed event with a
confidence and a citation is a real product, and the business case is quantifiable in the units
the business uses:

> *N filings ingested per night, X% classified with no index consequence and never touched by a
> human, Y% auto-extracted above the confidence threshold, Z% routed to review, at $C per filing.*

Those numbers come out of the scoreboard (§10), not out of a claim.

## 3. Scope

**In scope**

- SEC 8-K filings and their exhibits, US registrants, from EDGAR.
- Corporate actions with index consequence: forward and reverse splits, special dividends,
  completed mergers and acquisitions, spin-offs, rights issues, delistings, bankruptcies, ticker
  and name changes.
- Index treatment classification: what the event obliges an index calculator to do.
- Measurement: per-field precision/recall against a hand-labelled gold set, baseline vs model,
  cost and latency.
- Deployment: scheduled, live, publicly readable.

**Explicitly out of scope**

- Vendor-feed reconciliation. No free vendor corporate-actions feed exists, so there is nothing
  honest to reconcile against.
- UK / non-US markets. RNS full-feed access is commercially licensed.
- Index review prediction. Russell reconstitution is annual in June; there is no scored review
  cycle inside this project's window.
- Multi-model comparison, embeddings-based retrieval, and any agentic tool-use loop. Each adds
  evaluation surface without adding a claim the project needs.
- Authentication. The dashboard is public and read-only.

## 4. Known limits, stated up front

These are properties of the data, not defects to be fixed later. They belong in the README and in
any conversation about the tool.

1. **The 8-K is not the canonical source for every corporate action.** Splits and dividends are
   declared by the board and disseminated through the exchange and DTC; the 8-K is a disclosure
   that usually accompanies them, not the instrument. Some events never generate an 8-K at all.
   Coverage is therefore *incomplete by construction*, and the scoreboard must report recall
   against filings, not against the universe of corporate actions.
2. **Item codes do not identify corporate actions.** Item 2.01 (completed acquisition), 3.01
   (delisting notice), 1.03 (bankruptcy) and 5.01 (change in control) are genuine signals. But
   splits and special dividends overwhelmingly arrive under **Item 8.01 "Other Events"**, which is
   the dumping ground, or under 5.03 when a charter amendment is required. The baseline classifier
   cannot resolve 8.01 — and that is exactly where the model earns its cost (§6).
3. **The action is often in the exhibit, not the filing body.** A split announcement is typically
   an EX-99.1 press release attached to a near-empty 8-K. Document fetch must follow exhibits.
4. **EDGAR full-text search covers 2001 onward** and the daily index is the reliable enumeration
   path. Historical backfill beyond the gold set is not attempted.

## 5. Architecture

```
src/reviewradar/
  types.py              Cik, Accession, ExDate, Ratio, EventType, Treatment — domain primitives
  ingest/
    edgar.py            daily index -> filing metadata -> primary document + exhibits
    store.py            append-only event log (DuckDB), idempotent per (date, accession)
  extract/
    schema.py           CorporateActionEvent, Span, Extracted[T]; the JSON Schema for the model
    baseline.py         Item codes + phrase rules -> EventType or UNRESOLVED. Deterministic.
    llm.py              Vertex AI client behind a Protocol; retries, token/latency/cost accounting
    pipeline.py         baseline -> residual routing -> merge -> span validation -> confidence
  treatment/
    rules.py            event -> DIVISOR_ADJUST | SHARES_UPDATE | PRICE_ADJUST |
                        REMOVE_CONSTITUENT | NO_ACTION | MANUAL_REVIEW
  evals/
    gold.py             gold set loader; random and stratified strata kept separate
    score.py            per-field P/R/F1, baseline-vs-model delta, cost per filing
    forward.py          ex-post verification of past-dated extractions against market data
  service/
    app.py              FastAPI: /events, /scoreboard, /filing/{accession}
    job.py              nightly job entrypoint
  cli.py                typer
```

Each module has one job and a stated dependency direction. `ingest` knows nothing about
extraction. `extract` knows nothing about index treatment. `treatment` is a pure function of an
event. `evals` reads the log and never writes to it. `service` renders; it computes nothing.

That last boundary is enforced by a test, in the same spirit as
`test_desk_contains_no_index_arithmetic` in the miniftse reference build: **the service layer must
not recompute a published figure.**

## 6. The two-stage extraction, and why

**Stage 1 — baseline (deterministic, every filing).** Item codes plus a small phrase-rule table.
Its job is *cheap elimination*, not classification: Items 2.02 (results), 5.02 (officer changes),
7.01 (Reg FD) and 9.01 (exhibits only) carry no index consequence and account for the majority of
8-K volume. These terminate as `NO_INDEX_ACTION` and never reach the model.

Where the Item code *is* diagnostic — 2.01, 3.01, 1.03, 5.01 — the baseline assigns the event type
and passes the filing on for **field** extraction only.

Where the Item code is 8.01 or 5.03, the baseline returns `UNRESOLVED` and the model does both
classification and field extraction.

**Stage 2 — model (Vertex AI, residual only).** Structured output against `schema.py`. Returns
event type where unresolved, plus `ex_date`, `ratio`, `counterparty`, `affected_securities` — each
with a character span into the source document.

**Why this split rather than sending everything to the model:** it produces a *measured delta*.
The scoreboard can state what the model contributes, per field, against a baseline that runs for
free. A system that cannot answer "what would this cost you if you deleted the LLM?" has not been
engineered, it has been assembled.

## 7. Domain model

```python
Ratio = Fraction          # a 3-for-1 split is exactly 3/1, never 3.0
Cik = NewType("Cik", str)
Accession = NewType("Accession", str)

class EventType(StrEnum):
    SPLIT_FORWARD, SPLIT_REVERSE, SPECIAL_DIVIDEND, MERGER_COMPLETED,
    SPINOFF, RIGHTS_ISSUE, DELISTING, BANKRUPTCY, TICKER_CHANGE,
    NAME_CHANGE, NO_INDEX_ACTION, UNRESOLVED

class Treatment(StrEnum):
    DIVISOR_ADJUST, SHARES_UPDATE, PRICE_ADJUST, REMOVE_CONSTITUENT,
    NO_ACTION, MANUAL_REVIEW

@dataclass(frozen=True, slots=True)
class Span:
    doc_id: str      # accession, plus exhibit id where applicable
    start: int
    end: int
    text: str

@dataclass(frozen=True, slots=True)
class Extracted[T]:
    value: T
    span: Span | None          # None only for baseline-derived values
    source: Literal["baseline", "model"]
    confidence: float

@dataclass(frozen=True, slots=True)
class CorporateActionEvent:
    accession: Accession
    cik: Cik
    filed_at: datetime
    event_type: Extracted[EventType]
    ex_date: Extracted[date] | None
    ratio: Extracted[Ratio] | None
    counterparty_cik: Extracted[Cik] | None
    affected_securities: Extracted[tuple[str, ...]] | None
    treatment: Treatment
    confidence: float
    run_id: str
    supersedes: str | None
```

`Ratio` as `Fraction` is not decoration. A 1-for-8 reverse split held as `0.125` is fine; held as
`1/3` it is not, and float ratios are a recurring source of divisor error.

## 8. Design decisions

**D-001 — The event log is append-only.** An extraction that was wrong yesterday stays in the log;
a corrected extraction is a new record with `supersedes` set. The reason is the same one that makes
index providers keep a recalculation policy: the published history of what you *said* is itself a
record, and silently rewriting it destroys the ability to audit a decision made on it.

**D-002 — The baseline runs on every filing; the model runs on the residual.** Cost, and
measurability. Rejected: model-on-everything (simpler, ~10h cheaper, but yields no baseline and so
no defensible statement of what the model is worth).

**D-003 — Span grounding is mandatory.** Every model-derived field carries a character span, and
the pipeline **drops any field whose value does not appear within its own cited span**. Not flags —
drops. This is stricter than checking a number against a set of supplied facts, because it forces
the citation to be load-bearing rather than decorative. A dropped field lowers confidence and
pushes the filing toward `MANUAL_REVIEW`, which is the correct outcome.

**D-004 — Below-threshold confidence routes to MANUAL_REVIEW, never to a guess.** Abstention is a
measured metric, not a failure. A system that always answers is a system that guesses, and on a
corporate action a guess is a wrong divisor.

Confidence is **computed, not self-reported.** A model's own stated confidence is not a calibrated
probability and must not be treated as one. Confidence here is a deterministic function of three
observable facts: whether every field required by the assigned event type survived span validation
(§D-003), whether baseline and model agree where both produced an event type, and whether the
extracted `ex_date` is internally consistent with `filed_at`. The routing threshold is a
configuration value **calibrated against the random stratum of the gold set** — chosen to
hit a target precision on auto-accepted events — not picked in advance.

**D-005 — Vertex AI, not the direct vendor API.** One GCP project, service-account authentication,
no second vendor relationship, no key to leak into a repository. This is the answer a procurement
or infosec reviewer wants, and it costs nothing to choose correctly at the start.

**D-006 — The model is behind a Protocol with a deterministic offline implementation.** CI runs the
full pipeline with no credentials and no spend. Only the eval job, run manually, spends money.

**D-007 — The gold set is two strata, kept separate.** See §9.

## 9. The gold set

~200 hand-labelled filings. This is the least glamorous step of the build and the one that makes
every other number credible.

**Two strata, never pooled:**

- **Random stratum (~120 filings)** — a uniform sample of 8-K filings across a date range. Its
  purpose is the *population-level* rate: what fraction of real 8-K traffic has index consequence,
  what fraction the baseline eliminates correctly, what the true manual-review rate would be. This
  is the stratum the business case is computed from.
- **Stratified stratum (~80 filings)** — deliberately oversampled on Item 8.01 and 5.03 and on
  known split/spin-off/merger events. Its purpose is *per-event-type F1* with enough observations
  per class to mean anything. Rates computed on this stratum are not population rates and must
  never be quoted as such.

Pooling the two would produce a headline accuracy that is neither a population rate nor a per-class
rate. Reporting them separately is the whole point, and being able to explain why is worth more in
an interview than the accuracy figure itself.

The gold set is committed to the repository and is the regression suite: **CI fails if baseline F1
drops.**

## 10. The scoreboard

One page, and it is the deliverable.

| Metric | Baseline | + model | Delta |
|---|---|---|---|
| Event-type accuracy (stratified) | | | |
| Ex-date F1 | | | |
| Ratio F1 | | | |
| Affected-security F1 | | | |
| No-consequence elimination rate (random) | | — | — |
| Manual-review rate (random) | | | |
| Cost per filing | $0.00 | | |
| Latency p50 / p95 | | | |

Plus **forward verification**: for events whose extracted ex-date has now passed, check the
extraction against observed market data — a 3-for-1 split should show a ~3× share-count change and
a ~1/3 price step. Reported as *"of N past-dated events, M verified, K contradicted, L
unverifiable."* This half needs no hand-labelling and grows on its own every night the job runs.

## 11. Error handling

- **EDGAR rate limits at 10 requests/second and blocks IPs that omit a contact User-Agent.** The
  ingest client declares itself and backs off. This is a documented trap, not a hypothetical.
- **A filing that fails to parse is recorded as `FAILED` with the exception text — never skipped.**
  An empty result is indistinguishable from "this filing contained no event", and that ambiguity is
  precisely how a missed split reaches production.
- **A Vertex call that fails retries with backoff, then records `UNEXTRACTED`.** The nightly job
  never dies because one filing was pathological.
- **The job is idempotent per `(date, accession)`.** Re-running a date produces no duplicate
  events. This is a property test, not a convention.
- **Schema-invalid model output is a retry, then a drop to `MANUAL_REVIEW`** — never a partial
  parse of malformed JSON.

## 12. Testing

- **Recorded EDGAR fixtures.** No test touches the network.
- **The gold set is the regression suite.** A CI gate on baseline F1.
- **Property: every extracted field's value appears within its cited span.** Hypothesis-driven,
  over generated documents and spans.
- **Property: re-ingesting a date is a no-op.**
- **Offline model fake**, so CI runs green with no credentials.
- **Layer-boundary test:** the `service/` package contains no arithmetic that re-derives a stored
  figure.
- **Cost regression:** mean tokens per residual filing is tracked; CI warns on a material increase.

## 13. Deployment

- Cloud Run **job** (not a service) for the nightly ingest, triggered by Cloud Scheduler.
- Cloud Run **service** for the read-only dashboard, serving the committed log snapshot plus the
  live store.
- One GCP project. Service-account auth to Vertex. No API keys in the repository.
- GitHub Actions for CI; deploy is a manual `gcloud run deploy` documented in the README, not an
  automatic push-to-prod — the repository has one maintainer and an accidental deploy has no
  reviewer.

## 14. Build order

Repo and domain types → EDGAR ingest → append-only store → baseline classifier and first
measurement → gold set → model extraction with span grounding and cost accounting → eval
harness and CI gate → Cloud Run job + Scheduler (**live**) → dashboard → forward
verification and the write-up. Current state and remaining tasks live in the active build
plan under [`docs/plans/`](../plans/).

**The gold set is the step people skip and the one that makes every later number
credible. The live deployment is non-negotiable — if scope slips, cut the dashboard, not
the deployment.**

## 15. Relationship to the miniftse reference build

Review Radar is a separate repository and depends on no miniftse code. Three patterns are
deliberately carried across, and the reasoning — not the implementation — is what transfers:

- Provider behind a Protocol with an offline implementation, so CI needs no credentials.
- A failure that cannot be distinguished from an absence must raise, never return empty.
- The presentation layer renders stored figures and computes none of its own, enforced by a test.

The compressed miniftse training track (M1, M2, M3, M5, M6, M8, M10, M13, M15) is **suspended** for
the duration. miniftse stands as a domain reference to be read and defended at design level;
Review Radar is the code written line by line.
