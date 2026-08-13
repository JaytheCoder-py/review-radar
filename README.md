# Review Radar

Corporate-action extraction and index-treatment classification over SEC 8-K filings.

A nightly job reads new 8-K filings from EDGAR, extracts corporate-action events into a
typed schema where **every field is grounded in a character span of the source document**,
classifies what an index calculator must do about it, and appends the result to an
immutable log. A read-only dashboard renders the log and computes nothing of its own.

```bash
uv sync
uv run pytest                                    # 97 tests, no network, no credentials
uv run reviewradar replay --baseline-only        # the pipeline over 399 real filings
uv run reviewradar score                         # the scoreboard
uv run reviewradar serve                         # the dashboard on :8080
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

## Measured on 399 real filings

| | |
|---|---|
| Filings parsed | 399 / 399, **0 failures** |
| Unrecognised 8-K item descriptions | **0** |
| Eliminated by the baseline, no model call | **41.9%** |
| Residual sent to the model | 58.1% |
| Tests | **97**, no network, no credentials |
| `ruff`, `mypy --strict` | clean |

### The finding that matters

The elimination rate is not a result on its own. Against the hand-labelled gold set:

| Stratum | n | index-relevant | eliminated | **false eliminations** |
|---|---:|---:|---:|---:|
| random | 30 | 1 | 63.3% | 0 / 1 |
| stratified | 36 | 28 | 30.6% | **6 / 28 = 21.4%** |

A fifth of genuinely index-relevant filings are **silently discarded** by item-code
routing — including GE's April 2024 spin-off of GE Vernova, the largest US corporate
action of that year. All six were disclosed under Items 7.01, 9.01 or 2.02, which carry no
index consequence on their own, with the announcement in an attached press release.

This is the structural limit of routing on item codes, and it is the argument for the
second stage. A false elimination is invisible by construction: the filing never reaches a
queue, never reaches the model, never reaches a person. Nobody finds out until the index
prints wrong. So the CI gate is on the false-elimination rate, not on accuracy.

**The model column is unmeasured.** The Vertex client is written and the pipeline runs
against it end to end, but no run has been made — that needs the owner's own GCP project.
Every model figure here is absent rather than estimated.

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
  21.4% as "roughly a fifth", not a point estimate.
- **The stratified stratum was built by phrase search**, so it over-represents filings
  whose language is explicit. The baseline probably does worse than 21.4% on obliquely
  worded events, not better.
- **Nothing is deployed yet.** [`deploy/README.md`](deploy/README.md) is the Cloud Run
  runbook; running it needs the owner's GCP account.

## Layout

```
src/reviewradar/
  types.py            Cik, Accession, Ratio, EventType, Treatment
  ingest/edgar.py     daily index, submissions, exhibits, normalisation
  ingest/store.py     append-only DuckDB event log
  extract/baseline.py item-code routing. Elimination, not classification
  extract/schema.py   the event, and the JSON Schema the model is held to
  extract/llm.py      LlmClient Protocol, offline double, Vertex client
  extract/pipeline.py span validation, confidence, orchestration
  treatment/rules.py  event -> what an index calculator must do
  evals/gold.py       the two strata, kept apart
  evals/score.py      false-elimination rate, per-field F1, cost
  service/app.py      the dashboard. Renders; computes nothing
```

Reproduce the corpus: `uv run python scripts/fetch_corpus.py --contact you@example.com`,
then `scripts/fetch_stratified.py` and `scripts/build_manifest.py`. The raw filings
(~400 MB) are not committed; the manifest and the gold labels are.
