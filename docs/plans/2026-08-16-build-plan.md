# Review Radar — build plan

**Status:** active 2026-08-16 · **Owner:** Jason Chung · **Build mode:** agent-built.
Claude orchestrates and implements; the owner directs, reviews and owns every decision.

**Goal:** a deployed, scheduled service that extracts corporate-action events from SEC 8-K
filings, classifies their index treatment, and publishes a measured accuracy scoreboard.

**Spec:** [`docs/specs/2026-08-14-review-radar-design.md`](../specs/2026-08-14-review-radar-design.md)
— unchanged and still authoritative.

## Global constraints

Every task inherits these.

- `mypy --strict` passes on `src/` at all times. A task is not done if it does not typecheck.
- `ruff check` and `ruff format --check` pass on `src/` and `tests/`.
- **No test touches the network.** All EDGAR interaction in tests runs against committed fixtures.
- **CI runs with no credentials and no spend.** The model sits behind a Protocol with a
  deterministic offline implementation.
- Every extracted field carries a `Span`; a field whose value does not appear inside its own
  cited span is dropped, not returned (spec D-003).
- The event log is append-only (spec D-001). Ratios are `Fraction`, never `float`.
- Small commits, present-tense messages that say *why*. Every judgement call goes in
  `DECISIONS.md` with the alternative rejected.
- **Every claimed number is measured, not asserted.** A change to the baseline re-runs the
  replay and the scoreboard, and the README tables move with it.

## State as of 2026-08-16

Built and measured: ingest, append-only store, item-code baseline, gold set (66 filings,
two strata), span-grounded extraction pipeline with offline model double, scoreboard,
dashboard. 97 tests, no network, `ruff` + `mypy --strict` clean. 399-filing corpus replayed
with 0 parse failures; baseline eliminates 41.9% with a measured 21.4% false-elimination
rate on the stratified stratum.

Not built: the keyword rescue stage (tests exist and fail at import), the real Vertex run
(client written, never invoked), deployment, forward verification.

## Remaining tasks

### 1. Keyword rescue stage — in flight

`tests/test_keyword_screen.py` is the contract. Implement `CORPORATE_ACTION_PATTERNS`
(data, not branches — D-003's reviewability argument) and `screen()` in
`extract/baseline.py`, consulted before elimination. The screen may only rescue, never
eliminate. The hard requirement is the tense distinction: a completed split mentioned in an
earnings release must not be rescued, while a prospective one filed under Item 7.01 must be.
Reconcile `test_baseline.py::test_the_baseline_silently_discards_a_real_split` — it asserts
pre-screen behaviour and its docstring still carries the falsified "21.4% justifies the
model" claim; the measured argument for the model is field extraction with provenance, not
detection. Re-run the replay and update the measured elimination figures in the README.

### 2. Real Vertex run — blocked on owner's GCP project

Run the pipeline with `VertexLlm` over the gold set. Record per-field F1 against both
strata separately, mean tokens, p50/p95 latency, cost per filing. Fill the model column of
the scoreboard, which is currently absent rather than estimated. Calibrate the confidence
threshold against the random stratum and record the target precision in `DECISIONS.md`.

### 3. Deploy — blocked on owner's GCP project

Cloud Run **job** (not service) + Cloud Scheduler, weekdays after EDGAR's daily index
publishes. Multi-stage Dockerfile, service account holding only Vertex invoke and storage
write, no baked credentials. Job exits non-zero on any non-per-filing failure. Runbook in
`deploy/README.md`. Let it run unattended for three nights; write up the first real failure
in `docs/incident_001.md` — cause, detection, fix, prevention.

### 4. Forward verification

`evals/forward.py`: check extracted events against subsequent market data. A 3-for-1 split
whose price steps ~1/3 on the extracted ex-date is `verified`; the same split with no price
step is `contradicted`; an event whose ex-date has not passed is `unverifiable`, never
`contradicted`. Tolerance band justified in `DECISIONS.md`. Scheduled weekly, surfaced on
`/scoreboard`.

### 5. Gold set expansion — owner labels, agent tools

Grow 66 toward 200 (120 random / 80 stratified). Labels are hand-applied by the owner —
a gold set labelled by the model under test is not gold. Agent work is limited to the
labelling CLI and sampling. Owner re-labels 20 blind after a week to measure
self-agreement; the number goes in `DECISIONS.md`.

### 6. Write-up

README scoreboard with real model numbers, the business case in units (filings/night,
auto-handled %, review %, cost/filing, implied hours), known limits stated plainly, and
`docs/what_the_build_found.md` kept current — including everything that did *not* work
and what the measurement said about it.

## Sequencing

Task 1 has no dependencies and is in flight. Tasks 2 and 3 need the owner's GCP project
and can proceed in either order once unblocked. Task 4 is independent of 2–3. Task 5 runs
whenever the owner has labelling time. Task 6 lands last.
