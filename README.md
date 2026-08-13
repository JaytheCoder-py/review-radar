# Review Radar

Corporate-action extraction and index-treatment classification over SEC 8-K filings.

A nightly job reads new 8-K filings from EDGAR, extracts corporate-action events into a
typed schema where **every field is grounded in a character span of the source document**,
classifies what an index calculator must do about it, and appends the result to an
immutable log. A read-only dashboard renders the log and computes nothing of its own.

```bash
uv sync
uv run pytest
uv run reviewradar ingest --date 2025-04-15
```

Design and plan: [`docs/specs/`](docs/specs/) · [`docs/plans/`](docs/plans/) ·
judgement calls: [`DECISIONS.md`](DECISIONS.md)

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

A deterministic baseline runs on **every** filing and eliminates the majority that carry
no index consequence. Only the residual reaches the model.

That split exists so the scoreboard can state what the model is *worth*, per field,
against a baseline that runs for free. A system that cannot answer "what would this cost
you if you deleted the LLM?" has not been engineered, it has been assembled.

_Status: in build. The scoreboard, the deployed URL and the measured business case land in
weeks 9–12 — see the plan._
