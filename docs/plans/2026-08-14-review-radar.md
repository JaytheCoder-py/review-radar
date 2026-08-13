# Review Radar — implementation plan

> **Tutor mode. This plan is deliberately NOT executable by an agent.** The standard plan
> format supplies the implementation code; this one withholds it on purpose. For every task
> the tutor supplies the file layout, the exact interfaces, and the failing tests. **Jason
> writes every line of implementation.** The tutor reviews at the checkpoint and asks the
> self-check questions. Do not paste implementations into this document, and do not dispatch
> subagents to complete tasks.

**Goal:** A deployed, scheduled service that extracts corporate-action events from SEC 8-K
filings, classifies their index treatment, and publishes a measured accuracy scoreboard.

**Architecture:** Two-stage extraction. A deterministic baseline runs on every filing and
eliminates the majority with no index consequence; the residual goes to a model for field
extraction, with every extracted field grounded in a character span of the source document.
Results append to an immutable log. A read-only dashboard renders the log and computes nothing.

**Tech stack:** Python 3.12+, `uv`, DuckDB, FastAPI, `typer`, `pytest` + `hypothesis`, `ruff`,
`mypy --strict`, GitHub Actions, Cloud Run + Cloud Scheduler, Vertex AI.

**Spec:** [`docs/specs/2026-08-14-review-radar-design.md`](../specs/2026-08-14-review-radar-design.md)

## Global constraints

Every task inherits these. They are not restated per task.

- `mypy --strict` passes on `src/` at all times. A task is not done if it does not typecheck.
- `ruff check` and `ruff format --check` pass on `src/` and `tests/`.
- **No test touches the network.** All EDGAR interaction in tests runs against committed fixtures.
- **CI runs with no credentials and no spend.** The model sits behind a Protocol with a
  deterministic offline implementation.
- Every extracted field carries a `Span`, and a field whose value does not appear inside its own
  cited span is dropped, not returned (spec D-003).
- The event log is append-only. Nothing is ever updated in place (spec D-001).
- Ratios are `Fraction`, never `float`.
- Commit at the end of every step marked **Commit**. Small commits, present-tense messages that
  say *why*, not *what*.
- Every judgement call goes in `DECISIONS.md` with the alternative you rejected.

## How each task works

1. Tutor gives you the file list, the interfaces, and a failing test file.
2. You run the tests and confirm they fail for the right reason.
3. **You write the implementation.** No implementation code appears in this plan.
4. You make the tests pass, then extend them with at least one case the tutor did not give you.
5. You commit.
6. **Checkpoint:** tutor reviews the diff and asks the self-check questions. You answer without
   looking at the code. If you cannot, that is the signal to go back — not to move on.

**The training wheels come off progressively.** Weeks 1–8 give you complete failing test files.
From week 9 the tests are specified in prose and **you write them**, because by then test design
is the skill being examined. If you find yourself wanting the code handed over in week 9, that is
information about weeks 1–8, not about week 9.

---

# Week 1 · Task 1: Repository foundation and domain types

**Budget:** 7h · **Deliverable:** a repo that lints, typechecks, tests green in CI on a real
GitHub remote, with domain primitives that make a whole class of bug unrepresentable.

**Files**
- Create: `pyproject.toml`, `.gitignore`, `.python-version`, `README.md`, `DECISIONS.md`
- Create: `src/reviewradar/__init__.py`, `src/reviewradar/py.typed`
- Create: `src/reviewradar/types.py`
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_types.py`

**Interfaces — produces**

```python
# src/reviewradar/types.py
from fractions import Fraction
from typing import NewType, TypeAlias
from enum import StrEnum

Cik: TypeAlias = NewType("Cik", str)               # always 10 digits, zero-padded
Accession: TypeAlias = NewType("Accession", str)   # always 0000000000-00-000000
Ratio: TypeAlias = Fraction

class EventType(StrEnum):
    SPLIT_FORWARD = "split_forward"
    SPLIT_REVERSE = "split_reverse"
    SPECIAL_DIVIDEND = "special_dividend"
    MERGER_COMPLETED = "merger_completed"
    SPINOFF = "spinoff"
    RIGHTS_ISSUE = "rights_issue"
    DELISTING = "delisting"
    BANKRUPTCY = "bankruptcy"
    TICKER_CHANGE = "ticker_change"
    NAME_CHANGE = "name_change"
    NO_INDEX_ACTION = "no_index_action"
    UNRESOLVED = "unresolved"

class Treatment(StrEnum):
    DIVISOR_ADJUST = "divisor_adjust"
    SHARES_UPDATE = "shares_update"
    PRICE_ADJUST = "price_adjust"
    REMOVE_CONSTITUENT = "remove_constituent"
    NO_ACTION = "no_action"
    MANUAL_REVIEW = "manual_review"

def parse_cik(raw: str) -> Cik: ...
def parse_accession(raw: str) -> Accession: ...
```

**Steps**

- [ ] **Step 1: Scaffold the repo.** `uv init --package`, src layout, Python 3.12+. Add `ruff`,
      `mypy`, `pytest`, `pytest-cov`, `hypothesis` to the dev group. Configure `mypy` strict in
      `pyproject.toml`. Push to your GitHub remote.

- [ ] **Step 2: Take the failing tests.** Create `tests/test_types.py`:

```python
import pytest
from fractions import Fraction
from reviewradar.types import (
    Cik, Accession, Ratio, EventType, Treatment, parse_cik, parse_accession,
)


def test_ratio_is_exact_where_float_is_not():
    # This is the whole reason Ratio is a Fraction. A 1-for-3 reverse split
    # applied three times must return exactly to par.
    assert Ratio(1, 3) * 3 == 1
    assert Ratio(1, 10) + Ratio(2, 10) == Ratio(3, 10)
    assert 0.1 + 0.2 != 0.3  # for contrast


def test_cik_is_zero_padded_to_ten_digits():
    assert parse_cik("320193") == Cik("0000320193")
    assert parse_cik("0000320193") == Cik("0000320193")


@pytest.mark.parametrize("bad", ["", "AAPL", "32019X", "12345678901"])
def test_cik_rejects_anything_that_is_not_a_ten_digit_number(bad: str):
    with pytest.raises(ValueError):
        parse_cik(bad)


def test_accession_normalises_the_dashless_form():
    assert parse_accession("000032019324000123") == Accession("0000320193-24-000123")
    assert parse_accession("0000320193-24-000123") == Accession("0000320193-24-000123")


@pytest.mark.parametrize("bad", ["", "0000320193-24", "not-an-accession"])
def test_accession_rejects_malformed_input(bad: str):
    with pytest.raises(ValueError):
        parse_accession(bad)


def test_unresolved_is_distinct_from_no_index_action():
    # "I looked and there is nothing to do" and "I could not tell" are
    # different outcomes. Collapsing them hides every classifier failure.
    assert EventType.UNRESOLVED is not EventType.NO_INDEX_ACTION


def test_manual_review_is_a_treatment_not_an_error():
    assert Treatment.MANUAL_REVIEW in Treatment
```

- [ ] **Step 3: Run them, confirm they fail** with `ModuleNotFoundError` / `ImportError`, not with
      a syntax error in your test file. `uv run pytest tests/test_types.py -v`

- [ ] **Step 4: Implement `types.py`.** Your exercise. Constraints: `parse_cik` and
      `parse_accession` raise `ValueError` with a message naming the offending input; no regex
      you cannot explain out loud.

- [ ] **Step 5: Make them pass.** `uv run pytest -v && uv run mypy src/reviewradar`

- [ ] **Step 6: Add one test the tutor did not give you.** Something about `EventType` or
      `Treatment` that would catch a real mistake.

- [ ] **Step 7: Write `.github/workflows/ci.yml`** — jobs for lint, mypy, and pytest on 3.12 and
      3.13. Set `PYTHONHASHSEED: "0"`. Confirm it goes green on the remote.

- [ ] **Step 8: Commit.** Then start `DECISIONS.md` with **D-001**: why the event log will be
      append-only. Write it now, before the code exists, while the reasoning is fresh.

**Traps**
- `NewType` gives you no runtime validation. `Cik("banana")` typechecks. That is why `parse_cik`
  exists and why nothing else in the codebase may construct a `Cik` directly.
- `Fraction(0.1)` is not `Fraction(1, 10)` — it is the exact binary double. Never construct a
  `Ratio` from a float.

**Self-check (answer without the code open)**
1. Why is `Ratio` a `Fraction` and not a `Decimal`?
2. What does `NewType` actually do at runtime, and what does that imply about where validation
   has to live?
3. Give a concrete failure that `UNRESOLVED` vs `NO_INDEX_ACTION` prevents.

---

# Week 2 · Task 2: EDGAR ingest

**Budget:** 8h · **Deliverable:** given a date, a list of 8-K filings with their full document
text and exhibits, fetched politely, replayable from fixtures.

**Files**
- Create: `src/reviewradar/ingest/__init__.py`, `src/reviewradar/ingest/edgar.py`
- Create: `tests/fixtures/edgar/` (committed real responses)
- Create: `memos/W2_what_the_8k_header_actually_contains.md`
- Test: `tests/test_edgar.py`

**Interfaces — consumes:** `Cik`, `Accession`, `parse_cik`, `parse_accession` from Task 1.

**Interfaces — produces**

```python
# src/reviewradar/ingest/edgar.py
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True, slots=True)
class FilingRef:
    cik: Cik
    accession: Accession
    form_type: str
    filed_date: date
    company_name: str
    submission_url: str

@dataclass(frozen=True, slots=True)
class SubmissionDocument:
    doc_id: str        # f"{accession}:{sequence}"
    doc_type: str      # "8-K", "EX-99.1", ...
    filename: str
    text: str

@dataclass(frozen=True, slots=True)
class Submission:
    ref: FilingRef
    header: str
    documents: tuple[SubmissionDocument, ...]

    def primary(self) -> SubmissionDocument: ...
    def exhibits(self) -> tuple[SubmissionDocument, ...]: ...

class EdgarError(RuntimeError): ...

class EdgarClient:
    def __init__(self, contact: str, *, requests_per_second: float = 8.0) -> None: ...
    def daily_index(self, on: date, *, form_type: str = "8-K") -> list[FilingRef]: ...
    def fetch(self, ref: FilingRef) -> Submission: ...
```

**Steps**

- [ ] **Step 1: Probe first, code second.** Before writing anything, fetch three real 8-K full
      submission files by hand (`curl` with a proper User-Agent) — one plain earnings 8-K, one
      merger, one split. Save them under `tests/fixtures/edgar/`. Then write
      `memos/W2_what_the_8k_header_actually_contains.md` answering, from what you actually see:
      *Where do the Item numbers live — the SGML header, the body, or both? Is the item given as a
      number, a description, or both? Where does the exhibit text live relative to the primary
      document?* **Do not assume the answer.** This memo is the point of week 2 and the plan
      deliberately does not tell you what you will find.

- [ ] **Step 2: Take the failing tests.** `tests/test_edgar.py`:

```python
import pytest
from datetime import date
from pathlib import Path
from reviewradar.ingest.edgar import EdgarClient, EdgarError, Submission

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


def test_client_refuses_to_construct_without_a_contact():
    # The SEC blocks IPs that omit a contact User-Agent. Making this
    # unconstructible is cheaper than discovering it in production.
    with pytest.raises(ValueError):
        EdgarClient(contact="")


def test_client_declares_its_contact_in_the_user_agent():
    client = EdgarClient(contact="jason@example.com")
    assert "jason@example.com" in client.user_agent


def test_daily_index_parses_the_committed_fixture():
    refs = EdgarClient.parse_daily_index(
        (FIXTURES / "master.20260812.idx").read_text(encoding="latin-1"),
        on=date(2026, 8, 12),
    )
    assert refs, "fixture parsed to nothing"
    assert all(r.form_type == "8-K" for r in refs)
    assert all(len(r.cik) == 10 for r in refs)
    assert all(r.filed_date == date(2026, 8, 12) for r in refs)


def test_submission_splits_primary_from_exhibits():
    raw = (FIXTURES / "split_8k.txt").read_text(encoding="latin-1")
    sub = EdgarClient.parse_submission(raw)
    assert sub.primary().doc_type.startswith("8-K")
    assert any(d.doc_type.startswith("EX-") for d in sub.exhibits())
    assert sub.primary() not in sub.exhibits()


def test_every_document_has_a_unique_doc_id():
    raw = (FIXTURES / "split_8k.txt").read_text(encoding="latin-1")
    sub = EdgarClient.parse_submission(raw)
    ids = [d.doc_id for d in sub.documents]
    assert len(ids) == len(set(ids))


def test_a_truncated_submission_raises_rather_than_returning_empty():
    # An empty result is indistinguishable from "this filing had no documents".
    # That ambiguity is how a missed split reaches production.
    with pytest.raises(EdgarError):
        EdgarClient.parse_submission("<SEC-DOCUMENT>truncated")


def test_rate_limiter_spaces_calls(monkeypatch):
    ticks: list[float] = []
    now = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda s: (now.__setitem__(0, now[0] + s), ticks.append(s)))
    client = EdgarClient(contact="jason@example.com", requests_per_second=4.0)
    for _ in range(3):
        client._throttle()
    assert all(t >= 0.25 for t in ticks), ticks
```

- [ ] **Step 3: Run, confirm failure.**

- [ ] **Step 4: Implement `edgar.py`.** Your exercise. Constraints: `parse_daily_index` and
      `parse_submission` are **pure static methods over text** — no I/O, no network. All network
      lives in `daily_index` and `fetch`. That separation is what makes the fixtures possible.

- [ ] **Step 5: Green, plus your own test** covering something you found in the Step 1 probe.

- [ ] **Step 6: Commit.** Add **D-002** to `DECISIONS.md`: why parsing is separated from fetching.

**Traps**
- EDGAR files are `latin-1`, not UTF-8. Decoding as UTF-8 raises on perfectly valid filings.
- 10 requests/second is a hard limit. Sit under it — the default of 8 exists for that reason.
- Some 8-Ks have no exhibits at all. `exhibits()` returning empty is valid; the submission having
  *no documents* is not.

**Self-check**
1. What did the header probe actually show, and how did it change your classifier design?
2. Why are `parse_daily_index` and `parse_submission` static and pure?
3. What happens today if EDGAR returns a 403? What *should* happen?

---

# Week 3 · Task 3: The append-only event log

**Budget:** 7h · **Deliverable:** a store that cannot lose a failure and cannot double-count a
re-run.

**Files**
- Create: `src/reviewradar/ingest/store.py`
- Test: `tests/test_store.py`

**Interfaces — produces**

```python
# src/reviewradar/ingest/store.py
from collections.abc import Sequence
from datetime import date
from pathlib import Path
import pandas as pd

class EventStore:
    def __init__(self, path: Path) -> None: ...
    def append(self, events: Sequence["CorporateActionEvent"]) -> int: ...
    def record_failure(self, accession: Accession, stage: str, error: str, run_id: str) -> None: ...
    def already_ingested(self, on: date) -> bool: ...
    def mark_ingested(self, on: date, run_id: str, n_filings: int) -> None: ...
    def events(self, since: date | None = None) -> pd.DataFrame: ...
    def failures(self, since: date | None = None) -> pd.DataFrame: ...
```

`CorporateActionEvent` is defined in Task 7. For this week, work against a minimal stand-in you
define in `types.py` and widen later — record that as a deliberate choice in `DECISIONS.md`.

**Steps**

- [ ] **Step 1: Take the failing tests.** `tests/test_store.py`:

```python
import pytest
from datetime import date
from hypothesis import given, strategies as st
from reviewradar.ingest.store import EventStore


def test_a_failure_is_recorded_not_swallowed(tmp_path):
    store = EventStore(tmp_path / "log.duckdb")
    store.record_failure(
        accession=parse_accession("0000320193-26-000001"),
        stage="parse", error="truncated submission", run_id="r1",
    )
    failures = store.failures()
    assert len(failures) == 1
    assert "truncated" in failures.iloc[0]["error"]


def test_marking_a_date_ingested_makes_it_idempotent(tmp_path):
    store = EventStore(tmp_path / "log.duckdb")
    assert not store.already_ingested(date(2026, 8, 12))
    store.mark_ingested(date(2026, 8, 12), run_id="r1", n_filings=417)
    assert store.already_ingested(date(2026, 8, 12))


def test_appending_the_same_event_twice_stores_it_once(tmp_path):
    store = EventStore(tmp_path / "log.duckdb")
    ev = make_event(accession="0000320193-26-000001", run_id="r1")
    store.append([ev])
    store.append([ev])
    assert len(store.events()) == 1


def test_a_superseding_event_does_not_delete_its_predecessor(tmp_path):
    # D-001. What you said yesterday is itself a record.
    store = EventStore(tmp_path / "log.duckdb")
    first = make_event(accession="0000320193-26-000001", run_id="r1")
    store.append([first])
    corrected = make_event(
        accession="0000320193-26-000001", run_id="r2", supersedes=first.event_id,
    )
    store.append([corrected])
    rows = store.events()
    assert len(rows) == 2
    assert set(rows["run_id"]) == {"r1", "r2"}


@given(st.integers(min_value=1, max_value=5))
def test_reingesting_a_date_n_times_is_a_no_op(n, tmp_path_factory):
    store = EventStore(tmp_path_factory.mktemp("s") / "log.duckdb")
    events = [make_event(accession="0000320193-26-000001", run_id="r1")]
    for _ in range(n):
        store.append(events)
    assert len(store.events()) == 1
```

You write `make_event` as a test helper.

- [ ] **Step 2: Run, confirm failure. Step 3: Implement.** Your exercise. Constraint: idempotency
      comes from a key the *data* determines — not from a `SELECT` before every insert.

- [ ] **Step 4: Green. Step 5: Commit.**

**Traps**
- "Append-only" and "no duplicates" are in tension. Resolve it by making the event's identity a
  deterministic function of its content plus its run, so a re-run of the same run collides and a
  genuinely new run does not.
- DuckDB has no `UPDATE`-free enforcement. Your discipline is the constraint; write a test for it.

**Self-check**
1. What exactly is `event_id` a function of, and why that and not a UUID?
2. If tonight's job crashes halfway through a date, what does tomorrow's run do?
3. Why is a `FAILED` row more valuable than a log line?

---

# Weeks 4–5 · Tasks 4–5: The baseline classifier and its first measurement

**Budget:** 15h · **Deliverable:** a working end-to-end system with no model in it, and a number
that says how good it is.

**Files**
- Create: `src/reviewradar/extract/__init__.py`, `src/reviewradar/extract/baseline.py`
- Create: `src/reviewradar/treatment/__init__.py`, `src/reviewradar/treatment/rules.py`
- Create: `src/reviewradar/cli.py`
- Test: `tests/test_baseline.py`, `tests/test_treatment.py`

**Interfaces — produces**

```python
# src/reviewradar/extract/baseline.py
NO_CONSEQUENCE_ITEMS: frozenset[str]        # items that never imply index action
DIAGNOSTIC_ITEMS: dict[str, EventType]      # items that identify an event type outright

@dataclass(frozen=True, slots=True)
class BaselineResult:
    event_type: EventType
    items: frozenset[str]
    rationale: str          # human-readable, goes in the review queue
    needs_model: bool

def parse_items(submission: Submission) -> frozenset[str]: ...
def classify(submission: Submission) -> BaselineResult: ...

# src/reviewradar/treatment/rules.py
def decide(event_type: EventType, *, confidence: float, threshold: float) -> Treatment: ...
```

**Steps**

- [ ] **Step 1: Take the failing tests.** `tests/test_baseline.py`:

```python
import pytest
from reviewradar.extract.baseline import classify, parse_items, NO_CONSEQUENCE_ITEMS
from reviewradar.types import EventType


def test_an_earnings_only_8k_terminates_with_no_index_action(earnings_submission):
    result = classify(earnings_submission)
    assert result.event_type is EventType.NO_INDEX_ACTION
    assert result.needs_model is False


def test_a_completed_acquisition_is_identified_but_still_needs_fields(merger_submission):
    result = classify(merger_submission)
    assert result.event_type is EventType.MERGER_COMPLETED
    assert result.needs_model is True, "the counterparty and date still have to be extracted"


def test_an_item_801_filing_is_unresolved_and_routed_to_the_model(split_submission):
    # Item 8.01 is the dumping ground. The baseline must not guess here.
    result = classify(split_submission)
    assert result.event_type is EventType.UNRESOLVED
    assert result.needs_model is True


def test_a_filing_with_both_a_no_consequence_and_a_diagnostic_item_is_not_eliminated(
    mixed_submission,
):
    # 2.02 + 2.01 in one filing. Eliminating on the presence of a harmless
    # item loses the event sitting next to it.
    result = classify(mixed_submission)
    assert result.event_type is not EventType.NO_INDEX_ACTION


def test_every_result_carries_a_rationale(earnings_submission):
    assert classify(earnings_submission).rationale.strip()


def test_no_consequence_and_diagnostic_item_sets_are_disjoint():
    from reviewradar.extract.baseline import DIAGNOSTIC_ITEMS
    assert not (NO_CONSEQUENCE_ITEMS & DIAGNOSTIC_ITEMS.keys())
```

You build the fixtures from your week-2 probe files.

- [ ] **Step 2: Implement `baseline.py`.** Your exercise. The item tables are **data, not
      branches** — a dict and a frozenset, so they can be reviewed by someone who does not read
      Python.

- [ ] **Step 3: Implement `treatment/rules.py`.** Your exercise. Every `EventType` maps to a
      `Treatment`; below-threshold confidence overrides to `MANUAL_REVIEW`. Write the test first
      showing that a high-confidence `SPLIT_FORWARD` gives `DIVISOR_ADJUST` and the same event at
      low confidence gives `MANUAL_REVIEW`.

- [ ] **Step 4: Wire the CLI.** `reviewradar ingest --date YYYY-MM-DD` runs
      fetch → classify → treat → append, end to end, on real filings.

- [ ] **Step 5: Run it on five real dates** and report, in `DECISIONS.md`: how many filings, what
      fraction eliminated, what fraction `UNRESOLVED`. **This is your first real number.** It is
      also your baseline for everything that follows.

- [ ] **Step 6: Commit.**

**Traps**
- The elimination rule is over the *whole item set*, not any single item. Get this wrong and you
  silently drop every merger that happened to be announced alongside earnings.
- Resist adding phrase rules for splits. Item 8.01 filings are prose; that is the model's job, and
  a half-working regex here will contaminate the measured delta in week 9.

**Self-check**
1. What fraction of real 8-K traffic does your baseline eliminate, and what is your evidence?
2. Why does `MERGER_COMPLETED` still set `needs_model=True`?
3. Someone proposes a regex for "three-for-one stock split". Give two reasons to refuse.

---

# Week 6 · Task 6: The gold set

**Budget:** 8h · **Deliverable:** ~200 hand-labelled filings in two strata. Unglamorous. It is
what makes every later number mean anything.

**Files**
- Create: `src/reviewradar/evals/__init__.py`, `src/reviewradar/evals/gold.py`
- Create: `data/gold/random.jsonl`, `data/gold/stratified.jsonl` (committed)
- Test: `tests/test_gold.py`

**Interfaces — produces**

```python
# src/reviewradar/evals/gold.py
from typing import Literal

@dataclass(frozen=True, slots=True)
class GoldLabel:
    accession: Accession
    stratum: Literal["random", "stratified"]
    event_type: EventType
    ex_date: date | None
    ratio: Ratio | None
    affected_securities: tuple[str, ...]
    notes: str

def sample_random(refs: Sequence[FilingRef], n: int, *, seed: int) -> list[FilingRef]: ...
def sample_stratified(refs: Sequence[FilingRef], n: int, *, seed: int) -> list[FilingRef]: ...
def load_gold(path: Path) -> list[GoldLabel]: ...
```

**Steps**

- [ ] **Step 1: Failing tests.** `tests/test_gold.py`:

```python
def test_sampling_is_reproducible_from_the_seed(refs):
    assert sample_random(refs, 20, seed=7) == sample_random(refs, 20, seed=7)


def test_the_two_strata_do_not_overlap():
    random_set = {g.accession for g in load_gold(Path("data/gold/random.jsonl"))}
    strat_set = {g.accession for g in load_gold(Path("data/gold/stratified.jsonl"))}
    assert not (random_set & strat_set)


def test_the_random_stratum_is_large_enough_to_quote_a_rate():
    assert len(load_gold(Path("data/gold/random.jsonl"))) >= 120


def test_the_stratified_stratum_has_at_least_five_of_each_index_relevant_type():
    from collections import Counter
    counts = Counter(g.event_type for g in load_gold(Path("data/gold/stratified.jsonl")))
    for et in (EventType.SPLIT_FORWARD, EventType.SPLIT_REVERSE,
               EventType.MERGER_COMPLETED, EventType.SPINOFF, EventType.DELISTING):
        assert counts[et] >= 5, f"{et} has only {counts[et]}"


def test_a_ratio_is_only_present_where_the_event_type_implies_one():
    for g in load_gold(Path("data/gold/stratified.jsonl")):
        if g.ratio is not None:
            assert g.event_type in {EventType.SPLIT_FORWARD, EventType.SPLIT_REVERSE,
                                    EventType.RIGHTS_ISSUE, EventType.SPINOFF}
```

- [ ] **Step 2: Build a labelling CLI** — `reviewradar label --stratum random`. Shows you the
      filing text, takes your label, appends to the JSONL. Ten minutes of tooling saves four
      hours of labelling.

- [ ] **Step 3: Label 120 random filings.** Sample uniformly across at least 20 trading days.
      Label honestly, including the ones you are unsure about — mark those in `notes`.

- [ ] **Step 4: Label 80 stratified filings**, oversampling Items 8.01 and 5.03 and deliberately
      seeking splits, spin-offs, mergers and delistings.

- [ ] **Step 5: Re-label 20 of the random set a week later, blind, and measure your own
      agreement with yourself.** If you disagree with your own past labels more than ~5% of the
      time, your label definitions are ambiguous and no model can beat that ceiling. Write the
      number in `DECISIONS.md`.

- [ ] **Step 6: Commit both files.**

**Traps**
- Do not pool the strata. Ever. A rate from the stratified set is not a population rate.
- Label the *filing*, not what you remember about the company.

**Self-check**
1. What is your self-agreement rate, and what does it bound?
2. Why can't you quote per-class F1 from the random stratum?
3. Why can't you quote a manual-review rate from the stratified stratum?

---

# Weeks 7–8 · Tasks 7–8: Model extraction with span grounding

**Budget:** 15h · **Deliverable:** the residual gets extracted, every field is provably quoted
from the source, and you know what it cost.

**Files**
- Create: `src/reviewradar/extract/schema.py`, `src/reviewradar/extract/llm.py`,
  `src/reviewradar/extract/pipeline.py`
- Test: `tests/test_schema.py`, `tests/test_spans.py`, `tests/test_pipeline.py`

**Interfaces — produces**

```python
# src/reviewradar/extract/schema.py
EVENT_SCHEMA: dict[str, Any]        # JSON Schema handed to the model

@dataclass(frozen=True, slots=True)
class Span:
    doc_id: str
    start: int
    end: int
    text: str

@dataclass(frozen=True, slots=True)
class Extracted[T]:
    value: T
    span: Span | None
    source: Literal["baseline", "model"]

@dataclass(frozen=True, slots=True)
class CorporateActionEvent:
    event_id: str
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
    dropped_fields: tuple[str, ...]
    run_id: str
    supersedes: str | None

# src/reviewradar/extract/llm.py
class LlmClient(Protocol):
    def extract(self, prompt: str, schema: dict[str, Any]) -> LlmResponse: ...

@dataclass(frozen=True, slots=True)
class LlmResponse:
    payload: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float

class OfflineLlm:   # deterministic, keyed on a hash of the prompt. No network.
    def __init__(self, canned: Mapping[str, dict[str, Any]]) -> None: ...

class VertexLlm:    # real. Never imported at module scope in tests.
    def __init__(self, project: str, location: str, model: str) -> None: ...

# src/reviewradar/extract/pipeline.py
def validate_spans(
    payload: Mapping[str, Any], documents: Sequence[SubmissionDocument]
) -> tuple[dict[str, Any], list[str]]: ...     # (surviving fields, dropped field names)

def compute_confidence(
    *, baseline: BaselineResult, payload: Mapping[str, Any],
    dropped: Sequence[str], filed_at: datetime,
) -> float: ...

def extract(submission: Submission, *, client: LlmClient, run_id: str) -> CorporateActionEvent: ...
```

**Steps**

- [ ] **Step 1: Take the span tests — these are the heart of the project.** `tests/test_spans.py`:

```python
def test_a_field_quoted_correctly_survives():
    doc = SubmissionDocument(doc_id="a:1", doc_type="8-K", filename="f.htm",
                             text="The board declared a three-for-one stock split.")
    payload = {"ratio": {"value": "3/1", "span": {"doc_id": "a:1", "start": 26, "end": 41}}}
    kept, dropped = validate_spans(payload, [doc])
    assert "ratio" in kept and dropped == []


def test_a_field_whose_value_is_not_in_its_span_is_dropped_not_flagged():
    # D-003. The citation has to be load-bearing.
    doc = SubmissionDocument(doc_id="a:1", doc_type="8-K", filename="f.htm",
                             text="The board declared a three-for-one stock split.")
    payload = {"ratio": {"value": "5/1", "span": {"doc_id": "a:1", "start": 26, "end": 41}}}
    kept, dropped = validate_spans(payload, [doc])
    assert "ratio" not in kept
    assert dropped == ["ratio"]


def test_a_span_pointing_at_a_document_that_does_not_exist_is_dropped():
    doc = SubmissionDocument(doc_id="a:1", doc_type="8-K", filename="f.htm", text="text")
    payload = {"ex_date": {"value": "2026-09-15",
                           "span": {"doc_id": "a:99", "start": 0, "end": 4}}}
    kept, dropped = validate_spans(payload, [doc])
    assert dropped == ["ex_date"]


def test_out_of_bounds_spans_are_dropped_not_clamped():
    doc = SubmissionDocument(doc_id="a:1", doc_type="8-K", filename="f.htm", text="short")
    payload = {"ex_date": {"value": "2026-09-15",
                           "span": {"doc_id": "a:1", "start": 0, "end": 9_999}}}
    kept, dropped = validate_spans(payload, [doc])
    assert dropped == ["ex_date"]


@given(st.text(min_size=1, max_size=200), st.integers(), st.integers())
def test_validate_spans_never_raises_on_arbitrary_input(text, start, end):
    doc = SubmissionDocument(doc_id="a:1", doc_type="8-K", filename="f.htm", text=text)
    payload = {"ex_date": {"value": "x", "span": {"doc_id": "a:1", "start": start, "end": end}}}
    validate_spans(payload, [doc])   # must not raise


def test_confidence_falls_when_a_required_field_is_dropped(baseline_merger):
    high = compute_confidence(baseline=baseline_merger, payload=FULL, dropped=[],
                              filed_at=FILED)
    low = compute_confidence(baseline=baseline_merger, payload=FULL, dropped=["ex_date"],
                             filed_at=FILED)
    assert low < high


def test_an_ex_date_before_the_filing_date_lowers_confidence(baseline_split):
    # A split cannot go ex before it was announced. This is arithmetic, not a model call.
    ...
```

- [ ] **Step 2: Implement `schema.py`.** Your exercise. The JSON Schema must make `span`
      **required** on every model-derived field. If the model can omit the citation, it will.

- [ ] **Step 3: Implement `OfflineLlm` and the `LlmClient` Protocol.** Deterministic, keyed on a
      hash of the prompt, no network. Every test and all of CI runs on this.

- [ ] **Step 4: Implement `validate_spans` and `compute_confidence`.** Your exercise. Confidence is
      **computed from observable facts** — fields dropped, baseline/model agreement, date
      consistency — never read off the model's own self-report (spec D-004).

- [ ] **Step 5: Implement `VertexLlm`** and run it on 30 gold-set filings. Record mean input and
      output tokens, p50/p95 latency, and cost per filing. Import it lazily so CI never touches it.

- [ ] **Step 6: Commit.** Add **D-003** (span grounding) and **D-004** (computed confidence).

**Traps**
- Character offsets against HTML are treacherous. Decide once whether spans index the raw document
  or a normalised text, write it in the docstring, and never mix.
- A model told to cite will cite *something*. The test is whether the value is inside the citation.
- Do not let `VertexLlm` be imported at module scope anywhere `pytest` collects.

**Self-check**
1. Why drop a field rather than flag it?
2. Your confidence function — name its three inputs and why each one is evidence.
3. What does span validation *not* protect against?

---

# Week 9 · Task 9: The scoreboard

**Budget:** 8h · **Deliverable:** the table that is the interview.

**Files**
- Create: `src/reviewradar/evals/score.py`
- Modify: `.github/workflows/ci.yml` (add the baseline-F1 gate)
- Test: `tests/test_score.py`

**Interfaces — produces**

```python
@dataclass(frozen=True, slots=True)
class FieldScore:
    field: str
    precision: float
    recall: float
    f1: float
    n: int

def score_field(gold: Sequence[GoldLabel], predicted: Sequence[CorporateActionEvent],
                field: str) -> FieldScore: ...

def scoreboard(gold: Sequence[GoldLabel], baseline_only: Sequence[CorporateActionEvent],
               with_model: Sequence[CorporateActionEvent]) -> pd.DataFrame: ...

def calibrate_threshold(gold: Sequence[GoldLabel], events: Sequence[CorporateActionEvent],
                        *, target_precision: float) -> float: ...
```

**Steps**

- [ ] **Step 1: Failing tests** covering: a perfect predictor scores 1.0; a predictor that
      abstains on everything has recall 0 and **undefined**, not 1.0, precision; per-field scoring
      ignores filings where gold has no value for that field; `calibrate_threshold` returns the
      lowest threshold meeting the target precision, and raises if none does.

- [ ] **Step 2: Implement.** Your exercise.

- [ ] **Step 3: Produce the real scoreboard** — random stratum and stratified stratum reported
      **separately**, baseline vs +model, with the delta, plus cost per filing and p50/p95.

- [ ] **Step 4: Calibrate the confidence threshold** against the random stratum at a target
      precision you choose and justify in `DECISIONS.md`.

- [ ] **Step 5: Add the CI gate** — baseline F1 may not fall below its current value.

- [ ] **Step 6: Commit.**

**Trap:** if the model does *not* beat the baseline on some field, report that. It is a better
finding than a win, and week 12's write-up is stronger for containing one.

**Self-check**
1. What is precision when a system abstains on everything, and why does that matter here?
2. Which fields does the model improve, which does it not, and what do you do about it?
3. Justify your target precision to someone whose team handles the review queue.

---

# Week 10 · Task 10: Deploy — non-negotiable

**Budget:** 8h · **Deliverable:** it runs every night without you.

**Files**
- Create: `Dockerfile`, `src/reviewradar/service/job.py`, `deploy/README.md`
- Test: `tests/test_job.py`

**Steps**

- [ ] **Step 1: Failing test** — the job is idempotent: running the same date twice appends no
      duplicate events and no duplicate ingest marker.
- [ ] **Step 2: Implement `job.py`.** Reads yesterday's filings, runs the pipeline, appends,
      writes a run manifest (run id, date, counts, model version, cost, duration).
- [ ] **Step 3: Multi-stage `Dockerfile`.** Build it locally and run the job in the container.
- [ ] **Step 4: Deploy as a Cloud Run *job*** — not a service — in your GCP project, with a
      service account holding only Vertex invoke and storage write.
- [ ] **Step 5: Cloud Scheduler trigger**, weekdays after EDGAR's daily index publishes.
- [ ] **Step 6: Watch it run unattended for three nights.** Fix what breaks. Write up the first
      real failure in `docs/incident_001.md` — cause, detection, fix, prevention.
- [ ] **Step 7: Commit**, and record **D-005** (Vertex over the direct API) and the runbook.

**Traps**
- A Cloud Run *job* has no request lifecycle; do not reach for a service because it is familiar.
- Never bake credentials into the image. Service-account identity only.
- The job must exit non-zero on a failure that is not per-filing, so the scheduler surfaces it.

**Self-check**
1. Job vs service — why?
2. Exactly what permissions does the service account hold, and why no more?
3. If tonight's run silently ingests zero filings, how do you find out?

---

# Week 11 · Task 11: The dashboard

**Budget:** 7h · **Deliverable:** somewhere to send the link.

**Files**
- Create: `src/reviewradar/service/app.py`, `src/reviewradar/service/templates/`
- Test: `tests/test_app.py`

**Steps**

- [ ] **Step 1: Failing tests** — `/events` renders recent events with treatment and confidence;
      `/filing/{accession}` shows the source text with the cited spans highlighted; `/scoreboard`
      renders both strata; and the **layer-boundary test**:

```python
def test_the_service_layer_recomputes_no_stored_figure():
    # Same discipline as the miniftse ops desk. The service renders what the
    # log holds. A second source of truth for a published number is a defect.
    import pathlib, re
    banned = re.compile(r"\*\s*100|/\s*100|\*\s*252|\*\*\s*0\.5|sqrt\(")
    for path in pathlib.Path("src/reviewradar/service").rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if banned.search(line) and "# allow:" not in line:
                raise AssertionError(f"{path}:{i} recomputes a figure: {line.strip()}")
```

- [ ] **Step 2: Implement.** Your exercise. FastAPI + Jinja, server-rendered, no build step.
- [ ] **Step 3: Deploy the service** alongside the job. Put the URL in the README.
- [ ] **Step 4: Commit.**

**The span-highlighting view is the screen worth demoing.** It shows a claim next to the sentence
it came from. That is the whole governance argument, visible in one screenshot.

**Self-check**
1. Why does the service compute nothing?
2. What does the highlight view prove that the scoreboard does not?

---

# Week 12 · Task 12: Forward verification and the write-up

**Budget:** 7h · **Deliverable:** a project that keeps getting more credible on its own, and the
document you actually send.

**Files**
- Create: `src/reviewradar/evals/forward.py`
- Modify: `README.md`, `DECISIONS.md`
- Test: `tests/test_forward.py`

**Interfaces — produces**

```python
@dataclass(frozen=True, slots=True)
class Verification:
    event_id: str
    status: Literal["verified", "contradicted", "unverifiable"]
    evidence: str

def verify(events: Sequence[CorporateActionEvent], prices: pd.DataFrame) -> list[Verification]: ...
```

**Steps**

- [ ] **Step 1: Failing tests** — a 3-for-1 split whose price steps ~1/3 on the extracted ex-date
      verifies; the same split with no price step is `contradicted`; an event whose ex-date has not
      yet passed is `unverifiable`, never `contradicted`.
- [ ] **Step 2: Implement.** Your exercise. Free daily prices are adequate here; the tolerance band
      must be justified in `DECISIONS.md`.
- [ ] **Step 3: Schedule it weekly** and surface the counts on `/scoreboard`.
- [ ] **Step 4: Write the README.** Cover: what it is, the live URL, the scoreboard with real
      numbers, the known limits from spec §4 stated plainly, and the business case in units —
      filings/night, auto-handled %, review %, cost/filing, implied hours.
- [ ] **Step 5: Write `docs/what_the_build_found.md`** — every bug the tests caught, as a table.
      This is the most-read page in the miniftse repo and it will be the most-read page here.
- [ ] **Step 6: Final commit and tag `v1.0`.**

**Self-check — the interview rehearsal**
1. In two minutes: what does Review Radar do and what is it worth?
2. What does the model contribute over the baseline, in numbers, per field?
3. Where does it fail, and what would you build next with a licensed feed?
4. Why should anyone trust an extracted ex-date?

---

## Coverage against the spec

| Spec section | Task |
|---|---|
| §5 architecture / module boundaries | 1, 11 |
| §6 two-stage extraction | 4, 7, 8 |
| §7 domain model | 1, 7 |
| §8 D-001 append-only | 3 |
| §8 D-002 baseline-then-residual | 4, 9 |
| §8 D-003 span grounding | 8 |
| §8 D-004 computed confidence | 8, 9 |
| §8 D-005 Vertex | 8, 10 |
| §8 D-006 offline Protocol | 7 |
| §9 two strata | 6 |
| §10 scoreboard | 9, 12 |
| §11 error handling | 2, 3, 10 |
| §12 testing | every task |
| §13 deployment | 10, 11 |

## If you fall behind

Cut in this order: the dashboard (11), forward verification (12), the stratified stratum down to
40. **Do not cut week 6 and do not cut week 10.** A labelled gold set and a live deployment are
the two things this project has that your existing work does not.
