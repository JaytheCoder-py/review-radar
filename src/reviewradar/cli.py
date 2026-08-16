"""Command line."""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from reviewradar.evals.forward import (
    FixturePrices,
    PriceSource,
    Verdict,
    claims_from_rows,
    verdict_counts,
    verify_all,
)
from reviewradar.evals.gold import GoldLabel, Stratum, load_all, load_gold
from reviewradar.evals.labelling import (
    DEFAULT_SEED,
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
from reviewradar.evals.score import (
    cost_report,
    elimination_report,
    field_table,
    review_rate,
)
from reviewradar.extract.llm import LlmClient, OfflineLlm
from reviewradar.extract.pipeline import extract
from reviewradar.extract.schema import CorporateActionEvent
from reviewradar.ingest.edgar import EdgarClient, EdgarError
from reviewradar.ingest.store import EventStore

app = typer.Typer(add_completion=False, help="Corporate-action extraction over SEC 8-K filings.")

CORPUS = Path("data/corpus")
DEFAULT_DB = Path("data/events.duckdb")
GOLD = Path("data/gold")


def _client(use_vertex: bool, project: str | None) -> LlmClient:
    if not use_vertex:
        return OfflineLlm()
    if not project:
        raise typer.BadParameter("--project is required with --vertex")
    from reviewradar.extract.llm import VertexLlm

    return VertexLlm(project=project)


def _stratum(raw: str) -> Stratum:
    if raw not in ("random", "stratified"):
        raise typer.BadParameter(f"stratum must be 'random' or 'stratified'; got {raw!r}")
    return cast(Stratum, raw)


@app.command()
def ingest(
    date: Annotated[str, typer.Option(help="Dissemination date, YYYY-MM-DD.")],
    contact: Annotated[str, typer.Option(help="Contact email for the EDGAR User-Agent.")],
    db: Annotated[Path, typer.Option()] = DEFAULT_DB,
    limit: Annotated[int, typer.Option(help="Cap filings, for a smoke run.")] = 0,
    vertex: Annotated[bool, typer.Option(help="Use the real model.")] = False,
    project: Annotated[str | None, typer.Option(help="GCP project for Vertex.")] = None,
) -> None:
    """Fetch one day of 8-K filings, extract, and append to the log."""
    on = dt.date.fromisoformat(date)
    run_id = f"{on:%Y%m%d}-{'vertex' if vertex else 'offline'}"
    client = _client(vertex, project)
    edgar = EdgarClient(contact=contact)

    with EventStore(db) as store:
        if store.already_ingested(on):
            typer.echo(f"{on} already ingested; nothing to do.")
            return
        refs = edgar.daily_index(on)
        if limit:
            refs = refs[:limit]
        typer.echo(f"{on}: {len(refs)} 8-K filings")
        events, failures = [], 0
        for i, ref in enumerate(refs, 1):
            try:
                submission = edgar.fetch(ref)
                events.append(extract(submission, client=client, run_id=run_id))
            except (EdgarError, ValueError) as exc:
                # Recorded, never skipped: an absent row is indistinguishable from
                # "this filing carried no event".
                store.record_failure(ref.accession, "ingest", str(exc), run_id)
                failures += 1
            if i % 25 == 0:
                typer.echo(f"  {i}/{len(refs)}")
        inserted = store.append(events)
        store.mark_ingested(on, run_id, len(refs))
        typer.echo(f"appended {inserted} events, {failures} failures")


@app.command()
def replay(
    db: Annotated[Path, typer.Option()] = DEFAULT_DB,
    corpus: Annotated[Path, typer.Option()] = CORPUS,
    baseline_only: Annotated[bool, typer.Option()] = False,
    vertex: Annotated[bool, typer.Option()] = False,
    project: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run the pipeline over the committed corpus, with no network."""
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    client = _client(vertex, project)
    run_id = "replay-" + ("baseline" if baseline_only else "vertex" if vertex else "offline")
    events, failures = [], 0
    for accession in sorted(manifest):
        path = corpus / "submissions" / f"{accession}.txt"
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            submission = EdgarClient.parse_submission(raw, ref=EdgarClient.ref_from_header(raw))
            events.append(
                extract(submission, client=client, run_id=run_id, baseline_only=baseline_only)
            )
        except (EdgarError, ValueError):
            failures += 1
    with EventStore(db) as store:
        inserted = store.append(events)
    typer.echo(f"{len(events)} events ({inserted} new), {failures} failures")


@app.command()
def score(
    corpus: Annotated[Path, typer.Option()] = CORPUS,
    gold_dir: Annotated[Path, typer.Option()] = Path("data/gold"),
    gate: Annotated[bool, typer.Option(help="Fail if the baseline regresses.")] = False,
    vertex: Annotated[bool, typer.Option()] = False,
    project: Annotated[str | None, typer.Option()] = None,
) -> None:
    """The scoreboard. Strata reported separately, never pooled."""
    gold = load_all(gold_dir)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))

    def run(baseline_only: bool) -> dict[str, CorporateActionEvent]:
        client = _client(vertex and not baseline_only, project)
        out: dict[str, CorporateActionEvent] = {}
        for accession in manifest:
            path = corpus / "submissions" / f"{accession}.txt"
            if not path.exists():
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                sub = EdgarClient.parse_submission(raw, ref=EdgarClient.ref_from_header(raw))
                out[accession] = extract(
                    sub, client=client, run_id="score", baseline_only=baseline_only
                )
            except (EdgarError, ValueError):
                continue
        return out

    baseline = run(baseline_only=True)
    model = run(baseline_only=False) if vertex else None

    worst_false_elimination = 0.0
    for stratum, labels in gold.items():
        if not labels:
            continue
        report = elimination_report(labels, baseline, stratum)
        typer.echo(f"\n=== {stratum} stratum (n={report.n}) ===")
        typer.echo(f"  index-relevant filings      {report.n_index_relevant}")
        typer.echo(f"  eliminated by the baseline  {report.elimination_rate:.1%}")
        typer.echo(
            f"  FALSE eliminations          {len(report.false_eliminations)}"
            f"/{report.n_index_relevant} = {report.false_elimination_rate:.1%}"
        )
        for accession in report.false_eliminations:
            label = next(lab for lab in labels if lab.accession == accession)
            typer.echo(f"     {accession}  {label.event_type.value}")
        typer.echo(f"  manual-review rate          {review_rate(baseline):.1%}")
        typer.echo(field_table(labels, baseline, model).to_string(index=False))
        if stratum == "stratified":
            worst_false_elimination = report.false_elimination_rate

    typer.echo("\n=== cost ===")
    for key, value in cost_report(model or baseline).items():
        typer.echo(f"  {key:18s} {value:,.1f}")

    if gate:
        # The gate protects against silent regression of the cheap stage. It is
        # deliberately loose: the point is to catch a change that makes elimination
        # more aggressive without anyone noticing.
        ceiling = 0.40
        if worst_false_elimination > ceiling:
            typer.echo(
                f"\nGATE FAILED: false-elimination rate {worst_false_elimination:.1%} "
                f"exceeds {ceiling:.0%}"
            )
            raise typer.Exit(1)
        typer.echo(
            f"\ngate passed (false elimination {worst_false_elimination:.1%} <= {ceiling:.0%})"
        )


@app.command()
def serve(
    db: Annotated[Path, typer.Option()] = DEFAULT_DB,
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option()] = 8080,
) -> None:
    """The read-only dashboard."""
    import uvicorn

    from reviewradar.service.app import create_app

    uvicorn.run(create_app(db), host=host, port=port)


@app.command()
def stats(corpus: Annotated[Path, typer.Option()] = CORPUS) -> None:
    """Routing distribution over the corpus, with no model involved."""
    from collections import Counter

    from reviewradar.extract.baseline import classify

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    routes: Counter[str] = Counter()
    items: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for accession, meta in manifest.items():
        path = corpus / "submissions" / f"{accession}.txt"
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            sub = EdgarClient.parse_submission(raw, ref=EdgarClient.ref_from_header(raw))
        except EdgarError:
            continue
        result = classify(sub)
        routes[f"{meta['stratum']}:{result.event_type.value}"] += 1
        items.update(result.items)
        unmapped.update(result.unmapped_descriptions)
    for key, count in sorted(routes.items()):
        typer.echo(f"  {key:34s} {count:4d}")
    typer.echo(f"\nitems seen: {len(items)}   unmapped descriptions: {len(unmapped)}")
    for description, count in unmapped.most_common(10):
        typer.echo(f"  {count:3d}  {description[:90]}")


@app.command()
def verify(
    db: Annotated[Path, typer.Option()] = DEFAULT_DB,
    source: Annotated[str, typer.Option(help="Price source: fixtures | yahoo.")] = "fixtures",
    prices: Annotated[
        Path, typer.Option(help="Directory of committed price series, for --source fixtures.")
    ] = Path("data/prices"),
    as_of: Annotated[
        str | None, typer.Option(help="Treat this ISO date as today. Defaults to the clock.")
    ] = None,
) -> None:
    """Check past-dated extractions against the tape, and append the verdicts.

    Run weekly. Verdicts accumulate: an event that was `unverifiable` because its ex-date
    had not passed becomes verifiable on its own, with no hand-labelling.
    """
    today = dt.date.fromisoformat(as_of) if as_of else dt.date.today()
    if source == "yahoo":
        from reviewradar.evals.forward import YahooChartPrices

        client: PriceSource = YahooChartPrices()
    elif source == "fixtures":
        if not prices.is_dir():
            raise typer.BadParameter(
                f"no price series at {prices}. The committed fixtures live in "
                "tests/fixtures/prices; point --prices at a directory of the same shape, "
                "or use --source yahoo for live unadjusted closes."
            )
        client = FixturePrices.from_dir(prices)
    else:
        raise typer.BadParameter(f"unknown price source {source!r}; expected fixtures or yahoo")

    with EventStore(db) as store:
        rows = store.latest_per_accession()
        claims = claims_from_rows(rows)
        results = verify_all(claims, source=client, today=today, run_id=f"verify-{today:%Y%m%d}")
        inserted = store.append_verifications([result.to_row() for result in results])

    typer.echo(f"{len(rows)} stored extractions, {len(claims)} index-relevant claims")
    for verdict, count in verdict_counts(results).items():
        typer.echo(f"  {verdict:14s} {count:4d}")

    contradicted = [r for r in results if r.verdict is Verdict.CONTRADICTED]
    if contradicted:
        # Loud, and listed individually. A contradiction is the one output of this
        # command that asks somebody to go and look at a filing.
        typer.echo("\nCONTRADICTED")
        for result in contradicted:
            typer.echo(
                f"  {result.accession}  {result.ticker}  ex {result.ex_date}  "
                f"expected x{result.expected_step}  observed x{float(result.observed_step or 0):.4f}"
                f"  relative error {float(result.relative_error or 0):.1%}"
            )

    reasons: dict[str, int] = {}
    for result in results:
        if result.reason is not None:
            reasons[result.reason.value] = reasons.get(result.reason.value, 0) + 1
    if reasons:
        typer.echo("\nunverifiable because")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            typer.echo(f"  {reason:32s} {count:4d}")

    typer.echo(f"\nappended {inserted} new verdict rows to {db}")


# ----------------------------------------------------------------------------------
# Hand-labelling. The loop is a shell; every rule it applies lives in evals.labelling.


class _Quit(Exception):
    """The labeller stopped. A clean exit, not an error - everything accepted is saved.

    Raised by `q`, and by Ctrl-D or Ctrl-C, which are how a session actually ends. An
    interrupt that lost the run's closing summary would be a small punishment for using
    the terminal the normal way.
    """


def _ask(prompt: str, parse: Callable[[str], Any], *, controls: tuple[str, ...] = ()) -> str:
    """Prompt until the answer parses. Returns the raw text, validated.

    The raw text rather than the parsed value, so that the label is built by
    `parse_label` and there is exactly one place that turns typing into a `GoldLabel`.

    Control words are matched before parsing, because `s` is a legal prefix of four event
    types and would otherwise be rejected as ambiguous instead of skipping the filing.
    """
    while True:
        try:
            raw: str = typer.prompt(f"  {prompt}", default="", show_default=False)
        except typer.Abort as exc:
            raise _Quit from exc
        typed = raw.strip().lower()
        if typed == "q":
            raise _Quit
        if typed in controls:
            return typed
        try:
            parse(raw)
        except ValueError as exc:
            typer.echo(f"    ! {exc}")
            continue
        return raw


def _read_filing(accession: str, corpus: Path) -> str | None:
    path = corpus / "submissions" / f"{accession}.txt"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        submission = EdgarClient.parse_submission(raw, ref=EdgarClient.ref_from_header(raw))
    except (EdgarError, ValueError):
        return None
    return submission.full_text()


def _label_one(accession: str, stratum: Stratum, text: str, workdir: Path) -> GoldLabel | None:
    """Show one filing and collect one label. `None` means the labeller skipped it."""
    path = write_body(accession, text, workdir)
    typer.echo(f"\n=== {accession} ===")
    typer.echo(f"full text: {path}")
    typer.echo("-" * 78)
    typer.echo(preview(text))
    typer.echo("-" * 78)

    while True:
        first = _ask("event_type ('s' to skip, 'q' to quit)", parse_event_type, controls=("s",))
        if first == "s":
            return None
        answers = {
            "event_type": first,
            "ex_date": _ask("ex_date (ISO, blank if none stated)", parse_ex_date),
            "ratio": _ask("ratio (5/4, 5-for-4, 3, blank)", parse_ratio),
            "counterparty": _ask("counterparty (blank if none)", str),
            "affected_securities": _ask("tickers (comma-separated, blank)", parse_securities),
            "notes": _ask("notes", str),
        }
        try:
            return parse_label(accession, stratum, answers)
        except ValueError as exc:
            # `GoldLabel.__post_init__` is the validator. Its message names the rule.
            typer.echo(f"    ! {exc}")


def _label_run(
    accessions: list[str],
    stratum: Stratum,
    corpus: Path,
    out: Path,
    workdir: Path,
) -> int:
    """Label each accession in turn, appending as we go. Returns the number labelled."""
    labelled = 0
    for i, accession in enumerate(accessions, 1):
        text = _read_filing(accession, corpus)
        if text is None:
            typer.echo(f"\n=== {accession} === unreadable or absent from the corpus; skipping")
            continue
        typer.echo(f"\n[{i}/{len(accessions)}]")
        try:
            label = _label_one(accession, stratum, text, workdir)
        except _Quit:
            typer.echo("\nstopped. Everything accepted is already on disk.")
            break
        if label is None:
            continue
        # Appended now, not at the end: a closed laptop must not cost an evening.
        append_label(out, label)
        labelled += 1
    return labelled


@app.command()
def label(
    stratum: Annotated[str, typer.Option(help="random | stratified.")] = "random",
    corpus: Annotated[Path, typer.Option()] = CORPUS,
    gold_dir: Annotated[Path, typer.Option()] = GOLD,
    seed: Annotated[int, typer.Option(help="Fixes the order. Recorded in the output.")] = (
        DEFAULT_SEED
    ),
    limit: Annotated[int, typer.Option(help="Stop after this many filings. 0 for no cap.")] = 0,
    relabel: Annotated[
        int, typer.Option(help="Blind self-agreement: re-read this many already-labelled filings.")
    ] = 0,
) -> None:
    """Hand-label corpus filings into the gold set, or re-label blind to measure agreement.

    Resumable. Quitting and re-running continues where it left off, because the queue is
    a fixed seeded order with the already-labelled filings filtered out of it.
    """
    which = _stratum(stratum)
    gold_file = gold_dir / f"{which}.jsonl"
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    already = labelled_accessions(gold_file)
    workdir = Path(tempfile.mkdtemp(prefix="reviewradar-label-"))

    typer.echo(f"stratum {which}   seed {seed}   already labelled {len(already)}")
    typer.echo("event types:")
    for i, event_type in enumerate(LABELLABLE_TYPES, 1):
        typer.echo(f"  {i:2d}  {event_type.value}")
    typer.echo(f"bodies are written to {workdir}")

    if relabel:
        # Blind: the existing label is never printed, never passed in. Agreement measured
        # against a label you can see is a measurement of your memory.
        out = relabel_path(which, seed, gold_dir)
        done = labelled_accessions(out)
        chosen = [
            a for a in sample_for_relabel(sorted(already), relabel, seed=seed) if a not in done
        ]
        typer.echo(f"\nblind relabel of {relabel} filings -> {out}  ({len(done)} already done)")
        _label_run(chosen, which, corpus, out, workdir)
        if out.exists():
            report = agreement(load_gold(gold_file), load_gold(out))
            typer.echo(f"\n=== self-agreement, seed {seed}, n={report[0].compared} ===")
            for field in report:
                rate = "-" if field.rate is None else f"{field.rate:.1%}"
                typer.echo(f"  {field.field:22s} {field.agreed:3d}/{field.compared:<3d} {rate}")
            typer.echo("\nRecord the event_type figure in DECISIONS.md.")
        return

    queue = unlabelled(labelling_order(manifest, which, seed=seed), already)
    if limit:
        queue = queue[:limit]
    typer.echo(f"\n{len(queue)} filings to label -> {gold_file}")
    labelled = _label_run([str(a) for a in queue], which, corpus, gold_file, workdir)
    typer.echo(f"\nlabelled {labelled}; {len(already) + labelled} in {which} now")


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
