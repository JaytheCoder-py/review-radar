"""Command line."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from reviewradar.evals.gold import load_all
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


def _client(use_vertex: bool, project: str | None) -> LlmClient:
    if not use_vertex:
        return OfflineLlm()
    if not project:
        raise typer.BadParameter("--project is required with --vertex")
    from reviewradar.extract.llm import VertexLlm

    return VertexLlm(project=project)


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


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
