"""The read-only dashboard.

**This layer renders stored figures and computes none of its own.** A published number
must have exactly one source, and a dashboard that re-derives a rate from raw rows is a
second source that will eventually disagree with the first. A test greps this package for
the arithmetic that would create one and fails the build if it appears.

The screen worth demoing is `/filing/{accession}`: the extracted claim next to the exact
sentence it came from, highlighted. That is the whole governance argument in one
screenshot.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from reviewradar.ingest.store import EventStore

_STYLE = """
:root { --bg:#fbfbfa; --fg:#1a1a19; --mut:#6b6b66; --line:#e3e3df; --accent:#8a5a2b;
        --hi:#fdf2c9; --ok:#2f6b3d; --warn:#8a5a2b; --stop:#8c2f2f; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161614; --fg:#eceae4; --mut:#9a978e; --line:#2e2e2a; --hi:#4a3f16; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }
main { max-width:1000px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1 { font-size:1.35rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1rem; margin:2rem 0 .6rem; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); }
nav a { color:var(--accent); text-decoration:none; margin-right:1.1rem; font-weight:500; }
nav { border-bottom:1px solid var(--line); padding-bottom:.75rem; margin-bottom:1.25rem; }
.sub { color:var(--mut); margin:0 0 1.5rem; }
table { border-collapse:collapse; width:100%; font-size:.88rem; }
th,td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--mut); font-weight:600; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
.wrap { overflow-x:auto; }
code { font:.85em ui-monospace,SFMono-Regular,Menlo,monospace; }
.pill { display:inline-block; padding:.08rem .45rem; border-radius:3px; font-size:.75rem; font-weight:600; }
.t-divisor_adjust,.t-remove_constituent,.t-price_adjust,.t-shares_update { background:var(--hi); color:var(--warn); }
.t-manual_review { color:var(--stop); border:1px solid var(--stop); }
.t-no_action { color:var(--mut); }
mark { background:var(--hi); color:inherit; padding:.05rem .15rem; border-radius:2px; }
.quote { border-left:3px solid var(--accent); padding:.6rem .9rem; margin:.5rem 0 1.2rem;
         background:color-mix(in srgb, var(--fg) 3%, transparent); white-space:pre-wrap; font-size:.88rem; }
.empty { color:var(--mut); font-style:italic; }
"""

_NAV = """<nav><a href="/">events</a><a href="/scoreboard">scoreboard</a>
<a href="/failures">failures</a><a href="/healthz">health</a></nav>"""


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{_NAV}{body}</main></body></html>"
    )


def _table(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    if frame.empty:
        return "<p class=empty>Nothing recorded yet.</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for label in columns.values())
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for key in columns:
            value = row.get(key)
            text = "" if value is None or pd.isna(value) else str(value)
            if key == "treatment":
                cells.append(
                    f"<td><span class='pill t-{html.escape(text)}'>"
                    f"{html.escape(text.replace('_', ' '))}</span></td>"
                )
            elif key == "accession":
                cells.append(
                    f"<td><a href='/filing/{html.escape(text)}'><code>"
                    f"{html.escape(text)}</code></a></td>"
                )
            else:
                cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class=wrap><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def create_app(db: Path | str) -> FastAPI:
    api = FastAPI(title="Review Radar", docs_url=None, redoc_url=None)
    path = Path(db)

    def store() -> EventStore:
        return EventStore(path)

    @api.get("/healthz")
    def healthz() -> JSONResponse:
        with store() as st:
            return JSONResponse({"ok": True, "events": len(st.events())})

    @api.get("/", response_class=HTMLResponse)
    def events() -> HTMLResponse:
        with store() as st:
            frame = st.events(current_only=True).head(200)
        body = (
            "<h1>Corporate actions</h1>"
            "<p class=sub>Extracted from SEC 8-K filings. Every field is grounded in a "
            "character span of the source document; a field whose value is not inside its "
            "own citation is dropped, not flagged.</p>"
            + _table(
                frame,
                {
                    "filed_at": "filed",
                    "accession": "accession",
                    "company_name": "company",
                    "event_type": "event",
                    "treatment": "treatment",
                    "confidence": "conf",
                    "ratio": "ratio",
                    "ex_date": "ex-date",
                },
            )
        )
        return _page("Review Radar", body)

    @api.get("/failures", response_class=HTMLResponse)
    def failures() -> HTMLResponse:
        with store() as st:
            frame = st.failures().head(200)
        body = (
            "<h1>Failures</h1><p class=sub>Filings that could not be processed. Recorded, "
            "never skipped: an absent row is indistinguishable from &ldquo;this filing "
            "carried no event&rdquo;, and that ambiguity is how a missed split reaches "
            "production.</p>"
            + _table(
                frame,
                {
                    "recorded_at": "when",
                    "accession": "accession",
                    "stage": "stage",
                    "error": "error",
                },
            )
        )
        return _page("Failures", body)

    def _forward_section() -> str:
        """Forward verification, rendered from the verdict log.

        Counts come from `EventStore.verification_counts`, not from a `value_counts` on
        the frame below. Two renderings of the same figure are two figures.
        """
        with store() as st:
            counts = st.verification_counts()
            frame = st.verifications(latest_only=True).head(200)
        # Display precision only. The log keeps the value the verifier computed; four
        # places is where a step stops being readable and starts being float noise.
        frame = frame.round({"observed_step": 4, "relative_error": 4, "prior_close": 2})
        tally = "".join(
            f"<tr><td>{html.escape(verdict)}</td><td>{count}</td></tr>"
            for verdict, count in sorted(counts.items())
        )
        return (
            "<h2>forward verification</h2>"
            "<p class=sub>An extraction is a claim about the future: a 3-for-1 split with "
            "an ex-date says the <em>unadjusted</em> close steps to about a third on the "
            "first session on or after it. These verdicts are the market's answer, and "
            "they need no hand-labelling. An ex-date that has not passed is "
            "<code>unverifiable</code>, never <code>contradicted</code>.</p>"
            "<p class=sub><strong>Empty is the honest state here, not a gap.</strong> No "
            "model run has been made, so the events in the log are typed by item code "
            "alone and carry no ratio and no ex-date &mdash; nothing a price can test. "
            "These counts move the night the extractor first runs. The arithmetic behind "
            "them is measured against committed price fixtures in the test suite.</p>"
            + (f"<table>{tally}</table>" if tally else "<p class=empty>No verdicts recorded.</p>")
            + _table(
                frame,
                {
                    "accession": "accession",
                    "ticker": "ticker",
                    "verdict": "verdict",
                    "reason": "reason",
                    "ex_date": "ex-date",
                    "session": "session",
                    "expected_step": "expected step",
                    "observed_step": "observed step",
                    "relative_error": "rel. error",
                },
            )
        )

    @api.get("/scoreboard", response_class=HTMLResponse)
    def scoreboard() -> HTMLResponse:
        report = Path("data/scoreboard.json")
        sections = []
        if report.exists():
            # Read and render. Nothing is recomputed here - the numbers come from the
            # scoring run that produced them.
            data: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
            for stratum, block in data.get("strata", {}).items():
                rows = "".join(
                    f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                    for k, v in block.items()
                )
                sections.append(f"<h2>{html.escape(stratum)} stratum</h2><table>{rows}</table>")
        else:
            sections.append(
                "<p class=empty>No extraction scoreboard recorded. "
                "Run <code>reviewradar score</code> to produce one.</p>"
            )
        sections.append(_forward_section())
        return _page(
            "Scoreboard",
            "<h1>Scoreboard</h1><p class=sub>Strata are reported separately and never "
            "pooled: a rate from the stratified stratum is not a population rate.</p>"
            + "".join(sections),
        )

    @api.get("/filing/{accession}", response_class=HTMLResponse)
    def filing(accession: str) -> HTMLResponse:
        with store() as st:
            frame = st.events()
        rows = frame[frame["accession"] == accession]
        if rows.empty:
            return _page(
                "Not found", f"<h1>{html.escape(accession)}</h1><p class=empty>No event.</p>"
            )
        row = rows.iloc[0]
        spans = json.loads(row["spans"] or "{}")
        blocks = []
        for field, entry in spans.items():
            if not entry:
                continue
            value = entry.get("value")
            span = entry.get("span")
            if span is None:
                blocks.append(
                    f"<h2>{html.escape(field)}</h2><p><code>{html.escape(str(value))}</code>"
                    f" <span class=empty>&mdash; no citation (abstention or baseline-derived)</span></p>"
                )
                continue
            quoted = str(span.get("text", ""))
            blocks.append(
                f"<h2>{html.escape(field)}</h2>"
                f"<p><code>{html.escape(str(value))}</code> &mdash; "
                f"<span class=empty>{html.escape(str(span.get('doc_id')))} "
                f"[{span.get('start')}:{span.get('end')}]</span></p>"
                f"<div class=quote><mark>{html.escape(quoted)}</mark></div>"
            )
        header = (
            f"<h1>{html.escape(str(row['company_name']))}</h1>"
            f"<p class=sub><code>{html.escape(accession)}</code> &middot; filed "
            f"{html.escape(str(row['filed_at']))} &middot; items "
            f"{html.escape(str(row['items']) or 'none')} &middot; "
            f"<span class='pill t-{html.escape(str(row['treatment']))}'>"
            f"{html.escape(str(row['treatment']).replace('_', ' '))}</span></p>"
            f"<p class=sub>{html.escape(str(row['rationale']))}</p>"
        )
        dropped = str(row["dropped_fields"] or "")
        if dropped:
            header += (
                f"<p class=sub><strong>Dropped:</strong> <code>{html.escape(dropped)}</code>"
                " &mdash; the value was not inside its own cited span.</p>"
            )
        return _page(accession, header + "".join(blocks))

    return api
