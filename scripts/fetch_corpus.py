"""Fetch a reproducible corpus of real 8-K filings.

Two sampling paths, deliberately kept apart because they answer different questions
(see the design spec, section 9):

* **random**     - every 8-K disseminated on a set of trading days, sampled uniformly.
                   Produces population rates: what fraction of real traffic carries an
                   index consequence, what the true manual-review rate would be.
* **stratified** - EDGAR full-text search for the phrases that accompany index-relevant
                   events, so that per-event-type F1 has enough observations per class.
                   Rates computed on this stratum are NOT population rates.

Run:  uv run python scripts/fetch_corpus.py --contact you@example.com
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewradar.ingest.edgar import EdgarClient  # noqa: E402

EFTS = "https://efts.sec.gov/LATEST/search-index"

#: Phrases that accompany index-relevant events. Used only to *find candidates* for
#: the stratified stratum - never to classify them. Labelling is done by reading.
PHRASES: dict[str, list[str]] = {
    "split": [
        '"for-one stock split"',
        '"stock split effected in the form of a stock dividend"',
        '"reverse stock split"',
    ],
    "spinoff": ['"spin-off"  "record date"', '"distribution of all of the outstanding shares"'],
    "merger": ['"completion of the merger"', '"merger became effective"'],
    "delisting": ['"notice of delisting"', '"will be suspended from trading"'],
    "dividend": ['"special cash dividend"', '"special dividend"  "record date"'],
    "rights": ['"rights offering"  "subscription rights"'],
}


def search(phrase: str, contact: str, *, limit: int = 12) -> list[tuple[str, str]]:
    """Full-text search. Returns (cik, accession) pairs."""
    resp = requests.get(
        EFTS,
        params={"q": phrase, "forms": "8-K"},
        headers={"User-Agent": f"reviewradar-research/0.1 (contact: {contact})"},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    out: list[tuple[str, str]] = []
    for hit in hits[:limit]:
        # _id looks like "0001193125-25-080753:d835579d8k.htm"
        accession = hit["_id"].split(":")[0]
        ciks = hit.get("_source", {}).get("ciks", [])
        if ciks:
            out.append((ciks[0], accession))
    return out


def trading_days(start: dt.date, n: int) -> list[dt.date]:
    days, day = [], start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--per-day", type=int, default=45)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2025, 4, 14))
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    client = EdgarClient(contact=args.contact)
    rng = random.Random(args.seed)
    raw_dir = args.out / "submissions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}

    # -- random stratum --------------------------------------------------------
    for day in trading_days(args.start, args.days):
        try:
            refs = client.daily_index(day)
        except Exception as exc:  # noqa: BLE001 - weekends, holidays, transient 403s
            print(f"  {day}: skipped ({exc})")
            continue
        picked = rng.sample(refs, min(args.per_day, len(refs)))
        print(f"  {day}: {len(refs)} 8-Ks, sampling {len(picked)}")
        for ref in picked:
            dest = raw_dir / f"{ref.accession}.txt"
            if not dest.exists():
                try:
                    dest.write_text(client._get(ref.submission_url), encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    print(f"    {ref.accession}: {exc}")
                    continue
            manifest[ref.accession] = {
                "stratum": "random",
                "cik": ref.cik,
                "company": ref.company_name,
                "filed": day.isoformat(),
                "url": ref.submission_url,
            }

    # -- stratified stratum ----------------------------------------------------
    for label, phrases in PHRASES.items():
        for phrase in phrases:
            try:
                hits = search(phrase, args.contact)
            except Exception as exc:  # noqa: BLE001
                print(f"  search {phrase!r} failed: {exc}")
                continue
            print(f"  {label}: {phrase} -> {len(hits)} hits")
            for cik, accession in hits:
                if accession in manifest:
                    continue
                dest = raw_dir / f"{accession}.txt"
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession.replace('-', '')}/{accession}.txt"
                )
                if not dest.exists():
                    try:
                        dest.write_text(client._get(url), encoding="utf-8")
                    except Exception as exc:  # noqa: BLE001
                        print(f"    {accession}: {exc}")
                        continue
                manifest[accession] = {
                    "stratum": "stratified",
                    "cik": cik,
                    "company": "",
                    "filed": "",
                    "url": url,
                    "search_hint": label,
                }
            time.sleep(0.3)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_random = sum(1 for v in manifest.values() if v["stratum"] == "random")
    n_strat = len(manifest) - n_random
    print(f"\ncorpus: {len(manifest)} filings ({n_random} random, {n_strat} stratified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
