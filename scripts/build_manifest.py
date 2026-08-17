"""Rebuild the corpus manifest from EDGAR, not from whatever was on disk.

Stratum membership is *derived*: a filing is `random` if it appears in the daily index
for one of the sampled dates, and `stratified` otherwise. That makes the split
reproducible from the SEC's own records rather than from the order files happened to be
downloaded in - which is what was lost when the first fetch was interrupted.
"""
from __future__ import annotations
import datetime as dt, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reviewradar.ingest.edgar import EdgarClient  # noqa: E402

CONTACT = sys.argv[1] if len(sys.argv) > 1 else "jaythecoder.py@gmail.com"
DAYS = [dt.date(2025, 4, d) for d in (14, 15, 16, 17, 18, 21)]
client = EdgarClient(contact=CONTACT)

daily: set[str] = set()
for day in DAYS:
    try:
        refs = client.daily_index(day)
        daily |= {r.accession for r in refs}
        print(f"  {day}: {len(refs)} 8-Ks", flush=True)
    except Exception as exc:
        print(f"  {day}: {exc}", flush=True)

manifest = {}
for p in sorted(Path("data/corpus/submissions").glob("*.txt")):
    raw = p.read_text(encoding="utf-8", errors="replace")
    try:
        ref = EdgarClient.ref_from_header(raw)
    except Exception as exc:
        print(f"  {p.stem}: {exc}", flush=True); continue
    manifest[ref.accession] = {
        "stratum": "random" if ref.accession in daily else "stratified",
        "cik": ref.cik, "company": ref.company_name,
        "filed": ref.filed_date.isoformat(), "url": ref.submission_url,
    }

Path("data/corpus/manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
n_r = sum(1 for v in manifest.values() if v["stratum"] == "random")
print(f"\n{len(manifest)} filings: {n_r} random, {len(manifest)-n_r} stratified", flush=True)
