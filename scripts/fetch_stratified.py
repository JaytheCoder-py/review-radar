"""Fetch the stratified stratum: EDGAR full-text search for index-relevant phrases.

Separate from the random stratum because they answer different questions and must never
be pooled. Search only *finds candidates*; nothing here classifies them.
"""
from __future__ import annotations
import sys, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reviewradar.ingest.edgar import EdgarClient  # noqa: E402

EFTS = "https://efts.sec.gov/LATEST/search-index"
CONTACT = sys.argv[1] if len(sys.argv) > 1 else "jasonchung.ck@gmail.com"
OUT = Path("data/corpus/submissions"); OUT.mkdir(parents=True, exist_ok=True)

PHRASES = {
  "split_forward": ['"for-one stock split"', '"two-for-one stock split"',
                    '"three-for-one stock split"', '"four-for-one stock split"',
                    '"stock split effected in the form of a stock dividend"',
                    '"ten-for-one stock split"'],
  "split_reverse": ['"reverse stock split"', '"one-for-ten reverse stock split"',
                    '"one-for-twenty reverse stock split"'],
  "spinoff":       ['"the spin-off" "record date"', '"pro rata distribution of all of the outstanding shares"'],
  "merger":        ['"completion of the merger"', '"the merger became effective"',
                    '"merger was consummated"'],
  "delisting":     ['"notice of delisting"', '"will be suspended from trading"',
                    '"determined to delist"'],
  "dividend":      ['"special cash dividend"', '"declared a special dividend"'],
  "rights":        ['"subscription rights" "rights offering"'],
  "ticker":        ['"will begin trading under the ticker symbol"', '"change its ticker symbol"'],
}

client = EdgarClient(contact=CONTACT)
sess = requests.Session()
sess.headers.update({"User-Agent": f"reviewradar-research/0.1 (contact: {CONTACT})"})
seen = {p.stem for p in OUT.glob("*.txt")}
added = 0

for label, phrases in PHRASES.items():
    for phrase in phrases:
        try:
            r = sess.get(EFTS, params={"q": phrase, "forms": "8-K"}, timeout=30)
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])
        except Exception as exc:
            print(f"  search {phrase!r}: {exc}", flush=True); continue
        got = 0
        for hit in hits[:10]:
            acc = hit["_id"].split(":")[0]
            if acc in seen: continue
            ciks = hit.get("_source", {}).get("ciks", [])
            if not ciks: continue
            url = (f"https://www.sec.gov/Archives/edgar/data/{int(ciks[0])}/"
                   f"{acc.replace('-','')}/{acc}.txt")
            try:
                (OUT / f"{acc}.txt").write_text(client._get(url), encoding="utf-8")
            except Exception as exc:
                print(f"    {acc}: {exc}", flush=True); continue
            seen.add(acc); got += 1; added += 1
        print(f"  {label:14s} {phrase[:48]:50s} +{got}", flush=True)
        time.sleep(0.3)

print(f"\nadded {added}; corpus now {len(seen)} filings", flush=True)
