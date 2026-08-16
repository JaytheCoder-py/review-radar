# Price fixtures

Unlike the EDGAR fixtures next door, these are **constructed, not recorded.** Building
this repository must not require a market-data fetch, and no free source permits
redistributing its tape, so there is no honest way to commit a real one here.

Each series is written the way an exchange prints it: **unadjusted, as-traded closes**,
with the corporate action's step left in. That is the whole point — a back-adjusted
series has the split divided out of every price before the ex-date, so the step these
fixtures exist to exercise would not be in the file at all. See
`src/reviewradar/evals/forward.py` and D-008.

What they test is the arithmetic, the ratio orientation, and the trading-calendar
handling. What they cannot test is whether a live source returns adjusted prices — that
is a contract on `PriceSource`, written down and asserted about `YahooChartPrices` in the
code, because no test that refuses to touch the network can check it.

| file | encodes |
|---|---|
| `FSPL.json` | a 3/1 forward split; the close steps to about a third |
| `RSPL.json` | a 1/8 reverse split; the close steps to about eight times |
| `FLAT.json` | a 3/1 split that never happened; the close does not move |
| `HOLI.json` | an ex-date on a Saturday; the step lands on the following Monday |
| `NEAR.json` | a step 7.5% off the expected one — inside the band, deliberately |
| `WIDE.json` | a step 12% off the expected one — outside the band, deliberately |
| `SOON.json` | data that stops before the ex-date; no session has closed yet |

Format: `{"ticker": ..., "note": ..., "closes": [["YYYY-MM-DD", close], ...]}`.
