# What the build found

Kept as a list because it is the most honest summary of what the tests and the corpus are
for. Every one was found by running the thing against 399 real SEC filings, not by
inspection.

| Found by | Defect |
|---|---|
| Probing three real headers | The header carries the item **description**, never the number. The whole classifier design changed: description-string lookup, not number parsing. |
| Running the corpus | `^ITEM INFORMATION:\s*(.+)$` crosses newlines. Filings that leave the field empty captured the *next* header line, and 2003–2004 filings reported `FILED AS OF DATE` as an unrecognised 8-K item. `\s*` had to become `[^\S\n]*`. |
| Running the corpus | One filer agent leaks the raw SGML tag into the description: `ITEM INFORMATION: <ITEMS>1.05`. Malformed, but the number is right there. |
| Running the corpus | SEC item wording drifts. Five variants in 399 filings — singular/plural and inserted articles — plus a wholly different pre-2004 vocabulary that full-text search drags in. |
| A test I wrote | Jaccard over token sets scores "Financial Condition" against "Financial**s**" at **0.71**, because a plural is a different token. Wording drift is *exactly* singular/plural, so token similarity was the wrong tool. Character-level `SequenceMatcher` scores the same pair 0.99. |
| The gold set | **The headline finding — below.** |
| The gold set | My own `eliminated` test fixture was a real 25% stock split. The test asserted "Regulation FD, no index consequence" about a filing that announced a corporate action. The fixture is now called `false_elimination` and the test asserts the failure. |
| Wiring the verifier to the log | `str(Fraction(3, 1))` is `"3"`, not `"3/1"` — and `to_row` writes the ratio column through `str()`. A verifier that parsed on the slash would have silently dropped every whole-number forward split, which is most of them. |
| Reading how price series are published | Split-adjusted closes have the split divided out of every price *before* the ex-date, so the step forward verification exists to find is **absent from the adjusted series**. The convenient library adjusts by default. A verifier built on one contradicts every correct extraction and looks like it works — see D-008, and the AST test that now fails the build for `adjclose`. |

---

## The headline finding: 21.4% false elimination

The deterministic stage eliminates **41.9%** of 8-K traffic before any model call. That
number looks like a result. It is not, on its own — because the question that matters is
what it eliminates *wrongly*.

Against the hand-labelled gold set:

| Stratum | n | index-relevant | eliminated | **false eliminations** |
|---|---:|---:|---:|---:|
| random | 30 | 1 | 63.3% | 0 / 1 |
| stratified | 36 | 28 | 30.6% | **6 / 28 = 21.4%** |

The six:

| Filing | Event | Items present |
|---|---|---|
| `0000040545-24-000088` GE | spin-off of GE Vernova, 1 share per 4 | `9.01` |
| `0000066382-21-000082` Herman Miller | ticker change MLHR → MLKN | `7.01`, `9.01` |
| `0000885988-05-000066` IntegraMed | 30% stock split (13/10) | `7.01`, `9.01` |
| `0000885988-06-000069` IntegraMed | 25% stock split (5/4) | `7.01`, `9.01` |
| `0000885988-07-000009` IntegraMed | 25% stock split (5/4) | `7.01` |
| `0000950124-06-006105` Princeton National | $0.05 special dividend declared | `2.02`, `9.01` |

Every one of them is disclosed under items that carry **no index consequence on their
own** — Regulation FD, financial statements and exhibits, results of operations — with the
actual announcement in an attached press release.

This is not a bug in the classifier. It is the structural limit of routing on item codes,
and it is the argument for the second stage. GE's Vernova spin-off is the largest US
corporate action of 2024 and a pure item-code system throws it away.

**Why this matters more than an accuracy figure:** a false elimination is invisible by
construction. The filing never reaches a queue, never reaches the model, never reaches a
person. Nobody finds out until the index prints wrong. So the elimination rate is reported
*next to* its error rate, always, and the CI gate is on the false-elimination rate rather
than on accuracy.

---

## Measurement caveats, reported rather than tuned away

- **The gold set is 66 filings, not the 200 the plan called for.** 30 random, 36
  stratified. Per-class counts are thin: one spin-off, two ticker changes, three special
  dividends. The 21.4% has a wide interval around it and should be read as "roughly a
  fifth", not as a point estimate.
- **The random stratum contains one index-relevant filing.** At a ~3% base rate that is
  what 30 filings buys. It is enough to say the population rate is low and not enough to
  quote a false-elimination rate on it — which is why that cell reads 0/1 rather than 0%.
- **The stratified stratum was built by phrase search**, so it over-represents filings
  whose language is explicit. Events described obliquely are under-represented, and the
  baseline probably does *worse* than 21.4% on those, not better.
- **The model column is unmeasured.** The Vertex client is written and the pipeline runs
  end to end against it, but no run has been made — that needs the repository owner's own
  GCP project. Every model figure in this repo is therefore absent rather than estimated.
  The baseline column is real.
- **Labels were assigned by reading the decisive passage of each filing**, not by rule. The
  labelling principle that decided the hard cases: *label the event the filing announces,
  not every event it mentions.* An earnings release referring back to a split already
  effected is `no_index_action`. Six filings turn on that distinction, and a different
  reader could reasonably disagree on two of them.
