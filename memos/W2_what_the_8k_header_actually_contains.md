# What the 8-K header actually contains

**Probe, not assumption.** Written before the classifier, from three real filings pulled
by hand, then checked against 399.

---

## The question

The design assumed 8-K Item codes would give the event category cheaply — Item 2.01 is a
completed acquisition, 3.01 a delisting notice, and so on. Before building a classifier on
that assumption, the header had to be read.

## What is actually there

```
ACCESSION NUMBER:		0001193125-25-080753
CONFORMED SUBMISSION TYPE:	8-K
CONFORMED PERIOD OF REPORT:	20250410
ITEM INFORMATION:		Departure of Directors or Certain Officers; Election of
                                Directors; Appointment of Certain Officers: Compensatory
                                Arrangements of Certain Officers
ITEM INFORMATION:		Financial Statements and Exhibits
FILED AS OF DATE:		20250415
```

**The header carries the item *description*, never the number.** "Departure of Directors
or Certain Officers…" is Item 5.02; "Financial Statements and Exhibits" is 9.01. Nothing
in the header says `5.02`.

The number *does* appear in the document body — "Item 5.02 Departure of…" — but the body
is HTML, and filer agents put tags and entities between the word "Item" and the number.
A `grep -o "Item [0-9]\.[0-9][0-9]"` over the raw document returns nothing on filings that
plainly contain the heading. Recovering it needs the HTML normalised first, and is
unreliable across agents; some render headings as images.

**So: map description strings to numbers.** The strings come from a fixed SEC vocabulary,
so a lookup table is exact rather than heuristic. That became D-003.

## Four things the assumption got wrong

**1. Item codes do not identify corporate actions.** 2.01, 3.01, 1.03 and 5.01 are real
signals. But splits and special dividends overwhelmingly arrive under **Item 8.01, "Other
Events"** — a dumping ground that also carries buyback authorisations, litigation updates
and press releases about new hires — or, worse, under 7.01 (Regulation FD) with the
announcement in an attached press release. The baseline's job is therefore *elimination*,
not classification.

**2. The wording drifts.** Across 399 filings, five descriptions failed to map exactly:

| Observed | SEC's current text | Item |
|---|---|---|
| Material **Modifications** to Rights of Security Holders | Material Modification… | 3.03 |
| **Cost** Associated with Exit or Disposal Activities | **Costs** Associated… | 2.05 |
| **Amendments to the** Registrant's Code of Ethics… | Amendment to Registrant's… | 5.05 |
| Shareholder Nominations Pursuant to Exchange Act Rule 14a-11 | Shareholder Director Nominations | 5.08 |
| Acquisition or disposition of assets | Completion of Acquisition or Disposition of Assets | 2.01 |

The last is pre-2004 vocabulary. EDGAR full-text search reaches back to 2001, so any
corpus sampled by phrase pulls in filings written against an older item list.

Singular/plural drift is the dominant pattern, and it broke the first matcher: Jaccard
over token sets scores "Financial Condition" against "Financial Conditions" at **0.71**,
because a plural is a wholly different token. Character-level `SequenceMatcher` scores the
same pair at 0.99. That is why the similarity floor is character-based.

**3. One filer agent leaks the raw SGML tag.**

```
ITEM INFORMATION:		<ITEMS>1.05
```

Malformed, but the number is right there, so it is read rather than rejected.

**4. Older filings leave the field empty**, which broke my own regex. `^ITEM INFORMATION:\s*(.+)$`
with `re.M` looks correct and is not: `\s*` matches newlines, so on

```
ITEM INFORMATION:
FILED AS OF DATE:		20040722
```

the capture crosses the line break and returns `FILED AS OF DATE: 20040722` as an 8-K item
description. Two 2003–2004 filings surfaced this. The fix is `[^\S\n]*` — whitespace that
is not a newline.

## Document structure

```
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>d835579d8k.htm
<DESCRIPTION>8-K
<TEXT> … </TEXT>
</DOCUMENT>
```

SGML fields are unclosed and run to end of line. A submission commonly contains 15+
documents, most of them XBRL, spreadsheets or images; those are skipped rather than
normalised into noise the model would be billed to read.

**Exhibits matter.** A split or special dividend is typically announced in an EX-99.1
press release attached to a near-empty 8-K. An extractor that stops at the primary
document finds the event but not the record date — the field an index calculator actually
needs.

## Encoding

EDGAR archives are **latin-1**. Decoding as UTF-8 raises on perfectly valid filings.

Filings are also full of typographic characters — non-breaking spaces, smart quotes, em
dashes. These are folded to ASCII during normalisation, and that is not cosmetic: a smart
quote inside a cited span turns a correct citation into a substring match that silently
fails.

## What this changed

- Item numbers come from description strings via an exact table, then explicit aliases,
  then character-level similarity above 0.90, then a loud failure. Never a silent miss.
- The baseline is scoped to elimination, and its false-elimination rate is measured
  rather than assumed. It is **21.4%** on the stratified stratum — see
  `docs/what_the_build_found.md`.
- Exhibits are followed.
- Everything is decoded latin-1 and normalised deterministically before any span is taken.
