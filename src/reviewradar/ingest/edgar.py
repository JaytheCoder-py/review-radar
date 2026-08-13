"""EDGAR ingestion: the daily index, and full submission text.

Parsing is separated from fetching (D-002). `parse_daily_index` and `parse_submission`
are pure functions over text, so the entire parsing surface is testable against committed
fixtures with no network, and a pathological filing can be reproduced by committing its
bytes.

**Character offsets.** `SubmissionDocument.text` is *normalised plain text* — HTML
stripped, entities decoded, whitespace collapsed — and every span in this codebase indexes
that normalised text, never the raw markup. The normalisation is deterministic, so a span
recorded tonight resolves to the same characters when the dashboard renders it next month.
Mixing the two would make every citation silently wrong.
"""

from __future__ import annotations

import datetime as dt
import html
import re
import time
from dataclasses import dataclass
from typing import Final

import requests

from reviewradar.types import Accession, Cik, parse_accession, parse_cik

ARCHIVES: Final = "https://www.sec.gov/Archives"
DAILY_INDEX: Final = "https://www.sec.gov/Archives/edgar/daily-index"

#: Document types that carry no readable prose. XBRL, spreadsheets and images are
#: skipped rather than normalised into noise that the model would have to pay for.
SKIP_DOC_TYPES: Final = re.compile(r"\A(EX-101|EX-99\.SCH|XML|EXCEL|GRAPHIC|ZIP|JSON)", re.I)


class EdgarError(RuntimeError):
    """EDGAR could not serve or could not parse a request.

    Never return an empty result for this. An empty document list is indistinguishable
    from "this filing genuinely had no documents", and that ambiguity is exactly how a
    missed split reaches production.
    """


@dataclass(frozen=True, slots=True)
class FilingRef:
    """One line of the daily index."""

    cik: Cik
    accession: Accession
    form_type: str
    filed_date: dt.date
    company_name: str
    submission_url: str


@dataclass(frozen=True, slots=True)
class SubmissionDocument:
    """One `<DOCUMENT>` block, normalised to plain text."""

    doc_id: str
    doc_type: str
    filename: str
    text: str

    @property
    def is_exhibit(self) -> bool:
        return self.doc_type.upper().startswith("EX-")


@dataclass(frozen=True, slots=True)
class Submission:
    """A complete EDGAR submission: the SGML header plus its documents."""

    ref: FilingRef
    header: str
    documents: tuple[SubmissionDocument, ...]

    def primary(self) -> SubmissionDocument:
        """The filing itself, as opposed to its exhibits."""
        for doc in self.documents:
            if not doc.is_exhibit:
                return doc
        return self.documents[0]

    def exhibits(self) -> tuple[SubmissionDocument, ...]:
        """The exhibits. Often empty, and legitimately so.

        Worth knowing: a split or special dividend is typically announced in an EX-99.1
        press release attached to a near-empty 8-K. Reading only the primary document
        finds nothing.
        """
        primary = self.primary()
        return tuple(d for d in self.documents if d is not primary and d.is_exhibit)

    def full_text(self) -> str:
        """Every document's text, concatenated. Used for phrase search, never for spans."""
        return "\n\n".join(d.text for d in self.documents)


# ----------------------------------------------------------------------------------
# Text normalisation


#: Typographic characters filings are full of, folded to ASCII. Spelled as escapes so
#: the source stays ASCII-only. This matters more than it looks: a smart quote inside a
#: cited span turns a correct citation into a substring match that silently fails.
_TYPOGRAPHIC = str.maketrans(
    {
        0x00A0: " ",  # non-breaking space
        0x2018: "'",  # left single quotation mark
        0x2019: "'",  # right single quotation mark
        0x201C: '"',  # left double quotation mark
        0x201D: '"',  # right double quotation mark
        0x2013: "-",  # en dash
        0x2014: "-",  # em dash
    }
)

_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def normalise(raw: str) -> str:
    """Markup to plain text, deterministically.

    Block-level tags become newlines so that item headings and table rows do not run
    together into a single line; everything else becomes a space. Entities are decoded
    once - twice would turn a literal ``&amp;lt;`` in a filing into a bracket.
    """
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = re.sub(r"<\s*(br|/p|/div|/tr|/h[1-6]|/li)\s*/?>", "\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = text.translate(_TYPOGRAPHIC)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


# ----------------------------------------------------------------------------------
# The client


class EdgarClient:
    """Polite, rate-limited access to EDGAR.

    The SEC blocks IPs that omit a contact User-Agent and rate-limits at 10 requests per
    second. Both are enforced here rather than documented and hoped for: the client
    cannot be constructed without a contact, and cannot issue requests faster than its
    configured rate.
    """

    def __init__(
        self,
        contact: str,
        *,
        requests_per_second: float = 8.0,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not contact or "@" not in contact:
            raise ValueError(
                f"EDGAR requires a contact email in the User-Agent; got {contact!r}. "
                "Requests without one are blocked at the IP level."
            )
        if requests_per_second <= 0 or requests_per_second > 10:
            raise ValueError(
                f"requests_per_second must be in (0, 10]; got {requests_per_second}. "
                "The SEC's published limit is 10/s."
            )
        self.contact = contact
        self.user_agent = f"reviewradar-research/0.1 (contact: {contact})"
        self.timeout = timeout
        self._min_interval = 1.0 / requests_per_second
        self._last_call: float | None = None
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    # -- network ---------------------------------------------------------------

    def _throttle(self) -> None:
        now = time.monotonic()
        if self._last_call is not None:
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, url: str) -> str:
        self._throttle()
        try:
            response = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise EdgarError(f"GET {url} failed: {exc}") from exc
        if response.status_code != 200:
            raise EdgarError(f"GET {url} returned HTTP {response.status_code}")
        # EDGAR archives are latin-1. Decoding as UTF-8 raises on valid filings.
        response.encoding = "latin-1"
        return response.text

    def daily_index(self, on: dt.date, *, form_type: str = "8-K") -> list[FilingRef]:
        """Every filing of `form_type` disseminated on `on`.

        Weekends and federal holidays have no index and raise `EdgarError`.
        """
        quarter = (on.month - 1) // 3 + 1
        url = f"{DAILY_INDEX}/{on.year}/QTR{quarter}/master.{on:%Y%m%d}.idx"
        return self.parse_daily_index(self._get(url), on=on, form_type=form_type)

    def fetch(self, ref: FilingRef) -> Submission:
        """The complete submission text file, parsed into documents."""
        return self.parse_submission(self._get(ref.submission_url), ref=ref)

    # -- pure parsing ----------------------------------------------------------

    @staticmethod
    def parse_daily_index(raw: str, *, on: dt.date, form_type: str = "8-K") -> list[FilingRef]:
        """Parse `master.YYYYMMDD.idx`.

        Pipe-delimited, `CIK|Company Name|Form Type|Date Filed|File Name`, preceded by a
        preamble terminated by a rule of dashes.
        """
        lines = raw.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines) if ln.startswith("---")) + 1
        except StopIteration as exc:
            raise EdgarError("daily index has no header rule; not an EDGAR index file") from exc

        refs: list[FilingRef] = []
        for line in lines[start:]:
            parts = line.split("|")
            if len(parts) != 5:
                continue
            cik_raw, company, form, _filed, path = parts
            if form.strip() != form_type:
                continue
            accession_raw = path.rsplit("/", 1)[-1].removesuffix(".txt")
            refs.append(
                FilingRef(
                    cik=parse_cik(cik_raw.strip()),
                    accession=parse_accession(accession_raw),
                    form_type=form.strip(),
                    filed_date=on,
                    company_name=company.strip(),
                    submission_url=f"{ARCHIVES}/{path.strip()}",
                )
            )
        return refs

    @staticmethod
    def parse_submission(raw: str, *, ref: FilingRef | None = None) -> Submission:
        """Parse a complete submission text file into its header and documents.

        Raises rather than returning an empty document list - see `EdgarError`.
        """
        header_match = re.search(r"<SEC-HEADER>(.*?)</SEC-HEADER>", raw, re.S | re.I)
        if header_match is None:
            # Some older submissions use <IMS-HEADER>. Fall back to everything
            # before the first <DOCUMENT>.
            first_doc = raw.find("<DOCUMENT>")
            if first_doc == -1:
                raise EdgarError("submission contains no <DOCUMENT> block and no header")
            header = raw[:first_doc]
        else:
            header = header_match.group(1)

        documents: list[SubmissionDocument] = []
        accession = ref.accession if ref else EdgarClient._header_accession(header)

        for seq, block in enumerate(re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, re.S | re.I), 1):
            doc_type = EdgarClient._sgml_field(block, "TYPE") or "UNKNOWN"
            if SKIP_DOC_TYPES.match(doc_type):
                continue
            filename = EdgarClient._sgml_field(block, "FILENAME") or f"doc{seq}"
            sequence = EdgarClient._sgml_field(block, "SEQUENCE") or str(seq)
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, re.S | re.I)
            body = text_match.group(1) if text_match else ""
            text = normalise(body)
            if not text:
                continue
            documents.append(
                SubmissionDocument(
                    doc_id=f"{accession}:{sequence}",
                    doc_type=doc_type,
                    filename=filename,
                    text=text,
                )
            )

        if not documents:
            raise EdgarError(
                f"submission {accession} parsed to zero readable documents. "
                "An empty result is indistinguishable from 'no event'; refusing to return one."
            )

        if ref is None:
            raise EdgarError("parse_submission needs a FilingRef to build a Submission")
        return Submission(ref=ref, header=header, documents=tuple(documents))

    # -- header helpers --------------------------------------------------------

    @staticmethod
    def _sgml_field(block: str, tag: str) -> str | None:
        """SGML fields in EDGAR are unclosed: `<TYPE>8-K` runs to end of line."""
        match = re.search(rf"^<{tag}>(.*)$", block, re.M | re.I)
        return match.group(1).strip() if match else None

    @staticmethod
    def _header_accession(header: str) -> str:
        match = re.search(r"ACCESSION NUMBER:\s*(\S+)", header, re.I)
        return match.group(1) if match else "unknown"

    @staticmethod
    def header_items(header: str) -> tuple[str, ...]:
        """The `ITEM INFORMATION:` description strings, verbatim.

        The header carries the item *description*, never the number - established by
        probe, see `memos/W2_what_the_8k_header_actually_contains.md` and D-003. Mapping
        those strings to item numbers is the baseline classifier's job, not this one's.
        """
        # `[^\S\n]*` and not `\s*`: older headers leave the field empty, and `\s*`
        # crosses the newline and captures the *next* header line as an item
        # description. Found on 2003-2004 filings, which reported "FILED AS OF DATE"
        # as an unrecognised 8-K item.
        return tuple(
            stripped
            for m in re.finditer(r"^ITEM INFORMATION:[^\S\n]*(.*)$", header, re.M | re.I)
            if (stripped := m.group(1).strip())
        )

    @staticmethod
    def ref_from_header(raw: str) -> FilingRef:
        """Reconstruct a `FilingRef` from a submission's own SGML header.

        Every field the daily index carries is also in the header, so a submission file
        on disk is self-describing. That removes the need for a side-car manifest, which
        is one more thing that can be lost, go stale, or disagree with the filings it
        claims to describe.
        """
        header = raw[: raw.find("<DOCUMENT>")] if "<DOCUMENT>" in raw else raw[:8000]

        def grab(pattern: str) -> str | None:
            match = re.search(pattern, header, re.I | re.M)
            return match.group(1).strip() if match else None

        accession_raw = grab(r"ACCESSION NUMBER:\s*(\S+)")
        cik_raw = grab(r"CENTRAL INDEX KEY:\s*(\d+)")
        filed_raw = grab(r"FILED AS OF DATE:\s*(\d{8})")
        if accession_raw is None or cik_raw is None or filed_raw is None:
            raise EdgarError(
                "submission header is missing accession, CIK or filing date; "
                "cannot build a FilingRef from it"
            )
        accession = parse_accession(accession_raw)
        cik = parse_cik(cik_raw)
        return FilingRef(
            cik=cik,
            accession=accession,
            form_type=grab(r"CONFORMED SUBMISSION TYPE:\s*(\S+)") or "UNKNOWN",
            filed_date=dt.datetime.strptime(filed_raw, "%Y%m%d").date(),
            company_name=grab(r"COMPANY CONFORMED NAME:\s*(.+)$") or "",
            submission_url=(
                f"{ARCHIVES}/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}.txt"
            ),
        )

    @staticmethod
    def header_period(header: str) -> dt.date | None:
        """`CONFORMED PERIOD OF REPORT`, the date the event relates to."""
        match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header, re.I)
        if match is None:
            return None
        try:
            return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            return None
