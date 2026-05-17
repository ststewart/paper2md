"""Copyright / open-access metadata front-end for paper2md.

Resolves an article identifier (DOI or arXiv ID) from a PDF, queries free
public scholarly-metadata APIs to determine the declared license and any
open-access copy, and returns a typed result the main pipeline can embed
in its YAML front-matter.

Design contract: best-effort, never raises out of `resolve()`. On any
network or parse error the returned dataclass has `confidence="low"`,
`license=None`, `safe_to_distribute="false"` and the caller proceeds.

API call order (only what's needed; stops as soon as license is firm):

    DOI in PDF -->  OpenAlex  -->  Unpaywall (license fallback / OA URL)
                                 -->  Europe PMC (if biomed and still no PDF)
    arXiv ID  -->  arXiv API (per-paper license slug, may be the
                              non-redistributable arXiv default)
    no ID    -->  Crossref title search (heuristic title from page 1)

Polite-pool emails come from env vars: UNPAYWALL_EMAIL (required by
Unpaywall), OPENALEX_MAILTO, CROSSREF_MAILTO. Same address may be reused.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import fitz  # PyMuPDF
import requests

log = logging.getLogger("paper2md.metadata")


# ---------------------------------------------------------------------------
# Identifiers in the PDF
# ---------------------------------------------------------------------------

# Crossref's DOI grammar: "10." + registrant + "/" + suffix. Restrictive enough
# to skip false matches like version strings (10.1.0) but loose on the suffix.
# NOTE: trailing-punctuation strip happens in _extract_doi (some PDFs render a
# DOI followed by ")." or "," which the regex would otherwise swallow).
DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[^\s\"<>]+",
    re.IGNORECASE,
)

# arXiv IDs come in two flavors:
#   new-style: 2401.01234, 2401.01234v2, 2401.01234v12
#   old-style: cs.LG/0701001, math/0703245
# We only catch new-style here; old-style has a DOI form too and is rare in
# scientific journal corpora.
ARXIV_RE = re.compile(
    r"\barXiv:(\d{4}\.\d{4,5})(v\d+)?",
    re.IGNORECASE,
)

# DOIs that show up at the article level for entire publishers; never the
# DOI of a specific article. Filter out so we don't lock onto a journal-wide
# notice DOI (e.g., copyright statements that contain the publisher's own DOI).
_DOI_BLOCKLIST_PREFIXES = (
    "10.0/",          # placeholder
    "10.0000/",       # placeholder
    "10.5555/",       # IETF/ACM example DOIs
)


def _extract_doi_from_pdf(pdf_path: Path, *, page_limit: int = 2) -> Optional[str]:
    """Find a DOI in the PDF metadata dict or the first `page_limit` pages.

    Most journals print the DOI on page 1 (header, footer, or right under the
    title). We scan two pages to catch journals that put it on a banner page
    before the article text. Returns the first plausible match, lowercased."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            for key in ("doi", "subject", "keywords"):
                v = doc.metadata.get(key) or "" if doc.metadata else ""
                m = DOI_RE.search(str(v))
                if m:
                    return _clean_doi(m.group(0))
            for i in range(min(page_limit, doc.page_count)):
                text = doc[i].get_text("text") or ""
                # "doi.org/" prefix is the most reliable cue when present;
                # try it first to avoid matching a DOI inside a citation list.
                m = re.search(
                    r"doi\.org/(10\.\d{4,9}/[^\s\"<>]+)",
                    text,
                    re.IGNORECASE,
                )
                if m:
                    return _clean_doi(m.group(1))
                m = DOI_RE.search(text)
                if m:
                    return _clean_doi(m.group(0))
    except Exception as e:
        log.debug("DOI extraction failed for %s: %s", pdf_path, e)
    return None


def _clean_doi(doi: str) -> Optional[str]:
    doi = doi.strip().lower().rstrip(".,;)>]")
    for prefix in _DOI_BLOCKLIST_PREFIXES:
        if doi.startswith(prefix):
            return None
    return doi


# DOI prefix / suffix patterns -> publisher (and sub-journal where the DOI
# suffix encodes it). Order matters: more specific patterns must come before
# their bare-prefix fallback so e.g. `10.1103/physrevlett.*` resolves to
# `aps-prl` rather than the catch-all `aps`.
#
# Used by Phase 1 of the journal-aware reference-reconstruction work
# (instrumentation only): the slug is recorded in manifest / frontmatter /
# .meta.json so future per-journal rescue passes have a stable key. No
# behavior changes when the slug is set.
_JOURNAL_SLUG_RULES: list[tuple[re.Pattern, str]] = [
    # APS — DOI suffix encodes the journal (physrevlett, physrevb, ...).
    (re.compile(r"^10\.1103/physrevlett\b", re.I), "aps-prl"),
    (re.compile(r"^10\.1103/physrevb\b", re.I), "aps-prb"),
    (re.compile(r"^10\.1103/physreva\b", re.I), "aps-pra"),
    (re.compile(r"^10\.1103/physreve\b", re.I), "aps-pre"),
    (re.compile(r"^10\.1103/physrevd\b", re.I), "aps-prd"),
    (re.compile(r"^10\.1103/physrevx\b", re.I), "aps-prx"),
    (re.compile(r"^10\.1103/physrevresearch\b", re.I), "aps-prresearch"),
    (re.compile(r"^10\.1103/revmodphys\b", re.I), "aps-rmp"),
    (re.compile(r"^10\.1103/", re.I), "aps"),
    # Science family.
    (re.compile(r"^10\.1126/sciadv\b", re.I), "science-advances"),
    (re.compile(r"^10\.1126/science\b", re.I), "science"),
    (re.compile(r"^10\.1126/", re.I), "science"),
    # Nature family — modern (s4xxxx) IDs encode the journal in the prefix.
    (re.compile(r"^10\.1038/s41561\b", re.I), "nature-geoscience"),
    (re.compile(r"^10\.1038/s41550\b", re.I), "nature-astronomy"),
    (re.compile(r"^10\.1038/s41586\b", re.I), "nature"),
    (re.compile(r"^10\.1038/ngeo", re.I), "nature-geoscience"),
    (re.compile(r"^10\.1038/nature", re.I), "nature"),
    (re.compile(r"^10\.1038/", re.I), "nature"),
    # Elsevier — `j.<journal>.<year>` slug carries the journal abbrev.
    (re.compile(r"^10\.1016/j\.icarus\b", re.I), "elsevier-icarus"),
    (re.compile(r"^10\.1016/j\.epsl\b", re.I), "elsevier-epsl"),
    (re.compile(r"^10\.1016/j\.gca\b", re.I), "elsevier-gca"),
    (re.compile(r"^10\.1016/j\.pepi\b", re.I), "elsevier-pepi"),
    (re.compile(r"^10\.1016/j\.chemgeo\b", re.I), "elsevier-chemgeo"),
    (re.compile(r"^10\.1016/", re.I), "elsevier"),
    # Society / publisher catch-alls.
    (re.compile(r"^10\.1029/", re.I), "agu"),
    (re.compile(r"^10\.1073/pnas\b", re.I), "pnas"),
    (re.compile(r"^10\.1073/", re.I), "pnas"),
    (re.compile(r"^10\.1063/", re.I), "aip"),
    (re.compile(r"^10\.1098/", re.I), "royal-society"),
    (re.compile(r"^10\.5194/", re.I), "copernicus"),
    (re.compile(r"^10\.1093/", re.I), "oup"),
    (re.compile(r"^10\.1002/", re.I), "wiley"),
    (re.compile(r"^10\.1146/annurev", re.I), "annual-reviews"),
    (re.compile(r"^10\.3847/", re.I), "iop-aas"),
    (re.compile(r"^10\.1088/", re.I), "iop"),
]


def _resolve_journal_slug(doi: Optional[str]) -> Optional[str]:
    """Map a DOI to a journal/publisher slug for downstream dispatch.

    Returns None when the DOI is missing or its prefix isn't in the table.
    The slug is intentionally stable (no version suffix); sub-journal
    refinement happens here so consumers key off the slug directly."""
    if not doi:
        return None
    s = doi.strip().lower()
    if not s:
        return None
    for rx, slug in _JOURNAL_SLUG_RULES:
        if rx.search(s):
            return slug
    return None


def _extract_arxiv_id(pdf_path: Path, *, page_limit: int = 2) -> Optional[str]:
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i in range(min(page_limit, doc.page_count)):
                text = doc[i].get_text("text") or ""
                m = ARXIV_RE.search(text)
                if m:
                    return m.group(1)  # without the "vN" suffix
    except Exception:
        return None
    return None


def _extract_title_from_pdf(pdf_path: Path) -> Optional[str]:
    """Heuristic: largest-font text block on page 1 that looks like a title.

    Used only as a fallback for Crossref title search when no DOI is found.
    Skips obvious non-titles (single-word headers, all-caps journal names)."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            if doc.page_count == 0:
                return None
            # fitz metadata title is sometimes useful and sometimes garbage
            # (placeholder strings like "untitled-1" or the production filename).
            md_title = (doc.metadata or {}).get("title") or ""
            md_title = md_title.strip()
            if md_title and len(md_title) > 15 and not md_title.lower().startswith("untitled"):
                return md_title

            blocks = doc[0].get_text("dict").get("blocks", [])
            candidates: list[tuple[float, str]] = []
            for b in blocks:
                if b.get("type") != 0:
                    continue
                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if len(text) < 15 or len(text) > 300:
                        continue
                    if text.isupper():
                        continue  # all-caps is usually a section header / journal name
                    size = max(s.get("size", 0.0) for s in spans)
                    candidates.append((size, text))
            if not candidates:
                return None
            candidates.sort(reverse=True)
            return candidates[0][1]
    except Exception as e:
        log.debug("Title extraction failed for %s: %s", pdf_path, e)
        return None


# ---------------------------------------------------------------------------
# License normalization
# ---------------------------------------------------------------------------
#
# APIs return license info in two forms:
#   1. A normalized slug ("cc-by", "cc-by-nc-4.0", "cc0", "public-domain")
#   2. A URL ("https://creativecommons.org/licenses/by/4.0/")
#
# We accept both, normalize to lowercase slugs with versions stripped for
# safety classification, and keep the full versioned slug + URL in the
# returned dataclass so downstream consumers can apply finer policy.

# URL pattern -> slug. Order matters: more specific first (by-nc-sa before by-nc).
_LICENSE_URL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"creativecommons\.org/publicdomain/zero", re.I), "cc0"),
    (re.compile(r"creativecommons\.org/publicdomain/mark", re.I), "public-domain"),
    (re.compile(r"creativecommons\.org/licenses/by-nc-sa/(\d+\.\d+)", re.I), "cc-by-nc-sa-{ver}"),
    (re.compile(r"creativecommons\.org/licenses/by-nc-nd/(\d+\.\d+)", re.I), "cc-by-nc-nd-{ver}"),
    (re.compile(r"creativecommons\.org/licenses/by-nc/(\d+\.\d+)", re.I), "cc-by-nc-{ver}"),
    (re.compile(r"creativecommons\.org/licenses/by-nd/(\d+\.\d+)", re.I), "cc-by-nd-{ver}"),
    (re.compile(r"creativecommons\.org/licenses/by-sa/(\d+\.\d+)", re.I), "cc-by-sa-{ver}"),
    (re.compile(r"creativecommons\.org/licenses/by/(\d+\.\d+)", re.I), "cc-by-{ver}"),
    (re.compile(r"creativecommons\.org/licenses/by-nc-sa", re.I), "cc-by-nc-sa"),
    (re.compile(r"creativecommons\.org/licenses/by-nc-nd", re.I), "cc-by-nc-nd"),
    (re.compile(r"creativecommons\.org/licenses/by-nc", re.I), "cc-by-nc"),
    (re.compile(r"creativecommons\.org/licenses/by-nd", re.I), "cc-by-nd"),
    (re.compile(r"creativecommons\.org/licenses/by-sa", re.I), "cc-by-sa"),
    (re.compile(r"creativecommons\.org/licenses/by", re.I), "cc-by"),
    (re.compile(r"arxiv\.org/licenses/nonexclusive-distrib", re.I), "arxiv-default"),
    (re.compile(r"arxiv\.org/licenses/assumed-1991-2003", re.I), "arxiv-pre-2004"),
    (re.compile(r"opensource\.org/licenses/MIT", re.I), "mit"),
    (re.compile(r"www\.elsevier\.com/.*/tdm", re.I), "elsevier-tdm"),
]

# Canonical CC URL for each base slug (4.0 default; older versions emit their own).
_LICENSE_CANONICAL_URL = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "public-domain": "https://creativecommons.org/publicdomain/mark/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc-by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "arxiv-default": "https://arxiv.org/licenses/nonexclusive-distrib/1.0/",
    "arxiv-pre-2004": "https://arxiv.org/licenses/assumed-1991-2003/",
    "elsevier-tdm": "https://www.elsevier.com/about/policies/text-and-data-mining",
    # Synthetic slugs the resolver assigns when a public PDF is reachable but
    # no permissive license is declared. These map to the 'readable' safety
    # tier: read-only personal/research use is fine, redistribution is not.
    "pmc-author-manuscript": "https://pmc.ncbi.nlm.nih.gov/about/copyright/",
    # DOE PAGES (OSTI) deposit under the federal public-access mandate.
    # Same legal posture as a PMC author manuscript: freely readable,
    # not under a CC license, not redistributable.
    "osti-public-access": "https://www.osti.gov/about/policies",
    "green-oa-no-license": None,  # no canonical URL; depends on host repo
    # US 95-year rolling cutoff: a work first published in year Y enters US
    # public domain on Jan 1 of year Y+96. Synthetic slug, set by resolve()
    # when the API chain returns a confident publication year and no
    # license; maps to the GREEN tier (safe_to_distribute=true).
    "public-domain-us": "https://www.copyright.gov/help/faq/faq-duration.html",
}

# Safety tiers, by base slug (versioned forms round-trip into the same tier).
#   GREEN  -> safe_to_distribute = "true"        (CC0, PD, CC-BY, CC-BY-SA)
#   YELLOW -> safe_to_distribute = "restricted"  (CC-BY-NC*, CC-BY-ND*)
#   READABLE -> safe_to_distribute = "readable"  (public PDF, no permissive
#                                                 license declared — fair use
#                                                 reading and personal/research
#                                                 reference OK; bulk
#                                                 redistribution NOT OK)
#   anything else -> safe_to_distribute = "false" (paywalled, unknown,
#                                                  publisher-TDM-only)
GREEN = {"cc0", "public-domain", "public-domain-us", "cc-by", "cc-by-sa"}
YELLOW = {"cc-by-nc", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-nd"}
READABLE = {"arxiv-default", "arxiv-pre-2004",
            "pmc-author-manuscript", "osti-public-access",
            "green-oa-no-license"}


def _normalise_license(value: Optional[str]) -> Optional[str]:
    """Map a slug or URL to the lowercase canonical slug. None on no match.

    Versioned forms (cc-by-4.0) round-trip; unversioned (cc-by) stay as-is."""
    if not value:
        return None
    s = str(value).strip().lower()
    if not s:
        return None

    # Already a slug?
    if re.fullmatch(r"cc(0|-by|-by-sa|-by-nc|-by-nd|-by-nc-sa|-by-nc-nd)(-\d+\.\d+)?", s):
        return s
    if s in {"public-domain", "publicdomain", "pd"}:
        return "public-domain"
    if s in {"arxiv-default", "arxiv"}:
        return "arxiv-default"
    if s == "mit":
        return "mit"
    if s in {"pmc-author-manuscript", "green-oa-no-license",
             "osti-public-access", "public-domain-us"}:
        return s

    # URL form
    for rx, template in _LICENSE_URL_RULES:
        m = rx.search(s)
        if not m:
            continue
        if m.groups():
            return template.format(ver=m.group(1))
        return template

    return None


def _classify_safety(license_slug: Optional[str]) -> str:
    """Return one of 'true', 'restricted', 'readable', 'false' for the
    safe_to_distribute flag. Unknown / unrecognized always falls in 'false'."""
    if not license_slug:
        return "false"
    base = re.sub(r"-\d+\.\d+$", "", license_slug)
    if base in GREEN or base == "mit":
        return "true"
    if base in YELLOW:
        return "restricted"
    if base in READABLE:
        return "readable"
    return "false"


def _license_url(license_slug: Optional[str]) -> Optional[str]:
    if not license_slug:
        return None
    base = re.sub(r"-\d+\.\d+$", "", license_slug)
    return _LICENSE_CANONICAL_URL.get(base)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArticleMetadata:
    """Resolved copyright / OA result. Always safe to embed in YAML."""

    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None                 # publication year (used for the US 95-year PD rule)
    # Publisher / journal slug derived from the DOI prefix (e.g. "aps-prl",
    # "elsevier-icarus", "nature-geoscience", "science"). Set by resolve()
    # whenever a DOI is known. Phase-1 instrumentation for journal-aware
    # reference reconstruction; consumers should treat None as "unknown".
    journal_slug: Optional[str] = None

    license: Optional[str] = None             # versioned slug, e.g. "cc-by-4.0"
    license_url: Optional[str] = None
    safe_to_distribute: str = "false"          # "true" | "restricted" | "false"
    confidence: str = "low"                    # "high" | "medium" | "low"

    oa_status: str = "unknown"                 # gold | green | bronze | hybrid | closed | unknown
    oa_pdf_url: Optional[str] = None
    oa_pdf_used: bool = False
    oa_pdf_local_path: Optional[Path] = None   # set only when prefer_oa downloaded successfully
    oa_pdf_source: Optional[str] = None        # repository hostname

    resolved_via: Optional[str] = None         # which API set the license
    provenance: list[str] = field(default_factory=list)  # human-readable trace

    def to_yaml_lines(self) -> list[str]:
        """Emit a 'copyright:' YAML block matching the QualityReport style."""
        L = ["copyright:"]
        if self.doi:
            L.append(f"  doi: {self.doi}")
        if self.arxiv_id:
            L.append(f"  arxiv_id: {self.arxiv_id}")
        if self.journal_slug:
            L.append(f"  journal_slug: {self.journal_slug}")
        if self.year is not None:
            L.append(f"  year: {self.year}")
        L.append(f"  license: {self.license or 'unknown'}")
        if self.license_url:
            L.append(f"  license_url: {self.license_url}")
        L.append(f"  safe_to_distribute: {self.safe_to_distribute}")
        L.append(f"  confidence: {self.confidence}")
        L.append(f"  oa_status: {self.oa_status}")
        L.append(f"  oa_pdf_used: {'true' if self.oa_pdf_used else 'false'}")
        if self.oa_pdf_url:
            L.append(f"  oa_pdf_url: {self.oa_pdf_url}")
        if self.oa_pdf_source:
            L.append(f"  oa_pdf_source: {self.oa_pdf_source}")
        if self.oa_pdf_used and self.oa_pdf_local_path:
            L.append(f"  oa_pdf_local: {self.oa_pdf_local_path.name}")
        if self.resolved_via:
            L.append(f"  resolved_via: {self.resolved_via}")
        return L

    def manifest_dict(self) -> dict:
        """Compact dict for batch-mode manifest.jsonl."""
        return {
            "doi": self.doi,
            "journal_slug": self.journal_slug,
            "license": self.license,
            "safe_to_distribute": self.safe_to_distribute,
            "oa_pdf_used": self.oa_pdf_used,
        }

    def to_dict(self) -> dict:
        """Full structured dict for the .meta.json sidecar / HDF5 bundle.
        Mirrors the YAML 'copyright:' block but in machine-readable form
        (no string quoting, Path serialised to filename only)."""
        return {
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "journal_slug": self.journal_slug,
            "license": self.license,
            "license_url": self.license_url,
            "safe_to_distribute": self.safe_to_distribute,
            "confidence": self.confidence,
            "oa_status": self.oa_status,
            "oa_pdf_url": self.oa_pdf_url,
            "oa_pdf_used": self.oa_pdf_used,
            "oa_pdf_local": (self.oa_pdf_local_path.name
                             if self.oa_pdf_local_path else None),
            "oa_pdf_source": self.oa_pdf_source,
            "resolved_via": self.resolved_via,
            "provenance": list(self.provenance),
        }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_USER_AGENT = "paper2md-metadata-frontend/1.0 (https://github.com/anthropics/paper2md)"


def _http_get_json(url: str, timeout: float, *, session: requests.Session) -> Optional[dict]:
    try:
        r = session.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            log.debug("HTTP %d on %s", r.status_code, url)
            return None
        return r.json()
    except (requests.RequestException, ValueError) as e:
        log.debug("HTTP error on %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# API queries
# ---------------------------------------------------------------------------

def _query_openalex(doi: str, mailto: Optional[str], timeout: float,
                    session: requests.Session) -> Optional[dict]:
    url = f"https://api.openalex.org/works/doi:{quote(doi, safe='/.')}"
    if mailto:
        url += f"?mailto={quote(mailto)}"
    return _http_get_json(url, timeout, session=session)


def _query_unpaywall(doi: str, email: Optional[str], timeout: float,
                     session: requests.Session) -> Optional[dict]:
    if not email:
        # Unpaywall requires email; without one, the call is guaranteed 422.
        return None
    url = f"https://api.unpaywall.org/v2/{quote(doi, safe='/.')}?email={quote(email)}"
    return _http_get_json(url, timeout, session=session)


def _query_europepmc(doi: str, timeout: float,
                     session: requests.Session) -> Optional[dict]:
    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
           f"?query=DOI:{quote(doi)}&resultType=core&format=json")
    data = _http_get_json(url, timeout, session=session)
    if not data:
        return None
    hits = data.get("resultList", {}).get("result", [])
    return hits[0] if hits else None


def _query_arxiv(arxiv_id: str, timeout: float,
                 session: requests.Session) -> Optional[str]:
    """arXiv API returns an Atom feed, not JSON. Pull license out via regex."""
    url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
    try:
        r = session.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        if r.status_code >= 400:
            return None
        body = r.text
    except requests.RequestException:
        return None
    m = re.search(r"<arxiv:(?:doi|license)[^>]*>([^<]+)</arxiv:license>", body)
    if not m:
        # Fall back to <link rel="license">
        m = re.search(r'<link[^>]+rel="license"[^>]+href="([^"]+)"', body)
    return m.group(1) if m else None


_TITLE_NORMALISE_RE = re.compile(r"[^a-z0-9]+")


def _normalise_title(s: str) -> str:
    """Lowercase + strip punctuation/whitespace for cheap title comparison."""
    return _TITLE_NORMALISE_RE.sub("", (s or "").lower())


def _query_crossref_by_title(title: str, mailto: Optional[str],
                             timeout: float,
                             session: requests.Session) -> Optional[dict]:
    """Title-search Crossref. Crossref's bibliographic-search relevance
    score is unreliable on old / sparsely-indexed papers (e.g. PNAS 1920
    records can score below an unrelated 1960 Nature paper that shares a
    keyword). Strategy: widen to top 5 results, accept the first whose
    normalised title equals the query exactly; fall back to the score
    >= 50 rule on the top hit only when no exact-title match is found."""
    if not title:
        return None
    url = ("https://api.crossref.org/works?rows=5"
           f"&query.bibliographic={quote(title)}")
    if mailto:
        url += f"&mailto={quote(mailto)}"
    data = _http_get_json(url, timeout, session=session)
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    target = _normalise_title(title)
    for item in items:
        cand = (item.get("title") or [""])[0]
        if _normalise_title(cand) == target:
            return item
    # No exact-title match in the top 5: accept the top hit only if
    # Crossref's score is firmly high (modern, well-indexed papers).
    top = items[0]
    if top.get("score", 0) < 50:
        return None
    return top


def _crossref_year(work: dict) -> Optional[int]:
    """Pull a publication year from a Crossref work record. Crossref nests
    the year under several keys with varying reliability; we prefer
    `published.date-parts[0][0]` (issued or published-online), fall back to
    `issued`, then `published-print`."""
    for key in ("published", "issued", "published-print", "published-online"):
        block = work.get(key) or {}
        parts = block.get("date-parts") or []
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError, IndexError):
                continue
    return None


def _first_author_surname(authors: list) -> Optional[str]:
    """Pull the surname from the first author entry. Authors are stored
    as 'Given Family' strings (whatever the API returned). Surname is
    the last whitespace-separated token. Returns None on an empty
    list."""
    if not authors:
        return None
    first = (authors[0] or "").strip()
    if not first:
        return None
    # Tolerate "Last, First" form too — split on the comma if present.
    if "," in first:
        return first.split(",", 1)[0].strip() or None
    parts = first.split()
    return parts[-1] if parts else None


def _query_osti_pages(doi: str, title: Optional[str],
                      author_surname: Optional[str],
                      timeout: float,
                      session: requests.Session) -> Optional[dict]:
    """Search DOE PAGES (OSTI) for an article. PAGES indexes accepted
    manuscripts of DOE-funded research; many physical-science papers
    have a deposit there even when OpenAlex/Unpaywall report
    `oa_status: closed`.

    OSTI does not document a DOI query parameter, so we try general
    full-record search (`q`) first -- which usually catches DOIs --
    and fall back to title + author surname. Either way, we accept a
    record only if its `doi` field exactly matches the input DOI.

    Returns dict with `fulltext_url`, `article_type`, `osti_id`, or
    None when no DOI-matched record is found."""
    base = "https://www.osti.gov/pages/api/v1/records"

    def _search(params_dict):
        qs = "&".join(f"{k}={quote(str(v))}" for k, v in params_dict.items()
                      if v)
        return _http_get_json(f"{base}?{qs}", timeout, session=session)

    target = (doi or "").lower()

    # Strategy 1: q={doi}. PAGES's general full-record search often
    # indexes DOIs even though there's no dedicated parameter.
    candidates = _search({"q": doi, "rows": 5}) or []
    # Strategy 2: title + author search if the DOI search didn't
    # produce a matching record.
    matched = _osti_match(candidates, target)
    if matched is None and title:
        more = _search({"title": title,
                        "author": author_surname,
                        "rows": 10}) or []
        matched = _osti_match(more, target)
    return matched


def _osti_match(candidates, target_doi: str) -> Optional[dict]:
    """Walk an OSTI search result list, return the first record whose
    `doi` field exactly matches `target_doi` (case-insensitive). The
    returned dict surfaces only the fields the resolver needs."""
    if not isinstance(candidates, list):
        return None
    for rec in candidates:
        if not isinstance(rec, dict):
            continue
        rec_doi = (rec.get("doi") or "").lower().strip()
        if not rec_doi or rec_doi != target_doi:
            continue
        # Find the fulltext link in the typed-links array.
        fulltext_url = None
        for link in rec.get("links") or []:
            if not isinstance(link, dict):
                continue
            if (link.get("rel") or "").lower() == "fulltext":
                fulltext_url = link.get("url") or link.get("href")
                if fulltext_url:
                    break
        return {
            "osti_id": rec.get("osti_id"),
            "article_type": rec.get("article_type"),
            "fulltext_url": fulltext_url,
        }
    return None


# ---------------------------------------------------------------------------
# OA PDF download
# ---------------------------------------------------------------------------

def _download_oa_pdf(url: str, dest: Path, timeout: float,
                     session: requests.Session) -> bool:
    try:
        with session.get(url, timeout=timeout, stream=True,
                         headers={"User-Agent": _USER_AGENT}) as r:
            if r.status_code >= 400:
                return False
            ctype = r.headers.get("Content-Type", "")
            # Some repos serve HTML landing pages at the "PDF" URL when the
            # actual file is gated. Reject anything that isn't clearly a PDF.
            if "pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
            return dest.stat().st_size > 1024  # reject empty / tiny files
    except (requests.RequestException, OSError) as e:
        log.debug("OA download failed %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Reference-list fetch (Crossref -> OpenAlex fallback)
# ---------------------------------------------------------------------------

def _query_crossref_refs(doi: str, mailto: Optional[str], timeout: float,
                         session: requests.Session) -> Optional[list[dict]]:
    """Fetch a paper's deposited reference list from Crossref.

    Returns the raw `references` array (list of dicts) preserving the
    deposit order, or None on miss / network error / no deposited refs.
    Crossref preserves citation order when the publisher deposits in
    order (true for APS, AAAS, AGU, Nature, Elsevier in practice).
    """
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{quote(doi, safe='/.')}"
    if mailto:
        url += f"?mailto={quote(mailto)}"
    data = _http_get_json(url, timeout, session=session)
    if not data:
        return None
    refs = data.get("message", {}).get("reference") or []
    return refs if refs else None


def _query_openalex_refs(doi: str, mailto: Optional[str], timeout: float,
                         session: requests.Session) -> Optional[list[dict]]:
    """Fetch a paper's reference list via OpenAlex (fallback when
    Crossref has none).

    OpenAlex returns `referenced_works` as a list of OpenAlex IDs.
    We batch-fetch up to 50 per call via the `ids.openalex:` filter,
    then preserve the original order by mapping ID -> Work record.
    Returns the ordered list of resolved Work dicts, or None on miss.
    """
    if not doi:
        return None
    work = _query_openalex(doi, mailto, timeout, session)
    if not work:
        return None
    raw_ids = work.get("referenced_works") or []
    if not raw_ids:
        return None
    bare = [u.rsplit("/", 1)[-1] for u in raw_ids if u]
    fetched: dict[str, dict] = {}
    for i in range(0, len(bare), 50):
        batch = bare[i:i + 50]
        url = (f"https://api.openalex.org/works"
               f"?filter=ids.openalex:{'|'.join(batch)}"
               f"&per-page=50")
        if mailto:
            url += f"&mailto={quote(mailto)}"
        data = _http_get_json(url, timeout, session=session)
        if not data:
            continue
        for w in data.get("results", []) or []:
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            if wid:
                fetched[wid] = w
    ordered = [fetched[b] for b in bare if b in fetched]
    return ordered if ordered else None


def _format_crossref_ref(entry: dict, idx: int) -> str:
    """Render a single Crossref reference entry as a markdown bullet.
    Prefers the `unstructured` field (publisher-supplied citation
    string) when present; otherwise composes from structured fields."""
    if entry.get("unstructured"):
        text = " ".join(entry["unstructured"].split())
        return f"- {idx}. {text}"
    parts: list[str] = []
    if entry.get("author"):
        parts.append(str(entry["author"]))
    if entry.get("year"):
        parts.append(f"({entry['year']})")
    if entry.get("article-title"):
        parts.append(str(entry["article-title"]))
    elif entry.get("volume-title"):
        parts.append(str(entry["volume-title"]))
    if entry.get("journal-title"):
        parts.append(f"*{entry['journal-title']}*")
    if entry.get("volume"):
        parts.append(f"vol. {entry['volume']}")
    if entry.get("first-page"):
        parts.append(f"p. {entry['first-page']}")
    if entry.get("DOI"):
        parts.append(f"[doi:{entry['DOI']}](https://doi.org/{entry['DOI']})")
    body = ", ".join(p for p in parts if p)
    return f"- {idx}. {body}." if body else f"- {idx}. (incomplete reference)"


def _format_openalex_ref(work: dict, idx: int) -> str:
    """Render an OpenAlex Work record as a markdown bullet."""
    parts: list[str] = []
    auths = work.get("authorships") or []
    names = []
    for a in auths[:6]:
        nm = (a.get("author") or {}).get("display_name")
        if nm:
            names.append(str(nm))
    if names:
        if len(auths) > len(names):
            parts.append(", ".join(names) + ", et al.")
        else:
            parts.append(", ".join(names))
    if work.get("publication_year"):
        parts.append(f"({work['publication_year']})")
    if work.get("display_name"):
        parts.append(str(work["display_name"]))
    venue = (work.get("host_venue") or {}).get("display_name") \
        or ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    if venue:
        parts.append(f"*{venue}*")
    doi = work.get("doi")
    if doi:
        bare = doi.replace("https://doi.org/", "")
        parts.append(f"[doi:{bare}]({doi})")
    body = ", ".join(p for p in parts if p)
    return f"- {idx}. {body}." if body else f"- {idx}. (incomplete reference)"


def _default_refs_cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") \
        / "paper2md" / "references.json"


def fetch_references(doi: str,
                     *,
                     allow_network: bool = True,
                     timeout_s: float = 10.0,
                     cache_path: Optional[Path] = None
                     ) -> Optional[dict]:
    """Best-effort fetch of a paper's reference list, formatted as
    markdown bullets ready to splice into a `## References (from ...)`
    section. Never raises.

    Try order: Crossref `/works/{doi}`.references -> OpenAlex
    `referenced_works` (batched). First success wins.

    Returns:
        {"source": "crossref" | "openalex", "refs": list[str], "fetched_at": iso8601}
        on success; None on miss, no DOI, network disabled, or all-API
        failure. Result is cached on disk by DOI in
        ~/.cache/paper2md/references.json (override via cache_path).
    """
    if not doi:
        return None
    cp = cache_path or _default_refs_cache_path()
    cache = _load_cache(cp)
    if doi in cache:
        log.debug("fetch_references: cache hit for %s", doi)
        return cache[doi]
    if not allow_network:
        log.debug("fetch_references: network disabled, skipping %s", doi)
        return None
    crossref_mailto = os.environ.get("CROSSREF_MAILTO")
    openalex_mailto = os.environ.get("OPENALEX_MAILTO") or crossref_mailto
    session = requests.Session()
    try:
        cr_refs = _query_crossref_refs(doi, crossref_mailto,
                                       timeout_s, session)
    except Exception as e:
        log.debug("Crossref refs query failed: %s", e)
        cr_refs = None
    if cr_refs:
        formatted = [_format_crossref_ref(e, i + 1)
                     for i, e in enumerate(cr_refs)]
        result = {
            "source": "crossref",
            "refs": formatted,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        cache[doi] = result
        _save_cache(cp, cache)
        return result
    try:
        oa_refs = _query_openalex_refs(doi, openalex_mailto,
                                       timeout_s, session)
    except Exception as e:
        log.debug("OpenAlex refs query failed: %s", e)
        oa_refs = None
    if oa_refs:
        formatted = [_format_openalex_ref(w, i + 1)
                     for i, w in enumerate(oa_refs)]
        result = {
            "source": "openalex",
            "refs": formatted,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        cache[doi] = result
        _save_cache(cp, cache)
        return result
    log.debug("fetch_references: no refs from any API for %s", doi)
    return None


# ---------------------------------------------------------------------------
# Cache (DOI -> resolved metadata, JSON-on-disk)
# ---------------------------------------------------------------------------

def _default_cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") \
        / "paper2md" / "metadata.json"


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, default=str))
        tmp.replace(path)
    except OSError as e:
        log.debug("Cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Top-level resolver
# ---------------------------------------------------------------------------

def resolve(pdf_path: Path,
            *,
            allow_network: bool = True,
            prefer_oa: bool = False,
            timeout_s: float = 10.0,
            cache_path: Optional[Path] = None,
            oa_download_dir: Optional[Path] = None) -> ArticleMetadata:
    """Resolve copyright + OA metadata for one PDF. Never raises.

    Args:
        pdf_path: input PDF.
        allow_network: when False, only the local PDF (DOI/arXiv ID/title
            extraction) is consulted; no API calls are made.
        prefer_oa: when True and a downloadable PDF URL is found on a
            public repository, fetch it to `oa_download_dir` and set
            `oa_pdf_used=True`. License judgment is independent — the
            downloaded copy may still be marked safe_to_distribute=false
            (e.g. green-OA author manuscripts kept for reference only).
        timeout_s: per-API request timeout in seconds.
        cache_path: JSON file to memoize DOI->metadata. Defaults to
            ~/.cache/paper2md/metadata.json.
        oa_download_dir: where to write downloaded OA PDFs. Defaults to
            <cache_path>.parent / "oa_pdfs".
    """
    meta = ArticleMetadata()

    # Local extraction is always available, even offline.
    meta.doi = _extract_doi_from_pdf(pdf_path)
    meta.arxiv_id = _extract_arxiv_id(pdf_path)
    meta.journal_slug = _resolve_journal_slug(meta.doi)
    if meta.doi:
        meta.provenance.append(f"doi-from-pdf:{meta.doi}")
    if meta.journal_slug:
        meta.provenance.append(f"journal-slug:{meta.journal_slug}")
    if meta.arxiv_id:
        meta.provenance.append(f"arxiv-from-pdf:{meta.arxiv_id}")

    if not allow_network:
        meta.provenance.append("network-disabled")
        return meta

    cache_path = cache_path or _default_cache_path()
    cache = _load_cache(cache_path)
    cache_key = meta.doi or (f"arxiv:{meta.arxiv_id}" if meta.arxiv_id else None)
    if cache_key and cache_key in cache:
        cached = cache[cache_key]
        meta.provenance.append("cache-hit")
        for k, v in cached.items():
            if hasattr(meta, k) and v is not None:
                setattr(meta, k, v)
        # Cache lookup doesn't carry over to a fresh OA download decision
        meta.oa_pdf_used = False
        meta.oa_pdf_local_path = None

    crossref_mailto = os.environ.get("CROSSREF_MAILTO")
    openalex_mailto = os.environ.get("OPENALEX_MAILTO") or crossref_mailto
    unpaywall_email = os.environ.get("UNPAYWALL_EMAIL") or crossref_mailto

    session = requests.Session()
    try:
        if not meta.doi and not meta.arxiv_id:
            # Title fallback. Confidence stays medium even on a high-score match.
            title = _extract_title_from_pdf(pdf_path)
            meta.title = title
            cr = _query_crossref_by_title(title or "", crossref_mailto,
                                          timeout_s, session) if title else None
            if cr:
                meta.doi = (cr.get("DOI") or "").lower() or None
                meta.title = " ".join(cr.get("title") or []) or meta.title
                meta.authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in (cr.get("author") or [])
                ]
                if meta.year is None:
                    meta.year = _crossref_year(cr)
                if meta.journal_slug is None:
                    meta.journal_slug = _resolve_journal_slug(meta.doi)
                meta.confidence = "medium"
                meta.resolved_via = "crossref-title"
                meta.provenance.append(f"crossref-title:{meta.doi}")
            else:
                meta.provenance.append("title-search-no-match")

        # Resolve via OpenAlex (one call gets license + OA URL + metadata).
        if meta.doi:
            oax = _query_openalex(meta.doi, openalex_mailto, timeout_s, session)
            if oax:
                meta.title = meta.title or oax.get("title")
                meta.authors = meta.authors or [
                    a.get("author", {}).get("display_name", "")
                    for a in oax.get("authorships") or []
                ]
                if meta.year is None and oax.get("publication_year"):
                    try:
                        meta.year = int(oax["publication_year"])
                    except (TypeError, ValueError):
                        pass
                oa = oax.get("open_access") or {}
                meta.oa_status = oa.get("oa_status") or "unknown"
                meta.oa_pdf_url = oa.get("oa_url") or meta.oa_pdf_url
                # OpenAlex carries license on each location; primary first.
                lic = ((oax.get("primary_location") or {}).get("license")
                       or (oax.get("best_oa_location") or {}).get("license"))
                norm = _normalise_license(lic)
                if norm:
                    meta.license = norm
                    meta.license_url = _license_url(norm)
                    meta.confidence = meta.confidence if meta.confidence == "medium" else "high"
                    meta.resolved_via = meta.resolved_via or "openalex"
                meta.provenance.append("openalex-ok")
            else:
                meta.provenance.append("openalex-miss")

        # Unpaywall: license fallback + better OA URL.
        if meta.doi and (meta.license is None or meta.oa_pdf_url is None):
            up = _query_unpaywall(meta.doi, unpaywall_email, timeout_s, session)
            if up:
                if meta.oa_status == "unknown":
                    meta.oa_status = "gold" if up.get("is_oa") else "closed"
                best = up.get("best_oa_location") or {}
                if not meta.oa_pdf_url:
                    meta.oa_pdf_url = best.get("url_for_pdf") or best.get("url")
                if not meta.license:
                    norm = _normalise_license(best.get("license"))
                    if norm:
                        meta.license = norm
                        meta.license_url = _license_url(norm)
                        meta.confidence = meta.confidence if meta.confidence == "medium" else "high"
                        meta.resolved_via = meta.resolved_via or "unpaywall"
                meta.provenance.append("unpaywall-ok")
            elif unpaywall_email:
                meta.provenance.append("unpaywall-miss")
            else:
                meta.provenance.append("unpaywall-skipped-no-email")

        # Europe PMC fallback. Fires when:
        #   - the license is still unknown (the original case), OR
        #   - oa_status is 'green' but no direct PDF URL is in hand: OpenAlex
        #     returns NCBI PMC landing-page URLs that don't serve a PDF
        #     (Content-Type: text/html), so the OA-substitution download
        #     would fail; Europe PMC's ?pdf=render URL is reliable instead.
        needs_pmc = meta.doi and (
            (meta.license is None
             and meta.oa_status in {"unknown", "closed", "green"})
            or (meta.oa_status == "green"
                and (not meta.oa_pdf_url
                     or "ncbi.nlm.nih.gov/pmc" in (meta.oa_pdf_url or "")))
        )
        if needs_pmc:
            ep = _query_europepmc(meta.doi, timeout_s, session)
            if ep:
                if meta.year is None and ep.get("pubYear"):
                    try:
                        meta.year = int(ep["pubYear"])
                    except (TypeError, ValueError):
                        pass
                norm = _normalise_license(ep.get("license"))
                if norm and meta.license is None:
                    meta.license = norm
                    meta.license_url = _license_url(norm)
                    meta.confidence = meta.confidence if meta.confidence == "medium" else "high"
                    meta.resolved_via = meta.resolved_via or "europepmc"
                # Europe PMC's isOpenAccess=Y means the article is in EPMC's
                # open-access subset (CC-licensed); isOpenAccess=N is common
                # for self-archived author manuscripts that are still readable
                # but not under a redistributable license. Either way, when a
                # PMC ID exists with hasPDF=Y, EPMC's ?pdf=render URL is a
                # direct download — strictly more useful than the NCBI PMC
                # landing-page URL OpenAlex hands back.
                pmcid = ep.get("pmcid")
                if pmcid and ep.get("hasPDF") == "Y":
                    meta.oa_pdf_url = (
                        f"https://europepmc.org/articles/{pmcid}?pdf=render"
                    )
                    # When EPMC has the PDF but no CC license is declared
                    # (isOpenAccess=N), this is a green-OA author manuscript
                    # deposit: publicly readable, not redistributable.
                    if meta.license is None and ep.get("isOpenAccess") != "Y":
                        meta.license = "pmc-author-manuscript"
                        meta.license_url = _license_url("pmc-author-manuscript")
                        meta.confidence = meta.confidence if meta.confidence == "medium" else "high"
                        meta.resolved_via = meta.resolved_via or "europepmc"
                meta.provenance.append("europepmc-ok")
            else:
                meta.provenance.append("europepmc-miss")

        # DOE PAGES (OSTI) fallback for DOE-funded papers. Fires when
        # we have a DOI and either no license OR no OA URL. Returns a
        # synthetic 'osti-public-access' license slug that maps to the
        # 'readable' tier (federal public-access mandate, not CC).
        # ~1-2 s overhead per paper that reaches this fallback; non-
        # DOE papers get a fast OSTI miss and continue.
        if meta.doi and (meta.license is None or meta.oa_pdf_url is None):
            osti = _query_osti_pages(
                meta.doi,
                meta.title,
                _first_author_surname(meta.authors),
                timeout_s, session)
            if osti:
                if meta.oa_pdf_url is None and osti.get("fulltext_url"):
                    meta.oa_pdf_url = osti["fulltext_url"]
                    if meta.oa_status in ("unknown", "closed"):
                        meta.oa_status = "green"
                if meta.license is None:
                    meta.license = "osti-public-access"
                    meta.license_url = _license_url("osti-public-access")
                    meta.confidence = (meta.confidence
                                       if meta.confidence == "medium"
                                       else "high")
                    meta.resolved_via = meta.resolved_via or "osti"
                meta.provenance.append("osti-ok")
            else:
                meta.provenance.append("osti-miss")

        # arXiv fallback when there's no DOI but we found an arXiv ID.
        if meta.license is None and meta.arxiv_id:
            ax = _query_arxiv(meta.arxiv_id, timeout_s, session)
            norm = _normalise_license(ax)
            if norm:
                meta.license = norm
                meta.license_url = _license_url(norm)
                meta.confidence = meta.confidence if meta.confidence == "medium" else "high"
                meta.resolved_via = meta.resolved_via or "arxiv"
                meta.provenance.append(f"arxiv-license:{norm}")
            else:
                meta.provenance.append("arxiv-no-license")

        # Catch-all for the 'readable' tier: the APIs gave us a downloadable
        # PDF on a public OA repository but no permissive license was
        # declared anywhere. Mark as green-oa-no-license so downstream
        # consumers know the paper is publicly readable but the extraction
        # is not safe to redistribute. Distinct from a real CC license.
        if (meta.license is None
                and meta.oa_pdf_url
                and meta.oa_status in {"green", "bronze", "gold", "hybrid"}):
            meta.license = "green-oa-no-license"
            meta.confidence = meta.confidence if meta.confidence == "medium" else "high"
            meta.resolved_via = meta.resolved_via or "oa-url-default"

        # US 95-year public-domain rule. Any work first published in year Y
        # enters US public domain on Jan 1 of Y+96. Fires when we have a
        # confident year from the API chain AND the current license is
        # either unresolved or a synthetic readable-tier fallback
        # (pmc-author-manuscript, osti-public-access, green-oa-no-license).
        # Real CC / publisher licenses are left alone — even on a PD-era
        # paper, deferring to the rights-holder's stated declaration is
        # the safer choice. arXiv default slugs are excluded because arXiv
        # didn't exist before 1991.
        _PD_OVERRIDABLE = {None, "pmc-author-manuscript",
                           "osti-public-access", "green-oa-no-license"}
        if meta.year is not None and meta.license in _PD_OVERRIDABLE:
            from datetime import date as _date
            if meta.year + 95 < _date.today().year:
                prev = meta.license
                meta.license = "public-domain-us"
                meta.license_url = _license_url("public-domain-us")
                meta.confidence = (meta.confidence
                                   if meta.confidence == "medium"
                                   else "high")
                meta.resolved_via = "us-95-year-rule"
                tag = (f"public-domain-us:year={meta.year}"
                       if prev is None
                       else f"public-domain-us:year={meta.year} "
                            f"(override {prev})")
                meta.provenance.append(tag)

        # Final classification.
        meta.safe_to_distribute = _classify_safety(meta.license)

        # Persist to cache (without the OA-download bookkeeping; that's
        # decided per-run, not per-DOI).
        if cache_key:
            snapshot = asdict(meta)
            snapshot.pop("oa_pdf_used", None)
            snapshot.pop("oa_pdf_local_path", None)
            snapshot.pop("provenance", None)
            cache[cache_key] = snapshot
            _save_cache(cache_path, cache)

        # Optional OA-PDF substitution. The downloaded copy is kept next to
        # the markdown output (when oa_download_dir is set to out_dir) so the
        # user can inspect / archive it; filename mirrors the input stem.
        #
        # `--prefer-oa-source` is an explicit user opt-in: when a PDF URL is
        # in hand, fetch it. Distribution safety is recorded in the
        # safe_to_distribute field for downstream consumers, but it does NOT
        # gate the download — many "closed" papers have a green-OA author
        # manuscript on PMC that's freely readable for reference even though
        # the markdown produced from it is not redistributable.
        if prefer_oa and meta.oa_pdf_url:
            dl_dir = oa_download_dir or (cache_path.parent / "oa_pdfs")
            dest = dl_dir / f"{pdf_path.stem}_oa_source.pdf"
            if dest.exists() and dest.stat().st_size > 1024:
                meta.oa_pdf_used = True
                meta.oa_pdf_local_path = dest
                meta.provenance.append("oa-pdf-cached")
            elif _download_oa_pdf(meta.oa_pdf_url, dest, timeout_s, session):
                meta.oa_pdf_used = True
                meta.oa_pdf_local_path = dest
                meta.provenance.append("oa-pdf-downloaded")
            else:
                meta.provenance.append("oa-pdf-download-failed")
            if meta.oa_pdf_used and meta.oa_pdf_url:
                # Record the publishing repository hostname.
                from urllib.parse import urlparse
                meta.oa_pdf_source = urlparse(meta.oa_pdf_url).netloc or None
    finally:
        session.close()

    return meta
