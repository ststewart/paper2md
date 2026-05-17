"""Data-repository link extraction for paper2md.

Scans the post-pipeline markdown for DOIs / URLs that point at recognized
research-data repositories (Zenodo, Dryad, Dataverse, figshare, OSF,
PANGAEA, ESS-DIVE, Mendeley Data, ICPSR, CaltechDATA), deduplicates by
canonical DOI, flags whether each link sits inside a Data / Code
Availability section header, and optionally fetches a one-shot summary
(title, description, license, file list) from each repo's public API.

Always-on: link extraction (deterministic regex, zero network).
Opt-in:    summary fetch (one HTTP GET per unique link, behind
           --fetch-data-repos).

Design contract: best-effort, never raises out of `extract_data_links`
or `fetch_summary`. Network / parse errors are logged at debug and
recorded as `fetch_status` on the entry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from urllib.parse import quote

import requests

log = logging.getLogger("paper2md.data_repos")

_USER_AGENT = "paper2md-data-repos/1.0"
_DEFAULT_TIMEOUT = 8.0
_DESC_MAX_CHARS = 600
_FILES_MAX = 50  # cap per-deposit file list to keep YAML/JSON tractable


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Markdown headings (any depth) whose text indicates a data / code
# availability section. Tolerates bold-wrapping (`# **Data Availability**`)
# and trailing punctuation. Matches:
#   Data Availability / Statement
#   Code Availability
#   Data and Code Availability
#   Data Accessibility / Access
#   Data Sharing
_DATA_SECTION_RE = re.compile(
    r"(?im)^\s*#+\s*\**\s*"
    r"(?:data\s+and\s+code|code\s+and\s+data|data|code)\s+"
    r"(?:availability|accessibility|access|sharing)"
    r"(?:\s+statement)?"
    r"\s*\**\s*[:.]?\s*$"
)

# Bold paragraph that announces data availability without being a markdown
# heading. Some journals format the section title as bold body text.
_DATA_BOLD_RE = re.compile(
    r"(?im)^\s*\*\*\s*"
    r"(?:data\s+and\s+code|code\s+and\s+data|data|code)\s+"
    r"(?:availability|accessibility|access|sharing)"
    r"(?:\s+statement)?"
    r"\s*\**\s*[:.]?\s*\*\*\s*$"
)


def _section_spans(md: str) -> list[tuple[int, int, str]]:
    """Return [(start_offset, end_offset, header_text)] for each
    Data/Code Availability section in `md`. The end offset is the
    start of the next markdown heading at any depth, or len(md).
    """
    spans: list[tuple[int, int, str]] = []
    matches = list(_DATA_SECTION_RE.finditer(md))
    matches.extend(_DATA_BOLD_RE.finditer(md))
    matches.sort(key=lambda m: m.start())
    if not matches:
        return spans
    next_header = re.compile(r"(?m)^\s*#+\s+", re.MULTILINE)
    for m in matches:
        # Find the next heading after this one.
        nxt = next_header.search(md, m.end())
        end = nxt.start() if nxt else len(md)
        header_text = m.group(0).strip().lstrip("#").strip().strip("*").strip()
        spans.append((m.start(), end, header_text))
    return spans


# ---------------------------------------------------------------------------
# DataLink record
# ---------------------------------------------------------------------------


@dataclass
class DataLink:
    repository: str                 # canonical short name, e.g. "zenodo"
    url: str                        # canonical https URL (resolves the deposit)
    doi: Optional[str] = None       # canonical DOI (no scheme)
    record_id: Optional[str] = None # repo-native record/GUID, when available
    section: Optional[str] = None   # heading text the link sits under, if any
    confidence: str = "medium"      # "high" if inside a data-availability section
    # Populated only when fetch_summary() runs successfully.
    title: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    files: list[dict] = field(default_factory=list)
    fetched_at: Optional[str] = None    # ISO timestamp UTC
    fetch_status: Optional[str] = None  # ok | not_implemented | not_found | http_error | parse_error | network_error

    def to_yaml_lines(self, indent: str = "  - ") -> list[str]:
        L = [f"{indent}repository: {self.repository}"]
        sub = " " * len(indent)
        L.append(f"{sub}url: {self.url}")
        if self.doi:
            L.append(f"{sub}doi: {self.doi}")
        if self.record_id:
            L.append(f"{sub}record_id: {_yaml_str(self.record_id)}")
        if self.section:
            L.append(f"{sub}section: {_yaml_str(self.section)}")
        L.append(f"{sub}confidence: {self.confidence}")
        if self.fetch_status:
            L.append(f"{sub}fetch_status: {self.fetch_status}")
        if self.title:
            L.append(f"{sub}title: {_yaml_str(self.title)}")
        if self.description:
            L.append(f"{sub}description: {_yaml_str(self.description)}")
        if self.license:
            L.append(f"{sub}license: {_yaml_str(self.license)}")
        if self.files:
            L.append(f"{sub}files:")
            for f in self.files:
                L.append(f"{sub}  - name: {_yaml_str(f.get('name', ''))}")
                if f.get("size") is not None:
                    L.append(f"{sub}    size: {f['size']}")
                if f.get("format"):
                    L.append(f"{sub}    format: {_yaml_str(f['format'])}")
        if self.fetched_at:
            L.append(f"{sub}fetched_at: {self.fetched_at}")
        return L

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop empty file list / null fields to keep JSON tidy.
        if not d["files"]:
            d.pop("files")
        return {k: v for k, v in d.items() if v is not None}


def _yaml_str(s: str) -> str:
    """Quote a string for safe YAML emission. Mirrors paper2md._yaml_str."""
    if s is None:
        return '""'
    s = str(s)
    if any(c in s for c in ":#\n\"'`{}[]&*!|>%@") or s.strip() != s:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'
    return s


# ---------------------------------------------------------------------------
# Recognizers
# ---------------------------------------------------------------------------


@dataclass
class _Recognizer:
    name: str
    doi_re: Optional[re.Pattern]    # (?P<id>...) capturing the registrar-assigned ID
    url_res: list[re.Pattern]       # alternate URL forms; same (?P<id>...) group
    doi_template: str               # "10.5281/zenodo.{id}" etc.
    canonical_url: Callable[[dict], str]   # build the resolver URL from match dict
    fetch: Optional[Callable[["DataLink", requests.Session, float], None]] = None


def _zenodo_canonical(g: dict) -> str:
    return f"https://zenodo.org/records/{g['id']}"


def _dryad_canonical(g: dict) -> str:
    return f"https://datadryad.org/dataset/doi:10.5061/dryad.{g['id']}"


def _harvard_dv_canonical(g: dict) -> str:
    return f"https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/{g['id']}"


def _generic_dv_canonical(g: dict) -> str:
    # Generic Dataverse needs the host preserved.
    host = g.get("host", "dataverse.harvard.edu")
    persistent = g["persistent"]
    return f"https://{host}/dataset.xhtml?persistentId={persistent}"


def _figshare_canonical(g: dict) -> str:
    return f"https://figshare.com/articles/{g['id']}"


def _osf_canonical(g: dict) -> str:
    return f"https://osf.io/{g['id']}/"


def _pangaea_canonical(g: dict) -> str:
    return f"https://doi.pangaea.de/10.1594/PANGAEA.{g['id']}"


def _essdive_canonical(g: dict) -> str:
    return f"https://data.ess-dive.lbl.gov/view/doi:10.15485/{g['id']}"


def _mendeley_canonical(g: dict) -> str:
    return f"https://data.mendeley.com/datasets/{g['id']}"


def _icpsr_canonical(g: dict) -> str:
    return f"https://www.icpsr.umich.edu/web/ICPSR/studies/{g['id']}"


def _caltech_canonical(g: dict) -> str:
    return f"https://data.caltech.edu/records/{g['id']}"


# ---------------------------------------------------------------------------
# Fetchers (one per repo with a usable public API)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, session: requests.Session,
                   timeout: float) -> Optional[dict]:
    try:
        r = session.get(url, timeout=timeout,
                        headers={"User-Agent": _USER_AGENT,
                                 "Accept": "application/json"})
    except requests.RequestException as e:
        log.debug("network error on %s: %s", url, e)
        return None
    if r.status_code == 404:
        return {"__status__": 404}
    if r.status_code >= 400:
        log.debug("HTTP %d on %s", r.status_code, url)
        return {"__status__": r.status_code}
    try:
        return r.json()
    except ValueError as e:
        log.debug("JSON parse error on %s: %s", url, e)
        return {"__status__": "parse_error"}


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > _DESC_MAX_CHARS:
        s = s[:_DESC_MAX_CHARS].rstrip() + "..."
    return s


def _set_fetched(link: DataLink) -> None:
    link.fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_zenodo(link: DataLink, session: requests.Session, timeout: float) -> None:
    if not link.record_id:
        link.fetch_status = "not_implemented"
        return
    js = _http_get_json(f"https://zenodo.org/api/records/{link.record_id}",
                        session, timeout)
    _set_fetched(link)
    if js is None:
        link.fetch_status = "network_error"
        return
    if "__status__" in js:
        link.fetch_status = "not_found" if js["__status__"] == 404 else "http_error"
        return
    md = js.get("metadata", {}) or {}
    link.title = md.get("title") or None
    link.description = _strip_html(md.get("description", "")) or None
    lic = md.get("license", {})
    if isinstance(lic, dict):
        link.license = lic.get("id") or lic.get("title") or None
    elif isinstance(lic, str):
        link.license = lic
    files = js.get("files") or []
    for f in files[:_FILES_MAX]:
        link.files.append({
            "name": f.get("key") or f.get("filename"),
            "size": f.get("size"),
            "format": (f.get("type") or "").lstrip("."),
        })
    link.fetch_status = "ok"


def _fetch_dryad(link: DataLink, session: requests.Session, timeout: float) -> None:
    if not link.doi:
        link.fetch_status = "not_implemented"
        return
    encoded = quote(f"doi:{link.doi}", safe="")
    js = _http_get_json(f"https://datadryad.org/api/v2/datasets/{encoded}",
                        session, timeout)
    _set_fetched(link)
    if js is None:
        link.fetch_status = "network_error"
        return
    if "__status__" in js:
        link.fetch_status = "not_found" if js["__status__"] == 404 else "http_error"
        return
    link.title = js.get("title") or None
    link.description = _strip_html(js.get("abstract", "")) or None
    # License appears as an SPDX URL or slug at the top level.
    link.license = js.get("license") or None
    # Files live under the latest version: dataset -> stash:version -> stash:files.
    ver_href = (js.get("_links", {}) or {}).get("stash:version", {}).get("href")
    if ver_href:
        ver_url = ver_href if ver_href.startswith("http") \
            else "https://datadryad.org" + ver_href
        vjs = _http_get_json(ver_url, session, timeout)
        if isinstance(vjs, dict) and "__status__" not in vjs:
            files_href = (vjs.get("_links", {}) or {}).get("stash:files", {}).get("href")
            if files_href:
                files_url = files_href if files_href.startswith("http") \
                    else "https://datadryad.org" + files_href
                fjs = _http_get_json(files_url, session, timeout)
                if isinstance(fjs, dict) and "_embedded" in fjs:
                    for f in (fjs["_embedded"].get("stash:files") or [])[:_FILES_MAX]:
                        link.files.append({
                            "name": f.get("path"),
                            "size": f.get("size"),
                            "format": f.get("mimeType") or "",
                        })
    link.fetch_status = "ok"


def _fetch_dataverse(link: DataLink, session: requests.Session, timeout: float) -> None:
    """Works against any Dataverse install (Harvard, Borealis, etc.)
    Uses the canonical URL's host + the link's DOI."""
    if not link.doi:
        link.fetch_status = "not_implemented"
        return
    # Extract host from canonical URL.
    m = re.match(r"https?://([^/]+)/", link.url)
    host = m.group(1) if m else "dataverse.harvard.edu"
    persistent = quote(f"doi:{link.doi}", safe="")
    js = _http_get_json(
        f"https://{host}/api/datasets/:persistentId/?persistentId={persistent}",
        session, timeout)
    _set_fetched(link)
    if js is None:
        link.fetch_status = "network_error"
        return
    if "__status__" in js:
        link.fetch_status = "not_found" if js["__status__"] == 404 else "http_error"
        return
    data = js.get("data", {}) or {}
    latest = data.get("latestVersion", {}) or {}
    # license -- Dataverse 5+ exposes structured license metadata.
    lic = latest.get("license", {})
    if isinstance(lic, dict):
        link.license = lic.get("name") or lic.get("uri") or None
    elif isinstance(lic, str):
        link.license = lic
    # Title + description live in the citation metadata block.
    cite = (latest.get("metadataBlocks", {}) or {}).get("citation", {})
    for f in cite.get("fields", []) or []:
        name = f.get("typeName")
        val = f.get("value")
        if name == "title" and isinstance(val, str):
            link.title = val
        elif name == "dsDescription" and isinstance(val, list):
            for entry in val:
                d = entry.get("dsDescriptionValue", {}).get("value", "")
                if d:
                    link.description = _strip_html(d)
                    break
    files = latest.get("files", []) or []
    for f in files[:_FILES_MAX]:
        df = f.get("dataFile", {}) or {}
        link.files.append({
            "name": df.get("filename"),
            "size": df.get("filesize"),
            "format": df.get("contentType") or "",
        })
    link.fetch_status = "ok"


def _fetch_figshare(link: DataLink, session: requests.Session, timeout: float) -> None:
    if not link.record_id:
        link.fetch_status = "not_implemented"
        return
    js = _http_get_json(
        f"https://api.figshare.com/v2/articles/{link.record_id}",
        session, timeout)
    _set_fetched(link)
    if js is None:
        link.fetch_status = "network_error"
        return
    if "__status__" in js:
        link.fetch_status = "not_found" if js["__status__"] == 404 else "http_error"
        return
    link.title = js.get("title") or None
    link.description = _strip_html(js.get("description", "")) or None
    link.license = (js.get("license") or {}).get("name") or None
    for f in (js.get("files") or [])[:_FILES_MAX]:
        link.files.append({
            "name": f.get("name"),
            "size": f.get("size"),
            "format": f.get("mimetype") or "",
        })
    link.fetch_status = "ok"


def _fetch_osf(link: DataLink, session: requests.Session, timeout: float) -> None:
    if not link.record_id:
        link.fetch_status = "not_implemented"
        return
    # Try /nodes/, then /registrations/.
    for endpoint in ("nodes", "registrations"):
        js = _http_get_json(
            f"https://api.osf.io/v2/{endpoint}/{link.record_id}/",
            session, timeout)
        if js is not None and "__status__" not in js:
            break
    else:
        _set_fetched(link)
        link.fetch_status = "not_found"
        return
    _set_fetched(link)
    attrs = (js.get("data") or {}).get("attributes") or {}
    link.title = attrs.get("title") or None
    link.description = _strip_html(attrs.get("description") or "") or None
    nl = attrs.get("node_license")
    if isinstance(nl, dict):
        link.license = nl.get("id") or nl.get("name") or None
    # Files require an additional /files/ traversal that's expensive
    # (per-provider listing). Skip in the one-shot fetch; the link/DOI
    # is recorded and users can explore via the URL.
    link.fetch_status = "ok"


def _fetch_pangaea(link: DataLink, session: requests.Session, timeout: float) -> None:
    if not link.record_id:
        link.fetch_status = "not_implemented"
        return
    # PANGAEA's content negotiation: ?format=metadata_jsonld
    js = _http_get_json(
        f"https://doi.pangaea.de/10.1594/PANGAEA.{link.record_id}?format=metadata_jsonld",
        session, timeout)
    _set_fetched(link)
    if js is None:
        link.fetch_status = "network_error"
        return
    if "__status__" in js:
        link.fetch_status = "not_found" if js["__status__"] == 404 else "http_error"
        return
    link.title = js.get("name") or None
    link.description = _strip_html(js.get("description") or "") or None
    lic = js.get("license") or {}
    if isinstance(lic, dict):
        link.license = lic.get("name") or lic.get("@id") or None
    elif isinstance(lic, str):
        link.license = lic
    for d in (js.get("distribution") or [])[:_FILES_MAX]:
        link.files.append({
            "name": d.get("name") or d.get("contentUrl"),
            "size": d.get("contentSize"),
            "format": d.get("encodingFormat") or "",
        })
    link.fetch_status = "ok"


def _fetch_caltech(link: DataLink, session: requests.Session, timeout: float) -> None:
    """CaltechDATA runs Invenio (same family as Zenodo); same API shape."""
    if not link.record_id:
        link.fetch_status = "not_implemented"
        return
    js = _http_get_json(
        f"https://data.caltech.edu/api/records/{link.record_id}",
        session, timeout)
    _set_fetched(link)
    if js is None:
        link.fetch_status = "network_error"
        return
    if "__status__" in js:
        link.fetch_status = "not_found" if js["__status__"] == 404 else "http_error"
        return
    md = js.get("metadata", {}) or {}
    link.title = md.get("title") or None
    desc = md.get("description")
    if isinstance(desc, list) and desc:
        desc = desc[0].get("description") if isinstance(desc[0], dict) else desc[0]
    link.description = _strip_html(desc or "") or None
    rights = md.get("rights") or []
    if rights and isinstance(rights[0], dict):
        link.license = rights[0].get("id") or rights[0].get("title", {}).get("en")
    for f in (js.get("files", {}).get("entries") or [])[:_FILES_MAX]:
        link.files.append({
            "name": f.get("key"),
            "size": f.get("size"),
            "format": f.get("mimetype") or "",
        })
    link.fetch_status = "ok"


# ESS-DIVE, Mendeley Data, ICPSR have public APIs that require
# auth/registration for read access. We record the link but mark
# fetch_status="not_implemented" so users know it's intentional.
def _fetch_unsupported(link: DataLink, session: requests.Session,
                       timeout: float) -> None:
    _set_fetched(link)
    link.fetch_status = "not_implemented"


RECOGNIZERS: list[_Recognizer] = [
    _Recognizer(
        name="zenodo",
        doi_re=re.compile(r"\b10\.5281/zenodo\.(?P<id>\d+)\b", re.I),
        url_res=[re.compile(r"zenodo\.org/(?:records?|deposit)/(?P<id>\d+)", re.I)],
        doi_template="10.5281/zenodo.{id}",
        canonical_url=_zenodo_canonical,
        fetch=_fetch_zenodo,
    ),
    _Recognizer(
        name="dryad",
        doi_re=re.compile(r"\b10\.5061/dryad\.(?P<id>[a-z0-9.]+?)(?=[\s.,;)\]<>\"']|$)", re.I),
        url_res=[
            re.compile(
                r"datadryad\.org/(?:dataset/|stash/dataset/)?doi:10\.5061/dryad\.(?P<id>[a-z0-9.]+)",
                re.I),
        ],
        doi_template="10.5061/dryad.{id}",
        canonical_url=_dryad_canonical,
        fetch=_fetch_dryad,
    ),
    _Recognizer(
        name="harvard-dataverse",
        doi_re=re.compile(r"\b10\.7910/DVN/(?P<id>[A-Z0-9]+)\b"),
        url_res=[
            re.compile(
                r"dataverse\.harvard\.edu/dataset\.xhtml\?persistentId=doi:10\.7910/DVN/(?P<id>[A-Z0-9]+)",
                re.I),
        ],
        doi_template="10.7910/DVN/{id}",
        canonical_url=_harvard_dv_canonical,
        fetch=_fetch_dataverse,
    ),
    _Recognizer(
        name="borealis-dataverse",
        doi_re=re.compile(r"\b10\.5683/SP\d+/(?P<id>[A-Z0-9]+)\b"),
        url_res=[
            re.compile(
                r"borealisdata\.ca/dataset\.xhtml\?persistentId=doi:10\.5683/SP\d+/(?P<id>[A-Z0-9]+)",
                re.I),
        ],
        doi_template="10.5683/SP3/{id}",  # placeholder; canonical from match
        canonical_url=lambda g: f"https://borealisdata.ca/dataset.xhtml?persistentId=doi:{g.get('full_doi','')}",
        fetch=_fetch_dataverse,
    ),
    _Recognizer(
        name="figshare",
        doi_re=re.compile(r"\b10\.6084/m9\.figshare\.(?P<id>\d+)(?:\.v\d+)?\b"),
        url_res=[
            re.compile(r"figshare\.com/(?:articles/(?:[^/]+/)*[^/]+/)(?P<id>\d+)", re.I),
        ],
        doi_template="10.6084/m9.figshare.{id}",
        canonical_url=_figshare_canonical,
        fetch=_fetch_figshare,
    ),
    _Recognizer(
        name="osf",
        doi_re=re.compile(r"\b10\.17605/OSF\.IO/(?P<id>[A-Z0-9]{5,8})\b"),
        url_res=[re.compile(r"osf\.io/(?P<id>[a-z0-9]{5,8})/?", re.I)],
        doi_template="10.17605/OSF.IO/{id}",
        canonical_url=_osf_canonical,
        fetch=_fetch_osf,
    ),
    _Recognizer(
        name="pangaea",
        doi_re=re.compile(r"\b10\.1594/PANGAEA\.(?P<id>\d+)\b", re.I),
        url_res=[
            re.compile(r"(?:doi\.)?pangaea\.de/(?:10\.\d+/)?PANGAEA\.(?P<id>\d+)", re.I),
        ],
        doi_template="10.1594/PANGAEA.{id}",
        canonical_url=_pangaea_canonical,
        fetch=_fetch_pangaea,
    ),
    _Recognizer(
        name="ess-dive",
        doi_re=re.compile(r"\b10\.15485/(?P<id>\d+)\b"),
        url_res=[re.compile(r"data\.ess-dive\.lbl\.gov/view/doi:10\.15485/(?P<id>\d+)", re.I)],
        doi_template="10.15485/{id}",
        canonical_url=_essdive_canonical,
        fetch=_fetch_unsupported,
    ),
    _Recognizer(
        name="mendeley-data",
        doi_re=re.compile(r"\b10\.17632/(?P<id>[a-z0-9]+)\.\d+\b", re.I),
        url_res=[re.compile(r"data\.mendeley\.com/datasets/(?P<id>[a-z0-9]+)", re.I)],
        doi_template="10.17632/{id}",
        canonical_url=_mendeley_canonical,
        fetch=_fetch_unsupported,
    ),
    _Recognizer(
        name="icpsr",
        doi_re=re.compile(r"\b10\.3886/ICPSR(?P<id>\d+)(?:\.v\d+)?\b"),
        url_res=[re.compile(r"icpsr\.umich\.edu/web/[^/]+/studies/(?P<id>\d+)", re.I)],
        doi_template="10.3886/ICPSR{id}",
        canonical_url=_icpsr_canonical,
        fetch=_fetch_unsupported,
    ),
    _Recognizer(
        name="caltechdata",
        doi_re=re.compile(r"\b10\.22002/(?P<id>[a-z0-9.\-]+)\b", re.I),
        url_res=[re.compile(r"data\.caltech\.edu/records?/(?P<id>[a-z0-9\-]+)", re.I)],
        doi_template="10.22002/{id}",
        canonical_url=_caltech_canonical,
        fetch=_fetch_caltech,
    ),
]

# Generic Dataverse catch-all -- runs after the specific Dataverse hosts
# above so Harvard / Borealis / etc. are recognized as themselves first.
_GENERIC_DV_URL_RE = re.compile(
    r"(?P<host>[a-z0-9.\-]*(?:dataverse|borealis)[a-z0-9.\-]*)/dataset\.xhtml\?persistentId=(?P<persistent>doi:10\.\d+/[^\s&\"<>')\]]+)",
    re.I,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _heading_for_offset(md: str, offset: int,
                        spans: list[tuple[int, int, str]]) -> Optional[str]:
    for start, end, text in spans:
        if start <= offset < end:
            return text
    return None


def extract_data_links(md: str) -> list[DataLink]:
    """Scan markdown for known data-repository links. Always-on,
    deterministic, no network. Dedups by canonical DOI; falls back to
    canonical URL when no DOI prefix matched."""
    spans = _section_spans(md)
    by_key: dict[str, DataLink] = {}

    def _record(link: DataLink, offset: int) -> None:
        key = (link.doi or link.url).lower()
        # Promote confidence to "high" if any occurrence sits inside a
        # data-availability section.
        section = _heading_for_offset(md, offset, spans)
        if existing := by_key.get(key):
            if section and not existing.section:
                existing.section = section
                existing.confidence = "high"
            return
        if section:
            link.section = section
            link.confidence = "high"
        by_key[key] = link

    for rec in RECOGNIZERS:
        if rec.doi_re is not None:
            for m in rec.doi_re.finditer(md):
                gid = m.group("id")
                doi = rec.doi_template.format(id=gid)
                # For Borealis the id is post-prefix-only; build full DOI.
                if rec.name == "borealis-dataverse":
                    full = m.group(0)
                    doi = full
                url = rec.canonical_url({"id": gid, "full_doi": doi})
                _record(DataLink(repository=rec.name, url=url, doi=doi,
                                 record_id=gid), m.start())
        for url_re in rec.url_res:
            for m in url_re.finditer(md):
                g = m.groupdict()
                doi = rec.doi_template.format(id=g["id"]) if "id" in g else None
                if rec.name == "borealis-dataverse":
                    # Reconstruct full DOI from the URL form's path.
                    bm = re.search(r"doi:(10\.5683/SP\d+/[A-Z0-9]+)", m.group(0))
                    doi = bm.group(1) if bm else None
                url = rec.canonical_url({"id": g.get("id", ""),
                                         "full_doi": doi or ""})
                _record(DataLink(repository=rec.name, url=url, doi=doi,
                                 record_id=g.get("id")), m.start())

    # Generic Dataverse catch-all (any non-Harvard, non-Borealis install).
    seen_dv_dois = {l.doi for l in by_key.values()
                    if l.repository in ("harvard-dataverse", "borealis-dataverse")
                    and l.doi}
    for m in _GENERIC_DV_URL_RE.finditer(md):
        host = m.group("host").lower()
        persistent = m.group("persistent").rstrip(".,;)>")
        doi = persistent.split("doi:", 1)[-1] if persistent.startswith("doi:") else None
        if doi and doi in seen_dv_dois:
            continue  # already captured by host-specific recognizer
        url = _generic_dv_canonical({"host": host, "persistent": persistent})
        _record(DataLink(repository="dataverse", url=url, doi=doi),
                m.start())

    return sorted(by_key.values(), key=lambda l: (l.repository, l.url))


def fetch_summary(link: DataLink, session: Optional[requests.Session] = None,
                  timeout: float = _DEFAULT_TIMEOUT) -> None:
    """Populate title/description/license/files on `link` in-place by
    calling the repo's public API. No-op on unknown repos. Always
    sets `fetched_at` and `fetch_status`. Never raises."""
    own_session = session is None
    if own_session:
        session = requests.Session()
    try:
        for rec in RECOGNIZERS:
            if rec.name == link.repository and rec.fetch is not None:
                try:
                    rec.fetch(link, session, timeout)
                except Exception as e:  # broad: best-effort contract
                    log.debug("fetch failed for %s %s: %s",
                              link.repository, link.url, e)
                    _set_fetched(link)
                    link.fetch_status = "parse_error"
                return
        # repository="dataverse" (generic catch-all) uses the same fetcher.
        if link.repository == "dataverse":
            try:
                _fetch_dataverse(link, session, timeout)
            except Exception as e:
                log.debug("fetch failed for dataverse %s: %s", link.url, e)
                _set_fetched(link)
                link.fetch_status = "parse_error"
            return
        # Unknown repo -- shouldn't happen since we only call this on
        # links we recognized, but keep the contract.
        link.fetch_status = "not_implemented"
        _set_fetched(link)
    finally:
        if own_session:
            session.close()
