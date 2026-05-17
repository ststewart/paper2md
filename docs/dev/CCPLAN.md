# Plan: Copyright & Open-Access Front-End for paper2md

## Context

`paper2md` extracts scientific PDFs to markdown but currently has no awareness of
copyright or licensing. Outputs cannot safely be redistributed without manually
inspecting each source. This feature adds a metadata-resolution front-end that:

1. Identifies each article (DOI, title) from the local PDF.
2. Looks up its **declared license** via public scholarly metadata APIs.
3. Locates an **open-access version** on a public repository if one exists.
4. Optionally **substitutes** the OA PDF as the extraction source, so the
   resulting markdown derives from a redistributable copy.
5. Records the **actual license** (e.g., `CC-BY`, `CC0`, `CC-BY-NC`) and a
   `safe_to_distribute` flag in the markdown YAML front-matter, alongside the
   existing extraction quality score.

The stage runs **before** the GPU-bound marker pipeline so it adds no GPU time
and parallelises freely in batch mode.

---

## 1. Available APIs — Reference

### 1.1 DOI & metadata resolution

| API | Auth | Returns | Notes |
|---|---|---|---|
| **Crossref REST** (`api.crossref.org/works/{doi}`) | None (polite pool: `mailto=` query param) | DOI metadata, `license[]` array (URL + start-date + content-version), title, authors, container, issued date | Authoritative for *registered* journal DOIs. License field is publisher-declared and may be empty. Free, unrestricted reuse of metadata (CC0). |
| **Crossref title search** (`api.crossref.org/works?query.bibliographic=...&query.author=...`) | Same | Ranked candidates with score | Used as fallback when no DOI is in the PDF. Top-1 match is reliable when paired with first-author surname + year filter. |
| **DataCite REST** (`api.datacite.org/dois/{doi}`) | None | DOI metadata + `rightsList[]` (rightsURI, rightsIdentifier like `cc-by-4.0`) | Covers datasets and many preprints/repos that mint DOIs through DataCite (Zenodo, Figshare, some institutional repos). Crossref doesn't have these. |
| **OpenAlex** (`api.openalex.org/works/doi:{doi}` or `/works?search=...`) | None (polite pool: `mailto=`) | Aggregated metadata, `open_access` object with `oa_status` (`gold`/`green`/`bronze`/`hybrid`/`closed`), `oa_url`, `primary_location.license` | Combines Crossref + Unpaywall + DOAJ. Single call gets both license **and** OA URL. CC0 dataset. |
| **Semantic Scholar Graph API** (`api.semanticscholar.org/graph/v1/paper/DOI:{doi}`) | Optional API key (rate-limited without) | Metadata, `openAccessPdf.url`, `externalIds` | Useful as a tertiary fallback. Stricter rate limits without key. |

### 1.2 Open-access PDF location

| API | Auth | Returns | Notes |
|---|---|---|---|
| **Unpaywall** (`api.unpaywall.org/v2/{doi}?email=...`) | Email required (free) | `is_oa`, `best_oa_location` (with `url_for_pdf`, `license`, `version`, `host_type`, `repository_institution`), `oa_locations[]` | Industry standard for OA discovery. Indexes 50K+ repos. License field uses normalised slugs (`cc-by`, `cc0`, `cc-by-nc`, `public-domain`, etc.). |
| **OpenAlex** | None | Same OA fields embedded in work record | One call replaces both Crossref and Unpaywall in many cases. |
| **Europe PMC** (`europepmc.org/webservices/rest/search?query=DOI:{doi}&resultType=core&format=json`) | None | `pmcid`, `isOpenAccess`, `license`, full-text XML link for OA subset | Best for life-sciences and bioRxiv/medRxiv preprints (ingests license from preprint XML after 14 days). |
| **arXiv** (`export.arxiv.org/api/query?search_query=ti:"..."` or `id:{arxiv_id}`) | None | Atom feed; per-paper license in OAI-PMH metadata (`<arxiv:license>`) | Covers physics/CS/math/stats preprints. arXiv metadata is CC0; per-paper license varies (default arXiv non-exclusive license ≠ redistributable; CC-BY is opt-in). |
| **bioRxiv / medRxiv API** (`api.biorxiv.org/details/{server}/{doi}`) | None | Version history, license string | Direct preprint server API; useful when Europe PMC hasn't yet ingested. |

### 1.3 Recommended stack

For paper2md's mixed corpus (journal articles + preprints), the recommended call
order minimises requests while maximising coverage:

1. **Extract DOI** from PDF (regex on first 2 pages + fitz `doc.metadata` dict).
2. **Single OpenAlex call** by DOI → gets license, OA status, OA URL in one
   round trip. Handles ~95% of journal articles.
3. **Unpaywall fallback** when OpenAlex lacks a license but reports OA (it
   sometimes has fresher license slugs).
4. **Europe PMC** when OpenAlex says `closed` and the work is biomedical (heuristic:
   PMID present, journal is life-sci) — catches author-deposited PMC copies.
5. **arXiv direct** when DOI is missing but PDF text matches `arXiv:NNNN.NNNNN`.
6. **Crossref title search** when no DOI/arXiv ID at all — extract title (largest
   font block on page 1) + first-author surname → top-1 match if score > threshold.

All APIs are free with a contact email (polite pool); no paid tier needed.

---

## 2. License Taxonomy → `safe_to_distribute`

Map normalised license slugs to a three-tier flag:

| Tier | `safe_to_distribute` | Licenses |
|---|---|---|
| **Green** | `true` | `cc0`, `public-domain`, `cc-by`, `cc-by-sa`, `cc-by-4.0`, `cc-by-3.0`, US-government-work |
| **Yellow** | `restricted` | `cc-by-nc`, `cc-by-nd`, `cc-by-nc-sa`, `cc-by-nc-nd` (redistribute OK, but commercial / derivative limits apply — caller must respect) |
| **Red** | `false` | `all-rights-reserved`, publisher-specific (`elsevier-tdm`, `acs-specific`), unknown / not found, arXiv-default (non-exclusive license to distribute, NOT a redistribution grant) |

The full normalised slug (e.g., `cc-by-nc-4.0`) is preserved in the
`copyright.license` field so downstream consumers can apply finer policy.

The flag is **conservative**: anything unrecognised is `false`. A separate
`copyright.confidence` field (`high`/`medium`/`low`) records whether the license
came from a DOI lookup (high), title fallback (medium), or PDF-text heuristic
(low).

---

## 3. Architecture & Insertion Point

New module **`paper2md/src/metadata_frontend.py`** (sibling of `src/paper2md.py`).

Wired into `src/paper2md.py` as a pre-pass in `_do_convert()` (around line 2212),
**before** `configure_client()` and `run_marker()`. Runs without GPU, returns a
typed `ArticleMetadata` dataclass. If `--prefer-oa-source` is set and the
resolver finds a green/yellow OA PDF, the local `pdf_path` is swapped for the
downloaded OA PDF before marker runs. Provenance is preserved in the dataclass.

```
parse args
  ↓
metadata_frontend.resolve(pdf_path) → ArticleMetadata
  ↓                                     ├─ doi, title, authors
  ├─ if --prefer-oa-source and          ├─ license, license_url, license_source
  │  metadata.oa_pdf_safe:              ├─ safe_to_distribute (true/restricted/false)
  │   pdf_path = download(oa_pdf)       ├─ confidence (high/medium/low)
  │                                     ├─ oa_status (gold/green/bronze/hybrid/closed)
  ↓                                     ├─ oa_pdf_url, oa_pdf_used (bool)
configure_client()                      └─ provenance (which APIs were called, latencies)
  ↓
run_marker() ... (rest of pipeline unchanged)
  ↓
QualityReport.to_frontmatter() — extended to embed `copyright:` block alongside `quality:`
```

### Why a separate module
- **No GPU dependency** — can run in pure-CPU pre-pass and parallelise freely
  in batch mode (no marker GPU lock).
- **Independently testable** — pure function: `Path → ArticleMetadata`.
- **Reusable** — Mac variant (`paper2md-mac`) can import the same module.

---

## 4. Files to Modify / Create

| File | Change |
|---|---|
| `paper2md/src/metadata_frontend.py` | **NEW.** Public surface: `resolve(pdf_path: Path, *, allow_network: bool = True, prefer_oa: bool = False, timeout_s: float = 10.0) -> ArticleMetadata`. Internals: `_extract_doi_from_pdf()`, `_query_openalex()`, `_query_unpaywall()`, `_query_europepmc()`, `_query_arxiv()`, `_query_crossref_by_title()`, `_normalise_license()`, `_classify_safety()`, `_download_oa_pdf()`. Uses `requests` (already a transitive dep) with retry + caching. |
| `paper2md/src/paper2md.py` | **EDIT** at ~line 2212 (`_do_convert`): call `metadata_frontend.resolve()`; pass result through to `convert()`. **EDIT** `convert()` (line 1815): accept `metadata: ArticleMetadata` kwarg, optionally swap `pdf_path` to OA download, pass into `QualityReport`. **EDIT** `QualityReport` (line 1209): add `metadata: Optional[ArticleMetadata]` field; extend `to_frontmatter()` (line 1240) to emit `copyright:` block. **EDIT** argparse (line 2053): add `--prefer-oa-source`, `--no-metadata-lookup`, `--metadata-timeout`, `--require-license`. **EDIT** batch manifest writer (line ~2006): include `doi`, `license`, `safe_to_distribute`, `oa_pdf_used` per record. |
| `paper2md/USAGE.md` | **EDIT** lines 16–36: insert new "Stage 0: Metadata resolution" in pipeline diagram. Document the new flags and front-matter block. |
| `paper2md/memory.md` | **EDIT** (decisions section, ~line 115): record API stack choice and OA-substitution policy. |
| `paper2md/tests/test_metadata_frontend.py` | **NEW.** Unit tests: DOI regex on synthetic page-1 text fixtures; license-slug normalisation table; safety classifier; `requests`-mocked API responses for OpenAlex/Unpaywall/Europe PMC/Crossref-title each. One end-to-end test using a known-CC-BY PDF (e.g., a PLOS One sample committed under `tests/fixtures/`). |

### Key utilities to reuse from existing code
- `src/paper2md.py:1632 extract_citation()` — already pulls citation text from page 1
  via VLM. The metadata frontend should NOT duplicate this; instead, the new
  `_extract_doi_from_pdf()` runs purely textually (PyMuPDF text + regex) and
  is fast/CPU-only. The VLM citation hook stays as a separate (richer) layer
  and can read the resolved DOI when present to anchor its output.
- `src/paper2md.py:_PROVIDER_DEFAULT_MODEL` (line 58) — pattern for env-var-driven
  config; mirror it for `_API_DEFAULTS = {"crossref_mailto": ..., "unpaywall_email": ...}`.
- `QualityReport.to_frontmatter()` (line 1240) — extend the same YAML emitter
  rather than introducing a second serialisation path.

---

## 5. CLI & Environment Surface

### New flags
- `--prefer-oa-source` — if a green/yellow OA PDF is found, download and extract
  it instead of the local file. Default off.
- `--no-metadata-lookup` — skip the metadata stage entirely (offline mode).
- `--metadata-timeout SEC` — per-API timeout (default 10).
- `--require-license` — exit non-zero if `safe_to_distribute` is `false`.
- `--metadata-cache PATH` — JSON cache of DOI→metadata lookups (default
  `~/.cache/paper2md/metadata.json`).

### New env vars (polite pool)
- `CROSSREF_MAILTO` — e.g. `api.partake757@passfwd.com`. Required if making >1 req/s.
- `UNPAYWALL_EMAIL` — required by Unpaywall API.
- `OPENALEX_MAILTO` — recommended for higher rate limit.
- `S2_API_KEY` — optional, only if Semantic Scholar fallback enabled.

If env vars are missing, the resolver falls back to anonymous access with
1-req/s self-throttling and warns once.

---

## 6. Front-Matter Output

Extend the existing YAML block (currently `quality:` only) with a peer
`copyright:` block:

```yaml
---
copyright:
  doi: 10.1371/journal.pone.0123456
  license: cc-by-4.0
  license_url: https://creativecommons.org/licenses/by/4.0/
  safe_to_distribute: true            # true | restricted | false
  confidence: high                    # high | medium | low
  oa_status: gold                     # gold | green | bronze | hybrid | closed | unknown
  oa_pdf_used: true                   # true if OA copy was substituted
  oa_pdf_source: https://...          # repository URL when oa_pdf_used
  resolved_via: openalex              # openalex | unpaywall | europepmc | arxiv | crossref-title | pdf-text
quality:
  overall: 0.97
  grade: A
  ...
---
```

---

## 7. Failure Modes (best-effort, never blocks)

| Condition | Behavior |
|---|---|
| No network | `confidence=low`, `license=unknown`, `safe_to_distribute=false`. Pipeline continues. Warning logged once. |
| DOI not found in PDF | Try title-based Crossref fallback; mark `confidence=medium` if matched. |
| All APIs return 404 / no license | `license=unknown`, `safe_to_distribute=false`. |
| API timeout | Skip that source, try next. Record latency in provenance. |
| OA PDF download fails | Fall back to local PDF, set `oa_pdf_used=false`, keep license metadata. |
| `--require-license` set and resolution fails | Exit code 3 (distinct from quality-threshold exit 2). |

---

## 8. Verification

End-to-end checks the user can run after implementation:

1. **Unit tests** — `cd paper2md && python -m pytest tests/test_metadata_frontend.py`
   covering DOI regex, license normalisation, safety tiers, mocked API responses.
2. **Known-CC-BY paper** — run on a PLOS One PDF with embedded DOI:
   ```
   python src/paper2md.py path/to/plos.pdf -o out/ --no-vlm
   grep -A8 '^copyright:' out/plos/plos.md
   ```
   Expect `license: cc-by-4.0`, `safe_to_distribute: true`, `confidence: high`.
3. **Closed-access paper** — run on a paywalled Elsevier PDF:
   ```
   python src/paper2md.py path/to/closed.pdf -o out/ --no-vlm
   ```
   Expect `safe_to_distribute: false`, `oa_status: closed` (or `green` if a
   preprint exists).
4. **OA substitution** — same closed-access paper, `--prefer-oa-source`:
   ```
   python src/paper2md.py path/to/closed.pdf -o out/ --no-vlm --prefer-oa-source
   ```
   Expect `oa_pdf_used: true`, license from the OA copy, extraction text
   matching the OA version (compare page count / first-paragraph hash).
5. **No-DOI fallback** — strip DOI from a test PDF (or use one without one),
   verify title-based resolution sets `confidence: medium`.
6. **Offline mode** — disable network, run with default flags; expect
   `confidence: low`, `license: unknown`, but **the pipeline still completes**.
7. **Batch manifest** — run `--batch` over a folder of mixed papers; inspect
   `manifest.jsonl` for per-record `doi`, `license`, `safe_to_distribute`,
   `oa_pdf_used` columns.
8. **Quality block intact** — confirm existing `quality:` YAML block is
   unchanged in shape and present alongside the new `copyright:` block (no
   regression for downstream consumers).

---

## 9. Out of Scope (deferred)

- Full-text license-clause extraction (parsing the rights statement *inside*
  the PDF text). This is brittle and the API approach is more reliable.
- License negotiation / TDM agreement handling (Elsevier TDM, Wiley TDM API).
  These need institutional credentials and add complexity; flag as future work.
- Persisted, queryable metadata DB. The JSON cache covers re-runs; a real DB
  is overkill for the current corpus size.
- Auto-emailing authors for permission. Out of scope for an extraction tool.
