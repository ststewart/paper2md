# paper2md vs docling — capability comparison

[docling](https://github.com/docling-project/docling) is IBM
Research Zürich's open-source document converter, now an
LF AI & Data Foundation project. It's the closest peer to
paper2md on the "PDF → markdown for AI / RAG" axis and is itself
a sophisticated multi-stage pipeline (layout → TableFormer →
code/equation/picture classifiers → optional GraniteDocling VLM).

Disclosure: paper2md **uses docling internally** as one of its
table-detector backends (`--table-finder docling`). Docling's
TableFormer is the strongest table-region detector we ship; the
empirical 100% table-detection rate paper2md hits on its
test corpus depends on docling's bbox output. So this comparison
is about *standalone docling* vs *paper2md as a pipeline*, not
"docling tables vs paper2md tables" — the table component is
shared.

The two tools have **different layered scope**:

- **docling** is a general-purpose document converter (PDFs,
  Office, HTML, audio, images, LaTeX, plain text) with a unified
  intermediate representation (`DoclingDocument`) and many export
  formats. Optimized for breadth.
- **paper2md** is a journal-paper-specific pipeline that adds
  VLM-rewrite hooks for tables, figures, citations, and sparse-
  page rescues; copyright/OA metadata resolution; data-repository
  link extraction; reproducibility metadata; and quality scoring.
  Optimized for fidelity on one workload (scientific journal
  articles).

If you're building a multi-format document indexer, **use
docling**. If you're producing archival, citation-ready,
embedding-ready Markdown of journal articles where math is in
LaTeX and figures are paired with author captions, **use
paper2md** (which itself sits on top of docling for the table
phase).

## Capability matrix

| Capability | paper2md (v0.4.0) | docling (v2.92.0) |
|---|---|---|
| License | MIT | MIT |
| Runtime model | GPU + VLM (vLLM / LM Studio / OpenAI / Anthropic); CPU fallback under `--no-vlm` | CPU + GPU + Apple Silicon MLX; default mode is layout-models-only (no LLM); optional GraniteDocling VLM for higher fidelity |
| Wall-time per paper | minutes (LLM-bottlenecked; 5–15 min on a 32B VLM) | seconds (~6 s on canup.pdf default mode); minutes with GraniteDocling |
| Input formats | PDF (with supplement-pairing for journals) | PDF, DOCX, PPTX, XLSX, HTML, images, audio (WAV/MP3/WebVTT), LaTeX, plain text |
| Intermediate representation | none — markdown is the canonical artifact | `DoclingDocument` — a typed structured representation; export to Markdown / HTML / WebVTT / JSON / DocTags / domain XML (USPTO, JATS, XBRL) |
| Output formats | Markdown body + multi-block YAML front-matter, `.meta.json` sidecar, per-table / per-figure assets, optional HDF5 bundle | Markdown / HTML / WebVTT / JSON (lossless) / DocTags / domain-specific XML schemas |
| Two-column reading order | yes (marker / surya layout) | yes (Heron layout model + reading-order resolution) |
| Tables → markdown tables | uses **docling** as the table-finder backend → caption-page bypass with closest-to-caption candidate selection → VLM rewrite of the JPEG crop → per-table sidecar `.md`. Hook 1.5 orphan-caption rescue catches paragraph-rendered tables. **Multi-page continuation handling**: when marker emits N markdown tables for an N-page `(Continued.)` sequence, each is routed to its own page rather than collapsing to the original caption page. 100% on the curated test corpus. | TableFormer model → markdown tables directly. Strong table detection; fewer post-processing hooks (no caption pairing, no VLM rewrite of suspicious cells, no orphan-caption rescue, no continuation-page resolution). |
| Figures → cropped images + author captions | yes (5 caption regexes + `dup-detect` post-pass; non-figure images dropped) | yes — figures classified, optionally exported as separate images; no built-in author-caption pairing |
| Display equations (`$$...$$`) | LaTeX (surya equation recognizer); KaTeX-renderable | placeholder `<!-- formula-not-decoded -->` in default mode; LaTeX rendering possible with GraniteDocling VLM |
| Inline math + sub/superscripts | LaTeX inline + HTML `<sup>` / `<sub>` tags | inline subscripts / superscripts rendered as adjacent space-separated digits / characters (no `<sup>`, no LaTeX); equality signs in equations dropped |
| Citation synthesis (front-matter) | yes (Hook 0: VLM reads page 1 → journal-style citation prepended above H1) | no |
| References list — re-numbering / consolidation / tidy-up | deterministic, four passes: back-fill missing numbers, lift `<sup>N</sup>` footnote-refs, merge multi-section refs (e.g. Nature Methods refs), and `--no-tidy-refs`-toggleable in-section tidy that merges column-break continuations and pulls author addresses / `(Received…)` lines out of the bullet list | no |
| Article-boundary trim (concatenated PDFs) | yes (1 + log₂N VLM calls) | no |
| Sparse-page / broken-text-layer rescue | yes (re-renders + VLM rewrites) | OCR via EasyOCR / Tesseract / RapidOCR; VLM rescue available with GraniteDocling |
| Scanned / old-OCR'd PDF detection | yes — `--auto-force-ocr` runs a cheap (~100-300 ms / paper) PyMuPDF-based pre-check (Producer/Creator metadata against an OCR-tool fingerprint list, image-coverage ratio, font diversity, bold-character ratio) and applies `--force-ocr` only to flagged papers. `--scan-detect-only` does a TSV dry-run survey of a corpus without converting | no automatic scan detection at the document level — relies on the per-page OCR-engine selection (rapidocr / EasyOCR / Tesseract) regardless of whether the PDF is born-digital or scanned |
| Force-OCR override | `--force-ocr` (manual) and `--auto-force-ocr` (per-paper) bypass the embedded text layer entirely so every page goes through surya OCR — recommended on pre-2000 / scanned papers where the publisher's stale OCR poisons the text layer (spurious bold paragraphs, line-broken table headers, mis-located captions) | docling has its own per-page text-vs-OCR decision; no document-level "force OCR" override visible at the converter API level |
| Header / footer stripping | dedicated deterministic strippers per journal style | layout model marks headers/footers; default export keeps them as page furniture |
| Copyright / OA metadata resolution | 6-API resolution (OpenAlex, Unpaywall, EuropePMC, OSTI, arXiv, Crossref) → `safe_to_distribute` tier in front-matter; optional swap to OA copy | none |
| Data-repository link extraction | 11 repos (Zenodo, Dryad, Dataverse, figshare, OSF, PANGAEA, ESS-DIVE, Mendeley, ICPSR, CaltechDATA), optional API enrichment | none |
| Reproducibility metadata in artifact | full: paper2md_version, python_version, backend, VLM endpoint, full toggle snapshot, package versions, started_at, elapsed_sec | none in the standard markdown export; some provenance available on the `DoclingDocument` |
| Quality scoring / grade | A–F grade + per-element sub-scores | none |
| User annotations | `--user` / `--collection` / `--note` → `user:` YAML block | none |
| Batch / supplement pairing | yes (folder/glob + `_SI` regex; per-paper subdirs; manifest.jsonl); `--clean` removes pre-existing per-paper artifacts before re-runs | not first-class — user composes the loop |
| RAG-framework integrations | none built-in (artifacts go to any vector DB; sections chunkable) | native LangChain, LlamaIndex, Crew AI, Haystack integrations + an MCP server |
| Air-gapped / sensitive data | yes (local-first design; no required network for VLM hooks under self-hosted vllm/LM Studio) | yes (local execution for sensitive data and air-gapped environments) |

## Verified on canup.pdf (Nature Geoscience, 6 pages, math-heavy)

Same source PDF, default options for each tool. paper2md run with
`--table-finder docling` against local vllm + Qwen3-VL-32B; docling
v2.92.0 in default (no-VLM) mode; pymupdf4llm 1.27.2.3 included
for reference.

| Signal | paper2md | docling | pymupdf4llm |
|---|---|---|---|
| Output size (chars) | 44,206 | 36,178 | 41,111 |
| Wall time | minutes (VLM-bottlenecked) | ~6 s | seconds |
| Display-math blocks (`$$…$$`) | **8** | 0 (placeholders only — see below) | 0 |
| Inline-math spans (`$…$`) | **155** | 0 | 0 |
| LaTeX commands (`\frac`, `\sigma`, `\rho`, `\alpha`, `\beta`, `\sum`, `\sqrt`, `\approx`) | `\sigma×50, \frac×19, \alpha×11, \approx×12, \beta×4, \sqrt×3, \rho×2, \sum×1` | none | none |
| `<sup>` tags | **121** | 0 | 0 |
| `<sub>` tags | 1 | 0 | 0 |
| Greek letters in body (raw Unicode) | 22 across {σ, γ, Ω} (rest go through as `\sigma`, `\gamma` in math) | **0** — Greek glyphs from equations are dropped along with the equation | 86 across {α, β, γ, κ, π, ρ, σ} |
| Markdown tables (canup has 0 source tables) | 0 ✓ | 0 ✓ | 0 ✓ |

### Side-by-side: equation (3) from the Methods section

The source PDF has the equation `T_c ≈ T_1 (σ_T / 10⁷ g cm⁻²)^α`
with the prose "where T₁ and α are fitting factors".

**paper2md** — full LaTeX:

```latex
$$T_{\rm c} \approx T_{\rm 1} \left( \frac{\sigma_T}{10^7 \, \rm g \, cm^{-2}} \right)^{\alpha} \tag{3}$$

where  $T_1$  and  $\alpha$  are fitting factors, with  $T_1$  = 3,560 K and  $\alpha$  = 0.063 for  $x_c$  = 0.01, ...
```

**docling** — placeholder + flattened prose:

```
<!-- formula-not-decoded -->

where T 1 and are fitting factors, with T 1 D 3,560 K and D 0.063
for x c D 0.01, ...
```

Note three things in the docling output: (1) the equation is
detected (placeholder is emitted) but not decoded; (2) the Greek
`α` is silently dropped from "T₁ and α are fitting factors" — it
becomes "T 1 and are fitting factors"; (3) the `=` sign is mis-
OCR'd as `D` ("T 1 D 3,560 K" should be "T₁ = 3,560 K").

**pymupdf4llm** — equation completely missing; the surrounding
prose flows past unmarked.

This pattern holds across the Methods section: every equation
that paper2md preserves as LaTeX becomes either a `<!--
formula-not-decoded -->` placeholder (docling) or silently
disappears (pymupdf4llm). Greek letters and equation operators
that appear inline in prose (not just in equation environments)
are dropped or mis-OCR'd by docling and partially preserved
(but un-typeset) by pymupdf4llm.

GraniteDocling — docling's optional VLM mode — is expected to
close most of this gap, at a wall-time cost similar to paper2md.
We have not benchmarked it.

## Mixed old / new corpora — scan handling

A common scientific-corpus workload is a mix of pre-2000 papers
(scanned, with stale publisher OCR) and modern born-digital PDFs.
The two tools take different positions:

- **paper2md**: `--auto-force-ocr` runs a CPU-only PyMuPDF detector
  (~100-300 ms / paper) over each paper before marker. Strong
  signals (an OCR-tool fingerprint in `/Producer` or `/Creator`,
  or an image-cover-heavy + low-font-diversity combo) flip
  `--force-ocr` ON for that paper, bypassing the embedded text
  layer and re-running surya OCR on every page. Conservative
  thresholds (false-positive cost = wasted ~30-60 s of OCR on a
  clean paper). `--scan-detect-only` produces a TSV dry-run
  survey for tuning before a full batch.
- **docling**: per-page OCR engine selection happens inside the
  pipeline; no document-level scan-vs-digital classification is
  exposed at the converter API. Mixed corpora work, but the user
  doesn't have an explicit lever to force re-OCR on scans where
  the embedded text layer is poisoned but technically present.

If your corpus has the "1980s scan with publisher OCR" failure
mode (Boslough 1988, Lyzenga 1980/1983, Wackerle 1962 in our
test corpus), `--auto-force-ocr` is a meaningful capability.

## Where they differ in *philosophy*

- **docling** treats the **`DoclingDocument` intermediate
  representation** as the primary artifact. Markdown is just one
  of many exports. The tool is designed to plug into a broader
  document-processing graph (LangChain / LlamaIndex / Haystack
  consumers) where each downstream component can ingest the
  structured document directly.
- **paper2md** treats the **markdown artifact + YAML front-matter
  + per-element sidecars** as the primary deliverable. The
  unstated assumption is that someone will read or re-process
  this markdown six months from now and needs license provenance,
  reproducibility metadata, and a fidelity-graded score in the
  artifact itself — not in a separate metadata layer or an
  intermediate object that needs another tool to render.

Both philosophies are correct for their target workload.
docling's structured representation is the right substrate for a
multi-format enterprise document pipeline. paper2md's flat
markdown-with-rich-front-matter is the right deliverable for a
scientific corpus where each paper is its own archival object.

## When each is the right tool

- **Multi-format enterprise document pipeline** (PDFs + Office +
  HTML + audio transcripts) feeding a LangChain or LlamaIndex
  RAG stack, native MCP integration desired, default mode's
  equation-placeholder behavior is acceptable, GraniteDocling
  available for the math-heavy subset → **docling**.
- **Scientific journal articles** where math must be LaTeX,
  figures must be cropped + matched to author captions, citation
  must be synthesized, license/provenance must be in the
  artifact, and a quality grade per paper is needed → **paper2md**
  (which uses docling under the hood for table detection).
- **Both** in the same pipeline is reasonable: paper2md's
  `--table-finder docling` is the existing precedent. Future
  paths might pull docling's `DoclingDocument` for text + layout
  while keeping paper2md's VLM hooks, citation synthesis, and
  metadata resolution.

The MIT license alignment between the two tools makes this kind
of layered composition cost-free legally — both are permissively
licensed, suitable for closed-source commercial use without
license fees.

## Sources

- [docling on GitHub](https://github.com/docling-project/docling) —
  v2.92.0 (April 2026); LF AI & Data Foundation project; IBM
  Research Zürich origin.
- [paper2md README](../README.md), [USAGE.md](USAGE.md) —
  paper2md feature list, hooks, scoring; `tests/eval_table_extract.py`
  for the empirical 14/14 measurement on the curated table corpus.
- [COMPARISON_LIT_LAKE.md](COMPARISON_LIT_LAKE.md),
  [COMPARISON_PYMUPDF4LLM.md](COMPARISON_PYMUPDF4LLM.md) —
  companion comparisons against the Zotero-indexer family and the
  PyMuPDF-only converter respectively.
