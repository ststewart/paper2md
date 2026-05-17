# paper2md vs pymupdf4llm — capability comparison

[pymupdf4llm](https://pypi.org/project/pymupdf4llm/) is Artifex's
companion library to PyMuPDF for "PDF and other documents → clean,
LLM-ready data, in one line of code". It's the closest project on
the "PDF → markdown for LLM ingestion" axis but a fundamentally
different category of tool: a **single-call pure-CPU library** for
fast bulk RAG ingestion across mixed document formats, vs.
paper2md's **multi-hook pipeline** specifically tuned for scientific
journal articles where preserving tables-as-tables, figures-with-
captions, math-as-LaTeX, citations, and provenance matters more
than throughput.

If you have a corpus of mixed Office docs / EPUBs / arbitrary PDFs
and want them indexed for vector search, pymupdf4llm is the right
tool. If you have scientific journal articles where downstream
consumers need a faithful, archival, machine-readable artifact
with figures and tables preserved as such, paper2md is the right
tool. They compose: paper2md for the high-value journal subset,
pymupdf4llm for the long tail.

## Capability matrix

| Capability | paper2md (v0.4.0) | pymupdf4llm (1.27.2.3) |
|---|---|---|
| Runtime model | GPU + VLM (vLLM / LM Studio / OpenAI / Anthropic); CPU fallback under `--no-vlm` | Pure CPU, no LLM in the loop |
| Wall-time per paper | minutes (LLM-bottlenecked; 5–15 min on a 32B VLM for a dense paper) | seconds |
| Input formats | PDF (with supplement-pairing for journals) | PDF, XPS, EPUB, MOBI, images, Office (Pro license) |
| Output formats | Markdown body + multi-block YAML front-matter, `.meta.json` sidecar, per-table / per-figure assets, optional HDF5 bundle | Markdown / JSON / plain text / page chunks |
| Two-column reading order | yes (marker / surya layout) | yes (PyMuPDF Layout) |
| Tables → markdown tables | 4 detectors (`pymupdf` / `tatr` / `docling` / `all`) → caption-page bypass with closest-to-caption candidate selection → VLM rewrite of the JPEG crop → per-table sidecar `.md`. **Hook 1.5 orphan-caption rescue** catches paragraph-rendered tables. **Multi-page continuation handling** routes the i-th `(Continued.)` slice to the i-th continuation page rather than collapsing all to page 1. 100% detection on the curated test corpus (millot_si.pdf + root-supplemental.pdf, 14 tables across 8 pages each). | PyMuPDF's built-in table detector → GitHub-flavored markdown table; no continuation-page resolution |
| Figures → cropped images + author captions | yes (5 caption regexes covering Nature pipe / Elsevier period / bold-block / plain-span / Extended Data; `dup-detect` post-pass for same-caption duplicates; non-figure images dropped) | extracted and referenced as plain markdown image links; no caption pairing |
| Display equations (`$$...$$`) | LaTeX, via surya equation recognizer; rendered correctly by KaTeX (after the `\label{...}` / `\eqref{...}` cleanup pre-pass) | not preserved — equations come through as broken inline text where the PDF text layer had glyphs |
| Inline math (`$...$`: variables, fractions, sub/superscripts) | LaTeX inside `$...$`, plus HTML `<sup>` / `<sub>` tags for in-prose superscript citations and chemical formulas | not preserved as math; some Unicode subscript/superscript glyphs survive raw, but math semantics are lost |
| Special characters (Greek letters σ α β γ, ± ≈ ∞ ∂ ∇, °, em-dashes) | hybrid: Greek letters inside math go through as LaTeX (`\sigma`, `\alpha`, …); Greek in prose stays as Unicode; ASCII symbols (±, ≈, ∞) preserved verbatim | Unicode characters in the PDF text layer survive as Unicode (often more raw Unicode characters than paper2md, since paper2md converts in-equation Greek to LaTeX commands) |
| Citation synthesis | yes (Hook 0: VLM reads page 1 → journal-style citation with **bold** title and *italic* journal, prepended above the H1) | no |
| References list — re-numbering / consolidation / tidy-up | deterministic, four passes: back-fills missing leading numbers from `<span id="page-X-N">` anchors, lifts `<sup>N</sup>`-prefixed footnote-references into a single `## References`, merges multi-section ref lists (Nature Methods refs split) into one consolidated section at end of document, and a `--no-tidy-refs`-toggleable in-section tidy that merges column-break continuations and pulls author addresses / `(Received…)` lines out of the bullet list | no |
| Article-boundary trim (concatenated PDFs) | yes (1 + log₂N VLM calls; orphan figure assets cleaned up) | no |
| Sparse-page rescue (broken text layer) | yes (re-renders + VLM rewrites) | OCR via Tesseract / RapidOCR (selective; same-pass, no re-rewrite) |
| Scanned / old-OCR'd PDF detection | yes — `--auto-force-ocr` runs a cheap (~100-300 ms / paper) PyMuPDF-based pre-check (Producer/Creator metadata fingerprint, image-coverage ratio, font diversity, bold-character ratio) and applies `--force-ocr` only to flagged papers. `--scan-detect-only` does a TSV dry-run survey for tuning before a full batch | no — PyMuPDF text extraction returns whatever the embedded text layer says; on scanned papers with a poisoned publisher OCR layer, output inherits the noise (spurious bold runs, line-broken table headers, mis-located captions) |
| Force-OCR override | `--force-ocr` (manual) and `--auto-force-ocr` (per-paper) bypass the embedded text layer and re-run surya OCR on every page | no built-in re-OCR; user must run a separate OCR tool first and re-feed the output |
| Header / footer stripping | dedicated deterministic strippers: AGU `<AUTHOR> ET AL. N of M` running footers, Nature `LETTERS` / `**NATURE GEOSCIENCE DOI: ...**` page-headers, lineno-package digit lines, KaTeX-incompatible math labels | "configurable removal of repetitive elements" |
| Copyright / OA metadata | 6-API resolution (OpenAlex, Unpaywall, Europe PMC, OSTI / DOE PAGES, arXiv, Crossref) → `safe_to_distribute` classification in `copyright:` YAML; optional swap to OA copy via `--prefer-oa-source`; gating exit code via `--require-license` | none |
| Data-repository link extraction | 11 repositories (Zenodo, Dryad, Harvard / Borealis / generic Dataverse, figshare, OSF, PANGAEA, ESS-DIVE, Mendeley Data, ICPSR, CaltechDATA); optional API enrichment via `--fetch-data-repos` (one HTTP GET per deposit for title / license / file list) | none |
| Reproducibility metadata in artifact | full `run:` block: paper2md_version, paper2md_license, python_version, hostname, started_at, elapsed_sec, backend, VLM endpoint, full pipeline toggle snapshot, load-bearing package versions | none |
| Quality scoring / grade | A–F grade + per-table / per-figure / per-page sub-scores in `quality:` YAML | none |
| User annotations | `--user` / `--collection` / `--note` (with env-var fallbacks `PAPER2MD_USER` / `PAPER2MD_COLLECTION`) → `user:` YAML block | none |
| Batch mode | yes (folder / glob; supplement auto-pairing via `_SI` regex; per-paper subdirs; `manifest.jsonl`; configurable parallelism via `--workers`); `--clean` removes pre-existing per-paper artifacts before re-runs; `--auto-force-ocr` is per-paper so mixed old/new corpora work without manual segregation | not a built-in concept (one-call library; user composes their own loop) |
| License | MIT | AGPL v3 *or* Artifex Commercial License |
| Hardware required | NVIDIA GPU recommended (DGX Spark, A100, H100, GH200) or Apple Silicon for local VLM; pure-CPU operation works under `--no-vlm` but loses VLM hooks | any CPU |

## Verified on canup.pdf (Nature Geoscience, 6 pages, math-heavy)

Both tools were run on the same source PDF (`examples/canup.pdf`,
~41k chars of body text). pymupdf4llm 1.27.2.3 with default
options; paper2md v0.3.0 with `--table-finder docling` against
local vllm + Qwen3-VL-32B.

| Signal | paper2md | pymupdf4llm |
|---|---|---|
| Display-math blocks (`$$…$$`) | **8** | 0 |
| Inline-math spans (`$…$`) | **155** | 0 |
| LaTeX commands (`\frac`, `\sigma`, `\rho`, `\alpha`, `\beta`, `\sum`) | **\frac × 19, \sigma × 50, \alpha × 11, \beta × 4, \sum × 1** | none |
| `<sup>` tags (citations + chemical formulas) | **121** | 0 |
| `<sub>` tags | 1 | 0 |
| Approx (≈) glyph occurrences | 13 | 7 |
| Greek-letter occurrences in body text | 22 across {σ, γ, Ω} (the rest go through as `\sigma`, `\gamma` etc. inside math) | 86 across {α, β, γ, κ, π, ρ, σ} (raw Unicode, no math context) |

Side-by-side rendering of equation (1) from the Methods section:

**paper2md** — KaTeX-renderable LaTeX:

```latex
$$\sigma_T \approx \left(\frac{\pi}{x_c}\right)^{1/2}
  \frac{\overline{\mu} P_c H}{R T_c}
  \left[ 1 + \left(\frac{C_s T_c}{x_c l} - 2\right)
  \frac{T_c}{T_0} \right]^{-1/2} \tag{1}$$
```

**pymupdf4llm** — the equation does not appear; the surrounding
prose (`"condensation of potassium and more volatile elements…"`)
flows past where the equation should be without any marker that
math content was lost.

The same pattern holds across the document: every fraction,
subscripted variable, and integral that paper2md preserves as
LaTeX is silently dropped or mangled into adjacent prose by
pymupdf4llm's text-layer extraction. For RAG embedding pipelines
this often doesn't matter (the surrounding prose carries enough
signal to retrieve the section); for human-readable archives,
re-typesetting, equation search, or downstream LaTeX/MathML
conversion, it matters a lot.

## Mixed old / new corpora — scan handling

pymupdf4llm has no story for scanned PDFs: it reads the embedded
text layer and emits markdown from whatever's there. On a 1980s
scientific paper that's been OCR'd by the publisher with a
generation-old engine, that means the output inherits all of the
publisher OCR's noise (paragraph-shaped tables from broken
column wrap, spurious bold paragraphs from stale font-weight
metadata, line-broken captions like `TABLE\n2.`). The user has
to detect "this is a scan" themselves, run a separate OCR tool
(e.g. ABBYY, OCRmyPDF), and re-feed the cleaned PDF.

paper2md's `--auto-force-ocr` automates that judgment per paper:
a CPU-only PyMuPDF pre-check (~100-300 ms / paper) inspects
`/Producer` + `/Creator` metadata, image-coverage ratio, font
diversity, and bold-character ratio. Strong signals (an OCR-tool
fingerprint in metadata or an image-cover-heavy + low-font-
diversity combo) flip `--force-ocr` ON for that specific paper,
re-OCR'ing it through surya. Conservative thresholds: false-
positive cost is ~30-60 s of wasted OCR on a clean paper;
false-negative cost is bad output for a real scan.
`--scan-detect-only` does a dry-run TSV survey of a corpus
before a full batch.

For a corpus that's *all* born-digital, `--auto-force-ocr` is a
no-op (pure overhead); skip it. For a corpus with any pre-2000
papers mixed in, it's where paper2md decisively beats
pymupdf4llm on accuracy regardless of wall time.

## Where they differ in *philosophy*

- **paper2md** treats the markdown artifact as the **primary deliverable**, with everything in service of that artifact's accuracy and traceability: the YAML front-matter is part of the output, the per-table / per-figure sidecars are part of the output, the .meta.json is part of the output, and the VLM is a tool that buys higher fidelity at the cost of wall time. The unstated assumption is that someone will read or re-process this markdown six months from now and needs to know exactly how it was made.
- **pymupdf4llm** treats the markdown as **intermediate fuel for an LLM downstream**: get the text content out fast, hand it off to the embedding model or the chat-completion call, and never look at the intermediate markdown again. The unstated assumption is that the LLM will paper over any layout imperfections at query time.

Both assumptions are correct for their target workload. paper2md's quality / reproducibility / provenance machinery is overkill for a 10,000-PDF RAG sweep where each document gets read once by an embedding model. pymupdf4llm's "extract text and move on" model loses the things scientific readers actually care about (tables-as-tables, figures-as-figures-with-captions, equations-as-LaTeX, license attribution).

## When each is the right tool

- **High-volume RAG ingestion across mixed formats** (PDFs + Office docs + EPUB), seconds-per-doc is the constraint, no GPU available, AGPL license cost not a blocker → **pymupdf4llm**.
- **Scientific journal articles** where downstream consumers need tables preserved as actual markdown tables, figures cropped + matched to captions, math as LaTeX, citations synthesized, and a license/provenance trail → **paper2md**.
- **Hybrid corpus** with both → run paper2md on the journal subset (where the per-paper wall-time investment is justified by the artifact quality), pymupdf4llm on the long tail (books, technical reports, Office docs).

The AGPL license on pymupdf4llm is the load-bearing footnote for some users: incorporating it into a closed-source product (or a commercial SaaS) requires Artifex's commercial license, which is paid. paper2md is MIT — free for commercial reuse and redistribution.

## Sources

- [pymupdf4llm on PyPI](https://pypi.org/project/pymupdf4llm/) — feature list, license, version 1.27.2.3 (April 2026).
- [paper2md README](../README.md), [USAGE.md](USAGE.md) — feature list, hooks, scoring; `tests/eval_table_extract.py` for the empirical 14/14 measurement.
- [COMPARISON_LIT_LAKE.md](COMPARISON_LIT_LAKE.md) — companion comparison vs the lit-lake / zotero-mcp Zotero-indexer family.
