# paper2md vs lit-lake — capability comparison

[lit-lake](https://github.com/ElliotRoe/lit-lake) is a Zotero-Claude
bridge that indexes a personal Zotero library for semantic search.
It's the closest project on the "PDF text → searchable artifact" axis:
two extraction backends, `pymupdf` (default, plain-text) and a Gemini
API path for "higher-quality" extraction. The table below maps
paper2md's per-capability output against both lit-lake backends so
future users can decide which tool fits their workflow.

The two are different *categories* of tool, not direct competitors —
lit-lake's job ends at "text chunks indexed for semantic search";
paper2md's job ends at "structured Markdown artifact with figures,
tables, and citations preserved for downstream reuse." If you need
both, run paper2md to produce the Markdown, then index *that* with
lit-lake.

| Capability | paper2md | lit-lake (PyMuPDF default) | lit-lake (Gemini API) |
|---|---|---|---|
| Two-column reading order | yes (marker / surya layout) | weak — PyMuPDF returns text in xy-block order, not reading order; explicitly noted as needing line-wrap / hyphenation fixes | partial (Gemini sees the page image) |
| Tables → markdown tables | yes (VLM rewrite, dedicated bbox detection via pymupdf / TATR / docling, per-table sidecars, fallback rendering, **multi-page continuation handling** for `(Continued.)` sequences) | no — table cells emerge as ragged paragraphs | depends on Gemini's table handling; no explicit pipeline |
| Figures → cropped images + author captions | yes (extract → caption pair → drop banners; default `dup-detect` strategy validated to 100% on the test corpus) | "embedded images" mentioned, no caption-pairing | no |
| Math (inline + display) | yes ($\LaTeX$ via surya equation recognizer, preserved through marker) | none (PyMuPDF text strip) | depends on Gemini |
| Citations (front-matter) | yes (VLM-synthesized from page 1 + DOI lookup; YAML front-matter) | no — relies on Zotero's own metadata | no |
| References list re-numbering / tidy-up | yes — deterministic four-pass: number back-fill, footnote-reference consolidation, multi-section merge, and in-section tidy (column-break continuation merge + author-address / `(Received…)` extraction) | no | no |
| Article-boundary trim (concatenated PDFs) | yes (1 + log₂N VLM calls) | no | no |
| Sparse-page rescue | yes (re-renders + VLM rewrites pages with broken text layer) | no | no |
| Scanned / old-OCR'd PDF handling | `--auto-force-ocr` per-paper detection (~100-300 ms pre-check) + `--force-ocr` re-OCR via surya. `--scan-detect-only` for batch dry-runs. Recovers usable output from pre-2000 / scanned papers whose publisher OCR poisoned the text layer | no — PyMuPDF text path inherits whatever the embedded text layer says, including scan noise | depends on Gemini's vision; no explicit detection or re-OCR loop |
| Quality scoring + provenance | YAML front-matter with VLM model, per-figure score, per-page char density, full pipeline-toggle snapshot (including per-paper scan-detection signals when `--auto-force-ocr` triggers) | no | no |
| Local-first / copyright-aware | yes (works offline with vLLM/LM Studio; Gemma 4 31B MLX 8-bit hits 100% on the test corpus) | local PyMuPDF path is offline, but plain-text only; Gemini path sends content to Google | sends content to Google |
| Batch mode | yes (folder / glob; supplement auto-pairing; per-paper subdirs; `manifest.jsonl`; `--clean` to wipe per-paper artifacts on re-run; `--auto-force-ocr` per-paper for mixed old/new corpora) | depends on user wrapper around the lit-lake CLI | depends on user wrapper |
| Output format for downstream RAG | Markdown body (chunkable, `## References` consolidated and tidied) + structured asset metadata | Pre-chunked text + embeddings (their use case) | same |

## Predicted accuracy on jacquet.pdf (the project's test paper)

paper2md numbers are measured against ground truth (USAGE.md §13.3.1).
lit-lake numbers are predictions based on the architectural differences
above; happy to run a head-to-head if anyone wants to substitute
measurements for predictions.

| Task | paper2md (gemma-4-31b-it MLX 8-bit) | lit-lake (PyMuPDF) | lit-lake (Gemini) |
|---|---|---|---|
| Plain text recovery (born-digital paper) | ~99% (marker reading-order) | ~85% (xy-blocks; hyphenation issues) | ~95% (Gemini layout) |
| Plain text recovery (pre-2000 scan w/ publisher OCR) | recoverable via `--auto-force-ocr` → surya re-OCR; comparable to born-digital quality after the re-OCR pass | poor — PyMuPDF reads the publisher OCR text layer verbatim, inheriting scan noise (broken table headers, paragraph-shaped tables, spurious bold runs) | partial — Gemini may rebuild from the page image, but no explicit scan-detection loop |
| Figure extraction + caption matching | 100% (validated) | 0% (no figure handling) | partial — Gemini may describe figures inline, no pairing back to extracted JPEGs |
| Table → markdown table | ~95–100% (VLM table rewrite + sidecars) | ~10% (cells as ragged paragraphs) | ~70–85% (depending on Gemini's table reasoning) |
| Math preserved as LaTeX | yes (surya) | no (becomes ASCII) | partial |
| Citation synthesized | yes | no (Zotero metadata) | no |
| Copyright-compliant local processing | yes (Gemma 4 / Qwen3-VL) | yes (PyMuPDF) | no (Gemini API) |

## When each is the right tool

- **Query your Zotero library through Claude / ChatGPT, with citations
  and annotations** → use [zotero-mcp](https://github.com/54yyyu/zotero-mcp)
  (also evaluated; doesn't do PDF-to-Markdown at all, just Zotero
  integration).
- **Semantic search across the text bodies of your Zotero papers** →
  lit-lake. Purpose-built for that. paper2md *can* feed a vector DB but
  doesn't manage Zotero, embeddings, or query interfaces.
- **Faithful, archival, machine-readable Markdown of a specific paper**
  — preserving tables as tables, equations as LaTeX, figures as cropped
  JPEGs with author captions, citations as front-matter — for
  downstream uses that aren't just "search my library" → paper2md.

## Sources

- [paper2md USAGE.md §13.3.1](USAGE.md) — empirical figure-match
  accuracy on the jacquet test paper across 6 VLMs (Gemma 4 31B MLX
  8-bit, Qwen3-VL-32B BF16, Qwen2.5-VL-72B AWQ, OpenAI gpt-4.1,
  Anthropic claude-sonnet-4-6, and the lower-quant Gemma 4 31B
  default).
- [lit-lake README](https://github.com/ElliotRoe/lit-lake) — backend
  description (PyMuPDF + optional Gemini).
- [zotero-mcp README](https://github.com/54yyyu/zotero-mcp) — Zotero
  integration scope; explicitly does not convert PDFs to Markdown.
