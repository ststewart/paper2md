# Alternatives for science-paper PDF → Markdown without CUDA

paper2md is most effective on a CUDA GPU running a local VLM (vLLM /
LM Studio / OpenAI / Anthropic). Many researchers don't have that
hardware. This document collects the realistic alternatives for a
science-paper PDF → Markdown workflow on Apple Silicon, plain CPU,
or cloud — and where paper2md itself fits in that picture.

The comparison axes that matter for journal articles are different
from generic document conversion: equations as LaTeX, tables as
markdown tables, figures cropped and paired with author captions,
references consolidated into a navigable section, and provenance
recorded in the artifact. Tools that were designed for "any document
to text for an LLM" generally lose some or all of these.

If you have *any* local compute (Apple Silicon, AMD, or even just a
modern CPU), the recommendation is: **start with paper2md in CPU
mode**, see what the deterministic hooks recover, and only reach for
the alternatives below when paper2md isn't an option (no Python
environment, no time to set it up, or a need for a single-call
library inside a larger application).

---

## 0. paper2md without CUDA

Even without a GPU, paper2md retains most of its value-add:

```bash
python src/paper2md.py paper.pdf -o out/ --backend cpu --no-vlm
```

What still runs on CPU:

- **marker** spine — born-digital body extraction with two-column
  reading order, paragraph reflow, and surya OCR fallback for image-
  region pages. ~3× slower than CUDA.
- **All deterministic body-cleanup passes**: footnote-reference
  consolidation, multi-section reference merge, in-section ref tidy
  (column-break continuations, author-address extraction),
  publisher-stamp stripping, line-break-pulled-out-of-tables
  recovery, math-label cleanup.
- **Auto-OCR detection** (`--auto-force-ocr`) — pre-2000 / scanned
  papers get re-OCR'd through surya regardless.
- **Orphan-figure rescue** — captions whose figure marker missed
  get a synthetic image cropped from the PDF page (column-aware,
  text-density heuristic).
- **Journal-aware reference rescue** (`--use-journal-rescue`) —
  per-publisher recovery passes for known failure modes (APS bleed,
  Wiley/AGU `**Figure N.**` missing-heading, PRB superscript-style
  numbered-list mash, Crossref/OpenAlex API fallback).
- **Hook 2 page-locality validation** for figure-caption matching.
- **All metadata resolution** (Crossref, OpenAlex, Unpaywall, etc.)
  — pure HTTP, no GPU needed.
- **Reproducibility metadata, quality scoring, batch mode**.

What you lose under `--no-vlm`:

- Per-table VLM rewrite of detected table crops (you keep the
  text-finder's markdown table; you lose the higher-fidelity
  re-extraction). Table sidecars are also skipped.
- VLM-based figure-caption matching (Hook 2 falls back to dropping
  unmatched images deterministically).
- Citation synthesis from page 1 (Hook 0).
- VLM rescue of sparse-text pages.

Apple Silicon — `--backend mps` is **opt-in only** because surya's
MPS path produces silently corrupted body text on PyTorch 2.x (see
CLAUDE.md §6.0). `--backend cpu` is the safe default on Apple
Silicon and what `--backend auto` resolves to there.

If your alternative is "no PDF→markdown tool at all", paper2md
under `--no-vlm` is still the strongest journal-paper-specific
option this document evaluates. The detailed alternatives below
are for cases where paper2md's setup cost (conda environment,
~10 GB of marker/surya weights, optional VLM) isn't justified or
isn't possible.

---

## 1. Detailed alternatives (Tier 1)

These three tools are the realistic peers on the "scientific PDF →
structured markdown without VLM" axis.

### MinerU (opendatalab/MinerU)

[MinerU](https://github.com/opendatalab/MinerU) is the strongest
academic-paper-aware open-source pipeline we tested.
**Full per-capability comparison + 4-paper benchmark in
[COMPARISON_MINERU.md](COMPARISON_MINERU.md).** Quick summary
here:

- **License**: AGPL-3.0.
- **Provenance**: Shanghai AI Lab (opendatalab). For users /
  institutions where Chinese-origin software is a sensitivity,
  this rules MinerU out.
- **Two backends**: `pipeline` (no LLM, fast, weaker on tables /
  refs / captions); `hybrid-auto-engine` (2.3 GB local VLM,
  slower, better tables, includes speculative chart-data
  reconstruction).
- **Wall time on M5 Apple Silicon (128 GB)**: pipeline 28 s –
  1:29 per paper; hybrid 3:07 – 16:57 per paper.
- **Apple Silicon caveat**: MinerU 3.1.6's `get_vram()` has no
  MPS path — fall through to a 1 GB default. With
  `MINERU_VIRTUAL_VRAM_SIZE` ≥8 the run hangs at "Predict: 0/N".
  `MINERU_VIRTUAL_VRAM_SIZE=1` (forcing batch_size=1) is the
  only working configuration today. Worth filing upstream.

What it does well: equation detection breadth, table extraction
to HTML cells, multi-platform CPU/MPS/CUDA support.

Where it falls short on the science-paper workload: no figure
caption pairing (every `![](...)` has empty alt-text), no
citation synthesis, no copyright/OA metadata, no data-repo
links, no reproducibility metadata, no quality scoring,
**reference handling is uneven** (recovers refs on canup ✓,
partial on cuk, no-space-formatted on Knudson, **completely
missing on Wackerle1962** in both backends).

When MinerU is the right choice: you want a single-binary,
CPU-friendly pipeline that handles math reasonably well, you
don't need citation/copyright/reproducibility metadata, and
AGPL + Chinese provenance are both acceptable for your use.

### docling (IBM)

See [COMPARISON_DOCLING.md](COMPARISON_DOCLING.md) for the full
analysis. Quick recap: fast on CPU (~6 s on canup.pdf), strong
table extraction (paper2md uses docling internally as a
table-finder backend), MIT license. **Equations default to
`<!-- formula-not-decoded -->` placeholders** unless you opt into
GraniteDocling's VLM mode (which costs roughly the same wall-time
as paper2md). Greek letters in inline prose are sometimes dropped,
equality signs occasionally mis-OCR'd as letters. **Best choice
for high-volume corpora where math fidelity is secondary** to
breadth, structure, and integration with LangChain / LlamaIndex /
MCP frameworks.

### marker standalone

Marker is the spine paper2md wraps. Run on its own (without
paper2md's hooks) it's a strong general-purpose PDF→markdown
converter with two-column reading order, surya OCR, and equation
recognition. Apple Silicon support exists but the surya MPS path
has the silent-corruption bug (CLAUDE.md §6.0); recommend
`--device cpu` on macOS. License: GPL-3.0.

What you get vs paper2md `--no-vlm`:

- The same body extraction quality (both use marker)
- Less polished output: no reference consolidation, no orphan
  figure rescue, no journal-aware ref rescue, no copyright
  metadata, no citation synthesis, no data-repo links.

When marker standalone is the right choice:

- You want **just the converter** with no extra opinions about
  what the markdown should look like.
- Your downstream pipeline already handles refs / metadata / etc.

---

## 2. Cloud options (Tier 2 — no local hardware needed)

These are paid services accessed via HTTP API. We have not run
them against our test corpus; descriptions below are based on the
vendors' documentation and community reports.

- **Mathpix** ([mathpix.com](https://mathpix.com)) — best-in-class
  for **mathematical equations and complex notation**. Subscription
  or per-page (~$0.05/page). Output: Markdown / LaTeX / MathML /
  DOCX. The right choice when math fidelity is the constraint and
  the cost is acceptable. No copyright/citation/figure-caption
  pairing in the output.
- **LlamaParse** ([cloud.llamaindex.ai](https://cloud.llamaindex.ai))
  — RAG-focused; integrated with LlamaIndex / LangChain. Free tier
  ~1000 pages/day, paid above. Decent table handling, math
  unevenly preserved. The right choice if your downstream is
  already a LlamaIndex stack.
- **Mistral OCR API** ([mistral.ai](https://mistral.ai/news/mistral-ocr/))
  — released 2025. Strong layout + math claims. Paid. Modern
  alternative to Mathpix; we haven't independently verified math
  fidelity.

**Privacy note:** all three require uploading the PDF to the
vendor's servers. For closed-access journal articles or
unpublished work, this may be an issue.

---

## 3. Light extractors (Tier 3 — not science-paper-specific)

These tools work fine for plain prose but lose math, tables, and
figure structure in journal articles. Listed here so users
considering them can see the tradeoffs explicitly.

- **pymupdf4llm** — see [COMPARISON_PYMUPDF4LLM.md](COMPARISON_PYMUPDF4LLM.md)
  for the detailed comparison. Pure CPU, sub-second per paper, no
  math, no tables-as-markdown-tables, no figure pairing. AGPL or
  commercial license. Right choice for high-volume mixed-document
  RAG ingestion where setup speed beats per-document fidelity.
- **markitdown** ([microsoft/markitdown](https://github.com/microsoft/markitdown))
  — Microsoft's multi-format wrapper. PDF backend is pdfminer.six,
  so on PDFs you inherit pdfminer's text-only output: no math, no
  tables-as-tables, no figure pairing. Also handles Word, Excel,
  PowerPoint, HTML, audio, images. MIT. Right choice for a
  *mixed-format* corpus where science PDFs are a minority.
- **pdfminer.six** ([pdfminer/pdfminer.six](https://github.com/pdfminer/pdfminer.six))
  — Low-level pure-Python text + layout extraction. No structured
  output (raw text + bounding boxes). MIT. Right choice as a
  building block for higher-level tools, not as an end-user
  converter.
- **textract** ([deanmalmgren/textract](https://github.com/deanmalmgren/textract))
  — Wrapper around system tools (pdftotext, antiword, libreoffice).
  Plain text output across many formats. Largely unmaintained
  since ~2018. MIT. Right choice for **legacy** "extract text from
  this random file" workflows; not recommended for new projects.

---

## 4. Specialty mentions

- **GROBID** ([kermitt2/grobid](https://github.com/kermitt2/grobid))
  — Java server, TEI-XML output (not markdown). Excellent for
  **bibliographic metadata + reference list extraction** on academic
  PDFs. Apache-2.0. Right choice when reference structure is the
  primary deliverable (e.g., building a citation graph).
- **Nougat** ([facebookresearch/nougat](https://github.com/facebookresearch/nougat))
  — Meta's transformer-based academic-paper converter. MPS support
  exists. Last meaningful release 2023; quality has been surpassed
  by docling and MinerU on most papers. MIT.
- **Allen AI science-parse** — older Java tool; outputs JSON (not
  markdown). Limited math support. Mostly superseded by GROBID
  and docling.

---

## 5. Future plans for paper2md

The current paper2md tree (v0.4.0) is local-source-only.
Planned next steps:

- **Apple Silicon MPS fix → public pip module.** paper2md's
  `--backend mps` is gated behind a warning because surya's
  PyTorch MPS kernels silently corrupt body text on Apple Silicon
  (CLAUDE.md §6.0; upstream issue
  [datalab-to/marker#993](https://github.com/datalab-to/marker/issues/993)).
  Once that's resolved upstream, paper2md is a candidate for a
  proper PyPI package (`pip install paper2md`). Until then,
  installation is git-clone + conda environment, and `--backend cpu`
  is the safe path on Macs.
- **Zotero → MCP pipeline.** A development path is to wrap
  paper2md as an MCP server so a Zotero library can be ingested
  end-to-end into a structured markdown corpus accessible to
  Claude / ChatGPT / other MCP clients. Sketch: a Zotero plugin
  watches the library, hands new PDFs to paper2md, and exposes
  the resulting markdown + metadata through an MCP server. The
  MCP server may be how a "local" paper2md module first ships
  (one binary, drops into a user's Zotero workflow), independent
  of the broader pip-package timeline.
- **Same-page multi-figure disambiguation.** Hook 2 currently
  can't tell which of two captions on the same PDF page goes with
  which marker-extracted image. Position-pairing or recovery from
  Hook 2's dropped-image pool is sketched but not yet implemented.

If you'd like to follow or contribute, the public anchor is the
v0.4.0 release; the dev tree (this repo) tracks unreleased work
and will drop into a versioned release when the Mac story is
sorted.

---

## Sources

- This doc's MinerU numbers: empirical run on `examples/canup.pdf`
  using `mineru` v3.1.6 in `pipeline` backend (CPU + opportunistic
  GPU on the test machine; CPU-only would be ~5–10× slower).
- [COMPARISON_DOCLING.md](COMPARISON_DOCLING.md) — detailed
  paper2md vs docling head-to-head on canup.pdf, including
  GraniteDocling notes.
- [COMPARISON_PYMUPDF4LLM.md](COMPARISON_PYMUPDF4LLM.md) —
  detailed paper2md vs pymupdf4llm.
- [COMPARISON_LIT_LAKE.md](COMPARISON_LIT_LAKE.md) — paper2md vs
  the Zotero-indexer family (lit-lake, zotero-mcp).
- [paper2md README](../README.md) and
  [USAGE.md](USAGE.md) — feature list, hooks, scoring, full flag
  reference.
