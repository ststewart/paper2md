# paper2md

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20263035.svg)](https://doi.org/10.5281/zenodo.20263035) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PDF to Markdown converter with separate extraction of figures and tables. Tuned for scientific journal articles. Designed for fully offline extraction; runs on Apple Silicon (LM Studio + MLX) or NVIDIA (vLLM + Qwen3-VL-32B-Instruct).

This is not a fast converter. This workflow is designed for high accuracy, reproducibility, traceability, privacy, copyright compliance, and standardization of output formats.

## Why convert from the PDF, when LaTeX or HTML exist?

Both alternatives are appealing on the surface and frustrating in practice.

**LaTeX is a programming language, not a markup format.** Converting `\section{...}` is easy; converting AASTeX's `\deluxetable` or REVTeX's `\altaffilmark` requires reading the journal's `.cls` file, which pandoc and most general converters do not. Astronomy and physics macros (`\plotone`, custom unit definitions, `\citet`/`\citep` resolved against a separate `.bib`, `\includegraphics{file.pdf}` referencing assets that ship in the source tarball but not the published PDF) all depend on context outside the source file. For an AASTeX-heavy corpus, LaTeX-to-Markdown loses the things scientific readers actually care about: tables, figures, captions, math semantics.

**HTML is publisher-specific and often paywalled.** Each publisher (Elsevier, Wiley, Springer, Nature, AAS, PMC) ships its own DOM. PMC's is XML-clean; Elsevier embeds refhub cross-reference link wrappers and column-reflow-split DOI URLs; AAS uses custom MathJax-rendered fragments. There is no shared standard for equations (MathML, MathJax `<img>` tags, inline LaTeX, and Unicode glyphs all coexist), nor for table footnote linkages. Figures point at CDN URLs that often require institutional cookies. And HTML versions are paywalled for many papers when the user already has access to the PDF — a preprint, an author copy, an interlibrary loan, a public repository deposit.

**PDF is layout-first, but at least it is the canonical artifact.** The same reader can open a Nature paper (single column, side-bar figures), a Science article (two columns, margin notes), a Physical Review Letter (two columns, dense math), an eLife article (single column, sidebar), and a 1962 J. Appl. Phys. scan with Adobe Paper Capture OCR. Each layout requires its own heuristic: column reading order, table detection (ruled vs borderless vs inferred-from-spacing), figure-caption matching (above vs below vs Nature-pipe vs Elsevier-period), reference style (numbered vs author-year vs `<sup>`-prefixed footnote), math-glyph recognition. paper2md exists because every general-purpose conversion tool that worked on one journal failed on the next.

## Main challenges this pipeline targets

* Two-column reading order
* Tables without box rules
* Tables too small for layout heuristics to recognize
* Equation-heavy and LaTeX-heavy papers
* Author-line-numbered preprint manuscripts (`lineno` package artifacts)
* Mixed reference styles (numbered, author-year, footnote) in one corpus
* Scanned PDFs with Adobe Paper Capture-vintage OCR
* Author-deposited PMC manuscripts that are readable but not redistributable

The workflow combines a Python page-layout pass (Marker + PyMuPDF + Docling/TATR) with a vision LLM that re-transcribes located tables, captions matched figures, and rescues sparse pages. HDF5 output (`--hdf5`) bundles main markdown, optional supplement, and every asset into one self-contained file for downstream database aggregation and single-file portability.

## Quick start

### Apple Silicon

```bash
conda env create -f environment-mac.yml && conda activate paper2md
# Run LM Studio with a vision model loaded (default: qwen3-vl-32b-instruct-mlx)
python src/paper2md.py paper.pdf -o outputdir
```

### NVIDIA / CUDA

```bash
conda env create -f environment-gpu.yml && conda activate paper2md
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000 \
    --max-model-len 32768 --gpu-memory-utilization 0.80 &
python src/paper2md.py paper.pdf -o outputdir
```

Typical command line, with the supplement and an HDF5 bundle:

```bash
python src/paper2md.py paper.pdf --supplement paper_SI.pdf --hdf5 -o outputdir
```

The default `--table-finder docling` covers the hardest table cases (scanned, borderless, multi-row headers). For high-fidelity tables (subscripts, Greek, footnote markers, dense math), opt INTO the per-table VLM rewrite with `--vlm-tables --table-workers 4`. `--table-finder both` adds TATR on top of PyMuPDF for borderless tables in vector-text PDFs. Full flag reference: `python src/paper2md.py --help` or `docs/USAGE.md`.

## Project layout

```
paper2md/
├── README.md            this file
├── CLAUDE.md            project brief auto-loaded by Claude Code
├── environment-mac.yml  conda env, Apple Silicon (load-bearing pins; see docs/USAGE.md §6.0)
├── environment-gpu.yml  conda env, NVIDIA / CUDA 13
├── src/                 pipeline code (entry: src/paper2md.py)
├── docs/                detailed reference documentation
├── tests/               pytest suite, no GPU/network required (791 tests)
└── examples/            real journal PDFs used for end-to-end testing
```

## Detailed documentation

| Document | Read this when |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | learning the CLI, flag reference, hook-by-hook behavior, scoring, HDF5 schema, troubleshooting |
| [`docs/BATCH.md`](docs/BATCH.md) | running unattended on hundreds–thousands of PDFs |
| [`docs/design/FLOWCHART.md`](docs/design/FLOWCHART.md) | reading a stage-by-stage pipeline diagram |
| [`docs/design/CCPLAN.md`](docs/design/CCPLAN.md) | understanding the copyright/OA metadata frontend design |
| [`docs/design/COPYRIGHT_AND_REUSE.md`](docs/design/COPYRIGHT_AND_REUSE.md) | author memo on copyright posture, TDM working-copy lifecycle, journal-by-journal license advice (astronomy / physics / earth / biology / CS) |
| [`docs/comparisons/COMPARISON_LIT_LAKE.md`](docs/comparisons/COMPARISON_LIT_LAKE.md) | per-capability comparison vs lit-lake / zotero-mcp; deciding whether paper2md or a Zotero-indexer fits your workflow |
| [`docs/comparisons/COMPARISON_PYMUPDF4LLM.md`](docs/comparisons/COMPARISON_PYMUPDF4LLM.md) | per-capability comparison vs pymupdf4llm (the Artifex/PyMuPDF "PDF → markdown in one call" library); deciding whether paper2md's per-paper VLM cost is justified vs. fast pure-CPU bulk RAG ingestion |
| [`docs/comparisons/COMPARISON_DOCLING.md`](docs/comparisons/COMPARISON_DOCLING.md) | per-capability comparison vs docling (IBM's LF-AI multi-format document converter, MIT-licensed, used internally by paper2md as the `--table-finder docling` backend); side-by-side empirical equation rendering on canup.pdf |
| [`docs/comparisons/ALTERNATIVES_NO_CUDA.md`](docs/comparisons/ALTERNATIVES_NO_CUDA.md) | guide for users without a CUDA GPU: paper2md's CPU/MPS story, detailed Tier-1 alternatives (MinerU empirical run, docling, marker standalone), brief pointers to cloud (Mathpix, LlamaParse, Mistral OCR) and light extractors (markitdown, pdfminer.six, textract), specialty tools (GROBID, Nougat), plus paper2md's roadmap toward a public pip module + Zotero/MCP integration |
| [`docs/comparisons/COMPARISON_MINERU.md`](docs/comparisons/COMPARISON_MINERU.md) | head-to-head per-capability comparison vs MinerU (opendatalab; AGPL-3.0; pipeline + hybrid VLM backends) on a 4-paper benchmark spanning math density, scanned/OCR-era PDFs, and APS-style references; documents MinerU's chart-data reconstruction (`<details>` blocks of VLM-estimated curve values), reference-handling failure modes, and the Apple Silicon batch-size workaround |
| [`docs/setup/SPARK_SETUP.md`](docs/setup/SPARK_SETUP.md) | day-one setup + tuning for an NVIDIA DGX Spark (GB10 / sm_121 / 96 GB unified): vllm flags, concurrency verification (`--table-workers 4` smoke test), known sm_121 issues |
| [`docs/setup/SOL_SETUP.md`](docs/setup/SOL_SETUP.md) / [`docs/setup/SOL_ARM_INTERACTIVE.md`](docs/setup/SOL_ARM_INTERACTIVE.md) | running on ASU's Sol HPC (A100 batch and GH200 interactive paths) |
| [`LICENSE`](LICENSE) | MIT license terms (Sarah T. Stewart 2026; Claude Anthropic Opus 4.7 acknowledged as collaborator) |

## Cost estimate

For local-only extraction the cost is electricity + GPU/Mac time. With a frontier-model API (the multi-provider plumbing supports `--provider openai|anthropic`), the combination of page-layout analysis plus VLM hooks averages around one cent per paper.

## Tests

```bash
conda activate paper2md
python -m pytest tests/ -q     # 791 tests, no GPU/network
```

End-to-end smoke runs use the PDFs under `examples/`, e.g. `python src/paper2md.py examples/canup.pdf --metadata-only`.

---

paper2md v0.4.0 (May 2026). Developed by Sarah T. Stewart with Claude Code (Anthropic, Opus 4.7). MIT License — see [`LICENSE`](LICENSE).

**Cite this software:**
Stewart, S. T., & Claude (Anthropic, Opus 4.7). (2026). *paper2md (v0.4.0)* [Software]. Zenodo. https://doi.org/10.5281/zenodo.20263036

The concept DOI [`10.5281/zenodo.20263035`](https://doi.org/10.5281/zenodo.20263035) always resolves to the latest release; use the version-specific DOI above when citing for reproducibility.
