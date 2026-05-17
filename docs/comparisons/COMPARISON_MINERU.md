# paper2md vs MinerU — head-to-head on a 4-paper science corpus

Detailed comparison of paper2md against [MinerU](https://github.com/opendatalab/MinerU) v3.1.6 on four
representative scientific papers. Companion to
[COMPARISON_DOCLING.md](COMPARISON_DOCLING.md) and
[COMPARISON_PYMUPDF4LLM.md](COMPARISON_PYMUPDF4LLM.md), and the
detail layer for [ALTERNATIVES_NO_CUDA.md](ALTERNATIVES_NO_CUDA.md).

MinerU is an academic-paper-aware open-source pipeline from the
Shanghai AI Lab (opendatalab). It bundles layout detection, math
formula recognition (UniMERNet), table detection, and OCR. It has
two operating modes that bear directly on this comparison:

- `pipeline` — small specialized models only, no LLM in the loop.
- `hybrid-auto-engine` — adds a 2.3 GB local VLM that handles
  layout reasoning and structured-element extraction.

Both backends run on Apple Silicon via PyTorch MPS (with caveats
documented below). License: AGPL-3.0.

## Test configuration

Four papers were chosen to span the range of scientific-PDF
extraction failure modes:

| Paper | Pages | Source | Stress-tests |
|---|---:|---|---|
| canup (Canup et al. 2015) | 6 | *Nature Geoscience* | math density, single-column-ish layout |
| cuk (Cuk & Stewart 2012) | 7 | *Science* | 3-col layout, citations, figure-caption pairing |
| Knudson2013 | 18 | *Phys. Rev. B* | 2-col, tables, math-heavy, `[N]`-style refs |
| Wackerle1962 | 16 | *J. Appl. Phys.* | pre-2000 scan, OCR-era stale text layer |

Three tool configurations:

- **paper2md v0.3.1**: NVIDIA Spark (GH200), Qwen3-VL-32B-Instruct
  via vLLM, full pipeline (`--vlm-tables --auto-force-ocr
  --use-journal-rescue --table-workers 4 --hdf5`).
- **MinerU hybrid**: M5 (Apple Silicon, 128 GB unified memory),
  PyTorch MPS, batch_size=1 forced via
  `MINERU_VIRTUAL_VRAM_SIZE=1` (see Caveats).
- **MinerU pipeline**: same M5 hardware, no LLM.

paper2md numbers are from a Spark CUDA run; MinerU numbers are
from M5 MPS. Different hardware, but the comparison reflects
realistic deployment choices for each tool today.

## Wall time

| Paper | paper2md (Spark CUDA) | MinerU hybrid (M5 MPS) | MinerU pipeline (M5 MPS) |
|---|---:|---:|---:|
| canup | 3:13 | 3:07 | 0:28 |
| cuk + cuk_sm | 12:11 | 3:52 (cuk only) | 0:33 |
| Knudson2013 | 26:36 | 13:59 | 1:07 |
| Wackerle1962 | 15:53 | 16:57 | 1:29 |

MinerU pipeline is **7–30× faster than hybrid** because it skips
both VLM passes (layout-VLM and extract-VLM). It's also 7–25×
faster than paper2md, but the trade-off cost on output quality is
significant — see below.

paper2md elsewhere on M5 MPS with Gemma 4 31B MLX 8-bit completes
canup in ~3:01–3:12, comparable to its Spark CUDA time. The Spark
advantage is throughput-at-scale (concurrent papers), not
single-paper latency. See [USAGE.md](USAGE.md) §13.

## Output quality matrix

Stats across all four papers. paper2md's `--vlm-tables` was on; if
disabled the table numbers would be lower for paper2md too.

```
                    paper2md   hybrid   pipeline
canup
  display math         8        14       14
  inline math        162       118      128
  refs section       yes       yes      yes
  ref entries         38        38       38
  fig caption pair     3         0        0     ← paper2md only
  details (chart)      0         7        0     ← VLM hallucination
  HTML <table>         0         0        0
  md pipe rows         0         0        0   (canup has 0 source tables)

cuk
  refs section       yes       yes      yes*
  ref entries         76        42       41*
  fig caption pair     4         0        0
  details              0         3        0
  md pipe rows        21        25        0
  HTML <table>         0         1        1

  * cuk pipeline does extract all 41 of cuk's references
    (numbered 1-41, continuous) but they sit at ~85% through the
    document because the cuk.pdf has the Canup 2012 article
    bleeding onto its last page and MinerU does not trim that
    boundary. paper2md's article-boundary trim cuts cleanly so
    its references land at the end. paper2md's 76 includes
    cuk + cuk_sm (paired SI) merged; MinerU was run on cuk only.

Knudson2013
  refs section       yes      none*   none*    ← see ref-format note
  ref entries (corrected) 58   58       56
  fig caption pair    11         0        0
  details              0        12        0
  md pipe rows       182       148        0
  HTML <table>         0        16       16

Wackerle1962
  refs section       yes       no       no    ← truly missing
  ref entries         22         0        0
  fig caption pair     9         0        0
  details              0         8        0
  md pipe rows       112        49        0
  HTML <table>         0         5        5
```

\* MinerU emits Knudson2013's references as a trailing block but
without a `## References` heading and with the digit-prefix glued
to the surname (`36P. E. Blochl, Phys. Rev. B...`). The entries
ARE all there (58/58 hybrid, 56/58 pipeline) — they just need a
formatting post-pass. See "Reference handling" below.

## Reference handling — the most important finding

**paper2md preserves the references section on all four papers.**
The references-rescue pipeline (Phase-1 score + journal-keyed
rescue + Crossref/OpenAlex API fallback + APS missing-heading
rescue) is the load-bearing piece on hard layouts:

- canup: clean numbered list, no rescue needed.
- cuk: 76 entries via marker + tidy passes.
- Knudson2013: 58 entries, recovered via the `[N]`-style
  numbered-mash + missing-numbered-refs rescues + section-heading
  rescue.
- Wackerle1962: 22 entries (with gaps reflecting the original
  paper) via marker + tidy passes. No DOI in the PDF so no API
  fallback fires; the body extraction recovers them anyway.

**MinerU's reference handling is uneven:**

| Paper | hybrid | pipeline |
|---|---|---|
| canup | ✓ 38 entries with `## References` heading | ✓ 38 entries |
| cuk | ⚠ 42 entries present but doc continues with the appended Canup 2012 paper (no boundary trim) | ⚠ 41 entries — same article-boundary issue; refs end up at ~85% of doc, then Canup bleeds in |
| Knudson2013 | ⚠ 58 entries present but no heading + no-space format | ⚠ 56 entries same format |
| Wackerle1962 | ✗ references absent entirely | ✗ references absent entirely |

The Knudson2013 case is recoverable with a thin post-processor:

```python
# Detect the trailing block of "^\d{1,3}[A-Z]" lines and reformat.
import re
md = re.sub(
    r'(?m)^(\d{1,3})([A-Z])',
    r'- \1. \2',
    md,
)
# Optionally: insert "## References" heading before the first
# such line by splitting the document at the boundary.
```

This converts:
```
36P. E. Blochl, Phys. Rev. B 50, 17953 (1994).
37G. Kresse and D. Joubert, Phys. Rev. B 59, 1758 (1999).
```
into:
```
- 36. P. E. Blochl, Phys. Rev. B 50, 17953 (1994).
- 37. G. Kresse and D. Joubert, Phys. Rev. B 59, 1758 (1999).
```

paper2md's `_BULLETED_NUMBERED_REF_RE` handles this style natively
for APS papers. Wackerle's missing references are a harder
problem — the layout doesn't trigger any rescue we have for
MinerU, and pipeline mode's layout classifier doesn't recognise
the section as references at all.

## Figure caption pairing

paper2md outputs `![Figure N | <caption>](path/img.jpg)` on every
extracted figure (Hook 2 with the page-locality validator). The
caption is searchable in the alt-text and embeddings can match
"Figure 3 of Knudson 2013" to the right image.

**MinerU's markdown drops the pairing, but it's preserved in
`middle.json`.** Every figure / table / chart block in MinerU's
intermediate JSON has nested children that group the caption
with its body:

```
type=image    →  [image_caption, image_body]
type=table    →  [table_caption, table_body, table_footnote]
type=chart    →  [chart_body, chart_caption]
```

Each child has a `bbox` and the caption's `text` is captured.
The markdown emission flattens this structure: figures become
`![](images/...)` with empty alt-text, captions become separate
body paragraphs. For RAG over figures the markdown is unusable
(nothing connects an image to its caption), but a ~30-line
post-processor that reads `middle.json` and rewrites
`![](path)` → `![Figure N | <caption>](path)` closes the gap
deterministically.

This means the figure-caption story is **not** a fundamental
detection failure — it's a markdown-emission choice. If you wrap
MinerU as a backend (the Zotero → MCP path discussed in
[ALTERNATIVES_NO_CUDA.md](ALTERNATIVES_NO_CUDA.md)), reading
`middle.json` directly to do caption pairing is straightforward.
paper2md's Hook 2 does similar work via regex + VLM match
against the markdown, but on top of MinerU you can use the
already-paired structured data.

## Tables

| Tool | Markdown pipe tables | HTML `<table>` | Per-table sidecar `.md` |
|---|---|---|---|
| paper2md | yes (clean GFM) | no | yes (one per table) |
| MinerU hybrid | partial (mostly synthetic chart-data) | yes | no |
| MinerU pipeline | none | yes | no |

paper2md's table-extraction pipeline (docling default + optional
TATR/PyMuPDF + per-table VLM rewrite) outputs clean GFM markdown
tables and per-table sidecar `.md` files for downstream tools.
Knudson2013 has 16 tables; paper2md outputs them all as pipe
markdown plus 16 sidecars.

MinerU pipeline emits HTML `<table>` blocks for every detected
table — they render in browsers but aren't pipe-grep-able and
won't render in plain markdown viewers. MinerU hybrid is similar
but ALSO emits chart-data reconstruction tables (see next
section) which inflate the pipe-row count to ~150 on Knudson.

For tables with stacked subscripts in headers (e.g. Knudson Table
II's covariance matrix `σ²_{a₀}`), MinerU's table extraction
loses the inner subscripts (`σ20`, `σ2`) and produces malformed
multi-row spans. paper2md's `--vlm-tables` path preserves these
because the per-table VLM rewrite re-extracts the cell text from
the cropped image.

## Chart-data reconstruction (`<details>` blocks, hybrid only)

MinerU's hybrid backend includes a feature that **looks at chart
figures and emits an estimated table of curve data inside a
`<details><summary>line</summary>...</details>` block** under the
figure's image reference.

Example from canup's hybrid output:

```markdown
![](images/9c83...jpg)

<details>
<summary>line</summary>

| Time (yr) | Mass of largest body (M_L) | Mass fraction from inner disk |
| --------- | -------------------------- | ----------------------------- |
| 0.001     | 0.0                        | 0.0                           |
...
</details>
```

The numbers are NOT in the source PDF. They are the VLM's
eyeballed estimates of the curve values. MinerU produces:

| Paper | `<details>` blocks |
|---|---:|
| canup | 7 |
| cuk | 3 |
| Knudson2013 | 12 |
| Wackerle1962 | 8 |

For some downstream uses this is useful (RAG over numerical
claims, automated meta-analysis where exact values aren't
required). For factual extraction it's risky — the markdown
doesn't flag these as estimated, and a downstream consumer
treating them as ground truth will be quoting the VLM, not the
paper. There's no MinerU flag specifically to disable this.

## What is MinerU's hybrid VLM actually doing?

From the canup hybrid run logs:

| Stage | Wall on M5 | Function |
|---|---:|---|
| VLM Layout Predict | 1:26 (~14 s/page) | Page-level reading-order, region-type classification, **chart→table reconstruction**, **table→HTML cell extraction** |
| VLM Extract Predict | 0:54 (~3.6 s/iter) | Per-region text extraction (headings, captions, body) — produces the `<details>` blocks and HTML table cells |
| Pipeline-side stages | <0:05 each | Layout small model, MFR (math formula recognition), OCR-det, OCR-rec |

The VLM is not "reading" the document like an LLM would. It's
two specialized vision passes that:

1. **Decide layout structure** — column boundaries, region types,
   reading order. The pipeline-mode small layout model does this
   too but with less fidelity (which is why pipeline mode misses
   the cuk reference section entirely).
2. **Extract structured elements** — table cells one-by-one into
   HTML, chart curves into estimated data tables.
3. **Resolve reading order** — for multi-column papers, decide
   which block follows which.

What hybrid mode buys you over pipeline mode:

- ✅ Better table extraction (HTML cell-level vs nothing)
- ✅ Better reading order on hard layouts
- ✅ Chart-data reconstruction (speculative — VLM estimates)

What MinerU does NOT emit in markdown that paper2md does (some
recoverable from `middle.json` post-processing, some not):

- ⚠ Figure-caption pairing in alt-text (`![Figure N | …]`) —
  data is paired in `middle.json` but flattened in the markdown
  emission; recoverable with ~30 lines of post-processing
- ⚠ Knudson-style no-space references as a clean list — entries
  present, format fixable with a regex post-processor
- ❌ `<sup>`/`<sub>` HTML citations / chemical formulas
- ❌ Article-boundary trim (cuk has Canup 2012's article
  appended on its last page — paper2md detects via VLM probe and
  cuts at the boundary; MinerU emits both papers as one stream
  with the cuk references buried mid-doc). Not in `middle.json`
  either — it's a true detection gap.
- ❌ Recover references on Wackerle-style scanned papers
- ❌ Synthesize a citation block at the top from page 1
- ❌ Resolve copyright / OA metadata / data-repo links
- ❌ Surface a quality score / reproducibility metadata in the
  artifact

The 7–30× wall-time premium over pipeline mode pays for chart
reconstruction and cell-level table extraction. It does NOT close
the citation/reference fidelity gap with paper2md.

## Caveats and operational notes

**MinerU on Apple Silicon — batch>1 hangs.** With
`MINERU_VIRTUAL_VRAM_SIZE` unset (or set to a value ≥8), MinerU
3.1.6's hybrid backend chooses batch_size 4 or 8 and the run
hangs at "Predict: 0/N" without progress. Setting
`MINERU_VIRTUAL_VRAM_SIZE=1` (forcing batch_size=1) is the only
working configuration on M5 MPS for hybrid mode as of writing.
Worth filing upstream; for now treat as a hard requirement.

**MinerU's `get_vram()` has no MPS path** — it queries CUDA, NPU,
GCU, MUSA, MLU, SDAA, and falls through to a 1 GB default for
Apple Silicon. The `MINERU_VIRTUAL_VRAM_SIZE` env var is the
documented workaround.

**Pipeline-mode batch size on Spark CPU** — the same OOM
behavior we saw on a 47-page batch (`MFR Predict: 0/1192`
crashed) was solved by running papers one at a time. MinerU's
default 64-page window is too aggressive for memory-constrained
hosts.

## China-provenance note

MinerU is developed by [opendatalab](https://github.com/opendatalab)
(Shanghai AI Lab). For users / institutions where Chinese-origin
software is a sensitivity (e.g., classified research, certain
government / aerospace / defense contexts), MinerU may be ruled
out regardless of license or output quality. Alternatives without
this concern, also covered in the comparison docs:

- **paper2md** — US-developed, MIT-licensed, journal-paper-specific.
- **docling** — IBM Research Zürich, MIT, multi-format (see
  [COMPARISON_DOCLING.md](COMPARISON_DOCLING.md)).
- **pymupdf4llm** — Artifex Software (US), AGPL-3.0 or commercial
  (see [COMPARISON_PYMUPDF4LLM.md](COMPARISON_PYMUPDF4LLM.md)).
- **marker** — datalab.to (US), GPL-3.0; the spine paper2md wraps.

The comparison doc set should treat MinerU as **one option among
several**, not a recommended default.

## When each tool fits

**paper2md** — primary choice for science papers when:
- references and citations matter (RAG over bibliographies,
  citation linking, hand-edited reference cleanup)
- figure-caption pairing matters (RAG over figures)
- tables-as-markdown-tables matter (downstream pandas / jq usage)
- copyright / OA metadata matters (compliance, redistribution)
- the corpus has any pre-2000 / scanned papers
  (`--auto-force-ocr` recovers them; MinerU silently misses
  references on these)
- the user has any local compute (CPU, MPS, or CUDA)

**MinerU pipeline** — best when:
- speed matters above all (1-2 min/paper on M5 MPS)
- the body text is the deliverable (not refs / captions / tables)
- the corpus is modern, well-formatted papers (post-2000, APS or
  Elsevier or Nature)
- a downstream pipeline can convert HTML tables and add caption
  alt-text on its own

**MinerU hybrid** — useful when:
- chart-data reconstruction is desired (and the speculative
  nature is acceptable — output should be flagged)
- table fidelity beats pipeline-mode HTML output
- you don't need references on Wackerle-style old / non-standard
  papers

For the **Zotero → MCP pipeline** path discussed in
[ALTERNATIVES_NO_CUDA.md](ALTERNATIVES_NO_CUDA.md): paper2md
remains the recommended primary backend. MinerU could be a
fast-path option for bulk pre-screening, but cannot substitute
for paper2md on the bibliography-extraction step on harder
papers — the reference-loss data on Wackerle and the partial-loss
on cuk pipeline mean it would silently produce incomplete
artifacts.

## Sources

- This comparison: empirical runs on four papers (canup, cuk,
  Knudson2013, Wackerle1962) using paper2md v0.3.1 on Spark
  GH200 + Qwen3-VL-32B, MinerU 3.1.6 hybrid + pipeline on M5
  Apple Silicon (128 GB unified memory).
- Per-paper outputs preserved at
  `collections/mineru-comparison/` (Spark side) and
  `mineru_tests_mac/` (M5 side); not committed (corpora are
  copyrighted; gitignored).
- Bench harness: [`tests/eval_mineru_bench.py`](../tests/eval_mineru_bench.py)
  — cross-platform timing + log capture script for repeatable
  comparisons.
- [COMPARISON_DOCLING.md](COMPARISON_DOCLING.md),
  [COMPARISON_PYMUPDF4LLM.md](COMPARISON_PYMUPDF4LLM.md),
  [COMPARISON_LIT_LAKE.md](COMPARISON_LIT_LAKE.md),
  [ALTERNATIVES_NO_CUDA.md](ALTERNATIVES_NO_CUDA.md) — the rest
  of the comparison doc set.
- [paper2md USAGE.md](USAGE.md) — full feature list and flag
  reference.
