# paper2md — user guide

Local PDF-to-Markdown pipeline tuned for scientific journal articles (two-column layout, mixed figures and tables, running headers/footers). Uses [marker](https://github.com/VikParuchuri/marker) as the spine with targeted vision-LLM assist passes.

Runs on two backends from a single codebase:

| Backend | Hardware | Default VLM provider | Env file |
|---|---|---|---|
| MPS | Apple Silicon (32 GB+ unified memory) | LM Studio (`qwen3-vl-32b-instruct-mlx`) | `environment-mac.yml` |
| CUDA | NVIDIA (DGX Spark, etc.) | vLLM (`Qwen/Qwen3-VL-32B-Instruct`, BF16) | `environment-gpu.yml` |

Backend is picked automatically by `--backend auto` (default). Override with `--backend {cuda,mps,cpu}`. The VLM provider auto-pairs with the backend: `cuda → vllm`, `mps/cpu → lmstudio`. Override with `--provider` or `$VLM_PROVIDER`.

---

## 1. Architecture

```
PDF ──► marker (layout, reading order, math, headers/footers, images)
           │
           ▼
      base markdown
           │
           ├──► strip line numbers (lineno-package artifacts)         [deterministic]
           ├──► fix reference numbering (back-fill from span anchors) [deterministic]
           ├──► consolidate footnote refs (gather <sup>N</sup> → ## References) [deterministic]
           ├──► trim to first article (binary-search from last page)  [VLM, 1–log N calls]
           ├──► hook 0: synthesize a journal-style citation from page 1 + DOI link
           ├──► hook 1: save JPEG + VLM-rewrite every located table + sidecar .md
           ├──► hook 2: match each image to an author figure caption (or drop it)
           ├──► hook 3: rescue sparse-text pages via whole-page VLM
           └──► hook 4: heuristic quality scoring → YAML front-matter
           │
           ▼
    final markdown + assets/
```

The diagram above shows the original marker-spine path. As of v0.3.x there are three layout engines (`--layout-source {mineru, marker, hybrid}`, default `mineru`) — see §3.5 for the full picture. The hooks below apply across all engines unless noted.

The spine is deterministic. The VLM only does content interpretation on regions that marker or PyMuPDF have already localized — it never has to guess bounding boxes. Quality scoring, line-number stripping, and reference renumbering are deterministic passes — no VLM calls.

Before any of this runs, a **VLM preflight check** (`GET /v1/models` against the configured provider) verifies the VLM is reachable. If it isn't, the script exits in ~3 seconds with a clear error rather than burning the multi-minute marker run on a doomed pipeline. Skip with `--skip-vlm-check` (or `--no-vlm`).

---

## 2. Install

```bash
pip install marker-pdf pymupdf pillow openai pyyaml h5py

# Optional: only if you plan to use --table-finder tatr or both
pip install transformers torch timm

# Optional: only if you plan to use --table-finder docling
pip install docling

# REQUIRED if you plan to use --layout-source mineru (the default) or
# --layout-source hybrid. PIN to 3.1.7 -- see warning below.
pip install "mineru[core]==3.1.7"

# Optional: only if you plan to use --provider anthropic
pip install anthropic
```

Or, preferred on macOS / Linux, use the pinned conda env:

```bash
conda env create -f environment.yml
conda activate paper2md
pip install "mineru[core]==3.1.7"   # required post-step for mineru/hybrid layouts
```

The conda env pins `marker-pdf==1.10.2` and `surya-ocr==0.17.1` (matching `environment-gpu.yml` for cross-machine output parity). On Apple Silicon, paper2md auto-resolves `--backend auto` to `cpu` because surya's MPS path produces silently corrupted output — see §6.0.

### ⚠️ MinerU version pin (3.1.7) — important

paper2md's hybrid splice and several layout-rescue hooks parse MinerU's `middle.json` schema directly. **MinerU minor-version bumps have historically changed the schema in ways that silently break parsing**:

- caption-detection regexes fail to match → figures/tables dropped from output
- table-body HTML structure shifts → cells lost or duplicated
- image-asset path convention changes → splice can't locate cropped JPGs
- counter values in `rescues.hybrid_splice` go to zero corpus-wide

The fix is to pin `mineru[core]==3.1.7` everywhere — `environment-gpu.yml`, `environment-mac.yml`, and any external install scripts you may have. **Don't `pip install --upgrade mineru` without auditing the splice fudges first.**

If you have a hard requirement to upgrade MinerU:

1. Update the pin in BOTH env files + this section in the same commit.
2. Walk the "When MinerU version changes" checklist in `docs/dev/HYBRID_IMPLEMENTATION.md`.
3. Re-run a known-good corpus against the new MinerU and diff the output against baseline. Inspect any `rescues.hybrid_splice` counters that flipped to zero — they may indicate fudges that became inert (MinerU fixed the underlying issue; delete the fudge) OR fudges that stopped firing because the regex no longer matches (broken; needs patching).
4. The frontmatter records `run.packages.mineru` in every output — `jq '.run.packages.mineru' *.meta.json` is the post-hoc forensic tool.

- `pyyaml` is used by batch-analysis recipes in §11.4 (reading the quality front-matter).
- `h5py` is required for `--hdf5` (see §12). Omit it if you'll never use the bundle output.
- First run of `marker` downloads its model weights (~1 GB) from Hugging Face. First run of `--table-finder tatr` downloads the Microsoft Table Transformer detection weights (~110 MB). First run of `--table-finder docling` downloads IBM's TableFormer + DocLayNet + RapidOCR weights (~500 MB).

LM Studio:

1. Install LM Studio.
2. Download a vision-language model (see §4).
3. Load the model and start the local server (default: `http://localhost:1234`).
4. Confirm the model ID shown in LM Studio matches what you pass to `VLM_MODEL`.

---

## 3. Run

```bash
# default: marker + all three VLM hooks, using qwen2.5-vl-32b-instruct
python src/paper2md.py paper.pdf -o out/

# marker only, no VLM (fastest; good for sanity-checking the spine)
python src/paper2md.py paper.pdf --no-vlm

# high-fidelity tables: opt INTO the per-table VLM rewrite. Adds 1-8 min
# per dense table; recovers subscript / Greek / footnote-marker fidelity
# that the default detector-text path loses. Pair with --table-workers
# for concurrent VLM calls on multi-table papers.
python src/paper2md.py paper.pdf --vlm-tables --table-workers 4

# lightweight built-in detector for ruled-line tables only (rare today;
# the docling default catches more tables on most journal corpora).
python src/paper2md.py paper.pdf --table-finder pymupdf

# Table Transformer only (no PyMuPDF or docling table pass)
python src/paper2md.py paper.pdf --table-finder tatr

# borderless tables in vector-text PDFs: union of pymupdf + tatr.
python src/paper2md.py paper.pdf --table-finder both

# unknown corpus, want maximum coverage: chain pymupdf + tatr + docling
# (cheapest first; first-match wins). Loads every detector model.
python src/paper2md.py paper.pdf --table-finder all

# screen copyright/OA status only -- no marker, no VLM, no extraction.
# Single-paper: prints YAML 'copyright:' block to stdout (~5s/PDF
# depending on API latency). Batch: writes manifest.jsonl with one
# record per PDF. Honors --prefer-oa-source and --require-license.
python src/paper2md.py paper.pdf --metadata-only
python src/paper2md.py --batch corpus/ -o screen/ --metadata-only

# pick a different VLM
VLM_MODEL=qwen2.5-vl-72b-instruct python src/paper2md.py paper.pdf

# point at a non-default LM Studio URL
LM_STUDIO_URL=http://192.168.1.10:1234/v1 python src/paper2md.py paper.pdf

# use OpenAI instead of LM Studio (requires $OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python src/paper2md.py paper.pdf --provider openai

# use Anthropic (requires $ANTHROPIC_API_KEY + anthropic package)
ANTHROPIC_API_KEY=sk-ant-... python src/paper2md.py paper.pdf --provider anthropic

# override model per-provider
ANTHROPIC_API_KEY=... VLM_MODEL=claude-opus-4-7 python src/paper2md.py paper.pdf --provider anthropic

# verbose logging (shows which tables/pages are being redone)
python src/paper2md.py paper.pdf -v

# batch gate: fail (non-zero exit) if the conversion scores below 0.80
python src/paper2md.py paper.pdf --quality-threshold 0.80

# suppress the quality front-matter at the top of the .md
python src/paper2md.py paper.pdf --no-quality

# skip the citation-synthesis pass (one VLM call on page 1); useful in batch
# mode if you already have citations from another source
python src/paper2md.py paper.pdf --no-citation

# skip the deterministic reference-renumbering pass (default ON; uses
# surviving span anchors to back-fill missing numbers in the bibliography)
python src/paper2md.py paper.pdf --no-fix-refs

# skip the article-boundary trim (default ON; one VLM call per PDF; asks
# whether page 2 is a different article and truncates the markdown there
# if so). Only effective when the VLM is enabled.
python src/paper2md.py paper.pdf --no-trim-articles

# skip the deterministic footnote-reference consolidation pass (default ON;
# lifts <sup>N</sup>-prefixed footnote lines out of the body into a single
# ## References section at the end, so a downstream vector DB chunking by
# section heading sees one references chunk instead of refs scattered into
# body chunks). No VLM, no GPU; pure regex.
python src/paper2md.py paper.pdf --no-consolidate-footnotes

# skip the preflight VLM reachability check (default ON; pings the configured
# provider before starting the multi-minute marker run so a misconfigured
# provider fails in seconds instead of after the whole extraction)
python src/paper2md.py paper.pdf --skip-vlm-check

# also convert a supplement PDF. Writes paper.md + paper_SI.md (or whatever
# the supplement's stem is), sharing the same out/assets/ folder. Supplement
# assets are prefixed 'si_' so they don't collide with main-paper assets.
python src/paper2md.py paper.pdf --supplement paper_SI.pdf -o out/

# additionally write a single compressed HDF5 bundle (paper.h5) containing
# the main markdown, supplement markdown (if any), and every file under
# assets/. Loose files are still written; the bundle is a convenience for
# transport / archival / corpus ingestion.
python src/paper2md.py paper.pdf --supplement paper_SI.pdf -o out/ --hdf5

# safety net: force marker / surya / TATR off MPS onto CPU. Not needed on
# the current pinned stack (see §6.0) -- kept in place in case you ever
# relax the marker/surya pins and hit MPS bugs. Has NO effect on the VLM.
python src/paper2md.py paper.pdf --cpu
```

Output layout:

```
out/
├── paper.md                         # main: YAML quality front-matter + body
├── paper_SI.md                      # supplement (only if --supplement was used)
└── assets/
    ├── figure_1_p3.jpg              # main-paper Figure 1 (single panel)
    ├── figure_2a_p5.jpg             # main-paper Figure 2 panel (a)
    ├── figure_2b_p5.jpg             # main-paper Figure 2 panel (b)
    ├── table_1_p4_1.jpg             # main-paper Table 1 (page 4, body index 1)
    ├── table_1_p4_1.md              # main-paper Table 1 sidecar markdown
    ├── si_figure_S1_p2.jpg          # supplement figure (si_ prefix)
    ├── si_table_S1_p5_1.jpg         # supplement Table S1
    ├── si_table_S1_p5_1.md          # supplement Table S1 sidecar
    └── ...
```

**Semantic asset naming** (since v0.4.0). Tables are named
`table_{id}_p{page}_{idx}.{md,jpg}` and figures
`figure_{id}{letter?}_p{page}.{ext}`, where `{id}` is the paper-relative
table/figure number from the caption (e.g. `1`, `A.4`, `S1`) and
`{letter}` is the multi-panel letter (`a`, `b`, `c`, ...) when MinerU's
subpanel pass identifies separate panels. The sidecar `.md` and the
matching `.jpg` always share a stem, so `ls assets/ | sort` groups
related files together. Dots in ids (e.g. `A.4`) are replaced with `_`
in filenames for filesystem safety; body link text keeps the original
`.` for human readability. Under `--layout-source marker`, table assets
fall back to positional naming (`table_p{page}_{idx}`) because the
marker spine doesn't expose table-id metadata.

Unmatched images that hook 2 classified as non-figures (journal banners,
icons, decorations) are **deleted from both the markdown and `assets/`**
— see §5.2.

When `--supplement` is used, every asset produced by the supplement pass
is prefixed with `si_` so the two documents can share `assets/` without
filename collisions. The supplement gets its own quality front-matter and
runs all hooks independently (including the citation pass, which typically
returns SKIP for SI PDFs that don't reprint the parent citation). This
layout is designed to map cleanly onto a future HDF5 bundle (one asset
store, per-document markdown + metadata).

The top of `paper.md` looks like this (see §11 for quality details):

```yaml
---
quality:
  overall: 0.88
  grade: B
  vlm_enabled: true
  note: "Scores reflect pipeline confidence, not factual correctness."
  tables: [...]
  figures: [...]
  pages: [...]
---

Author A, Author B & Author C. **Paper title** *Journal Name* 42, 123-135 (2026) [https://doi.org/10.xxxx/yyyy](https://doi.org/10.xxxx/yyyy)

# Paper Title
... body ...
```

The citation line (hook 0) sits between the YAML front-matter and the H1 title. It is synthesized from page 1 using the VLM in the journal's own reference style — see §5.0.

### 3.1 Mode flags: `--no-vlm` vs `--vlm-tables`

The default run is "fast mode": detector text for table bodies, all other VLM hooks active. Two flags adjust this baseline. They turn off / on different amounts of work, and pair with different `--table-finder` choices.

| | default (no flag) | `--vlm-tables` | `--no-vlm` |
|---|---|---|---|
| Scope | All VLM hooks active EXCEPT per-table rewrite (hook 1) and sparse-page rescue (hook 3). | All VLM hooks active, including per-table crop rewrite (hook 1). Sparse-page rescue (hook 3) is still gated separately by `--rescue-sparse-pages`. | **All VLM hooks off** — citation (hook 0), table rewrite (hook 1), orphan-table rescue (hook 1.5), figure caption (hook 2), sparse-page rescue (hook 3), article-boundary trim (pre-pass). |
| Table contents | Sourced from the detector's text (`--table-finder docling` exports markdown directly). JPEG crop + sidecar `.md` + body link still produced. | VLM re-transcribes each crop to recover subscripts / Greek / footnote markers / dense math. | Untouched — marker's markdown body stays as-is. No JPEG crop, no sidecar `.md`. |
| Wall time per paper | Seconds for the table phase. Other VLM hooks still cost ~30 s–2 min for citation + figures + rescue. | 1–8 min per dense table on a 32B VLM, plus the non-table VLM hooks. | Marker + deterministic strippers only. Fastest mode that still produces a proper artifact. |
| When you'd use it | Default. Good fidelity for most papers; LaTeX/Greek/sub-script-heavy tables are the cases where you'd reach for `--vlm-tables`. | LaTeX-heavy or footnote-laden tables; manuscripts where table fidelity is the point of the extraction. | You don't have a VLM running, or want the fastest possible text-only first pass. |
| VLM endpoint required | Yes (preflight pings it) | Yes | No |
| Best partner | `--table-finder docling` (the default; required for the detector-text shortcut to produce markdown bodies). | `--table-finder docling` + `--table-workers 4` for concurrent VLM calls on multi-table papers. | None — it's self-contained. |

The **deprecated `--no-vlm-tables` flag** maps to the new default and is now a no-op alias that emits a warning. The default behavior already matches what the flag used to select, so most users can drop it; scripts can keep it for one release cycle.

One non-obvious detail: the default mode keeps the table **location pass** running (caption-page bypass, bbox finding, JPEG cropping). It only short-circuits the per-table VLM call. You still get a sidecar, an inline image link, and a `TableScore` in the YAML — just with the table body sourced from the detector instead of the model. Tables that fall through to the page-image fallback under `--vlm-tables` (no detector match + caption found nearby) stay as marker's body under the default, since the page-image fallback also requires the VLM.

#### `--vlm-tables-force` — force VLM rewrite for every table sidecar

`--vlm-tables` alone only fires the VLM on tables that `html_to_pipe_md` couldn't convert cleanly (rowspan, colspan, inconsistent column counts, >40% blank body rows — see "table_is_suspicious" gate). Clean tables skip the VLM entirely and the sidecar `.md` contains the cheap HTML-to-pipe-md conversion. For most papers that's fine — the cheap conversion preserves the structure correctly.

For publication-grade sidecars where you want the VLM's higher fidelity on sub/super, Greek, units, and math even for "clean" tables, pass `--vlm-tables-force`:

```bash
paper2md paper.pdf --layout-source mineru --vlm-tables-force
paper2md paper.pdf --layout-source hybrid --vlm-tables-force --table-workers 4
```

Behavior under `--vlm-tables-force`:

| Table state | Body | Sidecar `.md` |
|---|---|---|
| Clean (html_to_pipe_md OK) | cheap pipe-md (unchanged) | **VLM rewrite** (new — extra VLM call) |
| Suspicious / rowspan | VLM rewrite or HTML fallback (same as `--vlm-tables`) | VLM rewrite (same as `--vlm-tables`) |
| VLM call fails on clean table | cheap pipe-md (unchanged) | cheap pipe-md (fallback — sidecar always exists) |

Cost: roughly N extra VLM calls per paper where N is the count of tables that would otherwise have been "clean." For a 6-table paper that's ~30-180s extra on Spark + Qwen3-VL. Combine with `--table-workers 4` for concurrency — both `--layout-source mineru` and `--layout-source hybrid` parallelize the table-VLM calls when `--table-workers > 1`. (Single-table papers stay serial; the pool only spins up with ≥2 tables and workers ≥2.)

Under `--layout-source marker`, `--vlm-tables-force` is a **no-op**: marker's `process_tables` already calls VLM on every located table when `--vlm-tables` is set (no `table_is_suspicious` gating). You get the same behavior with or without the force flag.

Implies `--vlm-tables`. Incompatible with `--no-vlm` (errors at startup).

**Do NOT use `--vlm-tables-force` with `--batch`.** The flag is intended for one-off publication-grade sidecars on individual papers, not corpus runs. On Apple Silicon with LM Studio MLX, each forced VLM call takes ~80–180 s; a paper with 15 clean tables can easily exceed the default `--paper-timeout` (1800 s) and a multi-collection corpus run takes many hours, with mineru subprocess flakiness rising over the long wall time. The recommended workflow:

1. Run the corpus with the default `--vlm-tables` (auto VLM: fires only on tables `html_to_pipe_md` couldn't convert cleanly — typically a small subset).
2. Inspect sidecars; if a specific table's `.md` is unsatisfactory, transcribe it standalone:
   ```bash
   vlm-table assets/table_3_p7_2.jpg -o assets/table_3_p7_2.md
   ```
   See §19 for the standalone CLI. This is faster, has no batch-wall-time risk, and produces the same VLM output as `--vlm-tables-force` would have for that one table.

For users who genuinely want every table re-transcribed (small corpora on Spark CUDA, where each call is ~10–20 s instead of minutes), `--vlm-tables-force --batch` is fine — just budget the wall time.

`pre_redo_reason: "force"` in `quality.tables[].pre_redo_reason` tells you a VLM call was triggered by the force flag (vs `rowspan=N` or similar for the standard "suspicious" path). `post_redo_reason: "vlm-empty-fallback-to-pipe-md"` indicates a forced VLM call that failed; the sidecar fell back to the cheap pipe-md.

### 3.2 New flags in v0.3 and v0.4

All default-on; opt-out variants in parentheses. See §5 for the matching hook descriptions.

| Flag | Purpose | Default |
|---|---|---|
| `--force-ocr` | Bypass the embedded PDF text layer; force marker to run surya OCR on every page. Use on pre-2000 / scanned papers whose publisher OCR poisons the body (spurious bold paragraphs, line-broken table headers, mis-located captions). Adds ~30-60 s / page on a typical paper. | off |
| `--auto-force-ocr` | Per-paper detection: PyMuPDF-only sniff of `/Producer` + `/Creator` metadata, image-coverage ratio, font diversity, bold-character ratio (~100-300 ms / paper). Applies `--force-ocr` only to papers flagged as scans. Mutex with `--force-ocr`. Recommended for mixed-corpus batches. | off |
| `--scan-detect-only` | Dry-run: print the scan-detection verdict + supplement-pairing structure for each PDF and exit. TSV format suitable for spreadsheet review before launching a batch. No marker, no VLM. | off |
| `--clean` | Delete this paper's pre-existing outputs (`<stem>.md`, `<stem>.meta.json`, `<stem>.h5`, matching files in `assets/`) before running. Other files in `--out` are left alone. Without `--clean`, stale files from a prior run on the same paper persist alongside the fresh output and a one-line warning is logged. | off |
| `--paper-timeout SECONDS` (batch) | Abandon any single paper that exceeds this wall-time limit; the batch continues with the next paper. `0` disables. Caveat: the abandoned thread keeps running; for hangs that hold `_MARKER_LOCK` or GPU memory, split the batch across multiple paper2md invocations. | 1800 (30 min) |
| `--no-strip-publisher-stamps` | Skip the per-page download-watermark stripper. Currently handles Wiley Online Library's "DOI URL + downloading institution + Terms-and-Conditions" block stamped on every page of subscription downloads. | on |
| `--no-tidy-refs` | Skip the in-`## References` tidy-up: merge column-break continuation lines into their parent entries, pull author addresses / `(Received…)` lines out of the bullet list, apply uniform `- ` bulleting. | on |
| `--no-inject-orphan-refs` | Skip the orphan-reference-cluster injector. When marker splits a multi-page references list across a column boundary so some entries land outside the `## References` heading, this pass synthesizes a heading above the orphan cluster so `merge_reference_sections` can consolidate. | on |
| `--no-rotate-tables` | Skip the sideways-table rotation hook. When a cropped table looks rotated 90° on a portrait page (height/width > `PAPER2MD_TABLE_ROTATION_THRESHOLD`, default 2.5), hook 1 renders the whole page rotated upright (direction `PAPER2MD_TABLE_ROTATION_DIRECTION`, default `ccw`) and routes to the page-image VLM with a rotation-aware prompt. The rotated whole page is also saved as the human-visible JPG. | on |

### 3.3 Recommended batch workflow for mixed corpora

```bash
# 1. Dry-run survey: pairings + scan verdicts as TSV
python src/paper2md.py --batch path/to/corpus --scan-detect-only \
  -o /tmp/survey > corpus_survey.tsv 2>&1

# 2. Eyeball the TSV in a spreadsheet (filter by pair_role / verdict /
#    producer). Pairs are auto-attached for name.pdf + name_sm.pdf
#    (and _SI / _supp / _S1 / _supplement / etc. variants).

# 3. Kick off the actual batch
python src/paper2md.py --batch path/to/corpus -o out/ \
  --auto-force-ocr --paper-timeout 1800 --vlm-tables \
  --table-workers 4 --workers 1 --clean
```

Per-paper outputs land in `out/<paper-stem>/`; failures + summaries appear in `out/manifest.jsonl`. `--clean` is per-paper-scoped (only the current paper's stem is cleared).

Mental model: default is *"fast structured run; VLM polishes everything except table cells"*. `--vlm-tables` is *"high-fidelity tables; spend the wall time"*. `--no-vlm` is *"marker-only mode for fast text indexing"*.

---

## 3.5. Layout source (`--layout-source`, `--auto-layout-source`)

Since v0.3.x, paper2md ships **three layout engines** behind a single CLI. The default is `mineru`; the older marker path is still available for sup/sub-fidelity / scan papers; the `hybrid` path stitches marker's body text together with MinerU's figure / table layout.

### 3.5.1 The three options

| `--layout-source` | Body OCR | Figure / table / caption layout | When to use |
|---|---|---|---|
| `mineru` (DEFAULT) | MinerU's pipeline backend (PaddleOCR) | MinerU's middle.json | Born-digital modern PDFs. Fast (~30 s/paper), best layout on multi-figure / multi-table papers. |
| `marker` | marker + surya | paper2md's docling/pymupdf table-finder + caption_figures matcher | Scan / pre-2000 / sup-sub-heavy papers. Slower (~60-90 s/paper), but surya preserves `<sup>`/`<sub>` markup that PaddleOCR flattens. |
| `hybrid` | marker + surya (body text, caption text, reading order) | MinerU's middle.json (figure assets, table HTML, sub-panel grouping) — spliced into marker's body by figure / table NUMBER | Born-digital papers where you want surya's sup/sub fidelity on body AND MinerU's stronger figure / table extraction. Cost is additive (marker + MinerU sequentially per paper). |

The `text_engine` field in every output's frontmatter records which OCR actually produced the body (`mineru-paddleocr` or `marker-surya`); `layout_source` records the engine choice. Under hybrid, the frontmatter additionally records `figure_layout_source: mineru` and `caption_text_source: marker` so downstream filters can identify hybrid output without parsing `layout_source` directly.

### 3.5.2 Auto-routing for mixed corpora (`--auto-layout-source`)

The "good fast route" for mixed corpora — pairs born-digital papers to `mineru` and scan / flat-OCR papers to `marker`. `--auto-layout-source` does NOT route to `hybrid`; if you want hybrid you must pass `--layout-source=hybrid` explicitly (and it applies to every paper in the batch). For typical use just pass the flag:

```
paper2md --batch corpus/ -o out/ --auto-layout-source
```

Per paper, paper2md runs **two PyMuPDF probes** (~600 ms total) and **composes** their decisions:

| Probe | What it answers | Catches |
|---|---|---|
| `triage_pdfs.recommend_routing` | "Is the text layer rich enough for MinerU's layout extraction?" (super flags, distinct font sizes, char density) | Lyzenga-style flat-OCR (no super flags) |
| `detect_scan_ocr_pdf` | "Is this a SCAN that needs re-OCR?" (producer/creator metadata, image cover, bold-char ratio) | Wackerle-style scans whose OCR happens to leave triage-fooling text-layer artifacts |

**Routing rule**: marker if EITHER detector says marker is needed. Mineru otherwise.

**Force-OCR rule**: auto-applies when scan_detect fires AND routing=marker (skips force-OCR on triage-marker-routed-but-not-a-scan papers like a tables-heavy SI). Override with `--no-auto-force-ocr` if you want raw marker on a scan-routed paper (rare).

Outcomes by paper type:

| Paper | Triage | Scan-detect | Layout | Force-OCR |
|---|---|---|---|---|
| canup (born-digital) | mineru | not a scan | mineru | n/a |
| Wackerle (1962 scan, OCR'd) | mineru | **scan** | **marker** | **auto** |
| Lyzenga (1980 scan, flat OCR) | marker | scan | marker | auto |
| alexander_sm (tables-heavy SI) | marker | not a scan | marker | **off** |

Per-paper signals (both `triage_signals` AND `scan_signals` when applicable) plus the composed routing decision land in each output's `pipeline.*` block for reproducibility. The top-level `layout_source` and `text_engine` fields reflect THIS paper's resolved engine.

`--auto-layout-source` is mutually exclusive with `--layout-source` (force one engine for everything vs route per paper).

To preview the routing decisions for a collection without running paper2md, use the standalone triage helper:

```
./workflow/triage.sh corpus_name   # writes workflow/triage-corpus_name.tsv
```

Note: the standalone triage helper only runs the triage probe (signal #1). Use `--auto-layout-source` for the full composed routing logic that catches the Wackerle case.

### 3.5.3 MinerU controls (when `--layout-source=mineru`)

| Flag | Default | What it does |
|---|---|---|
| `--mineru-backend {pipeline, hybrid-auto-engine}` | `pipeline` | `pipeline` is fast (PaddleOCR). `hybrid-auto-engine` uses MinerU's bundled VLM for cell-level tables and chart-data reconstruction (~3-17 min/paper). |
| `--mineru-arg ARG` (repeatable) | — | Pass-through to mineru's CLI. E.g. `--mineru-arg --device-mode --mineru-arg cpu`. |
| `--skip-mineru-run` + `--mineru-dir PATH` | off | Consume an existing MinerU output (auto/ or flat layout dir) instead of re-running. ~0.2 s rerun for editing post-passes. |
| `--strip-chart-details` | off | Strip MinerU hybrid-mode `<details><summary>line</summary>` chart-data blocks (VLM-estimated values; off by default). |
| `--no-rescue-orphan-captions` | rescue is on | Skip the caption / footnote adoption pass for a fast clean MinerU run. |

### 3.5.4 Hybrid layout — automatic post-passes (when `--layout-source=hybrid`)

Hybrid runs both engines sequentially (marker first for body+captions, then MinerU for figure/table assets), and applies several extra fix-ups before emitting the markdown. All are deterministic, all run unconditionally under `--layout-source=hybrid`, and all surface counters under `rescues.hybrid_splice` in the YAML frontmatter + `.meta.json` sidecar.

| Auto pass | What it does | Counter |
|---|---|---|
| Bipartite figure/table splice | Matches MinerU's figure / table blocks to marker's `Fig. N.` / `**Figure N** \|` / `Table N.` caption lines by NUMBER (not text similarity). Image link is inserted **above** marker's caption; table region is replaced with MinerU's HTML / pipe-md while marker's caption heading is preserved. | `figures_spliced`, `tables_swapped` |
| Sub-panel rendering | When `rescue_subpanel_groups` consolidated a multi-panel figure (e.g. 2-panel `a` + `b` Nature figures), all panels are emitted as separate image links with letter suffixes (`![Figure 3 panel a]...`, `panel b`, `panel c`). | — |
| EOD fallback splice | For MinerU figures / tables that didn't match a marker caption anchor: image + MinerU caption are spliced near the nearest body cross-ref (`Fig. N` / `Table N`) or appended at end of document. Sub-panels are rendered here too. | `unmatched_mineru_figs`, `unmatched_mineru_tbls` |
| Marker image-link strip | Removes marker's own `_page_*.jpeg` image links from the body (including any leading `<span id="page-X-Y"></span>` cross-reference anchors common on PRL/APS papers). **Files remain on disk in `assets/`** for evaluation / recovery; only the markdown links are dropped. | `marker_image_links_removed` |
| MinerU references swap | When BOTH marker's body and MinerU's `mineru/<stem>.md` have a recognizable references heading, marker's references section is replaced with MinerU's. Marker's 3-col column-aware serialiser fails on Science/Nature multi-column papers (missing numbers, glued-pairs, paragraph-collapse); MinerU's structural extractor produces cleaner numbered lists. Guarded by a heading-present check on both sides so papers where MinerU silently dropped the section (e.g. footnote-style 1962 papers) keep marker's version. | `mineru_refs_swapped` |

All the global reference cleanup hooks (§5: span-anchor normalisation, missing-heading rescue, numbering repair, footnote consolidation, multi-section merge, in-section tidy, API ref fallback) still run under hybrid. The hybrid-specific passes above are in addition to those.

Frontmatter additions under hybrid:

```yaml
pipeline:
  layout_source: hybrid
  text_engine: marker-surya
  figure_layout_source: mineru
  caption_text_source: marker
rescues:
  hybrid_splice:
    figures_spliced: 5
    tables_swapped: 2
    unmatched_mineru_figs: 0
    marker_image_links_removed: 7
    mineru_refs_swapped: true
```

### 3.5.5 docling is now optional

The default `--table-finder` is `pymupdf` (was `docling` in v0.2.x). docling is no longer installed by the env files; install it separately if you need the borderless / scanned-table specialist under `--layout-source=marker`:

```
pip install docling==2.92.0
paper2md paper.pdf --layout-source marker --table-finder docling -o out/
```

Under the default `--layout-source=mineru`, MinerU's own table extractor handles borderless and scanned tables, so docling adds nothing.

---

## 4. Model choices and tradeoffs

### 4.1 Spine: why marker?

| Spine | Best at | Weaknesses | Use when |
|-------|---------|------------|----------|
| **marker** | two-column reading order, running headers/footers, LaTeX math, Apple Silicon (MPS) | complex merged-cell tables; occasional figure/body confusion | scientific journal articles (default) |
| docling (IBM) | table structure (TableFormer is the strongest open table model) | less paper-tuned; reading-order occasionally drifts | dense, heavily tabular documents; mixed document types (DOCX/PPTX/HTML) |
| MinerU | formula recognition (UniMerNet) | less mature Python API | math-heavy physics/ML theory papers |
| olmOCR (AllenAI) | end-to-end VLM trained for PDF→MD | less configurable, single-model pipeline | when you want one model to do everything |
| Nougat (Meta) | academic PDFs with heavy LaTeX | older, narrower scope | legacy arXiv-style papers |

For scientific journal articles, marker is the right default. If tables are your main pain point, swap to docling; if the papers are math-heavy, try MinerU.

### 4.2 VLM: picks by available RAM

The VLM is the heaviest component after marker. On Apple Silicon with LM Studio, the model loads into unified memory and coexists with marker/surya (~5 GB) and TATR (~500 MB) in the same Python process, plus macOS itself (~6–10 GB idle). Rule of thumb: the VLM should occupy **no more than ~60%** of total system RAM to leave headroom for marker + OS + context cache.

#### 4.2.1 Recommended local VLMs by memory tier

| System RAM | VLM to load (LM Studio / MLX) | Disk size | Notes |
|---|---|---|---|
| **≤ 16 GB** | **Qwen2.5-VL-7B-Instruct** (MLX 4-bit) | ~5 GB | Quality drops on dense tables; pairings still work. Consider `--provider openai` or `--provider anthropic` if quality matters more than locality. |
| **24 GB** | **Qwen3-VL-30B-A3B-Instruct MoE** (MLX 4-bit) | ~18 GB | MoE architecture → only ~3 B active params per token. Fastest option at this tier. |
| **32 GB** | **Qwen3-VL-32B-Instruct** (MLX 4-bit) dense | ~20 GB | Solid default at 32 GB. Or Qwen3-VL-30B-A3B at 6-bit if you prefer MoE throughput. |
| **48–64 GB** | **Qwen3-VL-32B-Instruct** (MLX 6-bit) | ~28 GB | Better numerical fidelity than 4-bit. Room for marker + TATR. |
| **96–128 GB** (the reference setup this project was built on) | **gemma-4-31b-it** (MLX 8-bit) — **best on the test corpus**, or **Qwen3-VL-32B-Instruct** (MLX 8-bit) — historically tested default | ~32 GB / ~36 GB | Both are near-lossless and fit in the 60% rule with room for marker. Gemma 4 31B MLX 8-bit hit 100% on the jacquet test paper (matching gpt-4.1 / claude-sonnet-4-6) where Qwen3-VL-32B BF16 scored 92.9%; see §13.3.1. paper2md sends `reasoning_effort=none` so Gemma's thinking mode doesn't consume the per-call budget — works out of the box. |
| **128+ GB** | **Qwen3-VL-72B-Instruct** (MLX 4-bit) | ~40 GB | Empirically did NOT improve on Qwen3-VL-32B (same 92.9% on jacquet, 2.2× slower); skip unless your corpus has features the 32B specifically misses. |

#### 4.2.2 Other models worth knowing about

| Model | MLX 4-bit size | When to consider |
|-------|----------------|------------------|
| **Qwen2.5-VL-32B-Instruct** | ~20 GB | The previous default — still excellent. Use if Qwen3-VL has compatibility issues in your LM Studio version. |
| **InternVL 2.5 / 3 (38B or 78B)** | ~22 GB / ~45 GB | Competitive with Qwen-VL on DocVQA and ChartQA. Less ecosystem support in LM Studio than Qwen family. |
| **MiniCPM-o 2.6 (8B)** | ~5 GB | Fast OCR at 8 B params — reasonable fallback on 16 GB boxes where Qwen-7B feels slow. Weaker on complex tables. |
| **Pixtral-12B** | ~7 GB | Fine for figure captions. Not great on dense tables. |
| **Llama-3.2-90B-Vision-Instruct** | ~50 GB | Strong general vision, but weaker than Qwen-VL on structured-document tasks. Skip unless you're already running it for something else. |

#### A specialist model that does NOT fit: GOT-OCR2.0

**GOT-OCR2.0** (StepFun, late 2024, ~580 M params, ~1 GB on disk) is an end-to-end OCR → markdown/LaTeX model that looks attractive for document work at first glance. It is not a good fit for this pipeline. Two hard blockers:

1. **Single-column training.** GOT-OCR2.0 was trained primarily on single-column document OCR and has no robust reading-order logic for two-column journal articles. Most scientific journals (Nature, MNRAS, AASTeX, APS-family, Elsevier) are two-column — expect scrambled reading order.
2. **Not instruction-following.** GOT-OCR2.0 is a single-shot "image → markdown" model with task-specific modes (plain OCR, formatted OCR, fine-grained OCR). It does not accept chat-style prompts. Every hook in this pipeline depends on instruction-following (citation synthesis, figure classification, SKIP sentinels, caption pairing) — GOT-OCR2.0 cannot do any of it.

It also does not plug into LM Studio's OpenAI-compatible endpoint cleanly; integrating it would require a parallel code path through the `transformers` library.

**Where GOT-OCR2.0 would genuinely be strong** is LaTeX math transcription on single-column preprints. But marker's built-in surya equation recognizer already handles this for our pipeline, so there is no quality gap to close. Unless you're processing a corpus of single-column technical reports with heavy math and you need a sub-gigabyte specialist, skip it.

#### 4.2.3 Setup recommendations per tier

- **≤ 16 GB, quality-sensitive work:** skip local. Use `--provider openai` or `--provider anthropic`. API cost per paper is typically $0.01–$0.25 for the full pipeline (see §13.2), which is cheaper and higher-quality than a 7 B model on a memory-starved box.
- **16 GB strictly local:** Qwen2.5-VL-7B-Instruct MLX 4-bit is your ceiling. Expect degraded table transcription but functional figure captioning.
- **24–32 GB:** Qwen3-VL-30B-A3B-MoE 4-bit (faster) or Qwen3-VL-32B dense 4-bit (slightly higher quality). Either handles documents cleanly.
- **48–64 GB:** move to Qwen3-VL-32B at 6-bit. This is the quality-cost inflection point — materially better table output than 4-bit, no noticeable speed penalty on Apple Silicon.
- **96–128 GB (reference box):** **gemma-4-31b-it MLX 8-bit** is the new top recommendation — empirically tied gpt-4.1 / claude-sonnet-4-6 at 100% on the jacquet test paper while staying fully local. **Qwen3-VL-32B-Instruct MLX 8-bit** is a strong alternative (the historically-tested default; still excellent at 92.9% on jacquet and the project's environment-mac.yml ships configured for it). Either works; consider running both in parallel LM Studio slots and switching via `VLM_MODEL` per paper. Also keep a small fast model (MiniCPM-o 2.6 or Qwen2.5-VL-7B) loaded in a third slot for long-document triage.
- **128 GB+ with budget for latency:** Qwen3-VL-72B 4-bit was *not* materially better than the 32B 8-bit on jacquet (same 92.9%, 2.2× slower) — only worth it if you've identified a specific corpus failure mode the 32B misses. The accuracy / latency win is going *down* in quantization (32B at 8-bit beats 72B at 4-bit), not up in parameter count.

#### 4.2.4 Picking 4-bit vs 6-bit vs 8-bit

Quantization choice matters more for this project than for general chat: hooks 1 (table rewrite) and 3 (page rescue) generate long autoregressive outputs where small errors accumulate. Numeric fidelity in table cells is a real quality axis.

| Quant | Fidelity vs full | Speed vs 4-bit | Memory vs 4-bit | Pick when |
|-------|------------------|----------------|-----------------|-----------|
| 4-bit | ~95% | baseline (fastest) | 1× | Memory-constrained; figure captioning dominates |
| 6-bit | ~97–98% | ~20% slower | ~1.4× | Good quality/memory balance if you have 48 GB+ |
| 8-bit | ~99%+ (near-lossless) | ~30% slower | ~1.8× | **Table-heavy corpora; numeric-critical output.** Current default on the reference box. |

See §13.3 for notes on when API providers beat even the best local model.

#### 4.2.5 Thinking vs Instruct variants

Two distinct cases:

**Models that ship as separate `-Instruct` and `-Thinking` builds** (Qwen3-VL, GLM-4 family, etc.). **Always use `-Instruct` for this pipeline.** The `-Thinking` variant emits `<think>...</think>` blocks before the final answer — paper2md doesn't strip them, and they'd leak into table sidecars and citations. Our tasks (classification, transcription, short captions) don't benefit from reasoning chains, so the Thinking variant is pure overhead.

**Models that ship as a single build with thinking enabled by default and a runtime toggle to disable it** (Gemma 4, gpt-5 / o-series). paper2md sends `extra_body={"reasoning_effort": "none"}` on every OpenAI-compat call so these models emit the answer directly within the small per-hook budgets (`max_tokens=5` for the figpair classifier, etc.). Without that flag, all 5 tokens get consumed by internal reasoning and every figure is silently dropped. Verified working on Gemma 4 31B in LM Studio; non-reasoning models (Qwen-VL family, gpt-4.1, gpt-4o) silently ignore the field. For OpenAI's actual reasoning models (gpt-5, o-series), `reasoning_effort=none` is in the right shape but may not be sufficient — see §13.6.

---

## 5. How the hooks work

### Pre-marker — auto-detect scan + force-OCR (`--auto-force-ocr`)

When `--auto-force-ocr` is passed (default: off), a CPU-only PyMuPDF detector runs per paper before marker is invoked, deciding whether to flip `--force-ocr` on for that paper. Cost: ~100-300 ms / paper. No torch, no surya, no GPU.

Signals:
- `/Producer` and `/Creator` PDF metadata, matched against an OCR-tool fingerprint list (`ABBYY`, `FineReader`, `OmniPage`, `ScanSoft`, `Adobe Capture`, `AdobeOCR`, `Capture Plug-in`, `Iris OCR`, `Readiris`, `Tesseract`, generic `OCR`).
- Median image-coverage ratio: image bbox area / page area, sampled across up to 8 evenly-spaced pages. Scanned PDFs are typically a single full-page image per page with a thin OCR text overlay (image_cover ≈ 1.00).
- Font count in the text layer. Born-digital scientific PDFs span 5-15 fonts; old-OCR'd scans collapse to 1-3.
- Bold-character ratio: chars in bold-flagged spans / total chars. Old-OCR PDFs frequently set bold on every span via stale font-weight metadata; modern papers ≈ 0.05.

Decision (first-match-wins, conservative — false-positive cost is ~30-60 s of wasted OCR on a clean paper):
1. **STRONG**: any one OCR-tool fingerprint in metadata.
2. **STRONG-COMBO**: `image_cover_median > 0.6` AND `font_count <= 3`.
3. **MEDIUM-COMBO**: 2+ of {`image_cover > 0.4`, `fonts <= 4`, `bold > 0.25`}.

Verified on the silica-shock collection (20 PDFs): 4 correctly flagged as scan (`Wackerle1962`, `Boslough1988`, `Lyzenga1980`, `Lyzenga1983`); 16 correctly classified as digital. No false positives, no obvious false negatives.

`--scan-detect-only` runs the detector across a corpus and exits, printing one TSV row per PDF with the verdict, signals, and (in batch mode) the supplement-pairing structure. Use it to validate decisions before kicking off a multi-hour run.

### Preflight — VLM reachability check

Before any pipeline step (including marker), `src/paper2md.py` does a `GET /v1/models` against the configured provider with a 5-second timeout.

- On success: log `VLM preflight OK` and continue.
- On failure: log the endpoint URL and the underlying exception, then exit with code 2.

Skipped automatically when `--no-vlm` is set (the pipeline doesn't need a VLM in that mode). Bypass explicitly with `--skip-vlm-check`. Anthropic gets a soft pass — there's no free reachability check for it that doesn't burn a billable token; only the SDK initialization is verified.

Why this exists: marker is the single longest step in the pipeline (often 5–25 minutes on dense papers). Without preflight, a misconfigured `--provider` would burn the full marker run and only fail on the first VLM hook, by which point the figure-drop pass (hook 2) has already deleted unmatched images from `assets/`. Preflight prevents that whole class of "I just lost half an hour" failures.

### Pre-pass — body cleanup strippers (deterministic, no VLM)

Four small deterministic md → md passes that drop common journal-PDF artifacts marker preserves verbatim. Each is gated independently via a CLI flag and reflected in `run.pipeline` of the YAML front-matter.

- **`strip_line_numbers`** (toggle: `--no-fix-refs` is unrelated; this pass is always-on). Removes lone 1–4 digit lines that survive marker's text reflow on author-line-numbered manuscripts (LaTeX `lineno` package output — AASTeX, MNRAS, APS submission templates all do this).
- **`strip_running_footers`** (toggle: `--no-strip-footers`). Removes `<AUTHOR> ET AL.  N of M` running-footer lines that AGU / Wiley / society journals interleave between body paragraphs. Bounded match (≤60 chars before "et al" + the `N of M` tail) so legitimate prose isn't affected. Skips fenced code blocks and table rows.
- **`strip_journal_page_headers`** (toggle: `--no-strip-page-headers`). Removes Nature-style page headers: bare publication-type labels (`LETTERS`, `ARTICLES`, `REVIEWS`, `PERSPECTIVES`, etc., optionally bold-wrapped) and journal-name + DOI lines like `**NATURE GEOSCIENCE DOI: [10.1038/NGEO2574](url)** LETTERS` or `LETTERS NATURE GEOSCIENCE DOI: 10.1038/NGEO2574`. Heuristic gate (length ≤ 250 chars; bullets / list items / table rows / code blocks skipped) keeps reference entries with the same keywords intact.
- **`strip_math_labels`** (toggle: `--no-strip-math-labels`). Removes KaTeX-incompatible `\label{...}` from display-math blocks and rewrites `\eqref{name}` to `(name)`. KaTeX (the in-browser math renderer used by VS Code, GitHub, Obsidian, mdBook, Jupyter) doesn't implement LaTeX cross-reference commands, so leaving them in causes `ParseError: KaTeX parse error: Undefined control sequence: \label`. Surya's equation OCR preserves them verbatim from the PDF text layer when the source had them.

### Pre-pass — article-boundary trim (1–log₂(N) VLM calls)

Some PDFs concatenate multiple articles — Nature News & Views frequently bundles two adjacent pieces, journal-issue downloads sometimes include the full issue, OCR'd archives often glue articles together. Without intervention, marker concatenates everything into one markdown file and downstream hooks process content from the wrong article.

The trim pass:

1. Renders page 1 + the LAST page at low DPI (~100), stacks them vertically. Asks the VLM: SAME or DIFFERENT? If DIFFERENT, what's the new article's title?
2. If the last page is SAME: the whole PDF is one article. Done. **(1 VLM call — the common case)**
3. If the last page is DIFFERENT: binary-search backwards for the boundary. Maintains the invariant that page `lo` is SAME and page `hi` is DIFFERENT, halving the gap each iteration. Stops when `lo + 1 == hi`; PDF page `hi+1` is the first page of the different article. **(adds log₂(N) VLM calls)**
4. Find the new article's title (returned by one of the DIFFERENT calls) in the markdown via decreasing word-window fuzzy matching (full title → 3-word slices; case-insensitive, punctuation-insensitive; takes the LAST occurrence so a forward-reference in the kept article doesn't match instead).
5. Cut the markdown at that line. Fallback if no title match: cut at the first image reference whose page index is ≥ the boundary page.
6. Delete `_page_N_*` image files from `assets/` that are no longer referenced in the trimmed markdown (so they don't end up in the HDF5 bundle).

Cost: 1 VLM call when the whole PDF is one article (~1–3 s on Qwen3-VL); roughly `1 + log₂(N)` calls when there's a boundary. For a 16-page PDF with one boundary: ~5 calls.

Default on; toggle with `--no-trim-articles`. Already skipped under `--no-vlm`.

Verified cases:
- `Elliott.pdf` (2 pages, page 2 is unrelated): 1 VLM call detects the boundary; trim cuts at the new article's title; final grade A.
- `canup.pdf` (6 pages, single article): 1 VLM call confirms last page = SAME; no trim.

Limitation in v1: still handles only ONE boundary per PDF. If the PDF contains three or more concatenated articles, only the boundary nearest the end is detected; re-run on the trimmed output to find earlier boundaries.

### Pre-pass — reference renumbering (deterministic)

Marker emits each numbered reference in the bibliography as a bullet line anchored by a span id (`- <span id="page-X-N"></span>`). The leading reference number is sometimes dropped during marker's column reflow (it gets confused with a footnote-link superscript), but the surviving `id-N` is always sequential per PDF page.

This pass uses surviving leading numbers as anchors:

1. Walk every line matching `- <span id="page-X-N">` and record `(page, id_n, captured_number)`.
2. For each PDF page that has at least one captured number, compute `offset = captured_number - id_n`.
3. For pages with no surviving anchor, infer the offset by forward sweep (`prev_page_offset + count(prev_page)`) then backward sweep (`next_page_offset - count(this_page)`).
4. Back-fill missing leading numbers using the per-page offset.

Idempotent — lines that already have a number are unchanged. Pages with no inferable offset are left alone (would require guessing). Toggle with `--no-fix-refs` (default on). No VLM, no GPU; pure regex.

Typical case: `young-full.pdf` had 45 references missing leading numbers on PDF page 10; surviving refs 64, 83, 84, 89 anchored the offset (`offset = 57`), repairing all 45 with correct numbering.

### Pre-pass — footnote-reference consolidation (deterministic)

Some journals (older J. Appl. Phys., older Phys. Rev., many engineering journals) use **numbered footnotes** for references — each citation appears at the bottom of the page where it's first cited, rather than collected in a single References section at the end. After marker conversion those refs end up scattered through the body as standalone lines like:

```
<sup>1</sup> F. W. Neilson and W. B. Benedick, Bull. Am. Phys. Soc. 5, 511 (1960).
```

Bad for vector-DB section-based chunking: the refs end up embedded in body chunks instead of in their own References chunk. This pass lifts them to a single `## References` numbered list at the end.

Algorithm:

1. **Idempotency check**: skip if a `## References` / `## Bibliography` / `## Notes` heading already exists.
2. **Pre-normalize** marker's known nested-`<sup>` garble: `<sup>&</sup>lt;sup>17</sup>` → `<sup>17</sup>`.
3. Walk lines. A line is a footnote-reference candidate iff it starts (after whitespace) with `<sup>N</sup>` where `N` is a digit. In-body `<sup>3</sup>` citation markers appear mid-line and are not touched.
4. On candidate lines, find ALL `<sup>N</sup>` segments. The leading one is always a ref. Subsequent `<sup>N</sup>` segments on the same line are treated as a NEW ref only if what follows looks like an author-name token (alphabetic, after optional whitespace and `*`/`_` formatting). This filters out embedded year/page superscripts like `<sup>749</sup>` mid-body.
5. Lift each `(N, body)` out of the line. Sort by N. Append as a numbered list under a new `## References` heading at the end (or before any `## VLM page rescues` block if hook 3 added one).

What is **NOT** moved or changed:

- In-body `<sup>3</sup>` citation markers (mid-line, not at column 0).
- Lines whose `<sup>` digit is unreadable (`<sup>::</sup>` — marker couldn't OCR the digit). They stay in place; you can fix them manually.
- Refs marker missed entirely (the pass can only consolidate what's in the markdown).

Toggle with `--no-consolidate-footnotes` (default on). No VLM, no GPU; pure regex.

Verified case: `wackerle.pdf` (Journal of Applied Physics, 16 pages, 31 footnote-references). Lifted 25 of 31 cleanly into `## References`. Of the missing six: three were lost because they had no `<sup>` tag and got glued onto the previous ref's line (refs 10, 22), three were marker-missed entirely (refs 4, 13, 14, 15 not in marker output at all). Three refs (2, 16, 26) had duplicates from marker re-OCRing the same source twice — both versions kept and a warning logged so the user can pick the correct one.

### Pre-pass — Elsevier/Icarus span-anchored ref normalisation (deterministic)

Sibling of the renumber pass; runs **before** it. Targets the messy bibliography format that appears on Icarus / Elsevier extracts: a mix of plain-`<span>` ref lines (no bullet) and bulleted `<span>` ref lines, with markdown-link clutter from refhub cross-references and column-reflow-split DOI URLs. Three transforms, all idempotent:

1. **Re-bullet plain-span ref lines.** A contiguous run (>=2) of lines matching `^<span id="page-X-N"></span>\S` get a `- ` prefix prepended. After this, the renumber pass can see them and back-fill leading numbers via the same anchor-offset arithmetic.
2. **Rejoin split DOI links.** Marker breaks DOIs like `[http://dx.doi.org/](u1) [10.1016/0012-821X\(80\)90038-2](u2)` into two markdown links during column reflow. Rejoined to a single `<https://doi.org/10.1016/0012-821X(80)90038-2>` autolink (escaped parens unescaped).
3. **Strip refhub link wrappers.** `[Title](http://refhub.elsevier.com/...)` becomes plain `Title`. The refhub URL encodes the publisher's internal cross-reference graph and is publisher-only — strips cleanly without losing user-visible information.

Toggle with `--no-normalise-refs` (default on). No VLM, no GPU; pure regex. Only fires on lines that look span-anchored or contain explicitly-recognised link patterns; safe to leave on for non-Elsevier corpora.

Verified case: `jacquet.pdf` (Icarus 2026). Before: ~15 refs un-bulletted and unnumbered, refhub-cluttered cells, fragmented DOI URLs. After: every ref is a numbered bullet, DOIs are clean autolinks, no refhub wrappers remain.

Limitations (deferred):
- Multi-line footnote bodies (continuation lines without `<sup>`): the continuation stays in the body. Cosmetic; doesn't break the consolidated list.
- Other reference styles (`[1] Author...`, `1. Author...` numbered list): not handled. Add patterns as real PDFs surface them.
- VLM-rescue for garbled `<sup>::</sup>` lines: not implemented. The fallback is manual cleanup.

### Pre-pass — merge multiple References sections (deterministic)

Many journals append a second references list AFTER the Methods section. Nature is the canonical case: the main-text references (1–N) sit at the end of the body, then a `### Methods` section follows, then a second `# **References**` block (refs N+1 onward) lists the methods-only references. Marker preserves both sections verbatim, leaving the markdown with two separate `## References` blocks separated by Methods content. For chunked-by-section embedding pipelines that expect one consolidated reference list at end-of-document, that's a problem.

The pass:

1. Walks every heading whose title matches `References` / `Bibliography` / `Notes and References` (any depth, optional bold-wrapping like `# **References**`, optional trailing colon).
2. For each match, captures the section span (heading line through the next heading at the same or higher depth — or end of file).
3. Removes every captured section in place, then appends a single `## References` at the very end of the document with the section bodies concatenated in original order.

Methods, Acknowledgements, Author contributions, Additional information, and any other sections are preserved in their original positions before the consolidated reference list. Toggle: `--no-merge-references` (default on). No VLM.

The heading regex covers `References` / `Reference` / `Bibliography` / `Cited References`, the multi-word variants `Notes and References` / `References and Notes` / `References & Notes` (Science journal style, including the all-caps `**REFERENCES AND NOTES**` form), and the `Method[s] References` aliases used by Nature's Methods-section convention.

Verified case: `canup.pdf` (Nature Geoscience). Before: `### References` after main text (refs 1–26), then `### Methods`, then `# **References**` (refs 27–38). After: one `## References` at end with refs 1–38 in order; Methods preserved between body and refs.

### Pre-pass — inject orphan reference clusters (deterministic)

Marker can split a multi-page References section across a column or page boundary so that some entries land *outside* any `## References` heading. Since `merge_reference_sections` only fires on ≥2 explicit headings, those orphan entries get silently dropped from the consolidated section.

Detection runs immediately before merge. Walks for contiguous runs of ≥3 bulleted lines whose bullet is followed by an optional `<span id="page-N-N"></span>` anchor and a digit (marker's signature for numbered ref entries even when the leading author name was joined to the number, e.g. `- 47A. F. Danilyuk, ...`). Confidence filter: ≥50% of cluster lines must contain a 4-digit year (1800-2099); rejects enumerated body lists (AASTEX feature lists "1. Declaring math mode", "1. Install Python" instructional steps, numbered author affiliations).

When an orphan cluster is found that's NOT inside an existing References-section line span, a synthetic `## References` heading is prepended above it. `merge_reference_sections` then naturally consolidates the synthetic + original sections.

Toggle: `--no-inject-orphan-refs` (default on). No VLM, pure regex.

Verified case: `Knudson2013.pdf` (Phys. Rev. B). Before: refs 1-32 + 35-45 in `## References`, refs 46-58 in two free-floating bulleted clusters above the heading. After: refs 46-58 absorbed into the consolidated section.

### Pre-pass — tidy in-section reference formatting (deterministic)

Within a `## References` (or alias) section, normalise the per-entry layout so the output is internally consistent regardless of how marker chunked the input. Three transforms, all idempotent:

1. **Merge column-break continuation lines** into their parent entries. Marker emits a split entry when a citation wraps from one column to the next, sometimes as a bulleted continuation, sometimes as a plain paragraph — either way they get re-stitched into a single bulleted entry.
2. **Pull addenda out of the bullet list**. The corresponding-author address line ("M. B. Boslough, Division 1131, Sandia National Laboratories, Albuquerque, NM 87185.") and the manuscript-history "(Received ... revised ... accepted ...)" parenthetical are common end-of-references material that marker emits as bullets even though they aren't citations. After this pass they sit as plain paragraphs after the bullet list.
3. **Apply uniform `- ` bulleting** to every citation entry. References already in this layout are unchanged (idempotent).

Citation detection covers two shapes:
- *Surname (or collab) first*, optionally numbered: `Lyzenga, G. A., ...` / `1. Smith, J., ...` / `Astropy Collaboration, Robitaille, T. P., ...` (multi-word capitalized leads handle AAS-style collaboration names).
- *Initials first, must be numbered*: `1. F. W. Neilson and W. B. Benedick, Bull. ...` (older physics convention; the leading number is what distinguishes a citation from a corresponding-author address).

Both shapes tolerate one or more leading `<span id="page-N-N"></span>` page anchors so Elsevier-style references with surviving anchor markup (jacquet's Icarus refs) are recognised.

Toggle: `--no-tidy-refs` (default on). No VLM, pure regex.

Verified case: `Boslough1988`. Before: 7 plain-paragraph citations + 8 bulleted citations + 1 column-break-split entry + 1 mis-bulleted author-address line. After: 15 cleanly bulleted citations + 2 plain-paragraph addenda (author address, "Received...").

### Post-pass — APS missing-heading rescue (deterministic, always-on)

Some APS papers (Phys. Rev. Letters, Phys. Rev. B family) emit reference lists as `[1] Author...` `[2] Author...` lines at the end of the document with no `## References` heading. Section-chunked embedding pipelines can't locate the refs section without a heading.

Trigger: `report.metadata.journal_slug` starts with `aps` AND no `## References` (or alias) heading exists AND ≥5 contiguous `[N]` lines run at the end of the doc with N starting at 1 and strictly monotonically increasing. The rescue inserts a single `## References` heading line before the trailing run; body content is unchanged.

Always-on, no flag. Idempotent.

Verified cases: `Knudson2009`, `Hicks2006`.

### Post-pass — API reference fallback (Crossref → OpenAlex, always-on)

When the locally extracted references section is poor or absent AND a DOI is known, paper2md fetches the paper's deposited reference list from a free scholarly API and **appends** it as a new `## References (from <source>)` section at the end of the body. The local section is preserved as-is.

Trigger: `report.metadata.doi` is set AND any of:
- `references.score < 0.5`
- `references.section_count == 0` (no References heading detected)
- `references.style == "numbered"` AND `not references.numbered_continuous` AND `entry_count >= 3` (visible gaps in a numbered list signal missing entries — Wackerle1962, Hicks2006-with-gaps pattern; the composite score may still look healthy because gap-presence currently weighs 0 in the score formula).

Try order: Crossref `/works/{doi}` (preserves citation order when the publisher deposited in order) → OpenAlex `referenced_works` (batched fallback). First success wins; both empty → silent skip.

Failure modes (all silent skip + DEBUG log):
- No DOI → no fetch
- Network unreachable → no fetch
- Crossref + OpenAlex both empty → no append
- API exception → swallowed

Cache: results memoized by DOI on disk at `~/.cache/paper2md/references.json` (override via `XDG_CACHE_HOME`). Survives across runs and across `--clean`. Reference lists for a published paper don't change, so cache entries don't go stale; if you ever need to wipe, `rm ~/.cache/paper2md/references.json`.

Polite-pool email: set `CROSSREF_MAILTO` (and optionally `OPENALEX_MAILTO`) in your environment for higher rate limits. Without these, the calls still work but with lower priority.

Frontmatter: when an API section is appended, `quality.references.api_source` ("crossref" | "openalex") and `quality.references.api_refs_appended` (count) are recorded.

Caveat: this fallback addresses the *recoverable middle* (mangled-but-extant Phys. Rev., AGU, Elsevier, etc.). Pre-Crossref papers (pre-2000, e.g. PNAS 1920, J. Appl. Phys. 1962) typically have no DOI or no deposited references; the fallback is silently a no-op for them.

Always-on, no flag.

### Pre-pass — strip per-page publisher download stamps (deterministic)

Subscription downloads from publisher servers are stamped on every page with a multi-line block giving the DOI URL, downloading institution, date, and Terms-and-Conditions / Creative Commons attribution. Marker preserves each stamp fragment as a body paragraph, where they accumulate per page (8× in an 8-page paper) and frequently interrupt body paragraphs and equations.

Currently handles **Wiley Online Library** stamps:
```
onlinelibrary.wiley.com/doi/<DOI> by <institution>, Wiley Online
Library on [<date>]. See the Terms and Conditions
(https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley
Online Library for rules of use; OA articles are governed by the
applicable Creative Commons License
```

Six line-level patterns flag stamp content; matched lines are dropped and runs of 3+ blank lines collapse back to 2. Pattern 0 (`onlinelibrary.wiley.com` URL) is anchored at line start so legitimate Wiley DOI citations in body prose ("Retrieved from http://onlinelibrary.wiley.com/...") and reference list entries are NOT stripped. Patterns 1-5 carry boilerplate phrases ("Wiley Online Library on [DATE]", "See the Terms and Conditions", "OA articles are governed by the applicable Creative Commons", etc.) that don't appear in normal scientific writing.

Skips fenced code blocks and markdown table rows (`|`-prefixed lines) so DOI URLs in real reference lists aren't dropped. Toggle: `--no-strip-publisher-stamps` (default on). No VLM, pure regex.

The function is publisher-plural by design; new publishers (Elsevier, Springer, Taylor & Francis) can be added as additional alternation branches in `_PUBLISHER_STAMP_LINE_RES` when their stamp shapes show up in the corpus.

### Hook 0 — citation synthesis

Ask the VLM to produce a single bibliographic citation line from page 1 of the PDF, in the journal's own reference-list style (Nature, Elsevier, ACS, APS, etc. — inferred from the page's visible publisher metadata, author list, DOI, and footer). The citation line is inserted between the YAML quality front-matter and the H1 title.

1. Render page 1 at `PAGE_DPI`.
2. Send to the VLM with `CITATION_PROMPT`, which asks for: authors in the journal's convention, title (bold), journal name (italic), volume, article/page range, year, and — if a DOI or URL is visible — a markdown link.
3. Parse the first non-empty line of the reply. If it reads `SKIP` (the VLM's escape hatch for pages without enough metadata), no citation is written.
4. Prepend the citation + blank line to the body markdown.

Cost: one extra VLM call per PDF (~5–15 s on Qwen3-VL-32B). Toggleable via `--no-citation` for batch runs or when you already have citations from another source (CrossRef, a sidecar `.bib`, etc.).

Fallbacks: if page 1 doesn't exist, the VLM returns SKIP, or the VLM call fails, the pipeline logs a warning and continues — citation is treated as best-effort metadata, never a hard failure.

### Hook 1 — table processing (JPEG + VLM rewrite + sidecar .md)

For every markdown table in marker's output:

1. Regex-scan marker's markdown for tables.
2. Locate the corresponding PDF region using the selected `--table-finder`. The matcher extracts the longest body cell as an anchor token and looks for a detected table whose extracted text contains that token.
3. **If the region is less than ~85% of the page** (i.e., not a full-page table), render the crop at 220 DPI, save it as `assets/table_{id}_p{N}_{M}.jpg` (where `{id}` is the paper-relative table number, e.g. `1`, `S2`, `A.4`), and prepend an image link above the markdown table so the asset is discoverable. Under `--layout-source marker` the spine doesn't expose table-id metadata, so the asset uses the positional fallback name `table_p{N}_{M}.jpg`.
4. **Send every located table to the VLM** (`TABLE_PROMPT`, max_tokens=2000) to re-transcribe it to GitHub markdown from the crop. The VLM's output replaces marker's inline markdown for that table. If the VLM call fails, the pipeline falls back to marker's original markdown. The `table_is_suspicious` heuristic still runs — it now reports pre/post-VLM state in the quality front-matter but does **not** gate the rewrite. The prompt asks the VLM to also include a `**Notes:**` block when the table has letter-marker (`a`, `b`, `c`, ...) or symbol-marker (`*`, `†`, `‡`) footnotes immediately below; this is paired with a small bbox-expansion heuristic (`_expand_bbox_for_footnotes`) that walks PyMuPDF text blocks below the table and pulls footnote-marked blocks into the rendered crop so the VLM can see them.
5. **Write a sidecar `.md`** to `assets/table_{id}_p{N}_{M}.md` containing just that table's markdown (same content as what's inline). The sidecar and the matching `.jpg` always share a stem, so `ls assets/ | sort` groups related files together. Main doc gets a `[Table N — separate markdown](assets/table_{id}_p{N}_{M}.md)` link after the inline block so the sidecar is discoverable. Dots in ids (`A.4`) become `_` in the filename for filesystem safety (`table_A_4_p11_2.md`); link text keeps the original `.` for human readability.

Full-page tables are skipped for the JPEG step (keeping them out of `assets/` avoids redundant full-page renders). They are still eligible for VLM rewrite and still get a sidecar `.md`.

**Limitation: very dense tables hit the 2000-token output cap.** The cap is load-bearing: empirically, vLLM with Qwen3-VL-32B-Instruct (BF16) hangs and times out the request at `max_tokens >= 2500` when given a ~1400×1200 dense scientific table image. For a table whose body alone exceeds ~1800 tokens of markdown (e.g. 28 rows × 11 columns + math superscripts), the VLM truncates mid-row and the `**Notes:**` block never fires. `examples/wackerle.pdf`'s Table II is the canonical case — Tables I, IV, V (smaller bodies) capture their footnotes cleanly, Table II currently does not. If you need the footnotes for a dense table, hand-transcribe from the saved JPEG (`assets/table_{id}_p{N}_{M}.jpg`) or rerun with a smaller model where the hang doesn't reproduce.

**Page-image fallback for unlocated tables.** When the selected `--table-finder` cannot locate a table's region in the PDF (typical for tables in scanned documents or with extremely borderless layouts), the pipeline searches the markdown context surrounding the table for the table's caption (`Table II`, `Table 3`, etc.) using `TABLE_CAPTION_RE`, then looks up which PDF page hosts that caption text in PyMuPDF's per-page text — preferring matches where `Table N` is line-leading (a real caption) over inline cross-references in body prose. That page is rendered at `PAGE_DPI` (170) and the VLM is given the full page image with `TABLE_PAGE_PROMPT`, which tells the model the page may have multiple tables and to extract only the captioned one (returning `SKIP` if the named table isn't visible). The result is written as a sidecar `assets/table_page{N}_{M}.md` (note the `page` prefix — distinct from the higher-confidence `table_p{N}_{M}.md` crop-based naming). No JPEG is saved; we used the whole page, not a tight bbox crop.

If no caption can be found in the surrounding markdown context, or if the VLM returns `SKIP`, marker's body is preserved and no sidecar is written — the "honest-over-confident" fallback.

**Multi-page table continuation handling (v0.3).** Marker emits a separate markdown table for each page of a "TABLE X. (Continued.)" sequence. Before v0.3, all continuations collapsed to the first page's bbox (the caption-page lookup returned the first occurrence of the heading every time). The fix: process_tables now counts each markdown table's caption-occurrence index, and `_find_caption_page` accepts an `occurrence` parameter to walk through the per-page line-leading "Table X" matches. Each "(Continued.)" header is itself a line-leading "Table X" so it contributes one entry. Result: the i-th markdown table routes to the i-th continuation page, producing distinct JPEG crops and sidecars per page.

For occurrences > 0, an anchor sanity check runs: the chosen candidate's text payload must contain a body-cell anchor from the markdown table (otherwise fall back to the all-page anchor scan). For the dominant single-occurrence case (occ=0), the caption-page lookup is authoritative and the anchor check is skipped — preventing false fallbacks when marker's surya-OCR'd cells and docling's rapidocr disagree on a few characters.

Verified case: `Knudson2013b.pdf` Table X spans pages 1-4 with `(Continued.)` markers. Each continuation now produces its own crop (`table_p1_*`, `table_p2_*`, `table_p3_*`, `table_p4_*`).

**Sideways-table rotation (v0.3).** Wide landscape tables rotated 90° onto a portrait page (e.g. `Millot-OSTI-1763189.pdf` Table S2 page 32) come out of the bbox detector as tall+narrow crops where the rotated header row sits at the LEFT page edge — frequently OUTSIDE the bbox. The detector ratio test (`crop.height / crop.width > PAPER2MD_TABLE_ROTATION_THRESHOLD`, default 2.5) flags these. When detected, hook 1 renders the WHOLE page (instead of the cropped bbox) and rotates it upright per `PAPER2MD_TABLE_ROTATION_DIRECTION` (default `ccw`, meaning the original publisher rotation was CCW; the helper applies the inverse). The rotated whole-page image becomes both the saved JPG asset (so the human-visible asset is upright) and the VLM input, with `TABLE_PAGE_PROMPT` carrying a generic rotation hint and the "extract the named table from this page; ignore other content" framing.

Both env vars are recorded in `run_info.pipeline.{rotate_tables, table_rotation_aspect_threshold, table_rotation_direction}`; toggle with `--no-rotate-tables`.

This path is genuinely slower than crop-based extraction (the VLM processes a whole page rather than a small image), so for corpora full of unlocated tables you'll get better wall-time and quality by upgrading the detector instead. Two upgrades, in increasing order of cost / coverage:

- `--table-finder both` adds Microsoft Table Transformer (TATR) on top of PyMuPDF. Good for borderless tables in text-layer (non-scanned) PDFs.
- `--table-finder docling` uses IBM's TableFormer instead. Strongest detector for scanned PDFs and complex multi-row-header tables; one whole-PDF scan (~36–50 s on a 16-page paper) is amortized across all pages, after which per-page candidate queries are O(1) cache lookups.

`--table-finder docling` is the right choice when you have a scanned-and-OCR'd corpus (Adobe Paper Capture vintage), tables with merged or spanning cells, or tables with multi-line headers that confuse simpler detectors. Docling's table-to-markdown export is structurally faithful but doesn't reinterpret subscripts / superscripts / math / footnote markers as semantic markup — for that the existing VLM crop-rewrite hook (`TABLE_PROMPT` on the located bbox) is what produces clean output. The value-add of docling here is **detection**, not extraction; the VLM still polishes whatever docling locates.

**Default fast path: detector-text mode (no per-table VLM).** Since v0.2 the default behavior is to substitute docling's pre-extracted markdown directly as the table body, bypassing the per-table VLM call entirely. Other VLM hooks (citation, figures, page rescue) still run — only the per-table rewrite is skipped. Tradeoffs:

- **Speed**: per-table VLM call (~1.5–6 min on a 32B model, depending on table density) replaced by zero VLM calls. On a 6-table scientific paper this saves 10–30 min.
- **Quality**: docling outputs structurally faithful markdown but doesn't render `±` as `\pm`, `μ` as `\mu`, or footnote markers (`a`, `b`) as `<sup>a</sup>`. Math-heavy tables come out raw. Acceptable for downstream search/RAG; not ideal for human-readable archives.
- **Coverage**: tables that docling can't locate fall through to marker's markdown unchanged (the page-image fallback also needs the VLM and is disabled in default mode). Keep `--table-finder docling` (the default; strongest detection of the available finders) to maximize the located fraction.
- **Pairing**: under `--table-finder pymupdf`, detector-text substitution is a no-op for most tables (PyMuPDF's candidate text is raw cell content, not markdown — paper2md detects this and keeps marker's body).

**Opting in to the VLM rewrite (`--vlm-tables`).** When subscript / Greek / footnote-marker fidelity matters (LaTeX-heavy tables, small archival batches), pass `--vlm-tables` to switch back to the VLM crop-rewrite path. Pair with `--table-workers 4` to dispatch the per-table VLM calls concurrently; vLLM and LM Studio both batch concurrent requests server-side, so this typically halves the table-phase wall time on multi-table papers.

The legacy `--no-vlm-tables` flag is now a no-op deprecation alias (the new default already matches what it used to select); the runtime warns and continues. It will be removed in a future release.

### Hook 1.5 — orphan-caption table rescue

Catches the **Mode A** table-extraction failure: marker rendered a small or unusually-shaped table as paragraph text (no markdown table syntax), so hook 1 had nothing to anchor on. Canonical example: `root-supplemental.pdf` Table I (1 data row, 5 columns) emitted as a ragged paragraph "QMC EA Expt EA QMC IP Expt IP Mg unbound unbound 7.562...".

After hook 1 finishes:

1. Walk line-leading `Table N.` captions in the markdown body via `TABLE_CAPTION_LINE_RE` (stricter than the inline-tolerant `TABLE_CAPTION_RE`: requires the caption be at the start of a line, optionally bold-wrapped, followed by `.` or `:` and then descriptor text).
2. For each caption, scan the next ~30 lines for a follow-up signal: a markdown-table separator row, an emitted `![Table N (page P)]` image link, or a `[Table N — separate markdown]` sidecar reference. The scan terminates at the next caption boundary so caption *N* doesn't claim caption *N+1*'s image link.
3. Captions without a follow-up are *orphans*. For each orphan: find the PDF page via `_find_caption_page`, render the page image, and call the VLM with the existing `TABLE_PAGE_PROMPT` that hook 1 uses for its page-image fallback (already names the caption verbatim, says "ignore other tables on the page", and has a SKIP escape for wrong pages).
4. On a successful extraction: save a JPEG of the page + a sidecar `.md`, splice an image-link + table body + sidecar-link triple into the body right after the orphan caption, and append a `TableScore` to `report.tables` with `pre_redo_reason="orphan-caption rescue"` so the front-matter records the rescue.

Toggle: `--no-rescue-orphan-tables` (default on). Needs the VLM, already skipped under `--no-vlm`.

Verified empirical impact (`tests/eval_table_extract.py` against `tests/table_extract_truth/` on local vllm + Qwen3-VL-32B):

| Stage | millot_si | root-supplemental | aggregate |
|---|---|---|---|
| Baseline                       | 4/6 (67%)   | 6/8 (75%)   | 10/14 = 71.4% |
| Hook 1 bypass relaxation       | 6/6 (100%)  | 7/8 (87.5%) | 13/14 = 92.9% |
| + Hook 1.5 orphan rescue       | 6/6 (100%)  | 8/8 (100%)  | 14/14 = 100% |

Table I in root-supplemental is now extracted with all 5 columns + 2 data rows + the `**Notes:**` block, column-Jaccard 1.00 vs ground truth.

### Hook 2 — figure captioning (author-caption matching with freeform fallback)

Two paths — the pipeline picks one automatically.

**Matched path (preferred, fires when the markdown has author figure captions):**

1. Extract figure captions from marker's markdown using **five** regex patterns to cover the conventions seen in the test corpus:
   - Nature-style `**Figure N | title...**` (`FIG_CAPTION_RE_PIPE`, pipe-separated; for both main and supplement-numbered Figs).
   - Elsevier-style `**Fig. N.** title...` (`FIG_CAPTION_RE_ELSEVIER`, period-separated, header-only bold).
   - AGU / Nature-Geoscience whole-bold-block `**Figure N. Title.**` (`FIG_CAPTION_RE_BOLD_BLOCK`, where the closing `**` sits *after* the title).
   - Plain unstyled `<span id="page-N-M"></span>Fig. N. Title.` or just `Fig. N. Title.` at line start (`FIG_CAPTION_RE_PLAIN`, used by some Elsevier journals like jacquet's Icarus paper).
   - Nature "Extended Data Fig. N" (`FIG_CAPTION_RE_ED_PIPE` and `FIG_CAPTION_RE_ED_BOLD_BLOCK`); IDs prefixed with `ED` so they don't collide with main-paper numbering.
   The id alternation `(?:S?\d+|[A-Z]\.\d+)` accepts plain integer, S-prefixed supplement, and letter-prefixed appendix forms (`B.13`, `D.14`). Inline body references like `(Extended Data Fig. 1)` are excluded from the main patterns by negative lookbehind. Title length cap is 500 chars; longer captions are silently dropped.
2. For each `![alt](path)` in the markdown (skipping the `Table ...` JPEGs we just inserted):
   - Show the VLM the image plus a numbered list of the extracted captions plus a `0. NONE` option (`PAIR_PROMPT_HEADER`, max_tokens=5).
   - Parse the first integer in the reply.
3. If the VLM picked a figure number: rewrite the alt-text to `Figure N | <title>` (author's own caption), keep the image file.
4. **If the VLM picked 0 / NONE / returned garbage:** remove the `![](path)` link from the markdown AND delete the JPEG from `assets/`. Journal banners, icons, logos, header-rule fragments, and figures marker extracted but can't be captioned disappear here.

**Cross-image post-pass — duplicate resolution (default `dup-detect` strategy):**

After all images are classified, identify any caption assigned to two or more images (a duplicate group). For each group:
- Run a focused YES/NO confirmation per claiming image: "this image and N others were classified as Figure X — confirm whether THIS image specifically depicts that figure." NO answers get demoted.
- If two or more still confirm, tiebreak by page-distance to the caption's list position (proxy for document order).
- The single retained image keeps the contested caption. Demoted images get a **second classification pass with the contested option excluded** — so an image that genuinely depicts a different figure (but happens to look similar to the contested one) can still find its true caption rather than being dropped.

The whole pass iterates up to 3 times so a reclassification that creates a new duplicate (image demoted from Fig 10 lands on Fig 8 which was already taken) gets resolved on the next iteration.

This was validated against ground truth on the project's 6-paper test corpus: aggregate accuracy went from 84.4% (single-pass classifier) to 96.9%, with no regressions on easy cases and the duplicate count dropping from 5 to 1. Override the default with `--figmatch-strategy {single, page-prior, dup-detect, vote}` or combinations like `page-prior+dup-detect`. Truth fixtures for the matcher live in `tests/figure_match_truth/` (one YAML per paper, listing expected figure→image pairings).

**Freeform path (fallback, fires when no author captions are detected or `--no-vlm` is set):**

1. For each image with empty or trivial alt-text, short-circuit if the file is under `CAPTION_MIN_BYTES` (10 KB, likely decoration).
2. Send to the VLM with `CAPTION_PROMPT` (one concise sentence, or `SKIP` if the image looks like a non-figure).
3. On SKIP / empty / sub-threshold: leave the image link with empty alt, file stays on disk.
4. On caption: write it back as the alt-text.

Images that already have meaningful alt-text from marker are left alone on both paths.

**Why two paths?** Author captions are ground truth. When marker extracts them, matching is a classification task (short, deterministic, low-hallucination). When it doesn't, we fall back to open-ended captioning — less accurate, but better than nothing.

### Hook 3 — sparse page rescue (opt-in via `--rescue-sparse-pages`)

**Default OFF since v0.2.** The density-only trigger fires on legitimate-but-short pages (end-of-references tails, figure-dominant pages, blank divider pages) where the text layer is fine, and the rescue output is appended as a new tail section rather than spliced into the body — so even on a true-positive the value-add over marker's existing output is limited. Pass `--rescue-sparse-pages` to enable on scanned PDFs / Adobe-Paper-Capture-vintage OCR corpora where the text layer is genuinely broken.

When enabled:

1. Compute `chars / area_in_points²` for each page using PyMuPDF's text extraction.
2. Flag pages below `SPARSE_CHARS_PER_PT2 = 0.005` chars/pt² (roughly <36% of normal journal-page density).
3. Render each flagged page at 170 DPI and send to the VLM with a "preserve two-column reading order" prompt.
4. **Append** the VLM's output under a `## VLM page rescues` section — not splice. This is intentional: the spine may already contain partial content for that page, and blindly replacing it risks losing good data.

Cost: ~30 s – 2 min per flagged page on a 32B VLM. Always skipped under `--no-vlm`.

**Known false-positive triggers** (planned future work — tighten the trigger to skip these):
- End-of-references pages with whitespace tail
- Pages whose visible content is a single figure already handled by hook 2
- Pages whose visible content is a single table already handled by hook 1 / 1.5

### Post-pass — data-repository link extraction (deterministic; optional API enrichment)

After all body mutations and the hooks above, the pipeline scans the markdown for DOIs and URLs that point at recognized research-data repositories and adds them to the YAML front-matter and `.meta.json` sidecar under a `data:` block. **Always-on, pure regex, no network.**

Recognized repositories (DOI prefix → repo):

| Repository | DOI prefix | Notes |
|---|---|---|
| Zenodo | `10.5281/zenodo.*` | full API summary supported |
| Dryad | `10.5061/dryad.*` | full API summary supported |
| Harvard Dataverse | `10.7910/DVN/*` | full API summary supported |
| Borealis (Canadian) Dataverse | `10.5683/SP*/*` | full API summary supported |
| Generic Dataverse (any other install) | various | URL-anchored catch-all (`*dataverse*/dataset.xhtml?persistentId=...`); summary supported |
| figshare | `10.6084/m9.figshare.*` | full API summary supported |
| OSF | `10.17605/OSF.IO/*` | metadata summary supported (file list omitted) |
| PANGAEA | `10.1594/PANGAEA.*` | full API summary supported (JSON-LD path) |
| ESS-DIVE | `10.15485/*` | link recorded; API requires auth (no summary) |
| Mendeley Data | `10.17632/*` | link recorded; API requires auth (no summary) |
| ICPSR | `10.3886/ICPSR*` | link recorded; no convenient public API |
| CaltechDATA | `10.22002/*` | full API summary supported (Invenio) |

For each match, the `data:` entry has `repository`, `url`, `doi`, `record_id`, plus `confidence: high` if the link sits inside a "Data Availability" / "Code Availability" / "Data Accessibility" / "Data and Code Availability" / "Data Sharing" section header (markdown heading **or** bold paragraph), else `confidence: medium`. Links repeated in references or other sections are deduplicated by canonical DOI.

**Opt-in API enrichment with `--fetch-data-repos`.** When this flag is on, the pipeline does one HTTP GET per unique deposit and appends `title`, `description` (truncated to 600 chars), `license`, `files: [{name, size, format}]`, and `fetched_at` (UTC ISO timestamp). Failures fall through silently — the link itself is still recorded, with `fetch_status` set to `network_error`, `not_found`, `http_error`, `parse_error`, or `not_implemented` (for repos whose APIs require auth). Default timeout is 8 s per call; tune with `--data-repos-timeout SEC`.

Off by default to keep runs hermetic — no surprise outbound network calls during batch ingestion, and no reproducibility drift if a deposit gets updated between two runs of the same paper. `--no-data-repos` disables the post-pass entirely (no scan, no `data:` block).

Example YAML emitted with `--fetch-data-repos` on the project's `millot.pdf` (Dryad-deposited shock-compression dataset):

```yaml
data:
  - repository: dryad
    url: https://datadryad.org/dataset/doi:10.5061/dryad.z08kprr8r
    doi: 10.5061/dryad.z08kprr8r
    record_id: z08kprr8r
    section: Data Availability Statement
    confidence: high
    fetch_status: ok
    title: "Recreating giants impacts in the laboratory: ..."
    description: "Understanding giant impacts requires accurate ..."
    license: "https://spdx.org/licenses/CC0-1.0.html"
    files:
      - name: Millot_GRL_MgSiO3_Dryad.xlsx
        size: 116966
        format: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
    fetched_at: 2026-05-01T15:31:36Z
```

The same content appears in `<stem>.meta.json` under the top-level `data` key (jq-friendly, no YAML parser needed).

### Hook 4 — quality scoring (heuristic, no VLM calls)

After hooks 1–3 run, the pipeline emits a YAML front-matter block summarizing how confident the conversion is. Everything here is derived from signals already collected during the preceding hooks — no extra VLM calls and no re-scanning the PDF. See §11 for the rubric and interpretation.

---

## 6. Known reliability issues

Be skeptical of output in these cases. They are not bugs in this script — they are inherent limits of the underlying models.

### 6.0 PyTorch MPS (Apple Silicon) and surya — why CPU is the default

**Status as of 2026-05: surya's MPS path produces silently corrupted body text on Apple Silicon, AND crashes sporadically on bigger papers. CPU is the default on Apple Silicon as of v0.2.2 — paper2md's `--backend auto` resolves to `cpu` even when MPS is available. `--backend mps` remains as an explicit opt-in for users who understand the trade-off; it logs a WARNING at startup.**

#### What `--backend {auto,cuda,mps,cpu}` actually does

The flag is, mechanically, a single env-var assignment in `main()` before marker imports torch:

```python
backend = "cpu" if args.cpu else args.backend
if backend == "auto":                                  # default
    if torch.cuda.is_available():    backend = "cuda"
    elif torch.backends.mps.is_available(): backend = "cpu"   # silent-corruption guard, see below
    else:                            backend = "cpu"
os.environ["TORCH_DEVICE"] = backend                   # ← the actual effect
log.info("Backend: %s (marker/surya/TATR)", backend)
```

(Earlier paper2md versions resolved Apple Silicon to `mps`. As of v0.2.2 the auto-resolution prefers `cpu` for the reasons documented below; users who want MPS pass `--backend mps` explicitly and see a startup warning.)

Three downstream consumers read `TORCH_DEVICE` and place their tensors on that device:

1. **Marker / surya** — at marker-import time, marker reads `TORCH_DEVICE` and puts surya's layout, text-recognition, equation-recognition, and table-recognition models on that device.
2. **TATR** (Microsoft Table Transformer, used when `--table-finder tatr|both|all`) — paper2md's `TATRFinder.__init__` reads `TORCH_DEVICE` and puts the detector model on the same device.
3. **Docling** (the v0.2 default `--table-finder`) — auto-detects MPS/CUDA itself, *independently* of `TORCH_DEVICE`. So `--backend cpu` does NOT pin docling to CPU; you'll see `INFO Accelerator device: 'mps'` in docling's startup log even when paper2md is otherwise on CPU. This is intentional — docling's MPS path is not affected by surya's bug.

What `--backend mps` does NOT affect:

- **The VLM** (LM Studio, vLLM, OpenAI, Anthropic). Runs as a separate server reachable over HTTP; paper2md is the client. The VLM's hardware is whatever the server is configured to use.
- **PyMuPDF** (`pymupdf`). Pure CPU C library; doesn't touch torch.
- **paper2md's orchestration** (regex passes, JSON writing, frontmatter emission, etc.). Pure Python on CPU regardless.

When MPS is healthy, the speed-up over CPU is concentrated in surya's text-recognition decoder (the OCR pass), which is the wall-time bottleneck for marker — typically 4–5× faster on MPS than CPU on Apple Silicon for that phase.

#### Why MPS is broken — TWO failure modes

The MPS bug class has two distinct surface symptoms. **The silent one is more dangerous than the loud one.**

##### Silent corruption (the dangerous one)

Even when MPS doesn't crash, surya produces materially worse output than the same model on CUDA / CPU. PyTorch's MPS float kernels are not bit-exact equivalents of CUDA / CPU kernels — slight differences in rounding, fused-multiply-add behaviour, and attention softmax precision accumulate over surya's transformer-decoder autoregressive generation. Every token the OCR decoder samples sees a slightly different logit distribution than CUDA / CPU would; most of the time the argmax is the same and the output looks identical, but some of the time the argmax differs and you get:

- **OCR character drops** — single-character corruptions: `"geoscience"` → `"peoscience"`, etc. Not flagged by anything; the text just *looks* like a typo in the source.
- **Reading-order scrambling** — the layout-region scoring picks different region orderings than CUDA / CPU, so the title, author byline, abstract, and body fragments come out interleaved in non-canonical order.
- **Lost equation bodies** — the equation decoder gives up on a long sequence and emits just the bare equation number (`(1)`) with no LaTeX expression.

Empirical on `examples/canup.pdf` (Nature Geoscience, 6 pages, math-heavy), same paper2md / marker / surya versions on both boxes:

| Check                    | Mac MPS (broken)  | Mac CPU (clean) | Spark CUDA (reference) |
|--------------------------|-------------------|-----------------|------------------------|
| `peoscience` OCR typo    | ✗ line 139        | ✓ absent        | ✓ absent               |
| Equation 1 LaTeX body    | ✗ bare `(1)`      | ✓ full LaTeX    | ✓ full LaTeX           |
| Reading order            | ✗ scrambled       | ✓ correct       | ✓ correct              |
| Quality grade            | 0.93 / A          | **0.97 / A**    | 0.97 / A               |
| `elapsed_sec`            | 192.7             | 557.0           | 191.3                  |

Mac CPU output is functionally equivalent to Spark CUDA. Mac MPS is not. The wall-time premium for MPS over CPU (192 s vs 557 s on canup) is real, but it buys silently corrupted output.

This is **insidious** because the artifact looks plausible — the user only catches it on side-by-side comparison with a known-good reference. For an unattended batch, every Mac-produced output would silently degrade.

##### Crash mode (the loud one)

PyTorch MPS also has an allocator bug class where freshly-allocated tensors aren't always zero-initialized. surya's prediction loop reads `input_ids` expecting valid token IDs; instead it reads garbage int64s like `-7430757241614981263`. That value gets passed to a tensor index op, which raises:

```
torch.AcceleratorError: index <huge-negative-number> is out of bounds: 1, range 0 to N
```

The huge negative int is the smoking gun: it's uninitialized memory interpreted as int64. The bug is **sporadic** — depends on which allocation gets recycled garbage. Larger workloads (more pages, longer OCR sequences) run more allocations and hit it more reliably. Empirical:

| Paper | Pages | OCR boxes | MPS crash on surya 0.15.4 (legacy pin) | MPS crash on surya 0.17.1 (current pin) |
|---|---|---|---|---|
| examples/canup.pdf | 6 | ~80 | ✓ completes (but silently corrupt) | ✓ completes (but silently corrupt) |
| examples/root-maintext.pdf | small | ~50 | ✓ completes | ✓ completes (167 s) |
| examples/root-supplemental.pdf | 8 | 207 | ✗ crashes at iter 182/207 | ✓ completes |
| examples/millot_si.pdf | (SI) | many | (untested) | ✓ completes |
| Larger / scanned papers | 10+ | 400+ | ✗ near-certain crash | unknown |

Upstream tracking: [datalab-to/marker#993](https://github.com/datalab-to/marker/issues/993) and [datalab-to/surya#493](https://github.com/datalab-to/surya/pull/493) ("fix: Apple Silicon MPS .max() kernel crash in encoder attention") address the crash flavour. **They do NOT fix the silent-corruption flavour.** Even when those upstream PRs merge and a clean surya release ships, the silent corruption may persist because it's rooted in float-precision differences between Metal and CUDA / CPU kernels, not in PyTorch allocator behaviour.

`PYTORCH_ENABLE_MPS_FALLBACK=1` does **not** help with either flavour. The failing op has an MPS kernel; the fallback only routes ops that aren't MPS-implemented through CPU.

#### Practical takeaway

- **Default Mac runs**: leave `--backend auto` alone. As of v0.2.2 it resolves to `cpu` on Apple Silicon. Output matches CUDA reference; wall time is ~3× a CUDA reference run (e.g. canup.pdf 557 s on Apple Silicon CPU vs 191 s on Spark CUDA — both produce identical body output).
- **Don't pass `--backend mps`**: paper2md will let you, but logs a WARNING. Output may contain silently corrupted text (OCR character drops, scrambled reading order, lost equations). Only useful for small papers where you're side-by-side comparing against a CUDA reference.
- **Don't pass `--backend mps` in batch**: even ignoring the silent-corruption risk, a single OCR-decoder allocation rolling snake-eyes can crash a multi-paper batch run.
- **Keep an eye on surya releases**: the upstream fix list in §6.0.1 below. But note that an upstream surya fix only addresses the crash flavour; the silent-corruption flavour is rooted in PyTorch MPS kernel precision and may persist independently.

#### Pinned stack on Apple Silicon

`environment-mac.yml` pins the SAME exact ML stack versions as `environment-gpu.yml` so paper2md outputs are bit-comparable across Mac CPU and Spark CUDA runs (modulo unavoidable per-machine fields like `run.backend`, `run.vlm_provider`, `run.vlm_model`, and `torch`'s CUDA-build suffix):

```yaml
- marker-pdf==1.10.2
- surya-ocr==0.17.1
- transformers==4.57.6
- pymupdf==1.27.2.3
- pillow==10.4.0
- docling==2.92.0
- torch==2.10.0           # plain PyPI build on Mac (CPU + MPS); +cu130 on Spark
```

The Mac and Spark env files MUST be bumped together; paper2md's reproducibility story across the two boxes depends on the version-string equality. The marker + surya pins are not a workaround for the MPS bug class (the bug is in PyTorch MPS kernels, not in any specific marker / surya release we tested) — they're a reproducibility pin.

History note: earlier paper2md versions shipped `environment-mac.yml` with a conservative `marker-pdf==1.8.5 / surya-ocr==0.15.4` pin in the (mistaken) belief that surya 0.16+ introduced the MPS bug. v0.2.2 promotes the latest pins to the production Mac env after empirically confirming that (a) the MPS bug is in surya's OCR decoder regardless of version, and (b) Mac CPU + the latest pins matches Spark CUDA output exactly on canup.pdf and root-maintext.pdf.

#### 6.0.1 Watching for the upstream fix

⚠ **Before relaxing these pins, check for a fix:**

1. **Watch [datalab-to/surya#493](https://github.com/datalab-to/surya/pull/493)** ("fix: Apple Silicon MPS .max() kernel crash in encoder attention"). As of 2026-04-26 this PR is open, mergeable, and the symptom in its body matches one of our crash variants (the encoder one). It adds a `safe_max_item()` helper that moves to CPU before `.max()` only on MPS (no-op on CUDA/CPU). When this PR merges AND a surya release is tagged on PyPI, that's the candidate to test — but expect to also need a fix to `embed_ids_boxes_images` in the OCR decoder for the bug class we hit on root-supplemental.
2. Check whether [marker#993](https://github.com/datalab-to/marker/issues/993) is closed, and whether its fix landed in a marker / surya release.
3. Check the [surya releases page](https://github.com/datalab-to/surya/releases) for any release notes mentioning "MPS", "Apple Silicon", "layout model", "AcceleratorError", "embedder", "uninitialized", or "input_ids".
4. If in doubt, keep the pins. Test on `examples/root-supplemental.pdf` (the canonical hard case — 8 pages, 207 OCR iterations, reliably exposes the MPS bug).

If you do bump and hit the crash, `--cpu` remains a working (slow) safety net — it's wired in permanently for exactly this scenario. See §3 for the flag.

**Related MPS caveats unchanged by the pin:**

- The `TableRecEncoderDecoderModel` inside surya is always pinned to CPU by surya itself (logged as a startup warning). This is surya's own MPS workaround for a *different* bug class and is fine.
- TATR (the Table Transformer we load for `--table-finder tatr|both`) runs on MPS without issue under both the legacy and current marker / surya pins — verified on jacquet.
- Docling's MPS path is independent and stable — it auto-detects MPS regardless of `--backend` and we have no reports of failures.

### 6.1 marker / spine

- **Borderless or whitespace-only tables** may not be recognized as tables at all. They come out as ragged paragraphs.
- **Image extraction is non-deterministic.** Running the same PDF twice can produce different image sets (observed: 3, 5, 6, 8 images on four identical runs of canup.pdf). Which panels of a multi-panel figure get extracted is also unstable run-to-run. Hook 2's drop path absorbs the noise but you may see figure counts vary between runs.
- **Multi-panel figures extract as multiple images.** A figure labeled "Figure 6" with sub-panels a/b/c may emerge as two or three `_page_N_Figure_*.jpeg` files. Hook 2's matched path will then write `![Figure 6 | title](...)` alt-text for each copy. Not wrong, just redundant.
- **Multi-column figures** (a figure that spans both columns) sometimes break reading order — text after the figure occasionally gets reordered.
- **Complex merged-cell table headers** (e.g., a group header spanning three sub-columns) tend to come out flattened or mangled. This is the single most common mode of failure on journal PDFs.
- **Rotated pages** (landscape supplementary tables) sometimes extract as scrambled text.
- **Footnotes** with reference marks are inconsistently handled — sometimes inline, sometimes dropped, sometimes appended at end.
- **Reference sections** are sometimes truncated or merged into the last paragraph of the body.

### 6.2 PyMuPDF table finder (`--table-finder pymupdf`, lightweight)

> The default since v0.2 is `--table-finder docling`; see §6.2c. PyMuPDF stays available as the lightweight built-in option for ruled-line tables.


- `page.find_tables()` misses tables without visible grid lines.
- It can also produce false positives on figure legends laid out in grid-like columns.
- `table.extract()` sometimes returns cells in the wrong order if the underlying text runs are reversed — which then breaks the anchor-token match.

### 6.2b Microsoft Table Transformer (`--table-finder tatr` or `both`)

- First invocation downloads ~110 MB of weights and takes a few seconds to initialize on MPS.
- Detection-only: TATR returns bounding boxes, not cell structure. Structure is handled downstream by the VLM on the crop, which is the intended division of labor.
- Can produce false positives on dense two-column prose that visually resembles a wide table. The anchor-token match in `find_pdf_table()` generally filters these out, but occasionally a long run of coincidentally-matching text will send the VLM a paragraph instead of a table. You'll notice: the VLM returns prose or a tiny malformed table.
- Detects tables that span both columns or overflow page margins more reliably than PyMuPDF; also picks up rotated (landscape) supplementary tables.
- Competes with marker and the VLM for unified memory. On 128 GB this is fine; on 32 GB, unload one of them if you OOM.
- `both` mode lists PyMuPDF candidates first, so when both detectors find the same table, PyMuPDF's tighter bbox wins the match.

### 6.2c IBM Docling (`--table-finder docling`, **DEFAULT**)

- First invocation downloads ~500 MB of model weights (TableFormer + DocLayNet + RapidOCR). Subsequent runs reuse the cache.
- Whole-PDF detection runs once per PDF on the first per-page candidate call and is cached for subsequent pages. Cost is roughly constant per PDF (~7–10 s on a 16-page paper on the Spark; longer on CPU-only machines) regardless of how many tables are present. PyMuPDF and TATR both run per-page and skip pages with no candidates; for very short documents docling's amortized cost may not pay back, but on mid-to-large papers it's competitive.
- TableFormer recognizes spanning cells, multi-row headers, and merged cells more reliably than TATR's detection-only output. Combined with the VLM crop-rewrite hook (which reads the located region as an image), this produces semantically clean markdown — subscripts, superscripts, math, footnote markers — that docling's own export does not produce on its own.
- Strong on scanned PDFs (Adobe Paper Capture–style OCR) where PyMuPDF finds nothing and TATR misses busy tables. Verified to locate all 5 real tables in a 16-page 1962-vintage J. Appl. Phys. scan that defeated both other detectors.
- Caveats:
  - **Matching layer is fragile on scanned PDFs.** `find_pdf_table()` uses an anchor-substring search: it takes a cell value from marker's markdown and looks for it in the candidate's extracted text. When marker's OCR and docling's OCR diverge (different OCR engines yielding different normalized cells on the same scanned page), the substring match can fail even though docling correctly located the table. Those tables fall through to the page-image fallback. On a clean text-layer PDF this isn't an issue; on a busy scan, expect 1–2 docling-located tables to be claimed by marker out of ~5 detected.
  - **Docling's own table export is raw.** If you bypass the VLM rewrite hook for any reason (`--no-vlm` etc.), docling's structure-faithful but visually-literal markdown shows up directly — subscripts as text, footnote markers as bullet glyphs.
- The `--table-finder both` flag is unchanged; it remains PyMuPDF + TATR.

### 6.2d Three-finder chain (`--table-finder all`)

- Chains PyMuPDF + TATR + Docling in that order. First-match-wins inside `find_pdf_table`, so the cheapest detector gets first shot at every table; only tables PyMuPDF and TATR both miss reach docling.
- Best for unknown / mixed corpora where you don't want to commit to a single detector. Loads all three model dependencies (~30 s extra at first run).
- Caption-page bypass works against all three: when a markdown table has a caption AND any of the chained detectors reports exactly one bbox on that page, the bypass takes it.
- Memory: PyMuPDF is small, TATR is ~110 MB, Docling adds ~500 MB. On 32 GB unified memory leave the VLM at 7 B; on 64+ GB you can run a 32 B VLM alongside.

### 6.2e Concurrent table VLM calls (`--table-workers`)

Independent of which detector you pick, AND independent of which layout source. The per-table VLM rewrite (and the page-image fallback when it fires) are independent network calls; with `--table-workers N`, paper2md dispatches up to N concurrently via a thread pool. The default is `1` (serial). All three layouts — `marker`, `mineru`, and `hybrid` — honor the flag, including under `--vlm-tables-force` where every table calls VLM.

- vLLM and LM Studio both batch concurrent requests server-side, so 4 workers typically cuts the table phase wall time roughly in half on a paper with ≥4 tables. Verified on `examples/wackerle.pdf` (16-page scanned scientific paper, 6 tables, Qwen3-VL-32B on vLLM): **32 min serial → 16 min with `--table-workers 4`**, same 6/6 located + JPEG sidecars, quality went from 0.96/A to 0.98/A. The bottleneck moves to the slowest single table (a dense 30 × 11 crystal-data table that takes ~9 min on its own).
- The bottleneck moves to the slowest single table when N ≥ table-count. Setting N higher than your VLM server's effective batch size adds queuing without speedup.
- **Interaction with VLM model size.** Big models (32 B-class on a 64 GB+ GPU) have enough headroom for vLLM's continuous-batching scheduler to actually run requests in parallel — `--table-workers 4` on Qwen3-VL-32B reliably halves table-phase wall time. Small models (7 B / 8 B-class on a single consumer GPU, or LM Studio with one model loaded) have a shallower internal batch and effectively serialize concurrent requests; here `--table-workers > 2` adds no speed and just lengthens client-side timeouts. Recommended: `4-6` on 32 B + vLLM with `--gpu-memory-utilization 0.80`; `1-2` on LM Studio or a 7 B model on a single GPU.
- **Interaction with `--table-finder all`.** The chained finder loads PyMuPDF + TATR (~110 MB) + Docling (~500 MB) and can issue more page renders during phase 1. With `--table-workers > 1`, phase 1 (locate + render) is still serial — only phase 2 (the VLM calls) parallelizes. `--table-finder all` and `--table-workers 4` compose cleanly; the finder's overhead is amortized across all tables in the same paper.
- The flag does NOT parallelize tasks across papers — that's `--workers` (batch mode). The two are orthogonal: `--workers 4 --table-workers 4` runs four papers in parallel, each dispatching four table calls in parallel. Watch GPU memory: on 64 GB you want `--workers 4 --table-workers 2` or `--workers 2 --table-workers 4`, not both at 4. Marker's serialization (one paper at a time) caps real concurrency anyway.

### 6.3 VLM — any model

- **Hallucination.** The VLM will sometimes invent plausible-but-wrong numbers, especially in tables with many similar-looking values. Always spot-check numerical cells.
- **Table structure drift.** The VLM may silently merge or split columns, change header phrasing, or drop units from cell contents.
- **Caption generalization.** Figure captions are often correct but generic ("a scatter plot showing two variables"). They should be treated as alt-text hints, not paper content.
- **Math rendering.** VLMs transcribe LaTeX inconsistently — sometimes `\mu`, sometimes `μ`, sometimes both in the same table. Prefer marker's math output; only rely on VLM math inside rescued pages.
- **Tool calls on local VLMs** are not reliable even when the API accepts them. This script deliberately does not use tool calls — every hook is a pure text-in/text-out invocation.

### 6.4 Matching (hook 1 table anchor)

- If the longest body cell is a short, generic token (e.g., "n/a", a single number), the match may land on the wrong table or fail entirely. In that case the table is left as-is and no sidecar `.md` is written.
- If two tables on different pages share identical content (rare but happens with duplicated summary tables), the first match wins and may not correspond to the right spot in the markdown.
- When the anchor match fails silently on several tables of a document, try `--table-finder both` (adds TATR) or `--table-finder docling` (uses IBM TableFormer instead). TATR often locates tables whose body text didn't overlap cleanly with PyMuPDF's bbox; docling is a stronger detector still and is the right choice on scanned PDFs (see §6.2c, including the matching-fragility caveat there).
- A table that no detector can locate falls through to the page-image fallback, which extracts it from a whole-page render rather than a tight crop. Sidecar names with the `page` prefix (`table_page{N}_{M}.md`) signal this — slightly lower trust than crop-based sidecars but typically acceptable. If even the page-image path fails (caption can't be found nearby, or VLM returns `SKIP`), marker's body markdown is preserved without a sidecar.

### 6.5 Figure–caption matching (hook 2)

- **Caption regex coverage.** Five patterns are supported:
  - Nature-style `**Figure N | title**` (pipe-separated)
  - Elsevier-style `**Fig. N.** title` (header-only bold, period inside)
  - AGU / Nature-Geo whole-bold `**Figure N. Title.**`
  - Plain `<span id="page-N-M"></span>Fig. N. Title.` (with optional anchor) at line start
  - Nature "Extended Data Fig. N | Title" / "**Extended Data Fig. N. Title.**"

  All patterns accept supplement-numbered (`S1`, `S12`) and appendix-numbered (`B.13`, `D.14`) figure ids; figure IDs flow through the pipeline as strings (`"1"`, `"S1"`, `"ED1"`, `"B.13"`). Inline body references like `(Extended Data Fig. 1)` are excluded by negative lookbehind. Journals with other conventions (`Figure N\ntitle` on two lines, HTML-like caption tags, etc.) will yield zero extracted captions and the pipeline will silently fall back to the freeform path. Sign to watch for: log line `Extracted 0 figure caption(s) from markdown` when you expected matches.
- **Line-broken captions don't match.** If marker introduces a newline inside a caption's title (common on narrow columns), the regex stops at the line break and the title truncates. That figure is still extractable — the number is captured — but the title in the alt-text may be a few words short.
- **Figures whose captions aren't extracted get dropped.** If the paper has 12 figures but the regex only catches 8, the 4 unmatched figures' images will be classified as NONE and deleted. Workaround: run once, spot the log to see which figure numbers were extracted, and either edit `FIG_CAPTION_RE_PIPE` / `FIG_CAPTION_RE_ELSEVIER` for that journal's caption style or run `--no-vlm` to preserve every image with empty alt-text.
- **Dropped images are deleted from disk.** Once hook 2 decides an image isn't a figure, the JPEG is `unlink()`'d. The output directory is designed to be rebuildable — re-running restores anything lost — but if you've manually edited files in `out/assets/` between runs, back them up first.
- **Multi-panel match collisions.** When marker extracts multiple panels of the same figure, each panel independently gets matched back to the same caption; you'll see `Matched _page_4_Figure_1.jpeg -> Figure 6` and `Matched _page_5_Figure_1.jpeg -> Figure 6` in the same run. The markdown ends up with two links bearing identical alt-text. Cosmetic, not wrong.

---

## 7. When the script fails — what to try

Match the symptom, apply the fix.

### macOS: "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized" + Abort trap 6

paper2md auto-sets `KMP_DUPLICATE_LIB_OK=TRUE` on Darwin at module import to work around this. If you still see it, your shell or wrapper script is unsetting the var. Restore with:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Root cause: macOS conda env + `pip install "mineru[core]"` ends up with two copies of `libomp.dylib` loaded in the same process (PyTorch / Pillow / pymupdf via OpenBLAS load one; PaddlePaddle + opencv-python brought in by MinerU load another). LLVM OpenMP aborts to prevent silent miscompilation in mixed-runtime parallel reductions. paper2md's process model doesn't trigger the failure mode the warning is guarding against (Paddle runs in a separate MinerU subprocess, not in-process with PyTorch), so the override is safe in practice.

If you want to fix the underlying duplicate instead of overriding the check: the cleanest path is `conda install -c conda-forge paddlepaddle` so it shares the env's `llvm-openmp` instead of pip's vendored copy. Often messy with paper2md's exact pin set — most users keep the env-var workaround.

### "torch.AcceleratorError: index ... is out of bounds" inside surya

- The error is in surya's OCR decoder running on MPS. Either you're using `--backend mps` explicitly (which logs a startup WARNING about silent corruption), or your paper2md is older than v0.2.2 and auto-resolved to MPS. Re-run with `--backend cpu` (or upgrade to v0.2.2+, which auto-resolves to CPU on Apple Silicon).
- If you deliberately relaxed the pins (e.g. testing a post-fix release), first check [marker#993](https://github.com/datalab-to/marker/issues/993) for resolution status. Until it's fixed upstream, either re-pin or use `--cpu` as the (slow) safety net.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` covers some but not all of the surya MPS bugs (equation encoder yes, layout encoder no — MPS returns corrupt data, not rejected ops).
- Note: `PYTORCH_ENABLE_MPS_FALLBACK` must be set as a shell env var before Python starts; setting it from inside Python doesn't work because PyTorch reads it once at MPS backend init.

### "LM Studio connection refused / timeout"

- Confirm LM Studio's server is running and the model is loaded (the main window shows a green indicator).
- Confirm the model ID in LM Studio matches your `VLM_MODEL` env var exactly.
- Test the endpoint: `curl http://localhost:1234/v1/models`.
- If using a remote host, check firewall and `LM_STUDIO_URL`.

### "VLM preflight failed: ... not reachable at ..." (exit code 2)

The preflight check ran and the configured VLM endpoint didn't respond within 5 seconds.

- The error line names the endpoint URL and the underlying exception. Most common cause: the wrong provider was selected for the host. On the Spark, `--provider vllm` (auto-paired when `--backend` resolves to cuda); on Apple Silicon, `--provider lmstudio` (auto-paired for mps/cpu). Pass `--provider X` explicitly if you've overridden `$VLM_PROVIDER`.
- Verify the server: `curl <endpoint>/models`. For vLLM the default is `http://localhost:8000/v1`, override with `$VLLM_BASE_URL`; for LM Studio `http://localhost:1234/v1`, override with `$LM_STUDIO_URL`.
- If you genuinely want to skip the preflight (e.g., because the VLM is on a slow network and 5s is too tight): `--skip-vlm-check`. Note: a real failure in a hook will still surface — preflight is just a fast-fail convenience.
- If you want to skip the VLM entirely (marker-only output, no citation/figures/page rescue): `--no-vlm` (which also skips the preflight).

### "marker fails to load / OOM at startup"

- You probably don't have `marker-pdf` installed — not `marker`. Re-run `pip install marker-pdf`.
- First run downloads weights; make sure the HF cache has disk space.
- If layout model OOMs, close LM Studio temporarily — marker's models and the VLM compete for unified memory.

### Tables come out mangled (merged cells, wrong column count)

- Run with `-v` and check whether hook 1 is firing on those tables. If the heuristic doesn't catch them, tighten it: edit `table_is_suspicious()` to flag more aggressively (e.g., drop the `>40% blank` threshold to 20%).
- If hook 1 fires but `find_pdf_table()` logs "no matching PDF table found", the table is likely borderless. Re-run with `--table-finder both` — Microsoft Table Transformer detects tables from visual layout rather than grid lines.
- If `--table-finder both` still misses a table, it's genuinely hard (e.g., cells separated only by whitespace in a figure caption). Workaround: open the PDF, manually snip the table region as an image, and call the VLM on it separately.
- For consistently bad tables across a corpus, consider swapping the spine to docling (TableFormer is stronger on merged cells) and keeping the same VLM hooks around it. This is a ~50-line change; the hook code is spine-agnostic.

### Citation line is wrong, missing, or has the wrong format

- **Missing entirely**: the VLM returned SKIP (not enough metadata on page 1 — happens with preprints, tech reports, or heavily-stripped PDFs), or the call failed (check log for `VLM returned no citation response` / `VLM declined to synthesize citation`). Re-running often helps; the VLM output is non-deterministic.
- **Wrong author order, wrong journal, wrong year**: the VLM misread page 1. The most common causes are (a) a journal-banner image of a different paper appearing in the header, which confuses the model, and (b) very small type on page 1. Verify manually against the PDF; re-running may fix it. If it's persistent on a particular journal, tighten `CITATION_PROMPT` with an example reference in that journal's style.
- **Wrong style** (e.g., Nature style on an Elsevier paper): the VLM picked the wrong convention. Usually still readable; fix manually or add a journal-family hint to the prompt for that corpus.
- **DOI missing** when you know it's on the page: the VLM either didn't see it or decided the link wasn't worth extracting. No clean fix other than re-running or copying the DOI in manually.
- **To disable entirely** on a batch where citations come from an external source: pass `--no-citation`.

### Figure captions are generic, wrong, or missing

Three distinct symptoms; each has its own fix:

- **Generic / vague** (e.g., "a graph showing two variables"): the pipeline took the freeform fallback path — no author captions were extracted from the markdown. Confirm by searching the log for `Extracted N figure caption(s)`; if it says 0, either the paper uses an unsupported caption convention, or marker's output isn't emitting caption blocks at all.
- **Wrong figure / hallucinated content**: on the *freeform* path this is Qwen-VL inventing text on a non-figure crop. Bump up to 72B, tighten `CAPTION_MIN_BYTES`, or make sure the matched path is firing (see above). On the *matched* path this almost never happens because the VLM's job is reduced to choosing a number from a short list.
- **Figures missing entirely from output**: the matched path dropped them. Check the log for `Dropped <file> (no caption match, deleting)` — either their captions weren't captured by the regex, or the VLM classifier said NONE. For the first case: inspect the paper's caption style and widen `FIG_CAPTION_RE_*`. For the second case: run with `--no-vlm` to keep every image.

Other knobs:
- Switching to Qwen2.5-VL-72B is a generic quality bump for both paths; biggest effect on the freeform path.
- If marker extracted only part of a figure (e.g., missed a legend), the classifier may still match correctly — it's identifying "which figure," not captioning.
- To disable captioning entirely on a run, comment out the `caption_figures` call in `convert()`.

### Page rescue appendix is bloated or hallucinated

- Current default is `SPARSE_CHARS_PER_PT2 = 0.005`, tuned for figure-heavy scientific papers. If your corpus is predominantly prose and pages are being flagged as sparse that shouldn't be, drop the threshold further (e.g., 0.003). If genuinely broken pages (image-only scans, missing text layer) aren't being flagged, raise it (0.010–0.015).
- If rescued pages contain hallucinated content, the VLM is filling in gaps. Switch to a stronger model (Qwen-VL-72B), or disable hook 3 entirely for that document.
- Rescued content is appended, not spliced — it's safe to delete the `## VLM page rescues` section from the output if it's unhelpful.

### Reading order scrambled in two-column output

- This is a marker-level failure. Options:
  - Re-run with `--no-vlm` first to confirm it's not a hook introducing the issue.
  - Try docling or MinerU on that specific paper.
  - As a last resort, run hook 3 on every page (edit `sparse_pages()` to return all indices) and compare the VLM's per-page output to marker's output manually.

### Math is broken

- marker uses LaTeX delimiters `$...$` / `$$...$$`. If math comes out as literal text, the PDF likely has math encoded as glyph outlines rather than characters — marker can't recover it. In this case, use MinerU as the spine (better formula handling) or run hook 3 on math-heavy pages.

### Whole script errors out mid-document

- Check the last log line — usually it's a specific page/table that tripped something.
- Re-run with `-v` to get the full trace.
- As a workaround, split the PDF with `pdftk` or `qpdf` and process pages in batches; combine the markdown afterward.

---

## 8. Tuning for your corpus

The heuristics in `src/paper2md.py` are starting points. After running on ~10 representative papers, review the output and adjust:

| Knob | File location | What it controls |
|------|---------------|------------------|
| `SPARSE_CHARS_PER_PT2` | top of file | how aggressively hook 3 flags pages (default 0.005, tuned for figure-heavy scientific papers) |
| `CAPTION_MIN_BYTES` | top of file | size threshold below which hook 2's freeform path short-circuits on decoration (default 10 KB; only used in the freeform fallback, not the matched path) |
| `table_is_suspicious()` thresholds | function body | post-VLM diagnostic only now — reported in YAML, no longer gates VLM |
| `FULL_PAGE_TABLE_FRAC` | top of file | bbox/page-area ratio above which a table is "full page" and skipped for JPEG |
| `TABLE_JPEG_QUALITY` | top of file | JPEG quality for saved table crops |
| `CROP_DPI` | top of file | quality vs speed of table crops |
| `PAGE_DPI` | top of file | quality vs speed of page rescues |
| `FIG_CAPTION_RE_PIPE`, `FIG_CAPTION_RE_ELSEVIER` | top of file | regexes that decide which captions hook 2's matched path sees — edit if your journal uses a non-standard caption convention |
| `TABLE_PROMPT`, `CAPTION_PROMPT`, `PAIR_PROMPT_HEADER`, `CITATION_PROMPT`, `PAGE_PROMPT` | constants | phrasing and strictness of VLM instructions |
| `--no-citation` flag | at runtime | skip hook 0 (citation synthesis); saves one VLM call per PDF |
| `--supplement FILE` flag | at runtime | also convert a supplementary PDF; writes a separate markdown file that shares the main paper's `assets/` folder (`si_` prefix on supplement assets) |
| `--hdf5` flag | at runtime | additionally write `<main-stem>.h5` bundling main + supplement markdown + every file under `assets/`. Loose files are still produced. See §12 |
| `--text-only` flag | at runtime | skip marker/surya; have the VLM convert each page directly. No assets folder, no layout analysis (VLM handles layout). Best for text-indexing / RAG |
| `--provider {lmstudio,openai,anthropic}` | at runtime | choose VLM provider. Defaults to `$VLM_PROVIDER` or `lmstudio`. See §13 |
| `$VLM_PROVIDER`, `$OPENAI_API_KEY`, `$ANTHROPIC_API_KEY` env vars | at runtime | provider selection + API keys. See §13 |
| `VLM_MODEL` env var | at runtime | quality vs speed of every VLM call |
| `--table-finder` flag | at runtime | `docling` (default; strongest detection — borderless, multi-row headers, scanned), `pymupdf` (lightweight, ruled-line only), `tatr` (borderless in vector-text PDFs), `both` (pymupdf + tatr), or `all` (chained, first-match-wins) |
| `--vlm-tables` flag | at runtime | opt INTO the per-table VLM crop rewrite. Default off (detector-text mode) since v0.2 |
| `--rescue-sparse-pages` flag | at runtime | opt INTO hook 3 (sparse-page VLM rescue, see §5). Default off since v0.2; useful on scanned / Adobe-Paper-Capture-vintage PDFs where the text layer is genuinely broken |
| `--cpu` flag | at runtime | force marker/surya/TATR off MPS onto CPU (MPS-bug workaround; no VLM effect) |
| `TATRFinder(detect_dpi=, threshold=)` | class ctor | TATR detection resolution and confidence floor |
| `PAGE_REFERENCE_DENSITY` | top of file | chars/pt² at which a page scores 1.0 in hook 4 (auto-computed as 2× `SPARSE_CHARS_PER_PT2`) |
| `_score_table`, `_score_figure`, `_score_page` | functions | per-artifact scoring rubric used by hook 4 |
| `--quality-threshold` flag | at runtime | non-zero exit below this overall score |
| `--no-quality` flag | at runtime | suppress the YAML front-matter entirely |

If you find yourself repeatedly editing these for the same journal, save the config as a wrapper script or a `.env` file.

---

## 9. Things this script deliberately does not do

- **No tool calling / function calling.** Local VLMs are unreliable at structured tool use. Every hook is a plain text-in/text-out call.
- **No automatic splicing of page rescues.** Rescues are appended as a clearly-marked appendix so you can review them before merging.
- **No OCR on scanned PDFs.** This pipeline assumes a text-layer PDF. For scans, run OCR first (e.g., `ocrmypdf`) and then pass the OCR'd PDF to this script.
- **No reference parsing.** The references section is preserved as text but not parsed into structured citations. Use a dedicated tool (GROBID, anystyle) for that.

---

## 10. Future improvements worth considering

- **Swap the spine to docling** for corpora dominated by complex tables.
- **Add a low-confidence-block detector** by reading marker's JSON output (when exposed) and checking layout-model confidence scores per block instead of the current heuristics.
- **Batch VLM calls** via async `openai` client — would cut end-to-end time roughly in half on a 20-page paper.
- **Add a diff view** that shows marker's original table/figure next to the VLM-rewritten version so you can accept/reject per block.
- **Add a TATR structure-recognition pass** (`microsoft/table-transformer-structure-recognition-v1.1-all`) after detection, to get cell-level bboxes that could be fed to the VLM as a grid hint — potentially more reliable than relying on the VLM to infer structure from the crop alone.
- **Optional VLM cross-check for scoring** — run the page-level VLM on a small random sample of non-sparse pages and compare word overlap with marker's output. Would catch silent marker failures that heuristics miss. Proposed flag: `--quality-check-sample N`.
- **JSON sidecar** (`paper.quality.json`) mirroring the front-matter, for corpus-level aggregation.
- **Move matched images next to their captions** in the output markdown. Right now we keep marker's placement; a pairing-aware pass could ensure every figure sits immediately above its `**Figure N | ...**` block for better human readability.
- **Dedup multi-panel matches.** When two images both match Figure N, label them `Figure N (panel a)`, `Figure N (panel b)` instead of repeating identical alt-text.
- **Discover tables marker missed.** Iterate `finder.candidates()` directly and inject new markdown blocks at page boundaries for tables not already in marker's output. Would particularly help with borderless tables in Elsevier-style papers.
- **Additional caption regexes** for journal conventions beyond Nature-pipe and Elsevier-period (some journals use `Figure N. Title` without any bold, or `FIG. N` in all caps).
- ~~HDF5 bundle output.~~ **Landed** via `--hdf5` (see §12).

---

## 11. Quality rating

Every conversion emits a YAML front-matter block at the top of the output markdown summarizing how confident the pipeline is in its own output.

### 11.1 What the score means (and does not mean)

The score measures **pipeline confidence**, not correctness. It answers "how clean was the conversion path?" — not "is the content accurate?" Concretely:

- **A high score means no hooks flagged anything.** Tables located, figures captioned, pages dense, no sparse-page rescues needed.
- **A low score means the pipeline had to paper over weak extraction** — suspicious tables, missing figure files, sparse pages.
- **Silent hallucinations by marker or the VLM cannot be detected by heuristics.** A VLM can confidently produce a well-formatted table with wrong numbers; that table scores 1.0. Always spot-check numerical content in tables and keep the side-by-side JPEG of any table you cite.

Treat the score as a triage signal: A/B means "probably safe to skim," C means "scan the flagged artifacts," D/F means "assume this needs manual review."

### 11.2 Front-matter shape

```yaml
---
run:
  command: "src/paper2md.py paper.pdf -o out/"
  hostname: "dgx-spark"
  elapsed_sec: 1063.5
  vlm_provider: vllm
  vlm_model: "Qwen/Qwen3-VL-32B-Instruct"
quality:
  overall: 0.91
  grade: A
  vlm_enabled: true
  note: "Scores reflect pipeline confidence, not factual correctness."
  tables:
    - index: 2
      page: 9
      score: 1.00
      located: true
      jpeg_saved: true
      vlm_redone: true
      matched_table: "1"
      sidecar: "table_1_p9_2.md"
    - index: 3
      page: 10
      score: 0.60
      located: true
      jpeg_saved: true
      pre_redo_reason: "inconsistent column count (5..8)"
      vlm_redone: true
      post_redo_reason: "inconsistent column count (5..7)"
      matched_table: "2"
      sidecar: "table_2_p10_3.md"
  figures:
    - filename: "figure_1_p2.jpg"
      score: 1.00
      caption_produced: true
      caption_length: 12
      matched_figure: "1"
    - filename: "_page_0_Picture_5.jpeg"
      score: 0.80
      caption_produced: false
      caption_length: 0
      dropped: true
  pages:
    - page: 1
      char_density: 0.0137
      sparse: false
      rescued: false
    - page: 7
      char_density: 0.0030
      sparse: true
      rescued: true
---
```

**Note (since 2026-05):** the per-page `score` field was removed and the
pages bucket no longer contributes to `quality.overall`. `char_density`
is a content-type proxy (figure-heavy vs text-heavy), not an
extraction-quality signal, and was systematically dragging
figure-heavy supplements down. Pages are now diagnostic-only —
`char_density`, `sparse`, and `rescued` flags are emitted for
inspection, but `overall` is computed from `tables` + `figures`
buckets alone (with `references` available as instrumentation,
not yet wired into `overall`).

New fields since this guide was last revised:
- `tables[].sidecar`: basename of the per-table `.md` file under `assets/`. Omitted when no sidecar was written (i.e., table not located).
- `figures[].matched_figure`: which author figure number (Figure N) the classifier paired this image with. Omitted in the freeform fallback and on dropped images.
- `figures[].dropped`: present and `true` when hook 2 classified the image as non-figure and deleted the JPEG from disk. Mutually exclusive with `matched_figure`.

The `run:` block (peer to `quality:` and `copyright:`) records reproducibility metadata for the conversion: the exact CLI invocation (shell-quoted so it can be copy-pasted), the `hostname` the run executed on (`socket.gethostname()`), the resolved VLM provider and model after env-var/auto-pair resolution (so you can tell which model produced the output even when `$VLM_MODEL` differs across machines), and `elapsed_sec` covering the convert() phase (marker + hooks + scoring + frontmatter emit; metadata pre-pass and HDF5 bundle write are excluded — both are typically sub-second). Useful for batch-mode forensics: jq-pipe the manifest to find the slowest 10% of files, group by `vlm_model` to compare model behavior across hardware, etc.

Every block parses as standard YAML — pipe it through `yq`, `python -m yaml`, or any front-matter-aware renderer.

### 11.3 Scoring rubric

**Per table** (`_score_table`) — VLM rewrite now runs on every located table; `pre_reason` is diagnostic only.

| Situation | Score |
|-----------|-------|
| Located + VLM rewrite succeeded + post-rewrite clean | 1.00 |
| Located + VLM rewrite succeeded + post-rewrite still suspicious | 0.60 |
| Located + VLM rewrite failed/skipped (`--no-vlm`) + marker markdown looks clean | 0.80 |
| Located + VLM rewrite failed/skipped + marker markdown looks suspicious | 0.40 |
| Not located + text-only VLM cleanup + post-rewrite clean | 0.85 |
| Not located + text-only VLM cleanup + post-rewrite still suspicious | 0.55 |
| Not located + no VLM / VLM returned SKIP + marker markdown looks clean | 0.70 |
| Not located + no VLM / VLM returned SKIP + marker markdown looks suspicious | 0.30 |

**Per figure** (`_score_figure`) — scoring now branches on the path (matched vs freeform):

| Situation | Score |
|-----------|-------|
| Matched path: image classified and matched to an author caption | 1.00 |
| Matched path: image classified as NONE and deliberately dropped | 0.80 |
| Freeform path: file on disk + caption ≥ 5 words | 1.00 |
| Freeform path: file on disk + caption < 5 words or empty | 0.70 |
| Freeform path: file on disk + VLM disabled | 0.80 |
| File missing (either path) | 0.30 |

A dropped figure at 0.80 is *not* a penalty for the pipeline — it reflects correct refusal to caption a non-figure. A high figure score isn't "every image kept"; it's "every image either kept with an author caption OR correctly recognized as decoration."

**Per page** (`_score_page`)

- Base = `density / PAGE_REFERENCE_DENSITY`, clamped to `[0, 1]`. Reference density is `2 × SPARSE_CHARS_PER_PT2` ≈ 0.030 chars/pt² (a typical well-extracted journal page).
- If sparse and rescued by hook 3 → floor at 0.70.
- If sparse and not rescued → floor at 0.30.

**Overall**

- Weighted mean: pages 0.5, tables 0.3, figures 0.2.
- Missing categories (e.g., a paper with no figures) have their weight redistributed across the others.

**Letter grade**

| Overall | Grade |
|---------|-------|
| ≥ 0.90 | A |
| ≥ 0.80 | B |
| ≥ 0.70 | C |
| ≥ 0.60 | D |
| < 0.60 | F |

### 11.4 Using the score in batch pipelines

```bash
# Gate a corpus conversion: non-zero exit if any paper scores below 0.80
for pdf in papers/*.pdf; do
  python src/paper2md.py "$pdf" -o "out/$(basename "$pdf" .pdf)" \
    --quality-threshold 0.80 || echo "NEEDS REVIEW: $pdf"
done

# Aggregate scores across a corpus (requires yq)
for md in out/*/*.md; do
  score=$(yq -r '.quality.overall' "$md")
  grade=$(yq -r '.quality.grade' "$md")
  echo "$score $grade $md"
done | sort -n
```

### 11.5 Known limits of the scoring

- **Heuristics only.** Silent errors in the content (wrong numbers in a well-formatted table, plausible-but-wrong VLM captions) are invisible to scoring.
- **Per-figure scoring doesn't inspect image quality.** A caption on a blank-ish image still scores 1.0 on the freeform path if the VLM returns a ≥5-word sentence, and 1.0 on the matched path if the classifier picks a figure number.
- **Matched-path score of 1.0 doesn't confirm the match was correct.** It confirms the VLM *chose* a valid caption number. If the classifier guessed wrong between two visually similar figures, the score is unchanged. Spot-check alt-text against the figures when that matters.
- **Page-level scores were removed (2026-05).** `pages[].char_density` is still emitted but does not contribute to `overall`. Reading-order problems on dense pages are invisible to the scoring entirely; pair scoring with visual spot-checks on long papers.
- **`--no-vlm` runs cap figure scores at 0.8** (no captions could be produced) and skip the table sidecar / VLM-rewrite steps. Overall score drops by a constant factor on otherwise clean docs. This is intentional: a conversion without VLM is materially less complete than one with it.
- **Tables we can't locate in the PDF score 0.7 when the markdown looks fine.** This is a guess — the markdown might be correct or might be silently wrong. The score reflects that uncertainty. No sidecar is written in this case.
- **Dropped figures score 0.8, not 1.0.** Even when the classifier correctly refuses a decoration, the score reflects residual uncertainty — a real figure marker extracted could also have been misclassified as NONE. If you see many drops in the YAML and grade starts sagging, cross-check a few of them visually.

### 11.6 Hand-edited content — the `edits:` front-matter block

Some references and captions will need manual correction. Reference sections in particular have many failure modes that the deterministic and journal-aware rescues can't cover (old-format papers, scanned documents, idiosyncratic typesetting). When you hand-edit the markdown body — fixing references, rewriting a caption, copy-editing a paragraph — record that fact in the front matter so downstream consumers (vector embedders, MCP tools, archive search) can discriminate human-verified content from auto-extracted content.

**The schema.** Add an `edits:` block as a peer to `quality:` / `copyright:` / `run:`. Two fields, both optional but at least one should be set:

```yaml
edits:
  edited: true
  note: "Manually inserted refs 33-34 from PDF text layer; rewrote Figure 3 caption"
```

For a multi-line note, use YAML's block-scalar form (`|` then 4-space indent on each line):

```yaml
edits:
  edited: true
  note: |
    References: inserted refs 33-34 from PDF text
    Figure 3: rewrote VLM caption from PDF
    Body: light copy-edit pass through Section 4
```

The block is **user-authored** — the pipeline never populates it. There is currently no CLI helper; you hand-edit the YAML at the top of the `<stem>.md` file, alongside the body changes you're recording.

**What downstream tools do with it.** The vector embedder, MCP query tools, and any other consumer can read `edits.edited` / `edits.note` from either the markdown frontmatter or the `<stem>.meta.json` sidecar. Useful patterns:

- **Retrieval ranking**: prefer chunks from human-verified papers when answering, OR surface them with a "verified" badge in the UI.
- **Provenance**: when a downstream summary cites a specific paper, the edit note becomes part of the citation lineage — useful when questions arise later.
- **Audit / triage**: filter the corpus to "papers that have been hand-checked" vs "raw machine output" for selective review.

**Three caveats to know before you start hand-editing.**

1. **Re-running paper2md OVERWRITES the .md file** — including your hand-edited body changes AND the `edits:` block you added to record them. The `edits:` block is currently *not* preserved across re-runs. Two practical workflows:
   - **Edit, then don't re-run.** Once a paper is annotated, leave its output alone unless you have a specific reason to regenerate. This is the simplest pattern and the one we recommend by default.
   - **Re-run with `--clean`, then re-add by hand.** If you intentionally regenerate (say, you re-pinned marker / surya, or you want to apply a new rescue), you'll need to re-apply your edits and re-add the `edits:` block. Cheap to copy-paste from a backup of the previous output.

   Until paper2md gains a "preserve hand-edits across re-runs" feature, plan your re-run schedule with this in mind.

2. **Body edits are silently lost on re-run.** The `edits:` block flag is just metadata about a fact ("this artifact was hand-edited") — it doesn't preserve *what* was edited. If you need to recover edits across re-runs, keep your own backup or use git on the output directory:
   ```bash
   cd <out>
   git init && git add . && git commit -m "auto-extracted from <pdf>"
   # ... hand edit cuk.md ...
   git add . && git commit -m "manual: refs 33-34 from PDF; Fig 3 caption"
   # later, after a re-run:
   git diff HEAD~1 cuk.md   # what the re-run changed
   ```
   For a small corpus, manual git is enough; for a batch, automate the snapshot before each re-run.

3. **HDF5 bundles (`--hdf5`) re-pack fresh on re-run.** The `<stem>.h5` bundle is built from the loose `.md` + `assets/` at write time. If you edit the loose `.md`, the bundle is **stale** until you regenerate it via `src/repack_h5.py`. Conversely, if you edit *inside* the bundle (via `src/unpack_h5.py` then edit then repack), make sure to also edit the loose `.md` if you keep both around. See §12.5 for the round-trip helpers.

**Recommended hand-edit workflow:**

1. Run paper2md normally. Get the output `<out>/<stem>/<stem>.md`.
2. Identify problem papers (references-score < 0.65, or any verdict from manual review).
3. For each problem paper:
   a. Open `<stem>.md` in a text editor.
   b. Fix the body content (paste in the correct refs / caption / paragraph).
   c. Add the `edits:` block to the front matter with a one-line note describing what changed.
   d. Save.
   e. (Optional) Re-pack the HDF5 bundle: `python src/repack_h5.py <stem>.h5`.
4. Don't re-run paper2md on the edited paper unless you've copied the edits out first.

**What we're explicitly NOT doing yet** (deferred until needed):

- A CLI helper (`paper2md edit add ...`) to inject `edits:` records without hand-writing YAML.
- Round-trip preservation: paper2md detecting an existing `edits:` block on overwrite and copying it into the new front matter.
- Per-item granularity (which refs / which figures), beyond the free-form note.
- Diff-based change tracking between auto and human versions.

These are tractable additions when you hit the pain — say so and we'll wire them in. For now the schema gets you through the immediate need: a stable, machine-readable signal that "this artifact has human verification" plus a free-text note describing what.

---

## 12. HDF5 bundle output (`--hdf5`)

With `--hdf5`, after the loose markdown + `assets/` are written, the pipeline packs everything into a single compressed HDF5 file at `<out>/<main-pdf-stem>.h5`. Intended for transport, corpus ingestion, and archival — one self-contained file per paper (+ optional supplement) that round-trips all the loose outputs.

Loose files are **still produced**. The bundle is additive; it doesn't replace or delete the loose outputs.

### 12.1 Bundle layout

```
/                              attrs: schema_version, tool, created_at, vlm_model
├── main/                      attrs: source_pdf, markdown_path, meta_json_path,
│   │                                 overall, grade
│   ├── markdown               utf-8 string (1-element, gzipped)
│   └── meta_json              utf-8 string (1-element, gzipped)
├── supplement/                (present only when --supplement was used)
│   ├── markdown               same shape as main/markdown
│   └── meta_json              same shape as main/meta_json
└── assets/                    attrs: count
    ├── figure_1_p3.jpg        uint8 byte array
    ├── figure_2a_p5.jpg
    ├── si_figure_S1_p2.jpg
    ├── table_1_p4_1.jpg
    ├── table_1_p4_1.md        utf-8 string (gzipped)
    ├── si_table_S1_p5_1.md
    └── ...
```

- **Markdown datasets are 1-D arrays of length 1** so HDF5 can apply gzip compression (scalar datasets don't support filters). Read with `f["main/markdown"].asstr()[0]`.
- **`meta_json`** is a structured mirror of the markdown frontmatter (`copyright:`, `run:`, `quality:` blocks) serialised as JSON. Same loose-file content lives at `<out>/<stem>.meta.json` next to the markdown. The `.md` frontmatter is canonical; the JSON is a derived artifact regenerated on every run, intended for batch / corpus consumers that prefer `jq` over a YAML parser. Read with `json.loads(f["main/meta_json"].asstr()[0])`.
- **Binary assets (JPEG/PNG)** are stored as raw `uint8` byte arrays with no compression — they're already compressed internally, so gzip would waste CPU for no size win.
- **Text assets (`.md` table sidecars)** use gzip level 9 — these compress well and dominate the savings.

### 12.2 Reading the bundle

```python
import h5py
with h5py.File("out/paper.h5", "r") as f:
    print("Grade:", f["main"].attrs["grade"],
          "Overall:", f["main"].attrs["overall"])
    md = f["main/markdown"].asstr()[0]            # full markdown
    meta = json.loads(f["main/meta_json"].asstr()[0])  # structured frontmatter
    print("License:", meta["copyright"]["license"],
          "year:", meta["copyright"].get("year"))
    if "supplement" in f:
        si_md = f["supplement/markdown"].asstr()[0]
    # Enumerate assets
    for name in f["assets"]:
        ds = f[f"assets/{name}"]
        if name.endswith(".md"):
            body = ds.asstr()[0]
        else:
            raw_bytes = ds[()].tobytes()          # JPEG bytes
```

### 12.3 Schema versioning

The root attribute `schema_version` starts at `1`. Bumped if the layout ever changes in a backward-incompatible way. Consumers should check it before assuming the structure above.

### 12.4 Caveats

- The bundle is written **only after both `convert()` calls succeed.** A failed run leaves no `.h5`.
- If an asset was deleted by hook 2 (unmatched figure drop), it's gone from `assets/` on disk and therefore absent from the bundle — consistent behavior.
- `--no-quality` removes the YAML from the markdown but the `overall` and `grade` attrs on `main`/`supplement` groups are still populated from the in-memory `QualityReport`. YAML presence is independent of the HDF5 metadata.

### 12.5 Repacking after editing (`src/repack_h5.py`)

If you edit a markdown file (or table sidecar) on disk after the bundle was written, the bundle goes stale. `src/repack_h5.py` rewrites it from the current loose-file content while preserving the per-document metadata (`source_pdf`, `markdown_path`, `overall`, `grade`) read from the existing bundle.

```bash
# read .md + assets/ from the directory containing paper.h5; rewrite paper.h5
python src/repack_h5.py paper.h5

# read loose files from a different dir
python src/repack_h5.py paper.h5 --from /path/to/edited/dir

# write to a different output, leave the input bundle untouched
python src/repack_h5.py paper.h5 -o paper-edited.h5
```

What changes vs. the original bundle:
- Markdown content (and `assets/*.md`) — read fresh from disk
- `meta_json` — read fresh from `<stem>.meta.json` if present on disk; else carried through unchanged from the source bundle
- Asset binaries (`*.jpg`, `*.png`) — read fresh from disk; deleted assets disappear from the bundle, added ones appear
- Root attr `tool` becomes `"src/repack_h5.py"` (provenance signal)
- Root attr `created_at` becomes the repack timestamp
- Root attr `original_created_at` is added, holding the previous bundle's `created_at`

What is preserved from the original bundle:
- `schema_version`, `vlm_model` on the root
- `source_pdf`, `markdown_path`, `overall`, `grade` on each doc group

Quality scores are NOT re-derived after editing — they reflect the original pipeline run. If you want fresh scores, re-run `src/paper2md.py` from the source PDF instead.

The write is atomic (writes to `<bundle>.h5.tmp` then renames) so a crash mid-write doesn't leave a half-written bundle in place of a working one.

---

## 13. VLM providers (`--provider`)

Three providers are supported. LM Studio (local) is the default; OpenAI and Anthropic swap in with a CLI flag + an API key.

### 13.1 Provider matrix

| Provider | CLI | API key env var | Default model | Notes |
|----------|-----|-----------------|---------------|-------|
| **lmstudio** (default) | `--provider lmstudio` | none (local) | `qwen3-vl-32b-instruct-mlx` | uses `$LM_STUDIO_URL` (default `http://localhost:1234/v1`) |
| **openai** | `--provider openai` | `$OPENAI_API_KEY` | `gpt-4o` | vision-capable; override model via `$VLM_MODEL`. **Use `gpt-4.1` or `gpt-4o`** — see §13.6 for the gpt-5 / o-series caveat. |
| **anthropic** | `--provider anthropic` | `$ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | requires `anthropic` Python package (`pip install anthropic`) |

The `$VLM_PROVIDER` env var provides a persistent default; the `--provider` flag overrides for a single invocation. Model choice cascades: `--provider` → default for that provider, unless `$VLM_MODEL` is set, which overrides.

### 13.2 Pricing and rate-limit awareness

LM Studio runs locally so VLM calls are free (other than electricity and time). OpenAI and Anthropic charge per token. Approximate per-paper cost at current list prices (late 2025/early 2026):

- **Full pipeline (with table + figure hooks)** on a typical 8-page scientific paper: ~10–15 VLM calls, ~50 K input tokens (images) + ~10 K output tokens. Roughly **$0.10–$0.30 on gpt-4o, $0.15–$0.40 on claude-sonnet-4-6**.
- **`--text-only`** on the same paper: 8 page-level calls with large outputs. Roughly **$0.30–$0.80 per paper** on either provider (output-token dominant).

For corpus-scale work (hundreds of papers) plan on $10s–$100s and respect rate limits. For iterative development, LM Studio is effectively free and should be your default.

### 13.3 Choosing a provider

- **LM Studio / vLLM (local)**: best default. Privacy (nothing leaves the box), no per-call cost, easy to swap models via the LM Studio UI. Qwen3-VL-32B at MLX 8-bit (Apple Silicon) or BF16 (CUDA) is strong on document tasks.
- **OpenAI (`gpt-4.1`, `gpt-4o`)**: use when local Qwen mis-classifies on visually-similar figures, when you need maximum accuracy on a high-value paper, or when you don't have GPU. **Avoid `gpt-5` and the o-series** for this pipeline today — see §13.6.
- **Anthropic (`claude-sonnet-4-6`, `claude-opus-4-7`)**: strong on reading-order/structural reasoning on complex layouts. Good fallback when local model mis-handles multi-panel figures or borderless tables.

#### 13.3.1 Empirical comparison — six VLMs on jacquet.pdf

A 20-page Elsevier paper (Jacquet et al., *Icarus*) with 14 real figures including 2 appendix figures (`B.13`, `D.14`) and a known-hard case where Figs 6, 7, 8 are visually similar (mesostasis chemistry plots). Same DGX Spark hardware for marker on the local-vLLM rows; remote LM Studio runs on a separate Apple Silicon machine. Same flags throughout (default detector-text table path — equivalent to the legacy `--no-vlm-tables`, default `dup-detect` strategy). Cold-start time (loading model weights into the local server) is excluded from the wall-clock — that's a one-time cost amortised across many papers in a real batch.

| Provider / model | Accuracy | Wall-clock | Cost / paper | Data egress |
|---|---:|---:|---:|---|
| Local **gemma-4-31b-it MLX 8-bit** (LM Studio) | **100% (14/14)** | 275.7 s | $0 | none |
| **OpenAI gpt-4.1** | **100% (14/14)** | **252.8 s** | ~$0.07 | sent to OpenAI |
| **Anthropic claude-sonnet-4-6** | **100% (14/14)** | 262.9 s | ~$0.10–0.20 | sent to Anthropic |
| Local **Qwen3-VL-32B BF16** (vLLM) | 92.9% (13/14) | 293.0 s | $0 | none |
| Local **Qwen2.5-VL-72B AWQ** (vLLM) | 92.9% (13/14) | 635.6 s | $0 | none |
| Local **gemma-4-31b** (LM Studio, lower-bit quant) | 85.7% (12/14) | 308.5 s | $0 | none |

Five findings:

1. **There IS a fully-local option that ties the commercial APIs.** Gemma 4 31B at MLX 8-bit hit 100% on jacquet, matching gpt-4.1 and claude-sonnet-4-6, with no data egress. dup-detect didn't even need to fire — the classifier got every image right on the first pass. This is the first time in the experiment a local model has saturated the truth.
2. **Quantization moves the needle far more than parameter count.** Same Gemma 4 31B at the LM Studio default (likely MLX 4-bit) scored only 85.7% — fourteen percentage points lower than the 8-bit version. Meanwhile Qwen2.5-VL-72B AWQ (4-bit) scored the same as Qwen3-VL-32B at BF16. **8-bit on a 30-billion-parameter model > 4-bit on a 70-billion-parameter model**, both empirically and in cost-of-inference. This is a stronger argument for §4.2.4's "8-bit when accuracy matters" guidance than the table-rewrite case originally made.
3. **Going from 32B to 72B locally did NOT improve accuracy.** Both Qwen variants hit 92.9% with different failure modes — 32B confuses Fig 6 vs Fig 8 (same-page mesostasis pair), 72B drops Fig 12 entirely. The 72B was 2.2× slower per paper (635 s vs 293 s).
4. **Both commercial APIs were faster than the best local model** by 13–23 s. Reason: when local LM Studio / vLLM serves the VLM, it shares hardware with marker (or marker shares with itself if running on a different machine and the LM Studio remote is the bottleneck); when the VLM lives on OpenAI / Anthropic, marker has dedicated hardware. The accuracy is now indistinguishable across the top tier; pick by the *constraint* axis (cost, privacy, hardware availability), not the quality axis.
5. **Reasoning-mode models need `reasoning_effort=none`** to work in this pipeline. Without it, Gemma 4 31B (in its default thinking mode in LM Studio) silently dropped every figure because the per-call max_tokens=5 budget was entirely consumed by internal reasoning. paper2md now sends this field on every OpenAI-compat call; non-reasoning models silently ignore it. See §4.2.5 and §13.6.

**Where the failures sat (for the < 100% rows):**
- *Qwen3-VL-32B (32B BF16):* `_page_5_Figure_1` (truth: Fig 6, "Rolling averages of Na/Al") consistently classified as Fig 8 even after dup-detect's reclassify-excluding pass. Visual-discrimination limit on similar mesostasis plots.
- *Qwen2.5-VL-72B AWQ:* dropped `_page_10_Figure_1.jpeg` (truth: Fig 12) — initial classifier returned 0, so it never entered a dup-detect group.
- *Gemma 4 31B (LM Studio default quant):* dropped `_page_5_Figure_3` (Fig 7) and `_page_5_Figure_5` (Fig 8) — the same mesostasis cluster, but the model returned 0/NONE rather than picking a wrong caption.

**Practical takeaways:**
- **For Apple Silicon / 96+ GB unified memory: serve gemma-4-31b-it MLX 8-bit in LM Studio.** Same accuracy as gpt-4.1 and claude-sonnet-4-6 on jacquet, $0 per paper, no data egress. The 13–23 s wall-clock penalty vs the APIs is a worthwhile trade for keeping copyrighted PDFs on your own machine (see §5 on TDM compliance).
- **For NVIDIA / vLLM users on the Spark or similar:** Qwen3-VL-32B BF16 stays the documented default (matches `_PROVIDER_DEFAULT_MODEL["lmstudio"]` and `environment-gpu.yml`). It scores ~93% on this kind of corpus; for the remaining cases either route to a commercial API (see §13.4) or run a separate LM Studio with Gemma 4 MLX 8-bit.
- **Don't reach for the 72B model.** Same accuracy as 32B, twice as slow.
- **Don't run Gemma 4 at default quant.** The 8-bit MLX is materially better than the 4-bit; only 1.5× the disk for a step-change in accuracy.
- **Commercial APIs are a clean per-paper escape hatch.** ~$0.07–0.20 per paper is rounding error compared to APC fees; useful when you want to spot-check a specific paper or the local result looks off.

### 13.4 Operational notes

- API errors (rate limit, invalid key, content-filter reject, timeout) are logged as warnings — the specific hook that fired returns `None` and the pipeline continues with marker's original output for that hook. No hard failure.
- The active provider + model is logged once at startup (look for `VLM provider: ..., model: ...`).
- Citation / table-rewrite / figure-classifier / page-rescue hooks all route through the same `vlm()` function, so provider selection is pipeline-wide: you cannot mix (e.g. lmstudio for captions, openai for tables) in a single run. If you need that, run the pipeline twice with different `--provider` and stitch outputs.
- `--provider anthropic` loads the `anthropic` SDK lazily (only when actually selected). Absence of the package is surfaced as a clear error; it doesn't affect `lmstudio` or `openai` runs.

### 13.5 Using Ollama (or any OpenAI-compatible local server)

The `lmstudio` provider name is slightly misleading — it really means "any OpenAI-compatible endpoint reachable via `$LM_STUDIO_URL`." LM Studio is the recommended default, but **Ollama works as a drop-in alternative** with zero code changes. Point the env vars at Ollama's port:

```bash
# one-time setup
ollama serve &                       # runs on :11434 by default
ollama pull qwen2.5vl:32b            # Ollama uses colon-separated tags

# run paper2md against Ollama
LM_STUDIO_URL=http://localhost:11434/v1 \
VLM_MODEL=qwen2.5vl:32b \
python src/paper2md.py paper.pdf -o out/
```

The `VLM_MODEL` string must match exactly what `ollama list` shows (colon-separated tag, not the HF-style name).

**LM Studio vs Ollama — which to pick for this project:**

| Criterion | Winner | Why |
|-----------|--------|-----|
| Apple Silicon raw speed on MLX quants | **LM Studio** | Ollama is llama.cpp+Metal only; no MLX. ~20–50% slower on Qwen-VL. |
| First-class vision model ecosystem | **LM Studio** | VLMs are front-and-center in the UI; Ollama often needs manual HF imports. |
| Image payload handling | **LM Studio** | Anecdotally more reliable on base64 page-size images; Ollama OpenAI-compat has had edge cases with large images. |
| Headless / server / Docker deployment | **Ollama** | Runs as a daemon, no GUI. Natural for CI, shared dev boxes, Linux servers. |
| Team reproducibility / setup scripting | **Ollama** | `ollama pull` is a one-liner; LM Studio is GUI-click-download. |
| Iterative model exploration / A/B | **LM Studio** | GUI chat playground + quick quant swap. |
| Licensing | **Ollama** | MIT open-source vs LM Studio proprietary freeware. |

**Recommendation:** default to LM Studio on Apple Silicon developer machines (speed advantage is real); switch to Ollama on Linux servers or for automated/headless workflows. No src/paper2md.py change needed in either case.

### 13.6 OpenAI reasoning models — gpt-5 and the o-series do not work today

Two compatibility issues block paper2md from using OpenAI's newer reasoning model families (`gpt-5`, `gpt-5-mini`, `o1`, `o3`, `o4`):

1. **API parameter name (fixed).** Reasoning models reject the legacy `max_tokens` parameter and require `max_completion_tokens` instead. paper2md auto-detects model family via prefix and sends the right one — gpt-4o / gpt-4.1 / vLLM / LM Studio all keep `max_tokens`; gpt-5 and o-series get `max_completion_tokens`.

2. **Reasoning-token budget consumption (NOT fixed).** Reasoning models spend tokens on internal "thinking" before emitting the final answer, all out of the same `max_completion_tokens` budget. paper2md's hook 2 figure-pair classifier passes `max_tokens=5` (the right budget for a one-digit answer on direct-output models). On gpt-5 those 5 tokens are entirely consumed by reasoning, leaving zero budget for actual output — every call returns `200 OK` with empty content, every figure gets dropped, the run silently produces a useless markdown.

   The same effect, smaller magnitude, happens on hook 0 citation (`max_tokens=500`) and the trim pre-pass (`max_tokens=80`). The only hook that survives is page rescue (`max_tokens=4000`), which is large enough that reasoning + answer both fit.

**Symptom in the log:**

```
WARNING VLM returned no citation response
INFO Dropped _page_2_Figure_1.jpeg (no caption match, deleting)
INFO Dropped _page_2_Figure_3.jpeg (no caption match, deleting)
... [every figure dropped]
```

with all HTTP requests returning `200 OK`. Quality score lands around 0.84 / B because marker's spine output is fine; the figures and citation are silently empty.

**Workaround until paper2md learns per-hook reasoning budgets:** stick to non-reasoning OpenAI models (`gpt-4.1`, `gpt-4o`, or older). These accept `max_tokens` (the old name still works for them) and are direct-output, so the existing hook budgets are correct.

**Future fix (not in scope):** detect reasoning models in `vlm()` and bump `max_tokens` by a per-model multiplier (e.g. ×40 for gpt-5 — empirically validated against the figpair, citation, and trim hooks). Or migrate to OpenAI's `responses` API which separates reasoning and output budgets cleanly. Either is a real project, not a one-line change.

---

## 14. vLLM on the DGX Spark (`--provider vllm`)

vLLM is the recommended provider on NVIDIA hardware. It serves an OpenAI-compatible HTTP API on port 8000 and batches requests server-side (so `--workers > 1` in batch mode produces real throughput gains).

**Default model is BF16, not FP8.** vLLM 0.19.1's CUTLASS FP8 GEMM kernel crashes on Spark's GB10 (sm_121) with `RuntimeError: Error Internal` during memory profiling. BF16 (~64 GB on disk, ~64 GB GPU memory) uses standard PyTorch matmul which works on sm_121 via PTX. Try FP8 again when a newer vLLM with sm_121 FP8 support lands.

### One-time setup

1. Install miniforge3 (aarch64) and create the env:
   ```bash
   conda env create -f environment-gpu.yml
   conda activate paper2md
   ```
2. Sanity-check CUDA:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # expected: True NVIDIA GB10
   ```
3. Launch vLLM in a tmux session (or systemd unit):
   ```bash
   vllm serve Qwen/Qwen3-VL-32B-Instruct \
     --port 8000 \
     --max-model-len 32768 \
     --gpu-memory-utilization 0.80
   ```
   First launch downloads the model (~64 GB) into `~/.cache/huggingface/`.
4. Smoke-test:
   ```bash
   curl http://localhost:8000/v1/models
   ```

### Run

```bash
python src/paper2md.py paper.pdf -o out/ --provider vllm
```

The `--provider vllm` flag is auto-suggested when `--backend auto` resolves to `cuda` (still pass it explicitly for now).

### Env vars

- `VLLM_BASE_URL` — override the default `http://localhost:8000/v1`. Useful when vLLM runs on a different host.
- `VLM_MODEL` — override the default served model identifier (e.g. to point at a local snapshot).

### Quantization

We default to BF16 (`Qwen/Qwen3-VL-32B-Instruct`) on Spark because vLLM 0.19.1's FP8 path is broken on sm_121 (see above). Once FP8 works on sm_121, switching is a one-flag change (`VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 vllm serve ...`); FP8 will roughly halve memory use and roughly double throughput on Blackwell's dedicated FP8 tensor cores. Quality is indistinguishable from BF16 on the document tasks paper2md performs (table OCR, figure captioning, citation extraction).

---

## 15. Batch mode (`--batch`)

Process a folder, glob, or single file in one invocation. Each paper lands in its own subdirectory under `-o`, with a `manifest.jsonl` summarizing the run.

```bash
# folder of PDFs
python src/paper2md.py --batch corpus/ -o out/ --provider vllm --workers 4

# glob
python src/paper2md.py --batch 'corpus/2024/*.pdf' -o out/

# resume after a crash (default behavior; --force re-runs)
python src/paper2md.py --batch corpus/ -o out/

# disable supplement auto-pairing
python src/paper2md.py --batch corpus/ -o out/ --no-pair

# custom SI-detection regex (must define a 'stem' named group)
python src/paper2md.py --batch corpus/ -o out/ --pair-regex '^(?P<stem>.+?)[-_]appx$'
```

**`--vlm-tables-force` is not a batch-friendly flag.** It's designed for one-off publication-grade sidecars on individual papers. Use the default `--vlm-tables` for corpus runs (auto-VLM fires only on tables the cheap converter couldn't handle cleanly), then touch up specific sidecars after the batch with the standalone `vlm-table` CLI (§19) if needed. See §3.1's `--vlm-tables-force` subsection for the wall-time numbers and the recommended workflow.

### Output layout

```
out/
├── manifest.jsonl
├── canup/
│   ├── canup.md
│   └── assets/
├── jacquet/
│   ├── jacquet.md
│   └── assets/
├── root-maintext/
│   ├── root-maintext.md
│   ├── root-supplemental.md   # auto-paired SI
│   └── assets/                # SI assets prefixed 'si_'
└── ...
```

### `manifest.jsonl` schema

One JSON object per line. Core fields:

| Field | Type | Meaning |
|---|---|---|
| `pdf` | string | Path to the main PDF |
| `supplement` | string \| null | Path to the auto-paired supplement, or null |
| `status` | string | `"ok"`, `"skipped"` (resume hit), `"low_quality"` (below `--quality-threshold`), `"timeout"` (exceeded `--paper-timeout`), or `"error"` |
| `grade` | string | Quality grade A/B/C/D/F (omitted on `error`/`skipped`) |
| `overall` | float | Quality overall score 0.0–1.0 (omitted on `error`/`skipped`) |
| `elapsed_sec` | float | Wall-clock time for this paper |
| `error` | string | `TypeName: message` (only present on `status=="error"`) |

**MinerU subprocess failure fields** (added when `status=="error"` AND the failure was inside the MinerU subprocess):

| Field | Type | Meaning |
|---|---|---|
| `error_kind` | string | Classified MinerU failure mode: `cuda_oom`, `model_load`, `torch_init`, `disk`, `pdf_parse`, or `unknown`. Use this to decide whether to lower vLLM `--gpu-memory-utilization`, free disk, etc. |
| `error_pdf` | string | Basename of the PDF that triggered the MinerU failure |
| `error_stderr_tail` | string | Last ~2000 chars of MinerU's stderr — enough to see the actual exception trace without re-running |

**Reference-vs-insertion audit fields** (added on every `status=="ok"` main paper):

| Field | Type | Meaning |
|---|---|---|
| `figures_referenced` | int | Distinct figure ids mentioned anywhere in the body (`Fig. 1`, `Figure 2a`, `Fig. S1`, etc.) |
| `figures_inserted` | int | Image-markdown lines actually placed in the body (excludes `table_*.jpg`) |
| `tables_referenced` | int | Distinct table ids mentioned in the body |
| `tables_inserted` | int | Distinct tables with a caption / sidecar in the body |
| `figure_mismatch` | bool | `true` when `figures_referenced > figures_inserted` — the body cites figures that were never spliced in. Manual inspection recommended. |
| `table_mismatch` | bool | Same for tables |

**Supplement fields** (added when `supplement` is non-null):

| Field | Type | Meaning |
|---|---|---|
| `si_status` | string | `"ok"` or `"error"` — the SI is processed AFTER the main paper. A failure here does NOT change the main `status`; the main result is preserved. |
| `si_error` | string | `TypeName: message` (only when `si_status=="error"`) |
| `si_error_kind` / `si_error_pdf` / `si_error_stderr_tail` | string | Same as the main MinerU failure fields, prefixed with `si_` |
| `si_figures_referenced` / `si_figures_inserted` / `si_tables_referenced` / `si_tables_inserted` | int | SI body audit counts |
| `si_figure_mismatch` / `si_table_mismatch` | bool | SI mismatch flags |

### Querying the manifest with `jq`

Common one-liners for triaging a batch run:

```bash
# All papers where MinerU crashed, grouped by failure kind
jq -r 'select(.status=="error" and .error_kind) | [.error_kind, .pdf] | @tsv' manifest.jsonl | sort

# All cuda_oom failures (likely fix: drop vllm --gpu-memory-utilization)
jq -r 'select(.error_kind=="cuda_oom") | .pdf' manifest.jsonl

# Papers where the supplement failed but the main paper succeeded
# (you still have a usable main output)
jq -r 'select(.si_status=="error") | [.pdf, .si_error_kind] | @tsv' manifest.jsonl

# Inspect the actual MinerU stderr tail for a failure
jq -r 'select(.error_kind=="pdf_parse") | .error_stderr_tail' manifest.jsonl

# Papers where the body cites figures but some weren't spliced in
# (manual inspection recommended -- compare PDF to the .md)
jq -r 'select(.figure_mismatch==true)
       | [.pdf, .figures_referenced, .figures_inserted] | @tsv' manifest.jsonl

# Same for tables -- common cause: MinerU detection missed a table
jq -r 'select(.table_mismatch==true)
       | [.pdf, .tables_referenced, .tables_inserted] | @tsv' manifest.jsonl

# Either-mismatch (figure OR table)
jq -r 'select(.figure_mismatch==true or .table_mismatch==true) | .pdf' manifest.jsonl

# Timeouts (paper exceeded --paper-timeout; output may still exist)
jq -r 'select(.status=="timeout") | .pdf' manifest.jsonl

# Low-quality flagged papers (below --quality-threshold)
jq -r 'select(.status=="low_quality") | [.pdf, .overall, .grade] | @tsv' manifest.jsonl

# Cleanly-finished papers (main AND si OK, no audit mismatches)
jq -r 'select(.status=="ok"
              and (.si_status==null or .si_status=="ok")
              and .figure_mismatch != true
              and .table_mismatch != true) | .pdf' manifest.jsonl

# Total wall time across the batch
jq -s 'map(.elapsed_sec) | add' manifest.jsonl
```

### Standalone audit script

The same reference-vs-insertion check is available as a CLI for ad-hoc
post-mortem on a single `.md`:

```bash
# Human-readable summary
python -m audit_md out/paper-stem/paper-stem.md

# JSON output for piping into jq
python -m audit_md out/paper-stem/paper-stem.md --json | jq
```

Exit code is `1` when any mismatch is flagged, `0` when refs match insertions. Useful in CI checks for a controlled corpus.

### Supplement auto-pairing

The default heuristic catches two filename conventions:

1. `foo.pdf` + `foo_SI.pdf` (also `foo-supp.pdf`, `foo_supplement.pdf`, `foo_S1.pdf`, etc., case-insensitive).
2. `foo-maintext.pdf` + `foo-supplemental.pdf` (also `foo_manuscript`, `foo-paper`, etc.).

When a paired SI is detected, the SI is processed inside the *main paper's* output subdir (`out/foo/foo_SI.md`, with `si_`-prefixed assets) — exactly mirroring single-paper `--supplement` behavior. The SI does *not* get its own top-level subdir.

Override with `--no-pair` (every PDF runs standalone) or `--pair-regex` (custom regex, must define `(?P<stem>...)`).

If a file matches the SI regex but no main paper is in the batch, it logs a warning and runs as a standalone paper.

### Concurrency model

Marker is GPU-bound and runs serialized across workers (held by an internal lock). The VLM hooks for different papers can overlap, because vLLM batches requests server-side. With `--workers 4`:

- Paper N's Marker run blocks paper N+1's Marker run.
- Paper N's VLM hooks (tables, figures, citation, page rescue) overlap paper N+1's Marker run and paper N+2's VLM hooks.

In practice this gets you ~2–3× throughput on a corpus dominated by VLM time vs. `--workers 1`. For Marker-dominated corpora (text-only papers, no tables), `--workers 1` is fine.

### Failure isolation

A crash on one paper does not abort the batch. The error is logged to `manifest.jsonl` with `status: "error"` and the next paper starts. Process exits 0 unless *every* paper failed.

### Quality threshold in batch mode

`--quality-threshold` does *not* abort a batch. It downgrades the manifest entry to `status: "low_quality"` so you can grep/jq for review later.

### Resume

Resume is on by default: a paper is skipped if `<out>/<stem>/<stem>.md` already exists. Use `--force` to re-run regardless.

---

## 16. Metadata / copyright front-end

Before marker runs, paper2md resolves the article's identity (DOI or arXiv ID) and looks up its declared license / open-access status via free public APIs (Crossref, OpenAlex, Unpaywall, Europe PMC, OSTI / DOE PAGES, arXiv). The result lands in the YAML front-matter as a `copyright:` block alongside `quality:` and `run:`. Disable with `--no-metadata-lookup`; run the resolver alone without any extraction with `--metadata-only`.

The API graph fans out conditionally: each fallback only runs when prior APIs left `license` or `oa_pdf_url` unfilled. The order is OpenAlex (DOI) → Unpaywall (DOI, license fallback) → Europe PMC (PMC-deposited biomedical) → OSTI (DOE-funded physical-science deposits) → arXiv (when no DOI but an arXiv ID is in the PDF) → Crossref title-search (when no DOI at all).

### `safe_to_distribute` tiers

The pipeline normalizes whatever license slug the APIs return into one of four safety tiers. Downstream consumers should treat these as the canonical signal:

| Tier | License examples | Meaning |
|---|---|---|
| `true` | `cc0`, `public-domain`, `cc-by`, `cc-by-sa`, `mit`, `us-government-work` | Permissive — safe to redistribute the markdown extraction. |
| `restricted` | `cc-by-nc`, `cc-by-nd`, `cc-by-nc-sa`, `cc-by-nc-nd` | Redistribution OK but with constraints (non-commercial, no-derivs, share-alike). The caller must respect the constraint downstream — paper2md does not enforce it. |
| `readable` | `arxiv-default`, `arxiv-pre-2004`, `pmc-author-manuscript`, `green-oa-no-license` | A public PDF is reachable and the article is legitimately readable for personal/research reference, but no permissive license has been declared. **Not safe to redistribute.** This tier covers the typical NIH/funder-mandated PMC author manuscript: free to read, no CC license. |
| `false` | `all-rights-reserved`, `elsevier-tdm`, unknown / not-found | Closed-access, publisher-TDM-only, or unresolvable. Conservative default for anything we couldn't classify. |

The `confidence` field (`high` / `medium` / `low`) records *how* the tier was resolved: `high` = an API gave us a license slug, `medium` = title-based Crossref fallback found a candidate, `low` = nothing resolved (unknown DOI, network off, etc.).

### Synthetic license slugs

Three `license` slugs are paper2md-internal synthesis (no API returns them verbatim):

- `pmc-author-manuscript` — set when Europe PMC has a PMC ID with `hasPDF=Y` but `isOpenAccess=N`. The classic NIH/NASA/funder-mandate green-OA deposit: readable, not redistributable.
- `osti-public-access` — set when OSTI / DOE PAGES has a deposit whose `doi` matches the article's DOI. Same legal posture as `pmc-author-manuscript`: federal public-access mandate, freely readable, not under a CC license, not redistributable. Useful for DOE-funded physical-science papers (Sandia / LANL / LBNL / fusion / particle-physics work) that Europe PMC doesn't cover.
- `green-oa-no-license` — catch-all when an OA URL exists (oa_status in `{green, bronze, gold, hybrid}`) but no API declared a CC license. Same readable-tier semantics; differentiates "we found a public PDF" from "we couldn't find anything."
- `public-domain-us` — set when the API chain returns a confident publication year and `year + 95 < current_year` (the US 95-year rolling cutoff: a work first published in year Y enters US public domain on Jan 1 of Y+96). Mapped to the `true` tier. **Overrides** the synthetic readable-tier fallbacks (`pmc-author-manuscript`, `osti-public-access`, `green-oa-no-license`) — those are "no license declared" assignments, but copyright on a >95-year-old work has actually expired, so PD beats the readable fallback. Does **not** override real CC / publisher licenses (even on a PD-era paper, deferring to the rights-holder's stated declaration is the safer choice). **US-only**: EU is life+70 from author death, which we cannot compute from publication year alone. Does not address the 1930–1963 grey zone where renewal status matters.
- The arXiv default license slugs (`arxiv-default`, `arxiv-pre-2004`) come from arXiv's own metadata but are mapped into `readable` because arXiv's default license grants arXiv permission to host, not downstream redistribution.

### Flags

- `--no-metadata-lookup` — skip the resolver entirely. No `copyright:` block in the front-matter; downstream license = unknown.
- `--prefer-oa-source` — when an OA PDF URL is found and the local PDF is closed-access, download the OA copy and run the rest of the pipeline against it. The OA copy is kept at `<out>/<stem>_oa_source.pdf` for reference (front-matter records `oa_pdf_used: true` and `oa_pdf_local: <stem>_oa_source.pdf`).
- `--require-license` — exit code 3 when `safe_to_distribute` is `false` *or* `readable`. Use in batch pipelines that should not produce markdown for non-redistributable papers.
- `--metadata-only` — run only the resolver and exit. Single paper: print the YAML to stdout. Batch: write `manifest.jsonl` with one record per PDF. Useful for screening a corpus before extraction.
- `--metadata-timeout SEC` — per-API request timeout (default 10).
- `--metadata-cache PATH` — JSON cache for DOI→metadata lookups (default `~/.cache/paper2md/metadata.json`). Re-runs of the same DOI are O(1).

### Polite-pool emails

The free APIs grant higher rate limits when callers identify themselves. paper2md picks these up from environment variables; missing values fall back to anonymous access with self-throttling.

| Env var | API |
|---|---|
| `UNPAYWALL_EMAIL` | Unpaywall (required by their API; resolver skips Unpaywall entirely without it) |
| `OPENALEX_MAILTO` | OpenAlex |
| `CROSSREF_MAILTO` | Crossref |

Same email may be used for all three. A `.env` next to `src/paper2md.py` is auto-loaded if `python-dotenv` is installed.

### Offline / air-gapped runs (`--offline`)

Pass `--offline` to run with **no scholarly-API network access**. Useful on a flight, on an air-gapped lab machine, or when you want predictable run times without 3–10 s timeouts on each network step.

What `--offline` skips:

| Path | Behavior under `--offline` |
|---|---|
| Copyright/preprint metadata pre-pass (Crossref, OpenAlex, Unpaywall, Europe PMC, OSTI, arXiv) | Skipped. Local DOI/arXiv-ID extraction from the PDF still runs. |
| References API fallback (`fetch_references()` — Crossref → OpenAlex) | Skipped unless the DOI is already in `~/.cache/paper2md/references.json` from a prior online run; cached entries are still served. |
| OA PDF download (`--prefer-oa-source`) | Auto-disabled with a warning. |
| Data-repository enrichment (`--fetch-data-repos`) | Auto-disabled with a warning. Detected DOI/URL strings are still recorded in the `data:` block — only the per-repo API summary fetch is skipped. |
| Local VLMs (`--provider lmstudio`, `--provider vllm`) | Unaffected. They ping `localhost`, not the open internet. |
| Cloud VLMs (`--provider openai`, `--provider anthropic`) | Warned about. The VLM calls themselves will fail; either re-run without `--offline`, or pair with `--no-vlm` for marker-only output. |

Where the "skipped" notes appear in the output (frontmatter only — no body markdown clutter):

- `copyright.provenance:` includes `network-disabled` (provided by `metadata_frontend.resolve()`).
- `run.pipeline.offline: true`.
- `quality.references.api_skipped_offline: true` — set only when the rescue's trigger gate would have fired (low score, no section, or visible numbered-list gaps) AND the network was unavailable. Disambiguates "skipped due to offline" from "fetched and got nothing".

Example invocation:

```
python src/paper2md.py paper.pdf -o out/ --offline --no-vlm
```

Cached metadata and references from prior online runs are still used. Cache is at `~/.cache/paper2md/metadata.json` and `~/.cache/paper2md/references.json`; not touched by `--clean` (which is per-paper output, not user-cache).

### Verifying a corpus

```
# Screen 100 PDFs for license status, no extraction
python src/paper2md.py --batch corpus/ -o /tmp/screen --metadata-only

# Find every CC-BY paper for safe redistribution
jq 'select(.safe_to_distribute == "true")' /tmp/screen/manifest.jsonl
```


## 17. Manual edit of tables and figures

`--replace-table`, `--replace-fig`, and `--revert-edit` operate on an existing paper2md output. They short-circuit the pipeline (no PDF needed) and are how you fix a paper whose table was garbled or whose figure was mis-spliced — without rerunning extraction.

The audit script (§15 "Standalone audit script") flags candidate papers; these flags reconcile the mismatches.

### When to use

| Symptom | Tool |
|---|---|
| Table renders as HTML fallback, missing rows, or columns swapped | `--replace-table` |
| Splice picked the wrong asset for a figure (mixed up with the supplement's, or wrong panel) | `--replace-fig` |
| Need to undo a prior replacement | `--revert-edit` |

### `--replace-table` — VLM re-transcribe a table from a user crop

```
paper2md --replace-table <ID> <image.png> <paper.md> [--note "..."]
```

The user crops a clean image of the table (any standard tool: macOS Preview, GIMP, screenshot) and points `--replace-table` at it. paper2md:

1. Copies the image into `<paper-dir>/assets/user_edit_table_<ID>.<ext>` (and computes a SHA-256).
2. Calls the VLM (same `TABLE_PROMPT` and `max_tokens=6000` as the in-pipeline rewrite) to transcribe it as a Markdown pipe-table.
3. Writes the transcription as a sidecar at `assets/user_edit_table_<ID>.md`.
4. Appends an entry to the `edits:` block in the paper's YAML frontmatter (creating the block if missing).
5. **Does not modify the body** — prints a "paste hint" pointing at the existing `**Table <ID>.**` caption line and the sidecar to copy from.

The user then opens `paper.md` and pastes the sidecar contents over the garbled block. This explicit two-step design avoids the silent-corruption failure modes of an auto-replace (region delimitation is fuzzy across HTML / pipe-md / multi-block tables).

Idempotent: re-running with the same `<ID>` replaces the prior `edits:` entry rather than appending a duplicate.

### `--replace-fig` — swap a figure asset in-place

```
paper2md --replace-fig <ID> <image.jpg> <paper.md> [--note "..."]
```

No VLM call. paper2md:

1. Locates the asset bound to `Fig. <ID>` in `paper.md` (two strategies: caption-proximity scan, then `assets/figure_<ID>.<ext>` fallback).
2. Archives the existing asset as `<asset>.orig` (only on first run; idempotent).
3. Overwrites the asset path with the new image.
4. Appends an `edits:` entry to the frontmatter.

The body is unchanged — the `![](...)` line still points at the same path, which now contains the corrected image. The page renders correctly the moment the command finishes.

If `--replace-fig` can't find the asset (no `![](...)` image line near the `Fig. <ID>` caption AND no `assets/figure_<ID>.<ext>` match), the command errors. Use it only when the audit shows the figure was actually inserted; if it's missing entirely, paste manually.

### `--revert-edit` — undo a prior replacement

```
paper2md --revert-edit <KIND> <ID> <paper.md>     # KIND ∈ {table, figure}
```

For **figure**: restores `<asset>.orig` over the replaced file and deletes the archive.

For **table**: drops the frontmatter `edits:` entry only. The sidecar `.md` and the image crop are kept on disk (you may already have pasted from them; the tool can't safely un-paste).

### Frontmatter schema for `edits:`

```yaml
edits:
- kind: table_replacement     # or figure_replacement
  id: '3'                     # the ID the user passed; matches caption number
  image: assets/user_edit_table_3.png    # or assets/figure_3.jpg for figs
  image_sha256: a3f8...
  sidecar: assets/user_edit_table_3.md   # tables only
  original_archived_as: assets/figure_3.jpg.orig   # figures only, first run
  vlm_model: Qwen/Qwen3-VL-32B-Instruct  # tables only
  timestamp: '2026-05-14T19:32:11Z'
  note: "splice garbled — manual rewrite from page 4 crop"   # if --note passed
```

Downstream consumers can query `edits:` to surface "human-touched" papers — e.g. `jq 'select(.edits != null)' meta.json` if you copy the frontmatter into your manifest pipeline.

### Worked example

```bash
# 1. Find papers flagged by the audit
for md in out/*/*.md; do python -m audit_md "$md" >/dev/null || echo "$md"; done

# 2. Open a flagged paper, crop the bad table to /tmp/table_fix.png

# 3. Re-transcribe via VLM
paper2md --replace-table 2 /tmp/table_fix.png out/canup/canup.md \
         --note "11-row table, MinerU lost rows 7-10"

# 4. paper2md prints:
#    Replaced Table 2 metadata; user paste required.
#    Sidecar written: assets/user_edit_table_2.md
#      • canup.md: Table 2 caption is at line 412.
#      • Replace the garbled table block below it with the contents of
#        assets/user_edit_table_2.md.

# 5. Open canup.md, paste sidecar contents over the garbled block. Done.
```


### `--recover-from-mineru` — auto-stage tables MinerU found but the hybrid splice dropped

After a hybrid run, the audit may show tables referenced-but-not-inserted. For tables where MinerU detected them correctly but the caption-anchoring splice failed, the table is already on disk under `<paper-dir>/mineru/<stem>_middle.json` (with its cropped image in `mineru/images/` or already moved to `assets/`). No re-extraction is needed — the recovery command stages it.

```
paper2md --recover-from-mineru <paper.md> [--recovery-vlm | --recovery-no-vlm]
```

What it does:

1. Audits the paper, finds tables referenced but not inserted.
2. For each missing id, walks the sibling `mineru/<stem>_middle.json` for a matching table block (matched by canonical id — Roman/arabic deduplicated, S-prefix preserved). Falls back to scanning the table HTML for the id when MinerU didn't classify a caption block (the actual failure mode for several silica-shock corpus papers).
3. Copies MinerU's table image into `assets/user_edit_table_<id>.<ext>`.
4. Produces the sidecar `assets/user_edit_table_<id>.md`:
   - **By default** the sidecar source matches the paper's original `run.pipeline.vlm_rewrite_tables` setting (read from frontmatter). If the original run used `--vlm-tables`, recovery calls VLM on the staged image too — same `TABLE_PROMPT`, same `max_tokens=6000`. If the original run did not, recovery converts MinerU's HTML via `html_to_pipe_md`.
   - **`--recovery-vlm`** forces VLM (useful when spot-upgrading specific tables).
   - **`--recovery-no-vlm`** forces HTML conversion (useful when the VLM server is offline).
5. Appends an entry to a new `pending_recoveries:` frontmatter block (idempotent by id — re-running replaces the prior entry).
6. **Does not modify the body.** Prints a paste hint pointing at the sidecar.

If VLM is requested but the call returns empty or errors, recovery falls back to HTML conversion and records the source as `*_fallback` so the user knows what they got.

### `--confirm-recovery` — mark a recovery applied

After pasting the sidecar contents into the body manually, mark it confirmed:

```
paper2md --confirm-recovery table <id> <paper.md>
```

The `pending_recoveries:` entry moves to `edits:` with two added fields:
- `recovered_from: mineru` — distinguishes audit-driven recoveries from `--replace-table` operations
- `confirmed_at: <UTC timestamp>`

Safety: subsequent `--recover-from-mineru` runs refuse to re-stage a confirmed table (you presumably pasted; silent clobber is prevented). Use `--revert-edit table <id> <paper.md>` first if you want to redo.

### Recovery source values (in the frontmatter)

| `source` | Means |
|---|---|
| `vlm_rewrite` | VLM transcribed the MinerU image successfully |
| `mineru_html` | `html_to_pipe_md` converted MinerU's HTML cleanly |
| `mineru_html_raw` | The HTML uses rowspan/colspan that pipe-md can't represent; raw HTML preserved in the sidecar with a comment header (user cleans up) |
| `mineru_html_empty` | MinerU's table block has no HTML body (sidecar is a placeholder comment) |
| `mineru_html_fallback` / `mineru_html_raw_fallback` | VLM was requested but failed; recovery fell back to HTML |

### Batch triage with the manifest

When `table_mismatch=true` AND a `mineru/` subdir exists for the paper, the batch manifest entry records a `recoverable_tables` count. Use it to drive triage:

```bash
# Papers where every dropped table is recoverable from MinerU --
# pure splice bug, no extraction loss
jq -r 'select(.recoverable_tables and .recoverable_tables == (.tables_referenced - .tables_inserted)) | .pdf' \
    out/manifest.jsonl

# Worst gaps to inspect first
jq -r 'select(.recoverable_tables and .recoverable_tables >= 2) | "\(.recoverable_tables)\t\(.pdf)"' \
    out/manifest.jsonl | sort -rn
```

A paper with `recoverable_tables == (tables_referenced - tables_inserted)` is a pure splice failure — everything dropped is on disk and one `paper2md --recover-from-mineru` recovers it all.

### Worked example (real corpus paper)

```bash
$ python -m audit_md out/feng/fengShockCompressionCoesite2024.md
Audit: fengShockCompressionCoesite2024.md
  Tables referenced:  1 (1)
  Table insertions:   0 (—)
  Flags:
    ! TABLE MISMATCH: 1 referenced > 0 inserted

$ paper2md --recover-from-mineru out/feng/fengShockCompressionCoesite2024.md --recovery-no-vlm
INFO 1 of 1 table(s) staged (VLM=off, source=override).
INFO   Table 1 staged via mineru_html; no caption line found in body --
       search for `Table 1` in fengShockCompressionCoesite2024.md and paste
       user_edit_table_1.md there
INFO After pasting, mark applied: paper2md --confirm-recovery table 1 ...

# User pastes assets/user_edit_table_1.md into the body, then:
$ paper2md --confirm-recovery table 1 out/feng/fengShockCompressionCoesite2024.md
INFO Confirmed: table 1 promoted to `edits:` (recovered_from=mineru)
```

### When recovery is NOT possible

- `--layout-source marker` runs (no MinerU subdir) → error: "no MinerU output for this paper".
- Table ids referenced but missing from MinerU's middle.json → genuinely unrecoverable, listed separately. Use `--replace-table <id> <crop.png> <paper.md>` with a manual crop instead.

The audit's `recoverable_tables` count tells you the split: if `tables_referenced - tables_inserted > recoverable_tables`, some tables need manual cropping.


## 18. AI / VLM disclosure for publications

Modern publication guidelines increasingly require a "Computational Research Integrity and AI Disclosure" section. paper2md records the AI-system facts it knows about into every output's YAML frontmatter so you can copy-paste them into your methods section.

### Deterministic sampling — defaults to `temperature=0`, `seed=42`

paper2md sends `temperature` and `seed` on **every** VLM request to make the output reproducible run-to-run on the same input. Defaults:

| Param | Default | Purpose |
|---|---|---|
| `temperature` | `0.0` | Greedy decoding; eliminates stochastic token sampling |
| `seed` | `42` | Reproducible engine RNG state on servers that accept the param (vLLM, LM Studio, OpenAI chat models) |

These were previously not set, so the server defaulted to `temperature=1.0` (stochastic) and `seed=None` (random). **The new defaults change paper2md's behavior** — re-running a paper now produces byte-identical output (modulo floating-point ordering inside the model on multi-GPU configs). If you previously relied on stochastic variation for some reason, opt back into it via `--vlm-temperature 1.0 --vlm-seed -1`.

### Changing temperature / seed

Three layers, last-write-wins:

```bash
# 1. CLI flags (highest precedence)
paper2md paper.pdf --vlm-temperature 0.2 --vlm-seed 7

# Pass a NEGATIVE seed to DISABLE seed (server picks one per request)
paper2md paper.pdf --vlm-seed -1

# 2. Environment variables
export VLM_TEMPERATURE=0.0
export VLM_SEED=42
paper2md paper.pdf

# 3. Module defaults (0.0 / 42) -- no action needed
paper2md paper.pdf
```

The resolved values are logged at startup (`VLM sampling: temperature=0.0, seed=42`) and recorded in every output's `run.vlm_temperature` / `run.vlm_seed` frontmatter fields.

### Provider behavior

| Provider | Honors `temperature`? | Honors `seed`? |
|---|---|---|
| vLLM (local) | yes | yes |
| LM Studio (local) | yes | yes (per-request) |
| OpenAI (cloud) | yes | yes on chat models; silently ignored on legacy models |
| Anthropic (cloud) | yes | **NO** — Messages API has no `seed` parameter |

paper2md detects the Anthropic case at startup: if `seed` is set under `--provider anthropic`, a one-shot WARN logs that the value is recorded in frontmatter for transparency but won't reach the API. The seed silently does nothing on Anthropic.

### What paper2md DOES control (recorded automatically)

Every output's `run:` block in the YAML frontmatter (and the `.meta.json` mirror) records:

| Field | Meaning |
|---|---|
| `vlm_provider` | `vllm` / `lmstudio` / `openai` / `anthropic` |
| `vlm_model` | model identifier as resolved at runtime (e.g. `Qwen/Qwen3-VL-32B-Instruct`) |
| `vlm_endpoint` | base URL the SDK called (tells you local vs cloud) |
| `vlm_system_prompt` | exact system prompt sent on every VLM call |
| `vlm_temperature` | sampling temperature (sent on every request) |
| `vlm_seed` | sampling seed (sent on every request, omitted from YAML when None) |
| `vlm_inference_params` | human-readable note summarizing the above + max_tokens range |
| `compute_backend` | `cuda` / `mps` / `cpu` |
| `hostname` | machine where the run executed |
| `command` | full CLI invocation (shell-quoted) |
| `started_at`, `elapsed_sec` | timestamps |
| `paper2md_version`, `python_version` | exact paper2md + python versions |
| `packages.*` | versions of every load-bearing package (`mineru`, `marker-pdf`, `surya-ocr`, `pymupdf`, `torch`, `transformers`, etc.) |

### What paper2md DOESN'T control (record separately for full disclosure)

paper2md sends `temperature`, `seed`, `max_tokens`, `model`, and `messages`. It does **not** set:

- `top_p` (server default — typically `1.0`)
- `top_k` (server default — typically `-1`, i.e., disabled)
- `presence_penalty` / `frequency_penalty` (server default — typically `0`)
- quantization choice (set when the server loaded the model)
- model file hash (only the model identifier is recorded)

**Server-side facts to record for a publication-grade disclosure:**

- The vLLM / LM Studio / Ollama serve command (or systemd unit) with all flags
- Quantization applied (`--quantization`, `--load-format`, model file path with quant suffix)
- The model's SHA-256 (or HuggingFace revision) if downloaded; container/image digest if served from a container
- Hardware: GPU model and VRAM, CPU/RAM if local; cloud provider region if remote

A vLLM operator can query:

```bash
curl -s http://localhost:8000/v1/models | jq          # loaded model + max_model_len
systemctl --user cat vllm | grep ExecStart            # launch flags (systemd setup)
tmux capture-pane -t vllm -p | head -5                # launch flags (tmux setup)
nvidia-smi -L                                          # GPU model + VRAM
```

### Extracting the recorded fields from frontmatter

From the YAML frontmatter of a single `.md`:

```bash
# Print just the VLM-related lines
sed -n '/^run:/,/^[a-z]/p' out/paper.md | grep -E "vlm_|paper2md_version|compute_backend"

# Or via the .meta.json mirror (structured)
jq '.run | {vlm_provider, vlm_model, vlm_endpoint, vlm_system_prompt,
            vlm_inference_params, compute_backend, hostname,
            paper2md_version, packages}' out/paper.meta.json
```

For a whole batch (one line per paper):

```bash
jq -r '. | [.pdf, .run.vlm_provider // "?", .run.vlm_model // "?",
            .run.compute_backend // "?"] | @tsv' \
    out/manifest.jsonl
```

To list every distinct (provider, model, backend) tuple used in the batch — useful for the methods section when multiple papers used the same setup:

```bash
jq -r '. | "\(.run.vlm_provider)\t\(.run.vlm_model)\t\(.run.compute_backend)"' \
    out/manifest.jsonl | sort -u
```

### Template disclosure text (adapt for your manuscript)

The template below is calibrated for paper2md's typical local-vLLM-on-Spark setup. Replace bracketed placeholders with the values from your frontmatter.

> **Computational Research Integrity and AI Disclosure**
>
> Scientific PDFs were converted to structured Markdown using **paper2md v[X.Y.Z]** (MIT License; https://doi.org/10.5281/zenodo.[VERSION_DOI]; project concept DOI https://doi.org/10.5281/zenodo.20263035) running on **[Linux / macOS, hostname]** with a **[CUDA / MPS / CPU]** backend. The pipeline uses the [Qwen/Qwen3-VL-32B-Instruct] open-weight vision-language model for targeted post-passes on figure captions, table transcription, and references rescue. To ensure data sovereignty and experimental reproducibility, the model was hosted locally on **[Hardware, e.g., NVIDIA DGX Spark GB10 with 124 GB unified memory]** using **[Runtime, e.g., vLLM v0.6.x]** with `--gpu-memory-utilization 0.65` and `--max-model-len 32768`.
>
> To eliminate stochastic variance and ensure result consistency, inference parameters were fixed at **`temperature=0` and `seed=42`** on every VLM request (sent by paper2md client-side; recorded in each output's `run.vlm_temperature` / `run.vlm_seed` YAML frontmatter). `top_p`, `top_k`, and the presence/frequency penalties used the server's defaults (typically `1.0`, `-1`, and `0` respectively for vLLM). The system prompt sent on every VLM request was `"You are a precise document digitizer. Output only what is asked, no commentary."`. Per-call `max_tokens` varied from 5 (article-trim probes) to 6000 (wide-table transcription) depending on the operation; the per-paper `run` block in each output's YAML frontmatter records the exact paper2md, Python, and model-stack versions used (e.g., `mineru 3.1.7`, `marker-pdf 1.10.2`, `surya-ocr 0.17.1`, `pymupdf 1.27.2.x`, `torch 2.x.x+cu130`).
>
> No paper text or image was transmitted to external cloud services; all VLM inference ran on local hardware. All AI-generated outputs (table transcriptions, figure captions, reference rescues) are subject to manual 'human-in-the-loop' verification at every stage; per-table and per-figure quality scores are recorded under each output's `quality.tables` and `quality.figures` YAML arrays. The authors maintain full accountability for the accuracy and originality of the manuscript content extracted via paper2md.

### Per-paper reproducibility check

To verify the recorded provenance for a single output:

```bash
jq '{
  paper2md: .run.paper2md_version,
  vlm: { provider: .run.vlm_provider,
         model: .run.vlm_model,
         endpoint: .run.vlm_endpoint,
         temperature: .run.vlm_temperature,
         seed: .run.vlm_seed,
         system_prompt: .run.vlm_system_prompt,
         inference_params: .run.vlm_inference_params },
  hardware: { backend: .run.compute_backend,
              hostname: .run.hostname },
  packages: .run.packages,
  cli: .run.command
}' out/paper.meta.json
```

This block + the server-side facts you record by hand gives a complete computational disclosure suitable for journal supplementary materials.


## 19. Standalone `vlm-table` CLI — one-off image → markdown / CSV

For ad-hoc table transcription outside the full PDF pipeline (e.g. a screenshot of a single table from a PDF you don't want to re-extract, a figure from a slide deck, a scan of a hand-drawn table), `vlm-table` is a thin standalone wrapper around paper2md's VLM client.

```bash
# After `pip install -e .` the console script is on PATH
vlm-table table.png                              # markdown to stdout
vlm-table table.png -f csv                       # RFC 4180 CSV to stdout
vlm-table table.png -o result.md                 # markdown to file
vlm-table table.png -f csv -o data.csv           # CSV to file
vlm-table table.png -c "Table 3. Shock impedance" # caption hint as VLM context
vlm-table table.png --max-tokens 8000            # bump for very wide tables

# Or as a module
python -m vlm_table table.png
```

**Provider configuration** — `vlm-table` reuses paper2md's `configure_client()`, so it picks up the same env vars: `VLM_PROVIDER`, `VLM_MODEL`, `VLLM_BASE_URL`, `LM_STUDIO_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. No CLI override at the moment — set the env vars to switch providers.

**Output format:**

| Format | Prompt | Output style |
|---|---|---|
| `md` (default) | paper2md's `TABLE_PROMPT` (the same prompt the in-pipeline hook uses) | GitHub-flavored markdown pipe-table; `**Notes:**` block when the image has footnote markers |
| `csv` | dedicated `TABLE_PROMPT_CSV` (RFC 4180 strict) | One header row + one row per data row; quotes any cell containing `,`, `"`, or newline; preserves footnote markers verbatim |

The same VLM determinism defaults apply (`temperature=0`, `seed=42` baked in via `paper2md.vlm()`), so re-running `vlm-table` on the same image produces byte-identical output.

**When to use vs `--replace-table`** — `--replace-table` does almost the same thing but writes outputs into a paper2md output directory (`assets/user_edit_table_<id>.md`) and records the edit in the paper's frontmatter `edits:` block. Use `--replace-table` when fixing a paper2md-produced paper; use `vlm-table` when you just need a transcription, with no paper context.
