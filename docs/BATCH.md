# BATCH.md — running paper2md on a large corpus (Spark)

Practical runbook for unattended batch ingestion of hundreds-to-thousands of PDFs from a Zotero library or similar corpus, on the DGX Spark.

For the underlying batch-mode CLI flags, manifest schema, SI-pairing rules, and per-paper output layout, see **USAGE.md §15 (Batch mode)** — this file is just the operational how-to.

---

## 1. Throughput model

`--workers N` runs `_do_convert()` in a `ThreadPoolExecutor`, but **Marker / surya OCR is fully serialized** by an internal `_MARKER_LOCK`. Only one paper runs Marker on the GPU at a time. The VLM hooks (network calls to vLLM) overlap because Python releases the GIL on socket reads and vLLM batches server-side.

**Per-paper time breakdown** (typical 16-page dense scientific paper):

| Phase | Wall time | Bound on |
|---|---|---|
| Marker (surya OCR + layout) | 10–20 min | GPU, serialized across workers |
| VLM hooks (citation + tables + figures + page rescues + trim) | 30–90 sec total | Network I/O, parallel |
| Deterministic passes (line-strip, ref-renumber, footnote-consolidate) | < 1 sec | CPU |

Marker dominates by ~10–20×. So **`--workers 2` already hides essentially all the VLM time** underneath the next paper's Marker run. Going from 2 → 4 buys very little throughput on a Marker-bound corpus. Use `--workers 4` only if the corpus has many text-only papers (`--text-only`, no Marker) or unusually heavy table/figure VLM passes.

## 2. Estimating wall time for your corpus

Steady-state throughput ≈ **1 / Marker_time_per_paper** at workers ≥ 2.

| Corpus mix | Avg per-paper wall | 1,000 papers |
|---|---|---|
| Short articles (5–8 pages) | ~6 min | **~4 days** |
| Mixed scientific (8–15 pages) | ~12 min | **~8–9 days** |
| Dense methods-heavy (20+ pages) | ~25 min | **~17 days** |

For a quick estimate on your specific library: run a small `--batch` of 10 representative papers, look at `manifest.jsonl` `elapsed_sec` mean, multiply by your corpus size.

## 3. One-time setup

- **Conda env active**: `conda activate paper2md`. See CLAUDE.md "Quick start" for the env install.
- **vLLM model cached** under `~/.cache/huggingface/hub/` (one-time ~64 GB for `Qwen/Qwen3-VL-32B-Instruct` BF16 — see USAGE.md §14).
- **Marker / surya weights cached** under `~/.cache/datalab/` (one-time ~5 GB, downloads on first run).
- **Disk budget for the batch output**:
  - ~5–10 MB per paper (markdown + `assets/`)
  - Add `--hdf5` to roughly double that (loose files + bundle)
  - 1,000 papers ≈ **10–20 GB** plus the one-time caches above.
- **GPU memory**: vLLM holds ~80 GB unified, Marker adds ~8–10 GB. Free RAM during steady state is tight (~9 GB on a 128 GB Spark) but stable.

## 4. The run

Two persistent tmux sessions: one for vLLM, one for paper2md. Detach with `Ctrl+B` then `D`; reattach with `tmux attach -t <name>`.

**Pre-flight survey (recommended before kicking off a multi-hour batch on a mixed corpus):**

```bash
python src/paper2md.py --batch /path/to/your/corpus --scan-detect-only \
  -o /path/to/out > /path/to/out/survey.tsv 2>&1
```

The survey TSV has one row per PDF with `pair_role` (main / si / solo / standalone-si), `pair_main` (the paired main's stem), scan-detection verdict + signals (producer/creator/image_cover_median/font_count/bold_char_ratio). Check pairings + scan flags before committing to the run.

```bash
# --- session 1: vLLM ---
tmux new -s vllm
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.80
# wait for "Application startup complete", then Ctrl+B D

# --- session 2: paper2md ---
tmux new -s paper2md
cd ~/datasets/paper2md/claude/paper2md
python src/paper2md.py --batch /path/to/your/corpus -o /path/to/out \
  --workers 2 --hdf5 --auto-force-ocr --paper-timeout 1800 \
  -v 2>&1 | tee /path/to/out/run.log
# Ctrl+B D to detach
```

Notes:

- Use `--workers 2`, not 4 (see §1).
- `--hdf5` produces a per-paper `.h5` bundle alongside loose files. Useful for archival / Zotero attachment.
- Keep the `tee … run.log` so you have a full transcript even if you forget to attach the tmux pane.
- The `--batch` argument can be a directory (lists `*.pdf` one level deep) or a glob.
- SI auto-pairing is on by default — `foo.pdf` + `foo_SI.pdf` / `foo_sm.pdf` / `foo_supplement.pdf` / `foo_S1.pdf` / `foo-maintext.pdf` + `foo-supplemental.pdf` get bundled into a single output dir.
- **`--auto-force-ocr` (v0.3+)**: per-paper detector flips `--force-ocr` on for scanned / pre-2000 papers without the user having to segregate the corpus. Cost: ~100-300 ms / paper for the detection sniff; ~30-60 s / page extra surya OCR cost on flagged papers. No effect on papers classified as digital. Mutex with `--force-ocr` (which forces unconditionally).
- **`--paper-timeout SECONDS` (v0.3+)**: abandons any single paper exceeding the deadline (default 1800 s = 30 min). Failed paper recorded as `status: "timeout"` in `manifest.jsonl`; the next paper starts immediately. Caveat: abandoned thread may hold marker lock / GPU memory; for long runs with frequent hangs, split the batch across multiple paper2md invocations so each one starts with a fresh Python process.
- **`--clean` (v0.3+)**: per-paper output wipe before re-running. Removes only this paper's `<stem>.md` / `.meta.json` / `.h5` and matching files in `assets/`; other corpus papers are untouched. Skip in batch unless you specifically want to discard prior runs.

## 5. Monitoring during the run

In a third tmux pane (or any shell):

```bash
# basic progress (count + last entry)
watch -n 30 '
  wc -l /path/to/out/manifest.jsonl
  echo
  tail -1 /path/to/out/manifest.jsonl | jq
'

# live status breakdown
jq -s '
  group_by(.status)
  | map({status:.[0].status, n:length})
' /path/to/out/manifest.jsonl
```

What "stuck" looks like: `manifest.jsonl` doesn't grow for 30+ min during steady state. Investigate:

```bash
# is vLLM still serving?
curl http://localhost:8000/v1/models

# is paper2md still running?
ps -p $(pgrep -f "src/paper2md.py.*--batch") -o pid,pcpu,pmem,etime,cmd

# what's the most recent log line?
tail -50 /path/to/out/run.log
```

If vLLM died: restart it in its tmux session. The paper2md run will recover on its own — pending VLM calls retry, and worst-case the current paper is logged with `status: "error"` and the next paper proceeds.

If paper2md died: just re-run the same command. Resume is built in — papers whose `<out>/<stem>/<stem>.md` exists are skipped in <1 second each.

## 6. After the batch

```bash
cd /path/to/out

# how did it go overall?
jq -s 'group_by(.status) | map({status:.[0].status, n:length})' manifest.jsonl

# papers that errored — investigate the .error field for each
jq -s 'map(select(.status == "error"))' manifest.jsonl

# the worklist for selective frontier-model review:
# papers with overall quality < 0.85, sorted ascending
jq -s '
  map(select(.overall and .overall < 0.85))
  | sort_by(.overall)
  | map({pdf:.pdf, grade:.grade, overall:.overall})
' manifest.jsonl

# total wall time
jq -s 'map(.elapsed_sec) | add / 3600 | "\(.) hours"' manifest.jsonl
```

## 7. Selective frontier-model fix-up

For the bottom-N% of the quality distribution (typically 5–10% of a corpus), the cost-effective workflow is **local pipeline produces the bulk + frontier model fixes the hard cases**, not "pure frontier model from scratch" (which costs ~$1–2/paper × 1,000 = $1k–$2k).

Workflow per problem paper:

1. Identify the problem from the worklist (above) — usually a specific table or figure.
2. Open the source PDF, screenshot the problem region.
3. Paste into a frontier-model conversation with a focused prompt ("convert this table crop to GitHub-flavored markdown, preserve every cell exactly").
4. Replace the affected sidecar in `<out>/<stem>/assets/`:
   - For a table: replace `assets/table_pN_M.md` with the corrected markdown.
   - For the main markdown body: edit `<out>/<stem>/<stem>.md` directly.
5. Repack the HDF5 bundle so it picks up the change:
   ```bash
   python src/repack_h5.py /path/to/out/<stem>/<stem>.h5
   ```
   See USAGE.md §12.5 for what repack does (preserves quality grade, source_pdf, etc.; updates `tool` to `"src/repack_h5.py"` and adds an `original_created_at` provenance attr).

This keeps frontier-model spend at ~$1–10 per problem paper × the 5–10% of papers that need it = ~$50–200 for a 1,000-paper corpus, vs ~$2,000 for full-corpus Opus.

## 8. When NOT to use this batch flow

- **Single paper or fewer than ~20 papers**: just run `python src/paper2md.py paper.pdf -o out/` directly. The batch mode adds nothing useful at small scale.
- **Latency-sensitive (need result in <5 min)**: not this pipeline. Marker dominates wall time; consider a `--text-only` run (skips Marker, VLM-converts each page directly — much faster but no figures/tables).
- **Quality budget allows full-corpus Opus**: if you have ~$2k to spend per ingest pass and need top-tier output without local infrastructure, image→markdown via Opus API is a real alternative.

## 9. Power / heat / sound

- Spark draws ~80 W under load. 7 days × 24 h × 80 W ≈ **13 kWh ≈ $2** of electricity.
- Fans audible but quiet at typical room ambient.
- Keep the box in a well-ventilated spot; thermal throttling on the GB10 will silently slow Marker if it overheats.
