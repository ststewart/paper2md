# paper2md — developer notes (dev-mode CLAUDE)

Internal notes for a Claude Code session driving paper2md development.
The public-facing orientation is at `../../CLAUDE.md`; this file
captures the dev-mode discipline that doesn't belong in the public
repo's project brief.

## Version + license

**Version: 0.4.x — MIT License** (see `LICENSE`). Authors: Sarah T. Stewart;
Claude (Anthropic, Opus 4.7). Cite as:

> *Stewart, S. T., & Claude (Anthropic, Opus 4.7). (2026). paper2md
> (v0.4.x) [Software]. MIT License.*

The runtime banner appends a Zenodo DOI (`https://doi.org/10.5281/zenodo.NNNNN`)
once `__doi__` in `src/paper2md.py` is filled in for the release. Every run
prints this banner in the first three INFO log lines.

The version is also recorded in every output's YAML front-matter and
`.meta.json` sidecar under `run.paper2md_version` / `run.paper2md_license`,
alongside the resolved Python version, backend, VLM endpoint, pipeline
toggle states, and version strings for the load-bearing packages —
`marker-pdf`, `surya-ocr`, `pymupdf`, `pillow`, `openai`, `anthropic`,
`requests`, `h5py`, `python-dotenv`. This is what makes a run reproducible
months later.

## Hybrid-layout fudges — tracking and maintenance

The `--layout-source hybrid` path has accumulated a sequence of shipped
workarounds (currently 11 plus diagnostic-only commits) for real-world
MinerU quirks. Each is targeted at a specific corpus pattern, has a
frontmatter counter or INFO log line, and is structurally tied to MinerU's
current pin.

**Tracking discipline:**

- Add new fudges as their own section in `HYBRID_IMPLEMENTATION.md`
  following the template (pattern, conditions, code locations, telemetry,
  what changes break it).
- Surface a counter in `rescues.hybrid_splice` so corpus-wide audits are
  observable.
- On MinerU version bumps: run the "When MinerU version changes" checklist
  in the doc. Counter going corpus-wide-zero → MinerU fixed it → propose
  deletion.

## Hybrid memory pressure (Spark unified-memory specific)

vLLM was sized for marker-only workflows on the Spark GB10 (124 GB unified
memory). Hybrid adds MinerU as a SECOND CUDA consumer in the same paper2md
process tree — marker holds surya in the parent; MinerU's subprocess loads
PaddleOCR + layout. Recommendation:

- **vLLM `--gpu-memory-utilization 0.65`** for hybrid runs (was 0.80 for
  marker-only). Leaves ~32 GB headroom vs 21 GB. Empirically the
  difference between "ocampo / boslough OOM" and "all clean".
- KV cache shrinks from ~36 GB to ~14 GB, supports `Maximum concurrency for
  32,768 tokens per request: 1.74×` — fine for paper2md's burst pattern
  (4 concurrent table calls × ~10K tokens each).
- The `run_mineru` wrapper calls `torch.cuda.empty_cache()` in the parent
  before subprocess.run to return marker's cached blocks to the driver.

## When MinerU fails

`layout_mineru.run_mineru` now raises `MineruRunError` (RuntimeError
subclass) carrying `returncode`, `pdf_path`, classified `kind`, and the
last ~50 lines of stderr. The batch driver surfaces these in
`manifest.jsonl` as `error_kind` / `error_pdf` / `error_stderr_tail`.

Classified kinds:

| kind | meaning | typical fix |
|---|---|---|
| `cuda_oom` | OOM during MinerU CUDA init | Lower vLLM `--gpu-memory-utilization`; clear parent cache (already done) |
| `model_load` | MinerU couldn't find a weight file | Network issue at first download; re-run |
| `torch_init` | NVIDIA driver / CUDA version mismatch | Setup problem; check `nvidia-smi` |
| `disk` | No space left on device | Free up `$HOME/.cache/{huggingface,paddle}` |
| `pdf_parse` | Malformed PDF | Try `qpdf --linearize` or skip |
| `unknown` | No pattern matched | Check `error_stderr_tail` manually |

Filter the manifest with `jq 'select(.error_kind == "cuda_oom") | .pdf'`.

## Memory references (Claude Code, dev-side)

Sarah's persistent memory keeps:

- `paper2md_state_YYYY_MM_DD.md` — end-of-session snapshots (HEAD commit,
  test count, what works, what's broken, open frontiers). Read the most
  recent one FIRST in any new session.
- `hybrid_fudges_inventory.md` — compact one-row-per-fudge index.
- `vlm_table_rewrite_mystery.md` — root cause + lessons from the
  sys.modules duplicate-import bug saga.
- `feedback_*.md` — Sarah's collaboration preferences (commit per task,
  API refs over VLM, etc.).

These memory files are NOT in the repo; they're at
`~/.claude/projects/.../memory/`.

## Open frontier items (2026-05-16)

- **`vlm-empty-or-failed` table rewrites** (Wackerle T2, cuk SI T1) —
  VLM returns empty / hangs on wide tables even at `max_tokens=6000`.
  Needs per-call wall-time instrumentation to distinguish hang from
  empty-response. Defer.
- **Audit cross-paper exclusion** — `Table V of Wackerle 1962` style
  citations still false-positive in the audit. Defer.
- **`--repair-output` mode** for already-extracted papers (apply new
  naming + conservative splice without re-running extraction). Defer.

## Conventions specific to dev sessions

- **Commit after every task.** Don't accumulate multi-task diffs in the
  working tree. Per Sarah's standing preference.
- **API ref-fetch (Crossref/OpenAlex) is canonical** for reference rescue
  on papers with a DOI. Prefer it over VLM-based ref reconstruction.
- **Author identity for commits**: `Sarah T. Stewart <sstewa56@asu.edu>`.
  `api.partake757@passfwd.com` is for API mailto only, never commits.
- **transfer/** is a separately-maintained frozen v0.3.0 snapshot in
  the workspace (not in this repo). Don't propagate dev changes there
  without explicit instruction.

## What's new (current dev branch)

See `git log --oneline -20` for the recent commit history. Notable since
v0.3.0 (now bundled into the v0.4.0 release):

- Conservative hybrid splice: body mutations are additive-only; unmatched
  tables collected into a start-of-doc `## Extracted tables` index
  (sidesteps refs-walks-from-end hooks)
- Semantic asset naming: `table_{id}_p{page}_{idx}.{md,jpg}`,
  `figure_{id}{letter?}_p{page}.{ext}` across all layouts (sidecars and
  matching JPGs share stems)
- Deterministic VLM by default: `temperature=0`, `seed=42` sent on every
  call; recorded in `run.vlm_temperature` / `run.vlm_seed`
- `--vlm-tables-force` to VLM-rewrite every table sidecar (clean
  pipe-md body kept; sidecar gets VLM)
- `--replace-table` / `--replace-fig` / `--revert-edit` for manual
  one-off fixes from a user-provided crop
- `--recover-from-mineru` / `--confirm-recovery` to auto-stage
  audit-flagged tables MinerU found but the hybrid splice dropped
- Standalone `vlm-table` CLI for one-off image → markdown / CSV
- MinerU pin **3.1.7** with three-layer defense: env-file pins, USAGE
  warning, runtime version check
- AI / VLM disclosure section in USAGE (§18) with template paragraph
  for publications

## Quick recipes for dev work

```bash
# Run the full test suite
python -m pytest tests/ -q

# Run only hybrid-related tests
python -m pytest tests/test_hybrid_layout.py tests/test_wrap_mineru.py -q

# Smoke a single paper end-to-end (use $TEST_FILES_DIR fixture)
python src/paper2md.py "$TEST_FILES_DIR/canup.pdf" \
    -o "$WORKFLOW_DIR/md_database/test/canup-smoke" --layout-source hybrid

# Diff a re-run against the baseline corpus reference
diff -r workflow/md_database/p2m-hybrid-baseline workflow/md_database/p2m-hybrid-rerun

# Classify all batch errors by kind
jq -r '[.pdf, .error_kind] | @tsv' manifest.jsonl | sort -k2
```
