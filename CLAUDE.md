# paper2md — orientation for Claude Code

This file tells an AI agent (Claude Code or similar) how the repo is
organised and where to look when helping a user with paper2md. For
hands-on usage by humans, the canonical reference is `docs/USAGE.md`.

## What this is

Local PDF → Markdown pipeline for scientific journal articles. Two
layout engines behind a single CLI:

- `--layout-source mineru` (default): MinerU's pipeline backend (PaddleOCR + layout)
- `--layout-source marker`: marker + surya OCR with paper2md's own table finder
- `--layout-source hybrid` (experimental): marker body+caption text + MinerU figure/table layout, spliced by figure/table number
- `--auto-layout-source`: triage detector routes per-paper (scans → marker, born-digital → mineru)

A vision LLM provides targeted post-passes for tables, figures, page
rescues, citations, and bundling. Backends:

| Hardware | VLM provider | Env file |
|---|---|---|
| NVIDIA CUDA (DGX Spark) | vLLM (default Qwen3-VL-32B) | `environment-gpu.yml` |
| Apple Silicon MPS | LM Studio | `environment-mac.yml` |

Single entry point: `src/paper2md.py`. The CLI is `python src/paper2md.py
paper.pdf -o out/` plus flags — see `docs/USAGE.md` §3 for the complete
reference.

## Repo layout

```
paper2md/
├── README.md                  ← project overview, install, quick start
├── docs/
│   ├── USAGE.md               ← definitive user guide. Start here for any "how do I…" question.
│   ├── BATCH.md               ← unattended large-corpus runbook
│   ├── setup/                 ← per-machine install (Spark, Sol HPC)
│   ├── comparisons/           ← paper2md vs MinerU / docling / pymupdf4llm / etc.
│   ├── design/                ← architectural / pipeline notes (FLOWCHART, COPYRIGHT_AND_REUSE)
│   └── dev/                   ← contributor / maintainer notes (see "If you're touching code")
├── src/
│   ├── paper2md.py            ← main pipeline + CLI
│   ├── layout_mineru.py       ← MinerU subprocess wrapper + middle.json parser
│   ├── wrap_mineru.py         ← MinerU-spine standalone entry (alternative to paper2md.py)
│   ├── metadata_frontend.py   ← Crossref / OpenAlex / Unpaywall lookups
│   ├── data_repos.py          ← Zenodo / Dryad / Dataverse link extraction
│   ├── triage_pdfs.py         ← scan-vs-born-digital detector for --auto-layout-source
│   ├── unpack_h5.py / repack_h5.py  ← --hdf5 bundle utilities
│   └── split_pdf.py           ← debug / one-off splitter
├── tests/
│   ├── test_*.py              ← unit + integration tests (run with pytest)
│   ├── bench_*.py             ← evaluation harnesses (not picked up by pytest)
│   ├── figure_match_truth/    ← truth fixtures for figure-caption pairing
│   └── table_extract_truth/   ← truth fixtures for table extraction
├── examples/                  ← stub README pointing to $TEST_FILES_DIR
├── CITATION.cff               ← Zenodo / GitHub-cite metadata
├── LICENSE                    ← MIT
└── pyproject.toml             ← pip install -e .
```

## Conventions for an AI agent

- **Don't auto-commit.** Make edits, run tests, then ask the user before `git commit`. Same for `git push`.
- **Prefer editing existing files** over creating new ones.
- **Don't create planning / summary `.md` files** unless asked.
- **License + CITATION are stable** — don't reformat or restructure them.
- **Test before changing user-facing behavior**: `python -m pytest tests/ -q`. The full suite should pass (~660 tests).
- **The pipeline default is `--layout-source mineru`** as of v0.3.x (carried through v0.4.x). Tests reflect this; if you find code paths that assume marker is the default, that's stale and worth fixing.

## When a user comes in with a "problem paper"

Typical things to check, in order:

1. **What layout source was used?** Different engines have different failure modes:
   - `mineru` issues: usually about MinerU's middle.json — check `<out>/mineru/<stem>_middle.json`
   - `marker` issues: usually about surya OCR or hook 1/2 — check the `_image*.jpeg` outputs
   - `hybrid` issues: see `docs/dev/HYBRID_IMPLEMENTATION.md` for the canonical fudges inventory; the splice has many known corpus-specific workarounds

2. **What does the YAML frontmatter say?** Open the output `.md`. The `run:` block records every toggle, the `rescues:` block records every rescue counter that fired, the `quality:` block gives per-table/per-figure scores.

3. **What does the `.meta.json` sidecar say?** Same content as frontmatter but structured (use `jq`). For batch runs, `manifest.jsonl` aggregates one line per paper.

4. **For VLM-related issues** (empty rewrites, connection errors), check:
   - vLLM / LM Studio server health: `curl http://localhost:{8000,1234}/v1/models`
   - The configured model: `$VLM_MODEL` env var
   - `docs/USAGE.md` §7 troubleshooting tree

5. **For batch errors**, the manifest now records `error_kind` (e.g. `cuda_oom`, `model_load`, `pdf_parse`) and `error_stderr_tail` for MinerU subprocess failures. `jq 'select(.status=="error")'` filters them.

## If you're touching code

- `docs/dev/HYBRID_IMPLEMENTATION.md` — canonical fudges inventory for the hybrid layout. Each fudge documents the pattern, conditions, code locations, telemetry counter, and what changes break it. **Read this first if touching the hybrid splice.**
- `docs/dev/CLAUDE.md` — dev-mode notes: fudge-tracking discipline, version-bump checklist, open-frontier items.
- `docs/dev/RELEASE_INSTRUCTIONS.md` — for maintainers cutting a tagged release.
- `docs/dev/EXAMPLES_FIXTURES.md` — how `tests/conftest.py` finds the `$TEST_FILES_DIR` PDFs.

## Common troubleshooting one-liners

```bash
# Re-grade an existing output without re-running (after a scoring bug fix)
python -c "import json; print(json.load(open('out.meta.json'))['quality'])"

# See what triggered for a paper
jq '.rescues' out.meta.json

# Find all batch errors of a given kind
jq -r 'select(.error_kind=="cuda_oom") | .pdf' manifest.jsonl

# Run only one test module
python -m pytest tests/test_hybrid_layout.py -q

# Pipeline state for a paper (toggles, versions, run-info)
jq '.run' out.meta.json
```
