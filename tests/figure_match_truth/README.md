# Figure-match experiment runbook

How to actually run the figure-match A/B (issue 4). All scaffolding
is in place after commits `76dfa1b` (harness + truth) and `20cf0e6`
(--figmatch-strategy variants).

## 1. Fill in the TODO fields in the truth files

Open each PDF and complete the `expected_matches` map plus the
`captions` dict in:

- `canup.yaml`        (4 images, 1 dup-pair on Fig 2)
- `young.yaml`        (9 images, 5 mapped to Fig 3)
- `jacquet.yaml`      (17 images, 5 mapped to Fig 8)
- `aastex.yaml`       (1 image, decide if expected match or DROP)

`millot.yaml`, `Elliott.yaml`, and `mnras.yaml` are already filled.

Each truth file's comment lists the current baseline classification
next to each marker filename, so you can see at a glance which ones
the baseline got wrong.

`expected_matches` value conventions:
- `"1"`, `"2"`, `"S1"` — the figure id as it appears in the PDF caption
- `DROP` — image is a banner / icon / logo / extraction fragment
- `TODO` — not yet annotated (rows skipped from accuracy until filled)

Estimated time: ~5 min per paper, ~20 min total for the four with TODOs.

## 2. Run the four conditions

For each of the seven test papers, run paper2md once per condition
into a separate output directory. The split_pdf helper makes
single-paper iteration fast — but for the experiment we want full
runs because hook 2 sees the same captions list in either case.

Spark / GH200 with vLLM already warm:

```bash
cd /scratch/$USER/paper2md-workspace/paper2md  # or wherever
PAPERS=(
  "examples/enstatite/millot.pdf:examples/enstatite/millot_si.pdf:millot"
  "examples/canup.pdf::canup"
  "examples/young-full.pdf::young"
  "examples/jacquet.pdf::jacquet"
  "examples/Elliott.pdf::Elliott"
  "examples/aastex-template.pdf::aastex"
  "examples/mnras-template.pdf::mnras"
)

for spec in "${PAPERS[@]}"; do
  IFS=: read -r pdf si name <<< "$spec"
  for strat in single page-prior dup-detect vote; do
    out="examples/eval/${name}_${strat}"
    [ -n "$si" ] && si_arg="--supplement $si" || si_arg=""
    python src/paper2md.py "$pdf" $si_arg \
      --figmatch-strategy "$strat" \
      -o "$out"
  done
done
```

7 papers × 4 conditions = 28 runs. Marker dominates wall-clock; on
the GH200 expect ~5–15 min/paper, so 4× re-runs is ~20–60 min/paper
serial. Use `--workers 4` in batch mode if you can parallelise.

**If you need it faster**, hook 2 is deterministic given (image,
captions list). You could implement a hook-2-only re-run mode that
re-uses the existing `out_*/assets/` from the baseline run — that
turns each variant into ~10 sec/paper instead of 10 min/paper. Not
implemented yet; flag if you want it before the next experiment.

## 3. Score the runs

```bash
python tests/bench_figure_match.py \
  --truth-dir tests/figure_match_truth \
  --condition single:examples/eval/millot_single/millot.md \
  --condition single:examples/eval/canup_single/canup.md \
  --condition single:examples/eval/young_single/young-full.md \
  --condition single:examples/eval/jacquet_single/jacquet.md \
  --condition single:examples/eval/Elliott_single/Elliott.md \
  --condition single:examples/eval/aastex_single/aastex-template.md \
  --condition single:examples/eval/mnras_single/mnras-template.md \
  --condition page-prior:examples/eval/millot_page-prior/millot.md \
  ... \
  --csv tests/figure_match_truth/results.csv
```

The harness prints a per-paper line plus an aggregate row per
condition. The CSV output is ready to pivot in pandas / a
spreadsheet.

## 4. Decide

Per the plan (`docs/FIGURE_MATCH_TEST_PLAN.md` § "Decision criteria"):

1. No regression on the easy-case sanity papers (Elliott,
   aastex, mnras) — every solution must keep accuracy at the
   baseline level on these.
2. ≥10 percentage-point aggregate accuracy gain on the four
   problem cases (millot, canup, young, jacquet) over baseline.
3. Duplicate rate driven to ~0 on those four.
4. Tiebreak by extra VLM cost: prefer single < page-prior <
   dup-detect < vote, and if A and B win on disjoint papers,
   try `--figmatch-strategy page-prior+dup-detect`.

If no single strategy clears the bar:
- Combinations (`page-prior+dup-detect`, `vote+dup-detect`) are
  worth trying.
- If even combinations don't fix it, the bug is upstream in
  marker (image extracted at wrong PDF coordinates, captions
  list missing entries) and hook 2 can't fix it. Escalation
  path is in the plan.

## 5. Land the winner

Once a strategy clears the bar:

1. Update the FIGMATCH_STRATEGY default in `src/paper2md.py`
   from `"single"` to the winner.
2. Update `CLAUDE.md` and `USAGE.md` to mention the new default.
3. Commit the truth files (now filled in) so future regression
   runs have ground truth.
