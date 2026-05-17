#!/usr/bin/env python3
"""Evaluate hook-2 figure-match accuracy against ground truth.

Reads:
- ground-truth YAML files in tests/figure_match_truth/
- the YAML front-matter from one or more paper2md runs

Reports per-paper and aggregate metrics: accuracy, drop precision/recall,
duplicate rate, missing-figure count. Writes one CSV row per
(paper, condition) so results can be pivoted later.

Usage:

    # Compare baseline against three solution variants
    python tests/bench_figure_match.py \\
      --truth-dir tests/figure_match_truth \\
      --condition baseline:examples/out_canup/canup.md \\
                  baseline:examples/out_young/young-full.md \\
                  baseline:examples/out_jacquet/jacquet.md \\
                  baseline:examples/enstatite/out_millot/millot.md \\
                  baseline:examples/out_elliott/Elliott.md \\
                  baseline:examples/out_aastex/aastex-template.md \\
                  baseline:examples/out_mnras/mnras-template.md

    # Or one condition at a time, then re-run with --append for solutions:
    python tests/bench_figure_match.py --truth-dir tests/figure_match_truth \\
        --condition baseline:examples/out_millot/millot.md \\
        --csv results.csv
    python tests/bench_figure_match.py --truth-dir tests/figure_match_truth \\
        --condition page-prior:out_millot_pp/millot.md \\
        --csv results.csv --append

The truth file's `expected_matches` map is keyed by marker filename
(`_page_N_Figure_M.jpeg`); values are figure id strings, "DROP", or
"TODO" (rows with TODO are skipped from accuracy computation but
counted in a "skipped" tally).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Metrics:
    paper: str
    condition: str
    total_truth_imgs: int = 0      # excluding TODO rows
    skipped_todo: int = 0
    total_run_imgs: int = 0
    accuracy_num: int = 0           # correct (image, fig_id) matches
    accuracy_den: int = 0           # truth rows where expected != DROP
    drop_correct: int = 0           # image was dropped AND truth says DROP
    drop_wrong: int = 0             # image was dropped BUT truth says some fig_id
    drop_missed: int = 0            # truth says DROP, image was kept (matched or unmatched)
    duplicate_groups: int = 0       # count of fig_ids assigned to ≥2 images
    duplicate_count: int = 0        # total extra images involved in duplicates
    missing_figures: list[str] = field(default_factory=list)  # truth fig_ids with no match
    extra_images: list[str] = field(default_factory=list)     # images in run not in truth

    @property
    def accuracy(self) -> Optional[float]:
        if self.accuracy_den == 0:
            return None
        return self.accuracy_num / self.accuracy_den


def read_front_matter(md_path: Path) -> dict:
    """Read YAML front-matter from a paper2md output .md file."""
    text = md_path.read_text()
    if not text.startswith("---"):
        return {}
    end = text.find("\n---\n", 3)
    if end == -1:
        return {}
    return yaml.safe_load(text[3:end]) or {}


def load_truth(truth_path: Path) -> dict:
    """Load and minimally validate a truth YAML file."""
    data = yaml.safe_load(truth_path.read_text()) or {}
    if "expected_matches" not in data:
        raise ValueError(f"{truth_path}: missing 'expected_matches'")
    return data


def evaluate(truth: dict, fm: dict, condition: str) -> Metrics:
    """Compare one run's figure data to the truth."""
    paper = truth.get("paper", "?")
    expected: dict[str, str] = truth.get("expected_matches", {})
    figs: list[dict] = (fm.get("quality") or {}).get("figures", []) or []

    m = Metrics(paper=paper, condition=condition)
    m.total_truth_imgs = sum(1 for v in expected.values() if v != "TODO")
    m.skipped_todo = sum(1 for v in expected.values() if v == "TODO")
    m.total_run_imgs = len(figs)

    # Index the run's figure data by filename
    run_by_fname: dict[str, dict] = {f.get("filename"): f for f in figs}

    # Walk the truth and score
    matched_ids: list[str] = []   # fig_ids the run produced (for dup detection)
    truth_fig_ids: set[str] = set()

    for fname, expected_val in expected.items():
        if expected_val == "TODO":
            continue
        run_entry = run_by_fname.get(fname)
        if run_entry is None:
            # Image is in truth but not in run output (regression or different marker run)
            if expected_val != "DROP":
                m.accuracy_den += 1  # would have counted toward accuracy
                # accuracy_num stays 0
            continue

        was_dropped = bool(run_entry.get("dropped"))
        matched = run_entry.get("matched_figure")

        if expected_val == "DROP":
            if was_dropped:
                m.drop_correct += 1
            else:
                m.drop_missed += 1
            continue

        # Truth says this image should be matched to a specific fig_id
        truth_fig_ids.add(expected_val)
        m.accuracy_den += 1
        if was_dropped:
            m.drop_wrong += 1
        elif matched is not None and str(matched) == str(expected_val):
            m.accuracy_num += 1
            matched_ids.append(str(matched))
        elif matched is not None:
            matched_ids.append(str(matched))
        # else matched is None (kept but no caption assigned) — counts as wrong

    # Duplicate analysis on the run's matched_ids
    counts = Counter(matched_ids)
    m.duplicate_groups = sum(1 for v in counts.values() if v > 1)
    m.duplicate_count = sum(v - 1 for v in counts.values() if v > 1)

    # Missing figures: truth said these fig_ids exist but the run produced 0 matches
    produced = set(matched_ids)
    m.missing_figures = sorted(truth_fig_ids - produced)

    # Extra images: in run but not in truth's expected_matches map (including TODOs)
    truth_filenames = set(expected.keys())
    m.extra_images = [f.get("filename") for f in figs
                      if f.get("filename") not in truth_filenames]

    return m


def fmt_metric(m: Metrics) -> str:
    acc = m.accuracy
    acc_s = f"{acc * 100:5.1f}%" if acc is not None else "  n/a "
    return (
        f"  {m.paper:30s}  acc={acc_s} ({m.accuracy_num}/{m.accuracy_den})  "
        f"drops={m.drop_correct}+/{m.drop_wrong}-/{m.drop_missed}m  "
        f"dups={m.duplicate_groups}g/{m.duplicate_count}n  "
        f"missing_figs={len(m.missing_figures)}"
        + (f"  TODO={m.skipped_todo}" if m.skipped_todo else "")
    )


def aggregate(rows: list[Metrics], condition: str) -> Metrics:
    """Sum a list of per-paper metrics into one row."""
    agg = Metrics(paper="AGGREGATE", condition=condition)
    for r in rows:
        agg.total_truth_imgs += r.total_truth_imgs
        agg.skipped_todo += r.skipped_todo
        agg.total_run_imgs += r.total_run_imgs
        agg.accuracy_num += r.accuracy_num
        agg.accuracy_den += r.accuracy_den
        agg.drop_correct += r.drop_correct
        agg.drop_wrong += r.drop_wrong
        agg.drop_missed += r.drop_missed
        agg.duplicate_groups += r.duplicate_groups
        agg.duplicate_count += r.duplicate_count
        agg.missing_figures.extend(r.missing_figures)
        agg.extra_images.extend(r.extra_images)
    return agg


def write_csv(rows: list[Metrics], csv_path: Path, append: bool) -> None:
    mode = "a" if append and csv_path.exists() else "w"
    write_header = mode == "w"
    with csv_path.open(mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "paper", "condition",
                "total_truth_imgs", "skipped_todo", "total_run_imgs",
                "accuracy_num", "accuracy_den", "accuracy_pct",
                "drop_correct", "drop_wrong", "drop_missed",
                "duplicate_groups", "duplicate_count",
                "missing_figures", "extra_images",
            ])
        for r in rows:
            acc = r.accuracy
            w.writerow([
                r.paper, r.condition,
                r.total_truth_imgs, r.skipped_todo, r.total_run_imgs,
                r.accuracy_num, r.accuracy_den,
                f"{acc*100:.2f}" if acc is not None else "",
                r.drop_correct, r.drop_wrong, r.drop_missed,
                r.duplicate_groups, r.duplicate_count,
                ";".join(r.missing_figures),
                ";".join(r.extra_images),
            ])


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate hook-2 figure-matching against ground truth.")
    p.add_argument("--truth-dir", type=Path, default=Path("tests/figure_match_truth"),
                   help="Directory of *.yaml truth files (default: tests/figure_match_truth)")
    p.add_argument("--condition", action="append", required=True,
                   metavar="NAME:PATH_TO_MD",
                   help="One condition spec per --condition flag, e.g. baseline:out_canup/canup.md. "
                        "Repeat for multiple papers / conditions.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional CSV output path; one row per (paper, condition) plus aggregate.")
    p.add_argument("--append", action="store_true",
                   help="Append to --csv instead of overwriting.")
    args = p.parse_args()

    if not args.truth_dir.is_dir():
        print(f"error: truth dir not found: {args.truth_dir}", file=sys.stderr)
        return 2

    # Build a lookup paper-stem -> truth file path
    truth_files = {t.stem: t for t in args.truth_dir.glob("*.yaml")}
    if not truth_files:
        print(f"error: no truth YAMLs found in {args.truth_dir}", file=sys.stderr)
        return 2

    # Group conditions by name for grouped reporting
    conditions: dict[str, list[Path]] = {}
    for spec in args.condition:
        if ":" not in spec:
            print(f"error: bad --condition spec '{spec}', expected NAME:PATH", file=sys.stderr)
            return 2
        name, path = spec.split(":", 1)
        conditions.setdefault(name.strip(), []).append(Path(path.strip()))

    all_rows: list[Metrics] = []

    for cond_name, md_paths in conditions.items():
        print(f"\n=== condition: {cond_name} ===")
        per_paper: list[Metrics] = []
        for md_path in md_paths:
            if not md_path.exists():
                print(f"  WARN: skipping missing run output: {md_path}", file=sys.stderr)
                continue
            stem = md_path.stem
            # Try direct stem match, then strip "-full" / "-template" suffixes
            truth_path = truth_files.get(stem)
            if truth_path is None:
                stem_short = stem.replace("-full", "").replace("-template", "")
                truth_path = truth_files.get(stem_short)
            if truth_path is None:
                print(f"  WARN: no truth file for {md_path.name} (looked for {stem}.yaml)",
                      file=sys.stderr)
                continue
            truth = load_truth(truth_path)
            fm = read_front_matter(md_path)
            m = evaluate(truth, fm, cond_name)
            per_paper.append(m)
            print(fmt_metric(m))

        if per_paper:
            agg = aggregate(per_paper, cond_name)
            print(fmt_metric(agg))
            all_rows.extend(per_paper)
            all_rows.append(agg)

    if args.csv:
        write_csv(all_rows, args.csv, args.append)
        print(f"\nwrote {len(all_rows)} row(s) to {args.csv}"
              + (" (appended)" if args.append and args.csv.exists() else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
