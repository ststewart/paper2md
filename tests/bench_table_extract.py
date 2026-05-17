#!/usr/bin/env python3
"""Evaluate hook-1 table-extraction coverage against ground truth.

Reads:
- ground-truth YAML files in tests/table_extract_truth/
- the .meta.json sidecar from one or more paper2md runs (canonical
  source of which tables produced sidecars and on which page)
- the per-table .md sidecars (assets/<prefix>table_p<N>_<i>.md), to
  check column-header coverage when the truth file lists expected
  columns

Reports per-paper and aggregate scores on three axes:
- detection: % of expected tables that produced a sidecar at all
- page accuracy: of detected tables, % with matching PDF page
- column match: of detected tables with truth columns listed,
  Jaccard similarity of column headers (sidecar vs truth)

Usage:

    python tests/bench_table_extract.py \\
        --truth-dir tests/table_extract_truth \\
        --condition baseline:out/out_root_prompt \\
                    baseline:out/out_millot_prompt

    # Compare conditions side by side via --csv
    python tests/bench_table_extract.py --truth-dir tests/table_extract_truth \\
        --condition baseline:out/out_root_prompt --csv table_results.csv
    python tests/bench_table_extract.py --truth-dir tests/table_extract_truth \\
        --condition rescue:out/out_root_prompt_rescue --csv table_results.csv \\
        --append

Each --condition argument is `<label>:<paper-output-dir>`. The harness
expects the output dir to contain <stem>.meta.json (where stem matches
truth.paper minus .pdf) and an assets/ subdir with the table sidecars.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TableScore:
    truth_id: str           # e.g. "S1", "I", "VIII"
    truth_page: int
    truth_label: str
    detected: bool = False
    detected_page: Optional[int] = None
    sidecar_path: Optional[Path] = None
    column_jaccard: Optional[float] = None  # only when truth.columns present


@dataclass
class PaperScore:
    paper: str
    condition: str
    rows: list[TableScore] = field(default_factory=list)
    extra_emitted: int = 0   # sidecars that don't map to any truth row

    @property
    def detection_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.detected) / len(self.rows)

    @property
    def page_accuracy(self) -> float:
        det = [r for r in self.rows if r.detected]
        if not det:
            return 0.0
        return sum(1 for r in det if r.detected_page == r.truth_page) / len(det)

    @property
    def column_avg(self) -> Optional[float]:
        with_cols = [r for r in self.rows if r.column_jaccard is not None]
        if not with_cols:
            return None
        return sum(r.column_jaccard for r in with_cols) / len(with_cols)


_TABLE_HEADER_RE = re.compile(r"^\|(.+)\|\s*$", re.MULTILINE)


def _parse_sidecar_columns(sidecar: Path) -> list[str]:
    """Read a per-table sidecar markdown and pull the first header
    row's column names (split on |, stripped, empty-cells preserved
    as ''). Returns [] on parse miss."""
    try:
        text = sidecar.read_text()
    except OSError:
        return []
    m = _TABLE_HEADER_RE.search(text)
    if not m:
        return []
    cells = [c.strip() for c in m.group(1).split("|")]
    return cells


def _normalize_col(s: str) -> str:
    """Lowercase + strip non-alphanumerics for tolerant column matching."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _column_jaccard(truth: list[str], detected: list[str]) -> float:
    t = {_normalize_col(c) for c in truth if c}
    d = {_normalize_col(c) for c in detected if c}
    if not t or not d:
        return 0.0
    inter = t & d
    union = t | d
    return len(inter) / len(union)


def _find_caption_in_md(md_text: str, label: str) -> bool:
    """Did paper2md's body markdown emit a recognizable caption for
    this table? Useful for distinguishing 'never seen' from 'seen
    but failed to extract'."""
    # Loose match: 'TABLE I.' / 'Table I:' / '**Table S6.**' etc.
    pat = re.compile(rf"\b{re.escape(label)}\b\s*[\.\:]", re.IGNORECASE)
    return bool(pat.search(md_text))


def evaluate(truth_path: Path, out_dir: Path,
             condition: str) -> Optional[PaperScore]:
    """Returns None if the out_dir does not contain this truth file's
    paper (so callers can silently skip rather than print empty rows
    for every cross-product mismatch)."""
    truth = yaml.safe_load(truth_path.read_text())
    paper_stem = Path(truth["paper"]).stem
    meta_path = out_dir / f"{paper_stem}.meta.json"
    if not meta_path.is_file():
        return None

    score = PaperScore(paper=paper_stem, condition=condition)
    md_path = out_dir / f"{paper_stem}.md"
    meta = json.loads(meta_path.read_text())
    tables_meta = meta.get("quality", {}).get("tables", [])

    # Build a map: detected_page -> list of (meta entry, parsed columns)
    by_page: dict[int, list[tuple[dict, list[str]]]] = {}
    for tm in tables_meta:
        if not tm.get("located"):
            continue
        page = tm.get("page")
        sidecar_name = tm.get("sidecar")
        if page is None or not sidecar_name:
            continue
        sidecar_path = out_dir / "assets" / sidecar_name
        cols = _parse_sidecar_columns(sidecar_path) if sidecar_path.is_file() else []
        by_page.setdefault(page, []).append((tm, cols))

    used_meta_ids: set[int] = set()
    expected = truth.get("expected_tables") or {}

    # Two-phase matching: first pass assigns each truth row to the
    # best-Jaccard sidecar on its page (so on a page with two truth
    # tables and one sidecar, the right truth row wins instead of
    # whichever appears first in YAML order). Second pass falls back
    # to "any unused sidecar on the page" for truth rows that don't
    # specify columns.
    for tid, t in expected.items():
        row = TableScore(truth_id=str(tid),
                         truth_page=t["page"],
                         truth_label=t["caption_label"])
        cands = [(tm, cols) for tm, cols in by_page.get(t["page"], [])
                 if id(tm) not in used_meta_ids]
        if not cands:
            score.rows.append(row)
            continue
        chosen = None
        chosen_jac = None
        if t.get("columns"):
            scored = [(_column_jaccard(t["columns"], cols), tm, cols)
                      for tm, cols in cands]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_jac, best_tm, best_cols = scored[0]
            # Only bind by columns if the best Jaccard meets a
            # minimum -- otherwise we'd happily attach S1 to S2's
            # sidecar just because they share a page.
            if best_jac >= 0.30:
                chosen = best_tm
                chosen_jac = best_jac
        if chosen is None:
            # Fall through: take any unused sidecar on the page
            # (truth rows without expected columns or where no
            # sidecar has a strong column match).
            chosen, chosen_cols = cands[0]
            if t.get("columns"):
                chosen_jac = _column_jaccard(t["columns"], chosen_cols)
        used_meta_ids.add(id(chosen))
        row.detected = True
        row.detected_page = chosen.get("page")
        sidecar_name = chosen.get("sidecar")
        row.sidecar_path = (out_dir / "assets" / sidecar_name
                            if sidecar_name else None)
        row.column_jaccard = chosen_jac
        score.rows.append(row)

    score.extra_emitted = sum(
        1 for tm in tables_meta
        if tm.get("located") and tm.get("sidecar") and id(tm) not in used_meta_ids
    )
    return score


def render_paper(score: PaperScore) -> str:
    L = [f"\n=== {score.paper} ({score.condition}) ==="]
    L.append(f"  detection: {score.detection_rate:.1%} "
             f"({sum(1 for r in score.rows if r.detected)}/{len(score.rows)})")
    L.append(f"  page accuracy: {score.page_accuracy:.1%}")
    avg = score.column_avg
    if avg is not None:
        L.append(f"  column jaccard (avg): {avg:.2f}")
    if score.extra_emitted:
        L.append(f"  extra emitted (no truth match): {score.extra_emitted}")
    L.append("")
    L.append(f"  {'id':<6} {'truth_pg':<8} {'detected':<8} "
             f"{'pg_match':<8} {'col_jac':<8} sidecar")
    for r in score.rows:
        det = "YES" if r.detected else "no"
        pg = (str(r.detected_page) if r.detected_page is not None else "-")
        pg_match = ("=" if r.detected and r.detected_page == r.truth_page
                    else ("X" if r.detected else "-"))
        cj = (f"{r.column_jaccard:.2f}" if r.column_jaccard is not None else "-")
        sc = r.sidecar_path.name if r.sidecar_path else "-"
        L.append(f"  {r.truth_id:<6} {r.truth_page:<8} {det:<8} "
                 f"{pg_match:<8} {cj:<8} {sc}")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth-dir", type=Path,
                   default=Path("tests/table_extract_truth"))
    p.add_argument("--condition", action="append", required=True,
                   metavar="LABEL:OUT_DIR",
                   help="condition to evaluate; repeat for multiple. "
                        "Example: baseline:out/out_root_prompt")
    p.add_argument("--csv", type=Path, default=None,
                   help="write per-paper-per-condition rows to CSV")
    p.add_argument("--append", action="store_true",
                   help="append to --csv instead of overwriting")
    args = p.parse_args(argv)

    truth_files = sorted(args.truth_dir.glob("*.yaml"))
    if not truth_files:
        print(f"no truth YAMLs in {args.truth_dir}", file=sys.stderr)
        return 2

    rows_for_csv: list[PaperScore] = []
    for cond in args.condition:
        if ":" not in cond:
            print(f"--condition needs LABEL:OUT_DIR; got {cond!r}", file=sys.stderr)
            return 2
        label, out_dir = cond.split(":", 1)
        out_dir = Path(out_dir)
        for tf in truth_files:
            score = evaluate(tf, out_dir, label)
            if score is None:
                continue  # truth file's paper isn't in this out_dir
            print(render_paper(score))
            rows_for_csv.append(score)

    # Aggregate
    by_cond: dict[str, list[PaperScore]] = {}
    for s in rows_for_csv:
        by_cond.setdefault(s.condition, []).append(s)
    print("\n=== aggregate by condition ===")
    for cond, papers in by_cond.items():
        total_truth = sum(len(p.rows) for p in papers)
        total_detected = sum(sum(1 for r in p.rows if r.detected) for p in papers)
        det_rate = total_detected / total_truth if total_truth else 0.0
        page_correct = sum(
            sum(1 for r in p.rows if r.detected and r.detected_page == r.truth_page)
            for p in papers
        )
        page_acc = page_correct / total_detected if total_detected else 0.0
        print(f"  {cond:<14} detection {det_rate:.1%} ({total_detected}/{total_truth})  "
              f"page accuracy {page_acc:.1%}")

    if args.csv:
        mode = "a" if args.append else "w"
        with open(args.csv, mode, newline="") as f:
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["condition", "paper", "truth_id", "truth_page",
                            "detected", "detected_page", "page_match",
                            "column_jaccard", "sidecar"])
            for s in rows_for_csv:
                for r in s.rows:
                    w.writerow([
                        s.condition, s.paper, r.truth_id, r.truth_page,
                        "yes" if r.detected else "no",
                        r.detected_page if r.detected_page is not None else "",
                        "yes" if (r.detected and r.detected_page == r.truth_page) else "no",
                        f"{r.column_jaccard:.3f}" if r.column_jaccard is not None else "",
                        r.sidecar_path.name if r.sidecar_path else "",
                    ])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
