#!/usr/bin/env python3
"""Audit a paper2md `.md` output for figure / table reference vs
insertion mismatch.

Use case: after a batch run, flag papers where the body text mentions
`Fig. 5` (and `Fig. 1` … `Fig. 4`) but only 3 figure images were
actually spliced into the markdown -- something dropped two figures
on the floor. Same for tables. The check is heuristic but cheap and
catches the common failure modes (MinerU missing a table on a page,
hybrid splice dropping a figure with garbled caption, marker-layout
caption-match shortfall) without re-running the pipeline.

CLI:
    python -m audit_md paper.md
    python -m audit_md paper.md --json     # machine-readable for jq

Library API (used by run_batch to populate manifest fields):
    audit_md(md_path) -> dict with keys
        figures_referenced  -- list of unique figure ids found in body
        figures_inserted    -- count of figure image markdown lines
        tables_referenced   -- list of unique table ids
        tables_inserted     -- count of table caption / sidecar links
        figure_mismatch     -- bool, True when refs > inserts (likely dropped)
        table_mismatch      -- bool, ditto for tables

The mismatch heuristic is one-directional: more references than
insertions = a likely problem; more insertions than references is
common and benign (a figure not cited inline still counts).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# Figure / table number patterns. Liberal to catch common variants:
# `Fig. 5`, `Figure 5a`, `Figs. 1 and 2`, `Fig. S1`, `Fig. A1`,
# `Figure 5(a)`, `Extended Data Fig. 3`. Capture the bare id (the
# `5` / `S1` part) so dedup-by-id works.
_FIG_ID = r"(?:S?\d+|[A-Z]\d+|[A-Z]\.\d+)"
_TBL_ID = r"(?:S?\d+|S?[IVXLC]+|[A-Z]\d+|[A-Z]\.\d+)"

# Body cross-references to figures: rejects image-alt-text (lines
# starting with `!`) and lines that ARE captions (start with `Fig.`
# at line start are captions, but in the middle of a paragraph they're
# cross-refs). We collect BOTH -- distinguishing caption vs body cite
# isn't worth the false-negative risk; what matters is "what fig
# numbers are mentioned anywhere in the body".
_FIG_REF_RE = re.compile(
    r"(?<![\w-])"  # don't match in identifiers like fig.path
    r"(?:Extended\s+Data\s+)?(?:Figure|Fig\.?)\s+"
    r"(" + _FIG_ID + r")",
    re.IGNORECASE,
)
_TBL_REF_RE = re.compile(
    r"(?<![\w-])"
    r"(?:Supplementary\s+|Supplement(?:ary)?\s+|Supp\.\s+)?"
    r"Table\s+(" + _TBL_ID + r")",
    re.IGNORECASE,
)

# Inserted figure: an image-markdown line whose href doesn't look like
# a table sidecar JPG. We use the path basename: marker emits
# `_page_N_Picture_M.jpeg`, MinerU emits hashes or our renamed
# `figure_N.jpg`, hybrid emits the same. Table images use the
# `table_p{page}_{idx}.jpg` convention from Phase 1 rename.
_IMG_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$",
                          re.MULTILINE)

# Table insertion signals (any of these in priority order):
#   - `**Table N.**` caption line
#   - `Table N.` line-anchored caption
#   - Pipe-md table headers preceded by a `Table N` line
#   - Sidecar link `[Table N — separate markdown](...)`
_TBL_CAPTION_RE = re.compile(
    r"(?im)^\s*\**\s*Table\s+(" + _TBL_ID + r")\b\s*[:.\*\|]",
)
_TBL_SIDECAR_RE = re.compile(
    r"\[Table\s+(" + _TBL_ID + r")\s+[-—]\s+separate\s+markdown\]",
    re.IGNORECASE,
)


def _is_table_image(path: str) -> bool:
    """Heuristic: image hrefs whose basename starts with `table_` are
    table images (matches the Phase 1 rename convention)."""
    base = path.rsplit("/", 1)[-1]
    return base.lower().startswith("table_")


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100,
                 "D": 500, "M": 1000}


def _roman_to_int(s: str) -> Optional[int]:
    """Convert roman numeral string to int, or None if not valid roman."""
    if not s or not all(c in _ROMAN_VALUES for c in s.upper()):
        return None
    total = 0
    prev = 0
    for c in reversed(s.upper()):
        v = _ROMAN_VALUES[c]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if total > 0 else None


def _canonical_tbl_id(raw: str) -> str:
    """Normalize Table id so `V` and `5` compare equal. Preserves an
    S-prefix (`Table SV` -> `S5`). Non-roman ids returned unchanged."""
    s = raw
    prefix = ""
    if s and s[0].upper() == "S" and len(s) > 1:
        # Distinguish "S5" (SI table 5) from a roman id starting with S
        rest = s[1:]
        if rest.isdigit():
            return s  # already canonical, e.g. "S5"
        prefix = s[0]
        s = rest
    a = _roman_to_int(s)
    if a is not None:
        return f"{prefix}{a}"
    return raw


def _has_si_sibling(md_path: Path) -> bool:
    """True iff a sibling SI markdown exists in the same directory.
    Detection covers the common conventions: `<stem>_sm.md`,
    `<stem>_SI.md`, `<stem>_si.md`, `si_<stem>.md`. Returns False when
    `md_path` itself looks like the SI file (so S-prefix refs in the
    SI are still counted against SI insertions)."""
    stem = md_path.stem
    if (stem.endswith("_sm") or stem.endswith("_SI")
            or stem.endswith("_si") or stem.startswith("si_")):
        return False
    parent = md_path.parent
    for cand in (f"{stem}_sm.md", f"{stem}_SI.md",
                 f"{stem}_si.md", f"si_{stem}.md"):
        if (parent / cand).exists():
            return True
    return False


def audit_md(md_path: Path,
             expect_si_separately: Optional[bool] = None) -> dict:
    """Audit a markdown file. Returns the counts + mismatch flags.
    File is read once; regexes scan it independently.

    `expect_si_separately`: if True, S-prefix refs in the body are
    skipped because they live in a separately-processed SI document.
    If None (default), auto-detect by looking for an SI sibling file."""
    if expect_si_separately is None:
        expect_si_separately = _has_si_sibling(md_path)

    text = md_path.read_text()

    # Strip YAML frontmatter -- the metadata may mention figs / tables
    # in commentary and would skew the count.
    body = text
    if body.startswith("---\n"):
        end = body.find("\n---\n", 4)
        if end > 0:
            body = body[end + 5:]

    fig_refs = {m.group(1) for m in _FIG_REF_RE.finditer(body)}
    tbl_refs = {m.group(1) for m in _TBL_REF_RE.finditer(body)}

    # Strip refs to S-prefixed figs/tables when the SI is a separate
    # document (the splice cannot insert them into this file).
    if expect_si_separately:
        fig_refs = {r for r in fig_refs if not r.upper().startswith("S")}
        tbl_refs = {r for r in tbl_refs if not r.upper().startswith("S")}

    # Inserted figures: image lines whose href isn't a table image.
    img_hits = [m.group(1) for m in _IMG_LINE_RE.finditer(body)]
    fig_inserted = sum(1 for href in img_hits if not _is_table_image(href))

    # Inserted tables: union of caption-line matches + sidecar links.
    # Each gets a unique table id; dedup by id.
    tbl_inserted_ids = set()
    for m in _TBL_CAPTION_RE.finditer(body):
        tbl_inserted_ids.add(m.group(1))
    for m in _TBL_SIDECAR_RE.finditer(body):
        tbl_inserted_ids.add(m.group(1))

    # Mismatch heuristic: more refs than inserts = likely-dropped.
    # The reverse (more inserts than refs) is common (uncited figure)
    # and not flagged.
    figure_mismatch = len(fig_refs) > fig_inserted
    # For tables, canonicalize roman <-> arabic before comparing
    # (else "Table V" + "Table 5" both count as two distinct refs).
    tbl_refs_canon = {_canonical_tbl_id(r) for r in tbl_refs}
    tbl_inserted_canon = {_canonical_tbl_id(r) for r in tbl_inserted_ids}
    table_mismatch = len(tbl_refs_canon) > len(tbl_inserted_canon)

    return {
        "figures_referenced": sorted(fig_refs),
        "figures_inserted": fig_inserted,
        "tables_referenced": sorted(tbl_refs),
        "tables_inserted": sorted(tbl_inserted_ids),
        "figure_mismatch": figure_mismatch,
        "table_mismatch": table_mismatch,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="audit_md",
        description=__doc__.split("\n\n", 1)[0],
    )
    p.add_argument("md_path", type=Path, help="paper2md output .md file")
    p.add_argument("--json", action="store_true",
                   help="emit JSON for machine consumption (jq)")
    args = p.parse_args(argv)

    if not args.md_path.exists():
        print(f"error: {args.md_path} not found", file=sys.stderr)
        return 2

    result = audit_md(args.md_path)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    # Human-readable summary.
    print(f"Audit: {args.md_path.name}")
    print(f"  Figures referenced: {len(result['figures_referenced'])} "
          f"({', '.join(result['figures_referenced']) or '—'})")
    print(f"  Figure insertions:  {result['figures_inserted']}")
    print(f"  Tables referenced:  {len(result['tables_referenced'])} "
          f"({', '.join(result['tables_referenced']) or '—'})")
    print(f"  Table insertions:   {len(result['tables_inserted'])} "
          f"({', '.join(result['tables_inserted']) or '—'})")
    flags = []
    if result["figure_mismatch"]:
        flags.append(
            f"FIGURE MISMATCH: {len(result['figures_referenced'])} "
            f"referenced > {result['figures_inserted']} inserted")
    if result["table_mismatch"]:
        flags.append(
            f"TABLE MISMATCH: {len(result['tables_referenced'])} "
            f"referenced > {len(result['tables_inserted'])} inserted")
    if flags:
        print("  Flags:")
        for f in flags:
            print(f"    ! {f}")
    else:
        print("  Flags: none (refs match insertions)")
    return 1 if (result["figure_mismatch"] or result["table_mismatch"]) else 0


if __name__ == "__main__":
    sys.exit(main())
