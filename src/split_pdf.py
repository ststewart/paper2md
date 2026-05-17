#!/usr/bin/env python3
"""Split a PDF into smaller PDFs by page range — fast iteration helper
for paper2md.

Marker (paper2md's spine) takes 5–25 minutes on a 16-page paper. When
you're debugging extraction issues (regex tweaks, prompt changes,
hook-2 misclassifications) you want to re-run on just the affected
pages, not the whole document.

Usage:

    # Single page (page 5, 1-indexed):
    python src/split_pdf.py paper.pdf --pages 5 -o paper_p5.pdf

    # Page range:
    python src/split_pdf.py paper.pdf --pages 3-7 -o paper_p3-7.pdf

    # Multiple ranges (comma-separated):
    python src/split_pdf.py paper.pdf --pages 1,3-5,8 -o paper_subset.pdf

    # Auto-name (writes paper_p3-7.pdf next to input):
    python src/split_pdf.py paper.pdf --pages 3-7

    # Each page as its own file (writes paper_p1.pdf, paper_p2.pdf, ...):
    python src/split_pdf.py paper.pdf --each-page

Then feed the smaller PDF to paper2md exactly like the full one:

    python src/paper2md.py paper_p3-7.pdf -o out_p3-7/

Pages are 1-indexed (matching how scientists talk about PDFs).
PyMuPDF is already a paper2md dependency, no new packages needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # pymupdf


def parse_page_spec(spec: str, total_pages: int) -> list[int]:
    """Parse "5", "3-7", "1,3-5,8" into a sorted list of 0-indexed page
    numbers. Validates against total_pages. Raises ValueError on bad
    input."""
    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < lo or hi > total_pages:
                raise ValueError(
                    f"page range {lo}-{hi} out of bounds (PDF has {total_pages} pages)"
                )
            pages.update(range(lo - 1, hi))
        else:
            n = int(chunk)
            if n < 1 or n > total_pages:
                raise ValueError(
                    f"page {n} out of bounds (PDF has {total_pages} pages)"
                )
            pages.add(n - 1)
    if not pages:
        raise ValueError("empty page spec")
    return sorted(pages)


def auto_name(src: Path, page_indices: list[int]) -> Path:
    """Build an output filename like paper_p3-7.pdf or paper_p1+3+5.pdf."""
    if len(page_indices) == 1:
        suffix = f"p{page_indices[0] + 1}"
    elif page_indices == list(range(page_indices[0], page_indices[-1] + 1)):
        suffix = f"p{page_indices[0] + 1}-{page_indices[-1] + 1}"
    else:
        suffix = "p" + "+".join(str(i + 1) for i in page_indices)
    return src.with_name(f"{src.stem}_{suffix}.pdf")


def write_subset(src: Path, page_indices: list[int], dst: Path) -> None:
    """Write a new PDF containing only the requested pages, preserving
    metadata."""
    src_doc = fitz.open(src)
    dst_doc = fitz.open()
    for idx in page_indices:
        dst_doc.insert_pdf(src_doc, from_page=idx, to_page=idx)
    dst_doc.set_metadata(src_doc.metadata)
    dst_doc.save(str(dst))
    dst_doc.close()
    src_doc.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Split a PDF by page range for fast paper2md re-extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf", type=Path, help="input PDF")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--pages",
        help="page spec: '5', '3-7', or '1,3-5,8' (1-indexed)",
    )
    g.add_argument(
        "--each-page",
        action="store_true",
        help="write each page as its own PDF (paper_p1.pdf, paper_p2.pdf, ...)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="output PDF path (default: auto-name next to input)",
    )
    args = p.parse_args()

    if not args.pdf.exists():
        print(f"error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    src_doc = fitz.open(args.pdf)
    total = len(src_doc)
    src_doc.close()

    if args.each_page:
        if args.output is not None:
            print("error: -o/--output cannot be combined with --each-page",
                  file=sys.stderr)
            return 2
        for i in range(total):
            dst = args.pdf.with_name(f"{args.pdf.stem}_p{i + 1}.pdf")
            write_subset(args.pdf, [i], dst)
            print(f"wrote {dst}  ({i + 1}/{total})")
        return 0

    try:
        indices = parse_page_spec(args.pages, total)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    dst = args.output or auto_name(args.pdf, indices)
    write_subset(args.pdf, indices, dst)
    print(f"wrote {dst}  ({len(indices)} page(s) of {total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
