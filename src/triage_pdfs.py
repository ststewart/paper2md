"""Triage helper: survey a collection of PDFs and recommend which
backend (mineru-paddleocr vs marker-surya) each should route through.

Stage 2 of the paper2md convergence plan. PyMuPDF-only, no GPU, no
network. Reads the first ~3 pages of each PDF, computes text-layer
quality signals, and emits a TSV recommendation file.

Routing decision (configurable thresholds, see `recommend_routing`):

    if text_density < 0.4              -> marker-surya (scan-dominated)
    elif style_flag_pct < 0.05         -> marker-surya (flattened OCR)
    elif distinct_font_sizes <= 3      -> marker-surya (no headings)
    else                                -> mineru-paddleocr

CLI:

    python -m triage_pdfs <collection-dir> [-o <tsv-out>]
    python -m triage_pdfs ~/datasets/pdf2md/claude/workflow/collections/moon

Output (`workflow/triage-<collection>.tsv` by default):

    stem  pages  density  sizes  super  style%  unk%  routing  reason

Borderline rows should be reviewed by eye; thresholds can be tuned
via env vars or by passing custom values to `recommend_routing`.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# Default routing thresholds. Tuned against the user's silica-shock /
# moon / chondrules collections (see plan Stage 2 verification).
# `chars_per_page` < 300 typically indicates a scanned PDF whose
# embedded OCR is sparse / missing.
DEFAULT_CHARS_PER_PAGE_MIN = 300
DEFAULT_STYLE_FLAG_MIN = 0.05
DEFAULT_DISTINCT_SIZES_MIN = 4
DEFAULT_PROBE_PAGES = 3


@dataclass
class PdfStats:
    stem: str
    pages: int
    chars_per_page: float          # total chars / pages probed
    distinct_font_sizes: int
    super_flag_count: int
    style_flag_pct: float          # 0..1
    unknown_char_pct: float        # 0..1
    error: Optional[str] = None


def analyze_pdf(path: Path,
                probe_pages: int = DEFAULT_PROBE_PAGES) -> PdfStats:
    """Open `path` with PyMuPDF and compute text-layer quality signals
    on the first `probe_pages` pages. Returns a PdfStats dataclass.

    On failure (missing file, fitz import error, corrupt PDF) returns
    a PdfStats with `error` set and zero metrics.
    """
    stem = path.stem
    empty = PdfStats(
        stem=stem, pages=0, chars_per_page=0.0,
        distinct_font_sizes=0, super_flag_count=0,
        style_flag_pct=0.0, unknown_char_pct=0.0,
    )
    try:
        import fitz
    except ImportError:
        empty.error = "pymupdf-import-failed"
        return empty
    try:
        doc = fitz.open(path)
    except Exception as e:
        empty.error = f"open-failed: {e}"
        return empty
    try:
        n_pages = len(doc)
        pages_to_probe = min(probe_pages, n_pages)
        if pages_to_probe == 0:
            empty.error = "empty-doc"
            return empty
        total_chars = 0
        sizes = set()
        super_n = 0
        styled_n = 0
        all_n = 0
        unknown_n = 0
        for i in range(pages_to_probe):
            page = doc[i]
            td = page.get_text("dict")
            for blk in td.get("blocks", []) or []:
                if blk.get("type") != 0:
                    continue
                for line in blk.get("lines", []) or []:
                    for span in line.get("spans", []) or []:
                        txt = span.get("text") or ""
                        size = span.get("size", 0)
                        flags = int(span.get("flags", 0))
                        all_n += 1
                        total_chars += len(txt)
                        if size > 0:
                            sizes.add(round(size * 2) / 2)  # 0.5pt buckets
                        if flags & 1:
                            super_n += 1
                        # Style discriminator: count spans with super
                        # (bit 0), italic (bit 1), or bold (bit 4)
                        # set. Mask out bit 2 (serif) and bit 3
                        # (monospace) — those describe font family,
                        # not typographic style, and are nearly always
                        # set on body PDFs (which would otherwise make
                        # this metric ~1.0 for every paper).
                        if flags & 0b10011:
                            styled_n += 1
                        if "�" in txt or any(
                                c in txt for c in ("□", "■")
                        ):
                            unknown_n += 1
        cpp = total_chars / pages_to_probe if pages_to_probe else 0.0
        style_pct = (styled_n / all_n) if all_n else 0.0
        unk_pct = (unknown_n / all_n) if all_n else 0.0
        return PdfStats(
            stem=stem,
            pages=n_pages,
            chars_per_page=cpp,
            distinct_font_sizes=len(sizes),
            super_flag_count=super_n,
            style_flag_pct=style_pct,
            unknown_char_pct=unk_pct,
        )
    finally:
        doc.close()


def recommend_routing(stats: PdfStats,
                      chars_per_page_min: float = DEFAULT_CHARS_PER_PAGE_MIN,
                      style_flag_min: float = DEFAULT_STYLE_FLAG_MIN,
                      distinct_sizes_min: int = DEFAULT_DISTINCT_SIZES_MIN,
                      ) -> tuple[str, str]:
    """Return (routing, reason). Routing is `mineru-paddleocr` (fast,
    modern PDFs) or `marker-surya` (scan-tolerant, sup/sub-preserving).

    Routing logic, top to bottom:
      1. error / empty doc                   -> marker
      2. chars_per_page < min                -> marker (sparse / scan)
      3. super_flag_count == 0               -> marker (no super flags
                                                preserved -- text layer
                                                was flattened by older
                                                OCR; sup/sub recovery
                                                won't fire)
      4. distinct_font_sizes < min           -> marker (no heading
                                                discrimination; layout
                                                signal is poor)
      5. otherwise                           -> mineru

    The `style_flag_pct` is reported in the TSV for diagnostic value
    but isn't used as a hard threshold — its denominator dominated by
    body-text spans makes it noisy, and the super_flag_count==0 check
    is a more reliable "flattened" signal.
    """
    if stats.error:
        return "marker-surya", f"error:{stats.error}"
    if stats.pages == 0:
        return "marker-surya", "empty-doc"
    if stats.chars_per_page < chars_per_page_min:
        return "marker-surya", "sparse-text-layer"
    if stats.super_flag_count == 0:
        return "marker-surya", "no-super-flags-preserved"
    if stats.distinct_font_sizes < distinct_sizes_min:
        return "marker-surya", "no-heading-discrimination"
    return "mineru-paddleocr", "clean-modern-text-layer"


def write_tsv(rows: list[dict], out_path: Path) -> None:
    """Write a list of dict rows to `out_path` as TSV."""
    if not rows:
        out_path.write_text("")
        return
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def triage_collection(collection_dir: Path,
                      out_path: Optional[Path] = None,
                      probe_pages: int = DEFAULT_PROBE_PAGES,
                      thresholds: Optional[dict] = None,
                      ) -> tuple[list[dict], dict]:
    """Walk `collection_dir` for `*.pdf` files, analyze each, and
    return (rows, summary) where rows is a list of dicts (one per
    PDF) and summary aggregates counts per routing label.

    If `out_path` is provided, also writes the TSV.
    """
    thresholds = thresholds or {}
    pdfs = sorted(p for p in collection_dir.glob("*.pdf") if p.is_file())
    rows: list[dict] = []
    summary: dict = {"mineru-paddleocr": 0, "marker-surya": 0}
    for pdf in pdfs:
        stats = analyze_pdf(pdf, probe_pages=probe_pages)
        routing, reason = recommend_routing(stats, **thresholds)
        summary[routing] = summary.get(routing, 0) + 1
        rows.append({
            "stem": stats.stem,
            "pages": stats.pages,
            "chars_per_page": f"{stats.chars_per_page:.0f}",
            "sizes": stats.distinct_font_sizes,
            "super": stats.super_flag_count,
            "style_pct": f"{stats.style_flag_pct:.2f}",
            "unk_pct": f"{stats.unknown_char_pct:.3f}",
            "routing": routing,
            "reason": reason,
        })
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_tsv(rows, out_path)
    return rows, summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triage_pdfs",
        description=(
            "Survey a collection of PDFs and recommend which backend "
            "(mineru-paddleocr or marker-surya) to use for each. "
            "PyMuPDF-only; no GPU, no network."
        ),
    )
    p.add_argument("collection_dir", type=Path,
                   help="Directory containing PDF files to triage.")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output TSV path. Default: "
                        "$WORKFLOW_DIR/triage-<collection-name>.tsv "
                        "or ./triage-<collection-name>.tsv.")
    p.add_argument("--probe-pages", type=int, default=DEFAULT_PROBE_PAGES,
                   help="Pages per PDF to scan (default: %(default)d).")
    p.add_argument("--chars-per-page-min", type=float,
                   default=DEFAULT_CHARS_PER_PAGE_MIN,
                   help="Minimum chars-per-page to route to mineru. "
                        "Below this, route to marker-surya. "
                        "Default: %(default)d.")
    p.add_argument("--style-flag-min", type=float,
                   default=DEFAULT_STYLE_FLAG_MIN,
                   help="Minimum style-flag fraction to route to mineru. "
                        "Default: %(default).2f.")
    p.add_argument("--distinct-sizes-min", type=int,
                   default=DEFAULT_DISTINCT_SIZES_MIN,
                   help="Minimum distinct font sizes to route to mineru. "
                        "Default: %(default)d.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    coll = args.collection_dir.resolve()
    if not coll.is_dir():
        print(f"ERROR: not a directory: {coll}", file=sys.stderr)
        return 2
    out = args.out
    if out is None:
        import os
        wf = os.environ.get("WORKFLOW_DIR", "")
        if wf:
            out = Path(wf) / f"triage-{coll.name}.tsv"
        else:
            out = Path.cwd() / f"triage-{coll.name}.tsv"
    rows, summary = triage_collection(
        coll, out_path=out, probe_pages=args.probe_pages,
        thresholds={
            "chars_per_page_min": args.chars_per_page_min,
            "style_flag_min": args.style_flag_min,
            "distinct_sizes_min": args.distinct_sizes_min,
        },
    )
    if not rows:
        print(f"No PDFs found in {coll}", file=sys.stderr)
        return 1
    # Print the TSV to stdout for immediate inspection.
    fields = list(rows[0].keys())
    print("\t".join(fields))
    for r in rows:
        print("\t".join(str(r[f]) for f in fields))
    total = sum(summary.values())
    print(file=sys.stderr)
    print(f"Wrote {out}", file=sys.stderr)
    print(f"  {summary.get('mineru-paddleocr', 0)} -> mineru-paddleocr",
          file=sys.stderr)
    print(f"  {summary.get('marker-surya', 0)} -> marker-surya",
          file=sys.stderr)
    print(f"  {total} total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
