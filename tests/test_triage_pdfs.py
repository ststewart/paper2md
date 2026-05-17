"""Unit tests for triage_pdfs.

Run with:
    cd paper2md && python -m pytest tests/test_triage_pdfs.py -v

PyMuPDF is exercised via the public API in `analyze_pdf`; for purely-
synthetic tests of `recommend_routing` and `write_tsv` we don't need
fitz at all, so those tests are import-light.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import triage_pdfs as tp  # noqa: E402


def test_recommend_routing_clean_modern():
    stats = tp.PdfStats(
        stem="krot", pages=4, chars_per_page=2400,
        distinct_font_sizes=14, super_flag_count=11,
        style_flag_pct=0.42, unknown_char_pct=0.0,
    )
    routing, reason = tp.recommend_routing(stats)
    assert routing == "mineru-paddleocr"
    assert reason == "clean-modern-text-layer"


def test_recommend_routing_sparse_text_layer():
    stats = tp.PdfStats(
        stem="lyzenga1980", pages=4, chars_per_page=120,
        distinct_font_sizes=2, super_flag_count=0,
        style_flag_pct=0.0, unknown_char_pct=0.05,
    )
    routing, reason = tp.recommend_routing(stats)
    assert routing == "marker-surya"
    assert reason == "sparse-text-layer"


def test_recommend_routing_no_super_flags_preserved():
    """chars_per_page is healthy and font sizes are varied, but the
    OCR engine emitted zero superscript flags across the document.
    That's the canonical "flattened OCR" case (Lyzenga 1980 pattern)
    -- route to marker so we don't try sup/sub recovery on a layer
    that has no metadata to recover from."""
    stats = tp.PdfStats(
        stem="lyzenga1980", pages=10, chars_per_page=5500,
        distinct_font_sizes=30, super_flag_count=0,
        style_flag_pct=1.0, unknown_char_pct=0.0,
    )
    routing, reason = tp.recommend_routing(stats)
    assert routing == "marker-surya"
    assert reason == "no-super-flags-preserved"


def test_recommend_routing_no_heading_discrimination():
    """Has supers but only 2 distinct font sizes — single-font OCR
    output, marker likely does better with surya layout model."""
    stats = tp.PdfStats(
        stem="weird", pages=8, chars_per_page=1200,
        distinct_font_sizes=2, super_flag_count=4,
        style_flag_pct=0.20, unknown_char_pct=0.0,
    )
    routing, reason = tp.recommend_routing(stats)
    assert routing == "marker-surya"
    assert reason == "no-heading-discrimination"


def test_recommend_routing_error_falls_back_to_marker():
    stats = tp.PdfStats(
        stem="broken", pages=0, chars_per_page=0,
        distinct_font_sizes=0, super_flag_count=0,
        style_flag_pct=0.0, unknown_char_pct=0.0,
        error="open-failed: corrupt",
    )
    routing, reason = tp.recommend_routing(stats)
    assert routing == "marker-surya"
    assert reason.startswith("error:")


def test_recommend_routing_threshold_overrides():
    """User can tune thresholds via kwargs."""
    stats = tp.PdfStats(
        stem="x", pages=4, chars_per_page=400,
        distinct_font_sizes=4, super_flag_count=2,
        style_flag_pct=0.10, unknown_char_pct=0.0,
    )
    # Default thresholds: routes to mineru.
    routing, _ = tp.recommend_routing(stats)
    assert routing == "mineru-paddleocr"
    # Tighten chars_per_page_min above 400: routes to marker.
    routing, reason = tp.recommend_routing(stats, chars_per_page_min=600)
    assert routing == "marker-surya"
    assert reason == "sparse-text-layer"


def test_write_tsv_round_trip(tmp_path):
    rows = [
        {"stem": "a", "pages": 4, "routing": "mineru-paddleocr"},
        {"stem": "b", "pages": 16, "routing": "marker-surya"},
    ]
    out = tmp_path / "out.tsv"
    tp.write_tsv(rows, out)
    text = out.read_text()
    assert text.startswith("stem\tpages\trouting\n")
    assert "a\t4\tmineru-paddleocr" in text
    assert "b\t16\tmarker-surya" in text


def test_write_tsv_empty(tmp_path):
    out = tmp_path / "empty.tsv"
    tp.write_tsv([], out)
    assert out.read_text() == ""


def test_triage_collection_skips_nonexistent_dir(tmp_path):
    """Empty directory yields zero rows + zero counts."""
    rows, summary = tp.triage_collection(tmp_path)
    assert rows == []
    assert summary == {"mineru-paddleocr": 0, "marker-surya": 0}


def test_analyze_pdf_handles_nonexistent_file(tmp_path):
    """Missing PDF returns a PdfStats with `error` set; doesn't raise."""
    s = tp.analyze_pdf(tmp_path / "nope.pdf")
    assert s.error is not None
    assert s.pages == 0


def test_analyze_pdf_real(test_pdf):
    """Smoke test against a real fixture (canup.pdf in test_files/).
    Uses the test_pdf fixture from conftest.py; skips when missing."""
    pdf = test_pdf("canup.pdf")
    s = tp.analyze_pdf(pdf, probe_pages=2)
    assert s.error is None
    assert s.pages > 0
    # canup is a modern Nature PDF; expect dense text + styled spans.
    assert s.chars_per_page > 1000
    assert s.distinct_font_sizes >= 4
    routing, _ = tp.recommend_routing(s)
    assert routing == "mineru-paddleocr"


def test_main_routes_to_default_output(tmp_path, monkeypatch, capsys):
    """End-to-end: empty collection runs cleanly with -o explicit."""
    # Empty collection (no PDFs) returns 1 ("No PDFs found").
    coll = tmp_path / "empty"
    coll.mkdir()
    rc = tp.main([str(coll), "-o", str(tmp_path / "out.tsv")])
    assert rc == 1


def test_cli_argparse_smoke():
    """Parser builds and accepts --help-related no-arg invocation."""
    parser = tp._build_parser()
    args = parser.parse_args(["/some/dir"])
    assert args.collection_dir == Path("/some/dir")
    assert args.probe_pages == tp.DEFAULT_PROBE_PAGES
    args = parser.parse_args(["/some/dir", "--chars-per-page-min", "500",
                              "--probe-pages", "5"])
    assert args.chars_per_page_min == 500.0
    assert args.probe_pages == 5
