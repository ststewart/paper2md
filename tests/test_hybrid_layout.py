"""--layout-source=hybrid: marker for body+caption text, MinerU for
figure assets and table HTML, matched by figure/table number.

Covers:
  * HYBRID_FIG_LINE_RE / HYBRID_TABLE_LINE_RE positive + negative cases
  * _normalise_fig_id, _extract_table_number id helpers
  * _render_image_only (single panel, subpanel group, extras)
  * _find_marker_table_region (atomic pipe-table, paragraph fallback,
    safety-net shrink when region would overlap a sibling caption)
  * _align_marker_to_mineru_layout splice driver: basic figure splice,
    table swap, no-marker-anchor fallback, marker-only-caption pass-
    through, reverse-order application, duplicate fig-number handling
  * argparse: --layout-source=hybrid accepted; garbage rejected
  * _collect_pipeline_state: figure_layout_source / caption_text_source
    keys present under hybrid; force_ocr included; table_finder omitted
  * Integration: run_marker_plus_mineru_layout end-to-end with mocked
    run_marker + a pre-baked middle.json
"""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import paper2md as p2m  # noqa: E402


# === regex anchors ========================================================


@pytest.mark.parametrize("line", [
    "Fig. 1. Caption text begins here.",
    "Figure 2 | Pipe-style title from Nature.",
    "**Figure 3.** Bold-prefix caption with body.",
    "**Fig. 4 |** Bold pipe header.",
    "FIG. 5 (color). APS-style with qualifier.",
    "Fig. A1. Appendix figure.",
    "Extended Data Fig. 1 | Plot of surface temperatures.",
    "Fig. S2. Supplementary figure.",
    # Nature-style with closing `**` before the pipe (canup Figs 1, 2).
    "**Figure 1** | **Simulation of the Moon's accretion**",
    "**Figure 2** | **Melt-vapour equilibria in BSE-composition disk",
    # Single-bold close before period.
    "**Figure 1**. Caption text in single bold.",
])
def test_hybrid_fig_line_re_positive(line):
    """Line-anchored caption forms must match."""
    assert p2m.HYBRID_FIG_LINE_RE.search(line) is not None, line


@pytest.mark.parametrize("text", [
    # Mid-sentence body refs (not line-start).
    "We see in Fig. 1 that the trend continues.",
    "(see Fig. 2 of Smith [1990]).",
    # Line-anchored body refs WITHOUT a period or pipe after the id.
    "Fig. 5 shows that the trend continues.",
    "Fig. 5 of Smith [1990] is the canonical reference.",
    # Image alt-text -- line starts with "!", not letters.
    "![Figure 1 | x](assets/y.jpg)",
])
def test_hybrid_fig_line_re_negative(text):
    """Body cross-refs and image links must NOT match."""
    assert p2m.HYBRID_FIG_LINE_RE.search(text) is None, text


@pytest.mark.parametrize("line,expected_id", [
    ("Table 1. Sample data.", "1"),
    ("**Table A1**Title without separator", "A1"),
    ("Table II: Roman numeral.", "II"),
    ("Supplementary Table S1. SI table.", "S1"),
])
def test_hybrid_table_line_re_positive(line, expected_id):
    m = p2m.HYBRID_TABLE_LINE_RE.search(line)
    assert m is not None, line
    assert m.group("id") == expected_id


@pytest.mark.parametrize("raw,expected", [
    ("1", "1"),
    ("1A", "1"),
    ("S2", "S2"),
    ("S2B", "S2"),
    ("ED1", "ED1"),
    ("A1", "A1"),  # appendix id (letter is part of id, no panel suffix)
    ("A.1", "A.1"),
])
def test_normalise_fig_id(raw, expected):
    assert p2m._normalise_fig_id(raw) == expected


@pytest.mark.parametrize("text,expected", [
    ("Table 1. Sample data.", "1"),
    ("**Table A1**Title", "A1"),
    ("Supplementary Table S2: Materials.", "S2"),
    ("Just body text not a caption.", None),
    (None, None),
    ("", None),
])
def test_extract_table_number(text, expected):
    assert p2m._extract_table_number(text) == expected


# === _MBlock stub for splice tests ========================================


@dataclass
class _StubBlock:
    """Mimics layout_mineru._MBlock fields the hybrid code reads."""
    type: str
    page_idx: int = 0
    bbox: tuple = (0, 0, 100, 100)
    text: str = ""
    image_path: str = ""
    html: str = ""
    figure_number: int = None
    panel_letter: str = None
    subpanel_paths: list = field(default_factory=list)
    subpanel_bboxes: list = field(default_factory=list)
    footnote: str = ""


@dataclass
class _StubParsed:
    blocks: list = field(default_factory=list)
    page_sizes: dict = field(default_factory=dict)
    pages: int = 1


# === _render_image_only ===================================================


def test_render_image_only_single_panel():
    blk = _StubBlock(type="image", image_path="fig1.jpg",
                     figure_number=1, text="MinerU caption (garble)")
    out = p2m._render_image_only(
        blk, "1", "Fig. 1. Marker caption (clean).",
        assets_rel="assets", asset_prefix="")
    assert out == "![Fig. 1. Marker caption (clean).](assets/fig1.jpg)"


def test_render_image_only_subpanels():
    """Subpanel letters come from layout_mineru._subpanel_letter_assignments
    (lowercase a-z)."""
    blk = _StubBlock(type="image", image_path="figA.jpg",
                     figure_number=3, panel_letter="a",
                     subpanel_paths=["figB.jpg", "figC.jpg"])
    out = p2m._render_image_only(
        blk, "3", "Fig. 3. Three panels.",
        assets_rel="assets", asset_prefix="")
    lines = out.split("\n\n")
    assert len(lines) == 3
    assert "panel a" in lines[0] and "figA.jpg" in lines[0]
    assert "panel b" in lines[1] and "figB.jpg" in lines[1]
    assert "panel c" in lines[2] and "figC.jpg" in lines[2]


def test_render_image_only_subpanels_reading_order_two_panel_lr():
    """canup Figs 1, 2: 2-panel figure where the primary is RIGHT side
    (panel b extracted from caption prefix). Sorted by reading order
    LEFT-to-RIGHT, the stub on the left becomes 'a' and the primary on
    the right becomes 'b' -- alphabetical, matching visual content."""
    primary = _StubBlock(type="chart", image_path="primary_right.jpg",
                         figure_number=1, panel_letter="b",
                         bbox=(306, 50, 506, 185),
                         subpanel_paths=["stub_left.jpg"],
                         subpanel_bboxes=[(85, 50, 291, 185)])
    out = p2m._render_image_only(
        primary, "1", "Fig. 1. Caption.",
        assets_rel="assets", asset_prefix="")
    lines = out.split("\n\n")
    assert len(lines) == 2
    # LEFT panel (stub) becomes 'a', RIGHT panel (primary) becomes 'b'.
    assert "panel a" in lines[0] and "stub_left.jpg" in lines[0]
    assert "panel b" in lines[1] and "primary_right.jpg" in lines[1]


def test_render_image_only_subpanels_reading_order_three_panel_lr():
    """canup Fig 3: 3-panel row at the same y, primary is RIGHTmost
    (no extracted letter -> _subpanel_letter_assignments returned
    [a, b, c] in primary-first order, so the primary was mislabelled
    'a' even though it was visually panel c). Reading-order sort fixes
    this: left=a, center=b, right=c."""
    primary = _StubBlock(type="chart", image_path="right.jpg",
                         figure_number=3, panel_letter=None,
                         bbox=(392, 328, 537, 468),
                         subpanel_paths=["left.jpg", "center.jpg"],
                         subpanel_bboxes=[(69, 327, 213, 468),
                                          (226, 328, 379, 468)])
    out = p2m._render_image_only(
        primary, "3", "Fig. 3. Three panels.",
        assets_rel="assets", asset_prefix="")
    lines = out.split("\n\n")
    assert len(lines) == 3
    assert "panel a" in lines[0] and "left.jpg" in lines[0]
    assert "panel b" in lines[1] and "center.jpg" in lines[1]
    assert "panel c" in lines[2] and "right.jpg" in lines[2]


def test_render_image_only_subpanels_reading_order_two_panel_tb():
    """cuk Fig 4: 2-panel figure stacked TOP/BOTTOM, primary is BOTTOM
    (no extracted letter). Reading-order sort: top=a, bottom=b. Without
    the fix the primary emitted first as 'a' even though it's the
    visually-lower panel b."""
    primary = _StubBlock(type="chart", image_path="bottom.jpg",
                         figure_number=4, panel_letter=None,
                         bbox=(262, 200, 507, 348),
                         subpanel_paths=["top.jpg"],
                         subpanel_bboxes=[(262, 39, 508, 188)])
    out = p2m._render_image_only(
        primary, "4", "Fig. 4. Two panels stacked.",
        assets_rel="assets", asset_prefix="")
    lines = out.split("\n\n")
    assert len(lines) == 2
    assert "panel a" in lines[0] and "top.jpg" in lines[0]
    assert "panel b" in lines[1] and "bottom.jpg" in lines[1]


def test_render_image_only_subpanels_fallback_no_bboxes_alphabetises():
    """Legacy path: no subpanel_bboxes provided -> fall back to
    _subpanel_letter_assignments, then sort by letter. Primary letter
    'b' + one subpanel -> labels [b, a]; alphabetised emission gives
    a (the subpanel) first, b (the primary) second."""
    primary = _StubBlock(type="chart", image_path="primary.jpg",
                         figure_number=1, panel_letter="b",
                         subpanel_paths=["sub.jpg"],
                         subpanel_bboxes=[])  # missing
    out = p2m._render_image_only(
        primary, "1", "Fig. 1. Caption.",
        assets_rel="assets", asset_prefix="")
    lines = out.split("\n\n")
    assert "panel a" in lines[0] and "sub.jpg" in lines[0]
    assert "panel b" in lines[1] and "primary.jpg" in lines[1]


def test_render_image_only_extras_for_duplicate_figs():
    primary = _StubBlock(type="image", image_path="dup1.jpg",
                         figure_number=2)
    extra = _StubBlock(type="image", image_path="dup2.jpg",
                       figure_number=2)
    out = p2m._render_image_only(
        primary, "2", "Fig. 2. Caption.",
        assets_rel="assets", asset_prefix="",
        extras=(extra,))
    assert "dup1.jpg" in out
    assert "dup2.jpg" in out


# === _find_marker_table_region ============================================


def test_find_marker_table_region_atomic_pipe():
    md = ("Table 1. Sample data.\n\n"
          "| h1 | h2 |\n"
          "|----|----|\n"
          "| 1  | 2  |\n"
          "\n"
          "Next paragraph.\n")
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    region = md[start:end]
    assert region.startswith("|")
    assert "1  | 2" in region
    assert "Next paragraph" not in region


def test_find_marker_table_region_paragraph_fallback():
    md = ("Table 2. Caption.\n\n"
          "Some paragraph that's actually a flat-text table.\n"
          "Multi-line continuation here.\n"
          "\n"
          "Next section.\n")
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    region = md[start:end]
    assert "Some paragraph" in region
    assert "Multi-line continuation" in region
    assert "Next section" not in region


def test_find_marker_table_region_safety_net_shrinks_on_sibling():
    """Region candidate that would swallow another caption -> insert-only."""
    md = ("Table 1. First caption.\n"
          "Fig. 9. Sibling caption that must NOT be swallowed.\n"
          "Body text.\n"
          "\n"
          "Next section.\n")
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    assert start == end  # insert-only; region is empty


def test_find_marker_table_region_extends_across_double_pipe_md():
    """Cuk Table 1 pattern: marker (surya) emits the SAME table TWICE
    in different OCR styles, separated by a blank line. The basic
    "stop at first \\n\\n" rule consumed only the first; the extension
    loop should pull both into the region so the splice replaces both
    renderings with MinerU's clean pipe-md."""
    md = ("**Table 1.** Caption.\n"
          "\n"
          "| $\\frac{a}{b}$ | $V$ | b |\n"
          "| --- | --- | --- |\n"
          "| 0.026 | 20 | -0.30 |\n"
          "| 0.050 | 25 | -0.30 |\n"
          "\n"
          "| M <sub>a</sub> | $V_i$ | b |\n"
          "|---|---|---|\n"
          "| 0.026 | 20.0 | -0.30 |\n"
          "| 0.050 | 25.0 | -0.30 |\n"
          "\n"
          "*Notes: footnote.*\n"
          "\n"
          "Some prose follows here.\n")
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    region = md[start:end]
    # Region must span BOTH pipe-md chunks.
    assert "$\\frac{a}{b}$" in region   # first chunk (LaTeX style)
    assert "<sub>a</sub>" in region     # second chunk (sub-tag style)
    # Region must NOT consume the *Notes:* footnote or the prose after.
    assert "Notes: footnote" not in region
    assert "Some prose" not in region


def test_find_marker_table_region_stops_at_non_pipe_paragraph():
    """Pipe-md, blank, regular paragraph -> region stops at the blank.
    Paragraph (no leading `|`) is not part of the table."""
    md = ("Table 1. Caption.\n"
          "\n"
          "| h1 | h2 |\n"
          "|---|---|\n"
          "| 1 | 2 |\n"
          "\n"
          "Next paragraph that is NOT a table row.\n")
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    region = md[start:end]
    assert "| 1 | 2 |" in region
    assert "Next paragraph" not in region


def test_find_marker_table_region_extension_stops_at_next_caption():
    """Pipe-md, blank, second pipe-md, blank, `Table 2.` caption ->
    region must NOT consume the next caption (the safety-check on
    each extension chunk's content rejects HYBRID_TABLE_LINE_RE
    matches). The next caption itself starts with `T` not `|`, so
    the chunk-start check fires first; this test guards the safety
    check in case the order changes."""
    md = ("Table 1. First.\n"
          "\n"
          "| h1 | h2 |\n"
          "|---|---|\n"
          "| 1 | 2 |\n"
          "\n"
          "| h3 | h4 |\n"
          "|---|---|\n"
          "| 3 | 4 |\n"
          "\n"
          "Table 2. Second caption.\n"
          "\n"
          "| x | y |\n"
          "|---|---|\n"
          "| a | b |\n")
    m = p2m.TABLE_CAPTION_LINE_RE.search(md)  # finds Table 1 (first match)
    start, end = p2m._find_marker_table_region(md, m)
    region = md[start:end]
    # First two pipe-md chunks consumed.
    assert "| 1 | 2 |" in region
    assert "| 3 | 4 |" in region
    # Table 2 caption + its body MUST NOT be consumed.
    assert "Table 2." not in region
    assert "| a | b |" not in region


def test_find_marker_table_region_extension_handles_three_chunks():
    """Three consecutive pipe-md chunks (paranoid OCR) -> all three
    consumed. Stops at the first non-pipe content."""
    md = ("Table 1. Caption.\n"
          "\n"
          "| a |\n|---|\n| 1 |\n"
          "\n"
          "| b |\n|---|\n| 2 |\n"
          "\n"
          "| c |\n|---|\n| 3 |\n"
          "\n"
          "Body paragraph.\n")
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    region = md[start:end]
    assert "| 1 |" in region
    assert "| 2 |" in region
    assert "| 3 |" in region
    assert "Body paragraph" not in region


def test_find_marker_table_region_first_chunk_runaway_still_bails():
    """Single 50-line pipe-md chunk -> still trips the 40-line cap and
    returns insert-only. The extension logic doesn't relax this safety
    net for the FIRST chunk."""
    rows = "\n".join(f"| {i} | x |" for i in range(50))
    md = f"Table 1. Caption.\n\n| h | h2 |\n|---|---|\n{rows}\n\nAfter.\n"
    m = p2m.HYBRID_TABLE_LINE_RE.search(md)
    start, end = p2m._find_marker_table_region(md, m)
    assert start == end  # insert-only


# === _align_marker_to_mineru_layout =======================================


def _empty_report():
    return p2m.QualityReport()


def test_align_basic_figure_splice():
    """Mineru figure block + matching marker `Fig. 1.` line ->
    image link spliced ABOVE the caption; marker caption preserved
    verbatim; counter increments; FigureScore appended to
    report.figures so overall() can grade figure-only papers
    instead of returning 0.0 -> F."""
    marker_md = ("Body intro.\n\n"
                 "Fig. 1. Marker's clean caption text.\n\n"
                 "More body.\n")
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="img1.jpg",
                   figure_number=1, text="MinerU garbled caption"),
    ])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # Image link spliced ABOVE the caption.
    img_idx = out.index("![")
    cap_idx = out.index("Fig. 1.")
    assert img_idx < cap_idx
    # Marker's clean caption preserved verbatim.
    assert "Marker's clean caption text." in out
    # MinerU's garbled text NOT in output.
    assert "MinerU garbled caption" not in out
    # Counter recorded.
    assert report.rescues["hybrid_splice"]["figures_spliced"] == 1
    # FigureScore appended so overall() can grade the figure-only paper.
    assert len(report.figures) == 1
    fig = report.figures[0]
    assert fig.filename == "img1.jpg"
    assert fig.matched_figure == "1"
    assert fig.caption_produced is True
    assert fig.score >= 0.9  # clean match with long caption
    assert fig.dropped is False


def test_align_figure_splice_short_caption_gets_lower_score():
    """Clean match but caption < 5 words -> 0.7 (mirrors marker's
    _score_figure caption-length gate)."""
    marker_md = "Fig. 1. Short.\n"
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="img1.jpg",
                   text="Fig. 1. anything"),
    ])
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert len(report.figures) == 1
    assert report.figures[0].score == 0.7
    # "Fig. 1. Short." -> split() -> ['Fig.', '1.', 'Short.'] -> 3 words
    assert report.figures[0].caption_length == 3


def test_align_duplicate_mineru_figs_records_extras_as_dropped():
    """Two MinerU image blocks for the same fig number -> primary
    gets score 1.0, extras get score 0.6 + dropped=True. Both
    appear in report.figures so the manifest can audit."""
    marker_md = "Fig. 1. Clean caption with enough words.\n"
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="primary.jpg",
                   figure_number=1, text="Fig. 1. caption"),
        _StubBlock(type="image", image_path="extra.jpg",
                   figure_number=1, text="Fig. 1. caption"),
    ])
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert len(report.figures) == 2
    primary = next(f for f in report.figures if f.filename == "primary.jpg")
    extra = next(f for f in report.figures if f.filename == "extra.jpg")
    assert primary.score >= 0.9 and primary.dropped is False
    assert extra.score == 0.6 and extra.dropped is True


def test_align_unmatched_mineru_fig_gets_eod_figure_score():
    """MinerU fig block has no marker `Fig. N` anchor -> EOD splice +
    FigureScore with score 0.6 (no caption match)."""
    marker_md = "Body without any Fig N reference.\n"
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="orphan.jpg",
                   figure_number=5, text="Fig. 5. mineru caption"),
    ])
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert report.rescues["hybrid_splice"]["unmatched_mineru_figs"] == 1
    assert len(report.figures) == 1
    assert report.figures[0].score == 0.6
    assert report.figures[0].matched_figure == "5"


def test_align_figure_score_honors_asset_prefix():
    """asset_prefix='si_' must show up on FigureScore.filename so a
    supplement run's figures are distinguishable in the manifest."""
    marker_md = "Fig. 1. Caption text with enough words here.\n"
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="img1.jpg",
                   figure_number=1, text=""),
    ])
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="si_", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert len(report.figures) == 1
    assert report.figures[0].filename == "si_img1.jpg"


def test_align_figure_only_paper_no_longer_grades_F():
    """End-to-end regression: a paper with figures but no tables used
    to grade F because `report.figures` was empty even when splices
    happened. Now overall() should reflect actual figure quality.
    Mimics the 7-paper batch finding (Hicks2006, canup, Knudson2009,
    tracy, krot, Elliott, feng all graded F with figures_spliced>0)."""
    marker_md = (
        "Fig. 1. First figure caption with plenty of words.\n\n"
        "Fig. 2. Second figure caption also has many words.\n"
    )
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="fig1.jpg",
                   figure_number=1, text="Fig. 1. mineru caption"),
        _StubBlock(type="image", image_path="fig2.jpg",
                   figure_number=2, text="Fig. 2. mineru caption"),
    ])
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert report.rescues["hybrid_splice"]["figures_spliced"] == 2
    assert len(report.figures) == 2
    # overall() now sees figures and produces a non-zero score.
    assert report.overall() > 0
    assert report.grade() != "F"


def test_align_table_conservative_preserves_marker(tmp_path):
    """Conservative splice: marker's pipe-md table is KEPT in body
    verbatim; only the image link + sidecar link are ADDED next to
    marker's caption. The MinerU table HTML is written to the
    sidecar `.md` -- NEVER inserted as inline body content."""
    marker_md = ("Table 1. Marker caption.\n\n"
                 "| a | b |\n"
                 "|---|---|\n"
                 "| 1 | 2 |\n"
                 "\n"
                 "After table.\n")
    blk = _StubBlock(
        type="table", text="Table 1. MinerU caption.",
        html=("<table><tr><td>col1</td><td>col2</td></tr>"
              "<tr><td>r1c1</td><td>r1c2</td></tr>"
              "<tr><td>r2c1</td><td>r2c2</td></tr></table>"),
        image_path="tbl1.jpg")
    parsed = _StubParsed(blocks=[blk])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", tmp_path,
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # Marker's caption + pipe-md preserved exactly.
    assert "Table 1. Marker caption." in out
    assert "| 1 | 2 |" in out
    # MinerU's HTML did NOT bleed into the body.
    assert "<td>col1</td>" not in out
    # Image link + sidecar link added.
    assert "assets/tbl1.jpg" in out
    assert "[Table 1 — separate markdown]" in out
    # Surrounding body preserved.
    assert "After table." in out
    # Counter: matched-and-linked (new), not the old tables_swapped.
    assert report.rescues["hybrid_splice"]["tables_linked_inline"] == 1
    assert report.rescues["hybrid_splice"]["tables_swapped"] == 0


def test_align_passes_pdf_doc_to_table_renderer(monkeypatch):
    """When pdf_doc is provided to _align_marker_to_mineru_layout, the
    table-render path calls render_crop(pdf_doc, page_idx, bbox) and
    forwards the resulting PIL image as pil_image_override to
    wrap_mineru.table_block_to_md. This is the test variant for
    fudge #10: render fresh from PDF instead of loading MinerU's JPG."""
    marker_md = ("Table 1. caption.\n\n"
                 "| a | b |\n|---|---|\n| 1 | 2 |\n")
    # Block with rowspan -> forces VLM path
    blk = _StubBlock(
        type="table",
        page_idx=2,
        bbox=(40, 68, 507, 249),
        text="Table 1. MinerU caption.",
        html=("<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
              "<tr><td>C</td></tr></table>"),
        image_path="t1.jpg",
    )
    parsed = _StubParsed(blocks=[blk])

    rendered_calls = []

    def _fake_render_crop(doc, page_idx, bbox, dpi=None):
        rendered_calls.append((doc, page_idx, tuple(bbox)))
        from PIL import Image as _Image
        return _Image.new("RGB", (100, 50), color="white")

    from PIL import Image as _Image
    captured = {}

    def _fake_table_block_to_md(block, table_idx, assets_rel,
                                vlm_tables, assets_abs, report,
                                asset_prefix="",
                                pil_image_override=None, **kwargs):
        captured["override"] = pil_image_override
        captured["override_size"] = (pil_image_override.size
                                     if pil_image_override else None)
        return "\n**Caption**\n\n| A | B |\n|---|---|\n| X | Y |\n"

    monkeypatch.setattr(p2m, "render_crop", _fake_render_crop)
    import wrap_mineru as _wm
    monkeypatch.setattr(_wm, "table_block_to_md", _fake_table_block_to_md)

    report = _empty_report()
    sentinel_doc = object()  # any non-None object plays the role of fitz.Document
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=True,
        strip_chart_details=False, report=report,
        pdf_doc=sentinel_doc)

    # render_crop was called with the block's page_idx and bbox.
    assert len(rendered_calls) == 1
    doc, page_idx, bbox = rendered_calls[0]
    assert doc is sentinel_doc
    assert page_idx == 2
    assert bbox == (40, 68, 507, 249)
    # The PIL override was threaded through to table_block_to_md.
    assert captured["override_size"] == (100, 50)


def test_align_no_pdf_doc_skips_render_crop(monkeypatch):
    """When pdf_doc is None (default), render_crop is NOT called and
    table_block_to_md receives pil_image_override=None -- preserving
    the existing MinerU-JPG path."""
    marker_md = ("Table 1. caption.\n\n"
                 "| a | b |\n|---|---|\n| 1 | 2 |\n")
    blk = _StubBlock(
        type="table", page_idx=2, bbox=(40, 68, 507, 249),
        text="Table 1.",
        html=("<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
              "<tr><td>C</td></tr></table>"),
        image_path="t1.jpg",
    )
    parsed = _StubParsed(blocks=[blk])

    def _boom_render_crop(*a, **kw):
        raise AssertionError("render_crop must not be called when "
                             "pdf_doc is None")

    captured = {}

    def _fake_table_block_to_md(block, table_idx, assets_rel,
                                vlm_tables, assets_abs, report,
                                asset_prefix="",
                                pil_image_override=None, **kwargs):
        captured["override"] = pil_image_override
        return "\n**Caption**\n\n| A | B |\n|---|---|\n| X | Y |\n"

    monkeypatch.setattr(p2m, "render_crop", _boom_render_crop)
    import wrap_mineru as _wm
    monkeypatch.setattr(_wm, "table_block_to_md", _fake_table_block_to_md)

    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=True,
        strip_chart_details=False, report=report,
        pdf_doc=None)
    assert captured["override"] is None


def test_align_render_crop_failure_falls_back_to_jpg(monkeypatch):
    """When render_crop raises (e.g. invalid bbox), the splice logs
    and falls back to None override so the MinerU-JPG path takes over.
    The table still renders successfully -- no exception escapes."""
    marker_md = ("Table 1. caption.\n\n"
                 "| a | b |\n|---|---|\n| 1 | 2 |\n")
    blk = _StubBlock(
        type="table", page_idx=2, bbox=(40, 68, 507, 249),
        text="Table 1.",
        html=("<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
              "<tr><td>C</td></tr></table>"),
        image_path="t1.jpg",
    )
    parsed = _StubParsed(blocks=[blk])

    def _broken_render_crop(*a, **kw):
        raise RuntimeError("simulated render failure")

    captured = {}

    def _fake_table_block_to_md(block, table_idx, assets_rel,
                                vlm_tables, assets_abs, report,
                                asset_prefix="",
                                pil_image_override=None, **kwargs):
        captured["override"] = pil_image_override
        return "\n**Caption**\n\n| A | B |\n|---|---|\n"

    monkeypatch.setattr(p2m, "render_crop", _broken_render_crop)
    import wrap_mineru as _wm
    monkeypatch.setattr(_wm, "table_block_to_md", _fake_table_block_to_md)

    report = _empty_report()
    # No exception should escape.
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=True,
        strip_chart_details=False, report=report,
        pdf_doc=object())
    # Override is None (fell back to MinerU JPG path).
    assert captured["override"] is None


def test_align_table_workers_concurrent_swap(monkeypatch):
    """With table_workers > 1 and multiple tables, table VLM tasks
    are dispatched through a ThreadPoolExecutor. Verify the executor
    is used and all tables still land in the right positions, with
    report.tables ordered by index."""
    marker_md = (
        "Table 1. Caption one.\n\n"
        "| a | b |\n"
        "|---|---|\n"
        "| x | y |\n\n"
        "Body.\n\n"
        "Table 2. Caption two.\n\n"
        "| c | d |\n"
        "|---|---|\n"
        "| p | q |\n\n"
        "Table 3. Caption three.\n\n"
        "| e | f |\n"
        "|---|---|\n"
        "| m | n |\n\n"
        "End.\n"
    )
    _ok_html = (
        "<table><tr><td>h1</td><td>h2</td></tr>"
        "<tr><td>v1</td><td>v2</td></tr>"
        "<tr><td>w1</td><td>w2</td></tr></table>"
    )
    blks = [
        _StubBlock(type="table", text="Table 1. MinerU.",
                   html=_ok_html, image_path="t1.jpg"),
        _StubBlock(type="table", text="Table 2. MinerU.",
                   html=_ok_html, image_path="t2.jpg"),
        _StubBlock(type="table", text="Table 3. MinerU.",
                   html=_ok_html, image_path="t3.jpg"),
    ]
    parsed = _StubParsed(blocks=blks)

    # Track whether ThreadPoolExecutor was invoked.
    used_pool = []
    import concurrent.futures as _cf
    orig_pool = _cf.ThreadPoolExecutor

    class _SpyPool(orig_pool):
        def __init__(self, *a, **kw):
            used_pool.append(kw.get("max_workers", a[0] if a else None))
            super().__init__(*a, **kw)

    monkeypatch.setattr(p2m, "ThreadPoolExecutor", _SpyPool)

    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report,
        table_workers=4)

    # Pool was invoked with max_workers=4.
    assert used_pool == [4]
    # Conservative splice: marker's pipe-md kept; only image +
    # sidecar link added per table. Verify each table got its link.
    assert "[Table 1 — separate markdown]" in out
    assert "[Table 2 — separate markdown]" in out
    assert "[Table 3 — separate markdown]" in out
    # Marker's data rows preserved verbatim (no replacement).
    assert "| x | y |" in out
    assert "| p | q |" in out
    assert "| m | n |" in out
    # MinerU's HTML must NOT bleed into the body.
    assert "<td>h1</td>" not in out
    # report.tables sorted by index (concurrent execution may have
    # appended out of order; the sort fixes that).
    indices = [getattr(t, "index", None) for t in report.tables]
    assert indices == sorted(indices)


def test_align_table_insert_has_blank_line_before_image(tmp_path):
    """The image link inserted after marker's pipe-md must be
    separated from the last pipe row by a BLANK LINE (\\n\\n), not
    just a single newline. Without this, some markdown renderers
    treat the image as a continuation of the table -> giant
    wide-column cell. This is the jacquet-Table-3 bug."""
    marker_md = (
        "**Table 3.** Sample.\n\n"
        "| Group | Na/Al |\n"
        "|-------|-------|\n"
        "| EH    | 1.0   |\n"
        "| OC    | 0.3   |\n"
        "\n"
        "Next paragraph follows.\n"
    )
    blk = _StubBlock(
        type="table", text="Table 3. MinerU caption.",
        html=("<table><tr><td>h1</td><td>h2</td></tr>"
              "<tr><td>a</td><td>b</td></tr>"
              "<tr><td>c</td><td>d</td></tr></table>"),
        image_path="tbl3.jpg")
    parsed = _StubParsed(blocks=[blk])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", tmp_path,
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)

    # Locate marker's last pipe row and the inserted image. There MUST
    # be at least one blank line between them (two consecutive \n's).
    last_pipe_row = "| OC    | 0.3   |"
    img_marker = "](assets/tbl3.jpg)"
    pipe_idx = out.index(last_pipe_row)
    img_idx = out.index(img_marker)
    assert pipe_idx < img_idx
    between = out[pipe_idx + len(last_pipe_row):img_idx]
    # Between the last pipe row's closing `|` and the image alt-text's
    # opening `![`, we need a paragraph break. Count consecutive
    # newlines -- at least 2 (i.e., a blank line) is required.
    nl_count = between.count("\n")
    assert nl_count >= 2, (
        f"only {nl_count} newline(s) between last pipe row and image; "
        f"renderer will treat the image as a wide table cell. "
        f"between bytes: {between!r}"
    )


def test_align_unmatched_table_lands_in_start_of_doc_index(tmp_path):
    """A MinerU table with NO matching marker caption is now collected
    into a `## Extracted tables` section at the START of the body
    (was: EOD splice at end of doc, which conflicted with refs-
    walks-from-end hooks)."""
    marker_md = (
        "# Paper title\n\n"
        "Intro body text. No Table N reference here.\n\n"
        "## References\n\nSome refs.\n"
    )
    # MinerU has Table 5 but marker has no `Table 5.` caption anywhere
    blk = _StubBlock(
        type="table", page_idx=4, bbox=(50, 50, 300, 200),
        text="Table 5. Orphan caption.",
        html=("<table><tr><td>h1</td><td>h2</td></tr>"
              "<tr><td>a</td><td>b</td></tr>"
              "<tr><td>c</td><td>d</td></tr></table>"),
        image_path="tbl5.jpg")
    parsed = _StubParsed(blocks=[blk])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", tmp_path,
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # The `## Extracted tables` heading appears at the START of body.
    assert out.startswith("## Extracted tables")
    # Sidecar link to Table 5 is present.
    assert "[Table 5 — separate markdown]" in out
    # References section still at END (not confused by the new index).
    refs_idx = out.find("## References")
    extracted_idx = out.find("## Extracted tables")
    assert extracted_idx < refs_idx
    # Marker's original body content preserved verbatim.
    assert "Intro body text." in out
    # Counter: tables_appended_index, not tables_swapped.
    hsp = report.rescues["hybrid_splice"]
    assert hsp["tables_appended_index"] == 1
    assert hsp["tables_linked_inline"] == 0


def test_align_unmatched_when_none_no_extracted_tables_section(tmp_path):
    """If every MinerU table matches a marker caption, no `## Extracted
    tables` section is emitted."""
    marker_md = (
        "Table 1. Marker caption.\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\nAfter.\n"
    )
    blk = _StubBlock(
        type="table", text="Table 1. MinerU.",
        html=("<table><tr><td>h1</td><td>h2</td></tr>"
              "<tr><td>x</td><td>y</td></tr>"
              "<tr><td>z</td><td>w</td></tr></table>"),
        image_path="t1.jpg")
    parsed = _StubParsed(blocks=[blk])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", tmp_path,
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert "## Extracted tables" not in out


def test_align_table_workers_one_uses_serial_path(monkeypatch):
    """With table_workers=1 (the default), no ThreadPoolExecutor is
    spawned -- the sequential loop runs directly."""
    marker_md = ("Table 1. caption.\n\n"
                 "| a | b |\n|---|---|\n| 1 | 2 |\n")
    blk = _StubBlock(type="table", text="Table 1. MinerU.",
                     html="<table><tr><td>x</td></tr></table>",
                     image_path="t1.jpg")
    parsed = _StubParsed(blocks=[blk])

    used_pool = []
    import concurrent.futures as _cf
    orig_pool = _cf.ThreadPoolExecutor

    class _SpyPool(orig_pool):
        def __init__(self, *a, **kw):
            used_pool.append(kw.get("max_workers", a[0] if a else None))
            super().__init__(*a, **kw)

    monkeypatch.setattr(p2m, "ThreadPoolExecutor", _SpyPool)

    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report,
        table_workers=1)
    # No pool spawned in the workers=1 case.
    assert used_pool == []


def test_warmup_vlm_connection_calls_chat_completions(monkeypatch):
    """The warmup helper sends a real /chat/completions call with a
    tiny placeholder image. This exercises vLLM's model worker (which
    /v1/models does NOT) so the first table call doesn't hit a cold
    worker -- see HYBRID_IMPLEMENTATION.md fudge #7."""
    captured = []

    def _fake_vlm(prompt, image, max_tokens=1500, timeout=None,
                  max_retries=None, raise_on_error=False):
        captured.append({
            "prompt": prompt,
            "image_size": image.size,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "timeout": timeout,
        })
        return "OK"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    monkeypatch.setattr(p2m, "_PROVIDER", "vllm")
    p2m._warmup_vlm_connection()
    # One call made.
    assert len(captured) == 1
    call = captured[0]
    # Tiny placeholder image (cheap to send and process).
    assert call["image_size"] == (32, 32)
    # Short prompt; small token budget.
    assert "OK" in call["prompt"]
    assert call["max_tokens"] == 5
    # Same retry behaviour as the table calls so transient rejections
    # don't break the warmup.
    assert call["max_retries"] == 5
    assert call["timeout"] == 30.0


def test_warmup_vlm_connection_swallows_errors(monkeypatch):
    """If the warmup itself fails, the helper logs and returns -- it
    must NOT raise. The real call after it will surface any persistent
    endpoint failure."""
    def _fake_vlm(*a, **kw):
        raise ConnectionError("vLLM down")

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    monkeypatch.setattr(p2m, "_PROVIDER", "vllm")
    # Must not raise.
    p2m._warmup_vlm_connection()


def test_warmup_vlm_connection_skips_anthropic(monkeypatch):
    """Anthropic provider isn't subject to the vLLM post-idle rejection
    pattern -- skip warmup to avoid an unnecessary API call."""
    def _boom(*a, **kw):
        raise AssertionError("should not be called for anthropic")

    monkeypatch.setattr(p2m, "vlm", _boom)
    monkeypatch.setattr(p2m, "_PROVIDER", "anthropic")
    p2m._warmup_vlm_connection()  # should be a no-op


def test_align_no_marker_anchor_fallback():
    """MinerU figure 4 with no `Fig. 4.` line in marker -> end-of-section
    fallback near body cross-ref. Counter increments."""
    marker_md = ("We discuss Fig. 4 below.\n\n"
                 "Some body.\n")
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="orphan.jpg",
                   figure_number=4, text="Orphan caption text."),
    ])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert "orphan.jpg" in out
    # Mineru caption used since marker had no caption line.
    assert "Orphan caption text." in out
    assert report.rescues["hybrid_splice"]["unmatched_mineru_figs"] == 1
    assert report.rescues["hybrid_splice"]["figures_spliced"] == 0


def test_align_no_marker_anchor_fallback_renders_subpanels():
    """Unmatched-anchor path must render subpanel_paths the same way
    the anchored path does. canup Figs 1+2 hit this: MinerU consolidated
    a+b panels (primary letter='b', subpanel_paths=[a]) but the
    Nature-style `**Figure 1** | **...` caption didn't match the anchor
    regex pre-fix, so EOD fallback dropped panel a."""
    marker_md = "We refer to Fig. 1 in the discussion.\n\nMore body.\n"
    parsed = _StubParsed(blocks=[
        _StubBlock(type="chart", image_path="fig1_panel_b.jpg",
                   figure_number=1, panel_letter="b",
                   subpanel_paths=["fig1_panel_a.jpg"],
                   text="Figure 1 | Multi-panel caption."),
    ])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # BOTH panels rendered, not just the primary.
    assert "fig1_panel_a.jpg" in out
    assert "fig1_panel_b.jpg" in out
    # Panel letters emitted in alt text.
    assert "panel a" in out
    assert "panel b" in out
    assert report.rescues["hybrid_splice"]["unmatched_mineru_figs"] == 1


def test_align_inferred_fig_number_pairing_single():
    """alexander Fig 2 / jacquet Fig 7 shape: MinerU has a real chart
    block whose caption text doesn't include a parseable `Fig. N`
    prefix (caption garbled or empty), and marker has exactly one
    unmatched `Fig. N` caption anchor. Pair them sequentially."""
    marker_md = ("Fig. 1. First figure caption text with words.\n\nBody.\n\n"
                 "Fig. 2. Second figure caption text only marker has.\n\n"
                 "After.\n")
    blocks = [
        _StubBlock(type="chart", image_path="fig1.jpg", page_idx=0,
                   bbox=(50, 100, 300, 250),
                   text="Fig. 1. First figure caption text with words."),
        # MinerU's caption text garbled -> no parseable fig number.
        _StubBlock(type="chart", image_path="fig2.jpg", page_idx=0,
                   bbox=(50, 300, 300, 450),
                   text="initial olivine FigNa $K_D$ data..."),
    ]
    parsed = _StubParsed(blocks=blocks)
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # Both images spliced + scored. Fig 1 matched normally (score 1.0),
    # Fig 2 paired via inference (score 0.85).
    fig1_score = next(f.score for f in report.figures if f.matched_figure == "1")
    fig2_score = next(f.score for f in report.figures if f.matched_figure == "2")
    assert fig1_score >= 0.9
    assert fig2_score == 0.85
    # Both images spliced into the body above their marker captions.
    assert "fig1.jpg" in out
    assert "fig2.jpg" in out
    # Inferred-pairing counter fired once for Fig 2.
    hsp = report.rescues["hybrid_splice"]
    assert hsp["inferred_fig_number_pairings"] == 1
    assert hsp["figures_spliced"] == 1  # Fig 1 matched normally
    # marker_caption_no_mineru_asset should NOT count Fig 2 since the
    # pairing covered it.
    assert hsp.get("marker_caption_no_mineru_asset", 0) == 0


def test_align_inferred_pairing_filters_decoration_size_blocks():
    """jacquet's blk[2] (29x29 publisher logo) was unmatched too but
    is below the size threshold. With 2 unmatched MinerU blocks vs
    1 unmatched marker number, naive counts-match would refuse to
    pair. The size filter rejects the logo so 1 real chart pairs
    cleanly with the 1 marker caption."""
    marker_md = "Fig. 7. Real figure caption marker has.\n\nAfter.\n"
    blocks = [
        # Publisher logo on title page -- 29x29, far below threshold.
        _StubBlock(type="image", image_path="logo.png", page_idx=0,
                   bbox=(500, 147, 529, 176), text=""),
        # Real Fig 7 chart on a later page with empty caption text.
        _StubBlock(type="chart", image_path="fig7.jpg", page_idx=5,
                   bbox=(49, 86, 270, 235), text=""),
    ]
    parsed = _StubParsed(blocks=blocks)
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert "fig7.jpg" in out
    assert "logo.png" not in out  # logo never paired, no EOD splice either
    hsp = report.rescues["hybrid_splice"]
    assert hsp["inferred_fig_number_pairings"] == 1


def test_align_inferred_pairing_skips_on_mismatched_counts():
    """3 unmatched MinerU blocks vs 2 unmatched marker numbers ->
    ambiguous pairing -> SKIP. Marker captions stay in body without
    images; the unmatched blocks go to the EOD fallback."""
    marker_md = "Fig. 1. A.\n\nFig. 2. B.\n\nAfter.\n"
    blocks = [
        _StubBlock(type="chart", image_path="a.jpg", page_idx=0,
                   bbox=(50, 50, 300, 200), text=""),  # no fid
        _StubBlock(type="chart", image_path="b.jpg", page_idx=0,
                   bbox=(50, 250, 300, 400), text=""),
        _StubBlock(type="chart", image_path="c.jpg", page_idx=0,
                   bbox=(50, 450, 300, 600), text=""),
    ]
    parsed = _StubParsed(blocks=blocks)
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    hsp = report.rescues["hybrid_splice"]
    assert hsp.get("inferred_fig_number_pairings", 0) == 0
    # Both Fig 1 and Fig 2 captions still have no mineru asset.
    assert hsp["marker_caption_no_mineru_asset"] == 2


def test_inferred_pairing_surfaces_in_frontmatter():
    report = p2m.QualityReport()
    report.rescues["hybrid_splice"] = {
        "figures_spliced": 1,
        "inferred_fig_number_pairings": 2,
    }
    yaml_text = "\n".join(report._rescues_yaml_lines())
    assert "inferred_fig_number_pairings: 2" in yaml_text
    d = report.to_dict()
    assert d["rescues"]["hybrid_splice"]["inferred_fig_number_pairings"] == 2


def test_align_inferred_tbl_number_pairing_single():
    """cuk Table 1 shape: MinerU has a real table block whose caption text
    starts with the table content (NOT `Table N.` prefix), so
    `_extract_table_number` returns None and the block dropped out of
    `mineru_tbls`. Marker has exactly one unmatched `Table N.` anchor.
    Pair them sequentially. Mirror of fudge #5 for the table path."""
    marker_md = ("Table 1. Marker caption.\n\n"
                 "| a | b |\n"
                 "|---|---|\n"
                 "| 1 | 2 |\n"
                 "\n"
                 "After table.\n")
    # MinerU's caption text doesn't start with "Table N." -> no number.
    blk = _StubBlock(
        type="table", page_idx=0, bbox=(50, 50, 300, 200),
        text="Potential Moon-forming giant impacts...",
        html=("<table><tr><td>c1</td><td>c2</td></tr>"
              "<tr><td>a</td><td>b</td></tr>"
              "<tr><td>1</td><td>2</td></tr></table>"),
        image_path="tbl1.jpg")
    parsed = _StubParsed(blocks=[blk])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # Marker's caption + pipe-md preserved verbatim (conservative
    # splice never replaces marker content).
    assert "Table 1. Marker caption." in out
    assert "| 1 | 2 |" in out
    # MinerU's HTML did NOT bleed into body; it lives in the sidecar.
    assert "<td>c1</td>" not in out
    # Image link + sidecar link added next to marker's caption.
    assert "[Table 1 — separate markdown]" in out
    # Inferred-pairing counter fired once for Table 1.
    hsp = report.rescues["hybrid_splice"]
    assert hsp["inferred_tbl_number_pairings"] == 1
    # marker_caption_no_mineru_asset should NOT count Table 1 since
    # the inferred pairing covered it.
    assert hsp.get("marker_caption_no_mineru_asset", 0) == 0


def test_align_inferred_tbl_pairing_skips_on_mismatched_counts():
    """2 unmatched MinerU table blocks vs 1 unmatched marker number ->
    ambiguous pairing -> SKIP. The marker caption still has no asset."""
    marker_md = "Table 1. A.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nAfter.\n"
    blocks = [
        _StubBlock(type="table", page_idx=0, bbox=(50, 50, 300, 200),
                   text="Random caption no number prefix one.",
                   image_path="t1.jpg"),
        _StubBlock(type="table", page_idx=0, bbox=(50, 250, 300, 400),
                   text="Random caption no number prefix two.",
                   image_path="t2.jpg"),
    ]
    parsed = _StubParsed(blocks=blocks)
    report = _empty_report()
    p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    hsp = report.rescues["hybrid_splice"]
    assert hsp.get("inferred_tbl_number_pairings", 0) == 0
    # Marker Table 1 caption still has no MinerU asset.
    assert hsp["marker_caption_no_mineru_asset"] == 1


def test_inferred_tbl_pairing_surfaces_in_frontmatter():
    report = p2m.QualityReport()
    report.rescues["hybrid_splice"] = {
        "tables_swapped": 1,
        "inferred_tbl_number_pairings": 1,
    }
    yaml_text = "\n".join(report._rescues_yaml_lines())
    assert "inferred_tbl_number_pairings: 1" in yaml_text
    d = report.to_dict()
    assert d["rescues"]["hybrid_splice"]["inferred_tbl_number_pairings"] == 1


def test_align_marker_caption_no_mineru_asset():
    """Marker has `Fig. 5.` but mineru has nothing -> caption left
    intact, counter increments."""
    marker_md = "Fig. 5. Marker has this caption with no mineru figure.\n"
    parsed = _StubParsed(blocks=[])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert out == marker_md
    assert (report.rescues["hybrid_splice"]
            ["marker_caption_no_mineru_asset"] == 1)


def test_align_reverse_order_application():
    """Three figures spliced at different offsets all land correctly
    (earlier offsets unaffected by later inserts)."""
    marker_md = ("Fig. 1. First.\n\n"
                 "Body A.\n\n"
                 "Fig. 2. Second.\n\n"
                 "Body B.\n\n"
                 "Fig. 3. Third.\n")
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="a.jpg", figure_number=1),
        _StubBlock(type="image", image_path="b.jpg", figure_number=2),
        _StubBlock(type="image", image_path="c.jpg", figure_number=3),
    ])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    # Find the image link and the original caption-line marker for each
    # figure. "First." / "Second." / "Third." are unique body tokens that
    # only appear in marker's caption lines (the alt-text uses the WHOLE
    # caption tail, but we'd find the body token at the SECOND
    # occurrence -- caption proper -- so locate the standalone caption
    # by matching the "\nFig. N. " line prefix.
    def _link_pos(name):
        return out.index(f"]({name}")
    def _caption_pos(label):
        return out.index(f"\nFig. {label}.")
    a_link = _link_pos("assets/a.jpg")
    b_link = _link_pos("assets/b.jpg")
    c_link = _link_pos("assets/c.jpg")
    fig1_cap = _caption_pos("1")
    fig2_cap = _caption_pos("2")
    fig3_cap = _caption_pos("3")
    # Each image link lands BEFORE its matching caption line.
    assert a_link < fig1_cap < b_link < fig2_cap < c_link < fig3_cap
    assert report.rescues["hybrid_splice"]["figures_spliced"] == 3


def test_align_duplicate_mineru_figs_no_subpanel():
    """Two mineru blocks with the same figure_number, neither marked
    as a subpanel primary -> both image links emitted; duplicate
    counter increments."""
    marker_md = "Fig. 7. The caption.\n"
    parsed = _StubParsed(blocks=[
        _StubBlock(type="image", image_path="seven_a.jpg", figure_number=7),
        _StubBlock(type="image", image_path="seven_b.jpg", figure_number=7),
    ])
    report = _empty_report()
    out = p2m._align_marker_to_mineru_layout(
        marker_md, parsed, "assets", Path("/tmp/assets"),
        asset_prefix="", vlm_rewrite_tables=False,
        strip_chart_details=False, report=report)
    assert "seven_a.jpg" in out
    assert "seven_b.jpg" in out
    assert report.rescues["hybrid_splice"]["duplicate_mineru_figs"] == 1


# === argparse ============================================================


def test_layout_source_hybrid_choice_accepted():
    """--layout-source=hybrid is in the choice list; garbage rejected."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-source",
                        choices=["marker", "mineru", "hybrid"], default=None)
    ns = parser.parse_args(["--layout-source", "hybrid"])
    assert ns.layout_source == "hybrid"
    with pytest.raises(SystemExit):
        parser.parse_args(["--layout-source", "garbage"])


# === pipeline state ======================================================


def test_collect_pipeline_state_hybrid():
    """Hybrid records BOTH the split (figure_layout_source/caption_text_source)
    AND force_ocr (marker side honors it) AND mineru sub-toggles."""
    state = p2m._collect_pipeline_state(
        table_finder="pymupdf",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=True,
        layout_source="hybrid",
        rescue_mineru_orphan_captions=True,
        rescue_mineru_subpanels=False,
    )
    assert state["layout_source"] == "hybrid"
    assert state["figure_layout_source"] == "mineru"
    assert state["caption_text_source"] == "marker"
    assert state["force_ocr"] is True
    assert state["rescue_mineru_orphan_captions"] is True
    assert state["rescue_mineru_subpanels"] is False
    # Marker-only fields (table_finder, figmatch_strategy, rotate_tables,
    # rescue_orphan_tables/figures) are NOT included under hybrid (those
    # marker-only hooks don't run).
    assert "table_finder" not in state
    assert "figmatch_strategy" not in state
    assert "rotate_tables" not in state


# === integration =========================================================


def test_run_marker_plus_mineru_layout_e2e(tmp_path, monkeypatch):
    """End-to-end: mocked run_marker + pre-baked middle.json -> spliced
    output contains marker's caption text once and mineru's image link
    once per figure."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)
    mineru_dir = tmp_path / "out" / "mineru"

    # Marker output (canonical body + caption text).
    marker_md = (
        "# Test Paper Title\n\n"
        "**Abstract.** This is the abstract.\n\n"
        "Introduction body text.\n\n"
        "Figure 1. Schematic of the apparatus.\n\n"
        "Body after fig 1.\n\n"
        "Table 1. Sample data.\n\n"
        "Old marker table fallback.\n\n"
        "References here.\n"
    )

    def _fake_run_marker(pdf, assets, asset_prefix="", force_ocr=False):
        return marker_md, {}

    monkeypatch.setattr(p2m, "run_marker", _fake_run_marker)

    # Stage a pre-baked mineru output (reuse the wrap_mineru fixture).
    from test_paper2md_mineru_layout import _stage_minimal_mineru_dir

    def _fake_run_mineru(pdf, mdir, **_kw):
        _stage_minimal_mineru_dir(Path(mdir), pdf.stem)
        return Path(mdir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    report = p2m.QualityReport()
    body = p2m.run_marker_plus_mineru_layout(
        pdf_path, assets_dir,
        report=report, mineru_dir=mineru_dir,
    )
    # Marker's clean caption text preserved.
    assert "Figure 1. Schematic of the apparatus." in body
    assert "Table 1. Sample data." in body
    # Mineru's image asset spliced above the figure caption.
    # Figure renamed: figure_{id}_p{page}.jpg (caption "Figure 1." -> id "1").
    assert "figure_1_p1.jpg" in body
    # Conservative splice: marker's body content stays. The MinerU
    # table content goes to the sidecar; only the link is in the body.
    assert "Old marker table fallback." in body
    assert "[Table 1 — separate markdown]" in body
    # Hybrid splice counters populated (new schema).
    assert report.rescues["hybrid_splice"]["figures_spliced"] == 1
    assert report.rescues["hybrid_splice"]["tables_linked_inline"] == 1
    # Mineru intermediates persisted.
    assert (mineru_dir / "paper_middle.json").exists()
    # Mineru images renamed: figure_{id}_p{page}.jpg + table_{id}_p{page}_{idx}.{jpg,md}.
    assert (assets_dir / "figure_1_p1.jpg").exists()
    assert (assets_dir / "table_1_p1_1.jpg").exists()
    assert (assets_dir / "table_1_p1_1.md").exists()
    assert not (assets_dir / "fig1.jpg").exists()
    assert not (assets_dir / "tbl1.jpg").exists()


# === marker image stripping ================================================


def test_strip_marker_image_links_basic():
    """Image-link lines whose basename is in the set are dropped; other
    image links (mineru's hash-named ones) are left in place."""
    body = (
        "Body text.\n\n"
        "![](page_1_Figure_2.jpeg)\n\n"
        "Figure 1. Caption.\n\n"
        "![Figure 1](assets/abc123.jpg)\n\n"
        "More body.\n"
    )
    out, n = p2m._strip_marker_image_links(
        body, {"page_1_Figure_2.jpeg"})
    assert n == 1
    assert "page_1_Figure_2.jpeg" not in out
    assert "abc123.jpg" in out
    assert "Figure 1. Caption." in out
    assert "More body." in out


def test_strip_marker_image_links_with_path_prefix():
    """Helper compares basenames, so `assets/<name>` prefix doesn't
    block matching."""
    body = "![alt](assets/_page_0_Picture_2.jpeg)\nNext.\n"
    out, n = p2m._strip_marker_image_links(
        body, {"_page_0_Picture_2.jpeg"})
    assert n == 1
    assert "_page_0_Picture_2.jpeg" not in out
    assert "Next." in out


def test_strip_marker_image_links_empty_set_noop():
    """Empty set returns body unchanged with count 0."""
    body = "![](page_1.jpeg)\nFoo.\n"
    out, n = p2m._strip_marker_image_links(body, set())
    assert out == body
    assert n == 0


def test_strip_marker_image_links_strips_leading_span_anchor():
    """Marker (PRL/APS papers) emits `<span id="page-X-Y"></span>` on
    the same line before the image link. The strip helper consumes the
    leading span(s) along with the matched image."""
    body = (
        "Body.\n\n"
        '<span id="page-1-0"></span>![](assets/_page_1_Figure_3.jpeg)\n\n'
        "After.\n"
    )
    out, n = p2m._strip_marker_image_links(
        body, {"_page_1_Figure_3.jpeg"})
    assert n == 1
    assert "_page_1_Figure_3.jpeg" not in out
    # Span anchor consumed too -- no dangling stub.
    assert "page-1-0" not in out
    assert "After." in out


def test_strip_marker_image_links_strips_multiple_leading_spans():
    """Some marker outputs chain multiple span anchors before one
    image link; all are consumed when the image is removed."""
    body = (
        '<span id="a"></span><span id="b"></span>'
        "![](assets/_page_2_Figure_4.jpeg)\nNext.\n"
    )
    out, n = p2m._strip_marker_image_links(
        body, {"_page_2_Figure_4.jpeg"})
    assert n == 1
    assert "_page_2_Figure_4.jpeg" not in out
    assert 'id="a"' not in out
    assert 'id="b"' not in out
    assert "Next." in out


def test_strip_marker_image_links_leaves_isolated_span_alone():
    """An isolated `<span id>` not followed by an image is left in
    place, even when its image-less line could be confused with one."""
    body = '<span id="anchor-only"></span>\nBody after.\n'
    out, n = p2m._strip_marker_image_links(
        body, {"_page_1.jpeg"})
    assert n == 0
    assert 'id="anchor-only"' in out
    assert "Body after." in out


def test_strip_marker_image_links_preserves_unknown_basenames():
    """Image links not in the set survive even if they look marker-ish."""
    body = "![](page_other.jpeg)\nBody.\n"
    out, n = p2m._strip_marker_image_links(
        body, {"page_1.jpeg"})
    assert n == 0
    assert "page_other.jpeg" in out


def test_run_marker_plus_mineru_layout_drops_marker_images(
        tmp_path, monkeypatch):
    """End-to-end: when marker emits an image link AND the matching
    figure number is spliced from MinerU, marker's image-link line is
    stripped from the body and the counter surfaces in the report."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)
    mineru_dir = tmp_path / "out" / "mineru"

    marker_md = (
        "Intro.\n\n"
        "![](_page_1_Figure_2.jpeg)\n\n"
        "Figure 1. Schematic of the apparatus.\n\n"
        "Body after fig 1.\n\n"
        "Table 1. Sample data.\n\n"
        "Old marker table fallback.\n\n"
        "References.\n"
    )

    def _fake_run_marker(pdf, assets, asset_prefix="", force_ocr=False):
        # Pretend marker wrote one image to disk.
        return marker_md, {"_page_1_Figure_2.jpeg":
                           assets / "_page_1_Figure_2.jpeg"}

    monkeypatch.setattr(p2m, "run_marker", _fake_run_marker)

    from test_paper2md_mineru_layout import _stage_minimal_mineru_dir

    def _fake_run_mineru(pdf, mdir, **_kw):
        _stage_minimal_mineru_dir(Path(mdir), pdf.stem)
        return Path(mdir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    report = p2m.QualityReport()
    body = p2m.run_marker_plus_mineru_layout(
        pdf_path, assets_dir,
        report=report, mineru_dir=mineru_dir,
    )
    # Marker's image-link line dropped from body.
    assert "_page_1_Figure_2.jpeg" not in body
    # Mineru's image still spliced (now under the renamed semantic name).
    assert "figure_1_p1.jpg" in body
    # Marker's caption preserved.
    assert "Figure 1. Schematic of the apparatus." in body
    # Counter surfaced in report.rescues.
    assert (report.rescues["hybrid_splice"]
            ["marker_image_links_removed"] == 1)


def test_marker_image_links_removed_surfaces_in_yaml_and_metajson():
    """The new counter shows up under rescues.hybrid_splice in both
    the YAML frontmatter and to_dict() outputs."""
    report = p2m.QualityReport()
    report.rescues["hybrid_splice"] = {
        "figures_spliced": 2,
        "marker_image_links_removed": 2,
    }
    yaml_lines = report._rescues_yaml_lines()
    yaml_text = "\n".join(yaml_lines)
    assert "hybrid_splice:" in yaml_text
    assert "marker_image_links_removed: 2" in yaml_text

    d = report.to_dict()
    assert (d["rescues"]["hybrid_splice"]
            ["marker_image_links_removed"] == 2)


# === MinerU refs swap ======================================================


def test_extract_mineru_refs_section_present(tmp_path):
    """When MinerU output has a `## References` heading, return the
    section text from heading to EOF."""
    p = tmp_path / "paper.md"
    p.write_text(
        "# Title\n\nBody.\n\n"
        "## References\n\n"
        "1. Smith, J. (2020). A paper. Journal, 1, 1.\n"
        "2. Doe, A. (2021). Another. Journal, 2, 2.\n",
        encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    assert out.startswith("## References")
    assert "1. Smith" in out
    assert "2. Doe" in out


def test_extract_mineru_refs_section_missing_returns_none(tmp_path):
    """When MinerU output has no `^##? References?` heading (Wackerle
    case), return None so caller falls back to marker's refs."""
    p = tmp_path / "paper.md"
    p.write_text("# Title\n\nBody only.\n\nAcknowledgments.\n",
                 encoding="utf-8")
    assert p2m._extract_mineru_refs_section(p) is None


def test_extract_mineru_refs_section_no_file_returns_none(tmp_path):
    """Missing file (mineru/ subdir not present) returns None."""
    p = tmp_path / "doesnt_exist.md"
    assert p2m._extract_mineru_refs_section(p) is None


def test_extract_mineru_refs_section_skips_when_followed_by_title(tmp_path):
    """alexander Science case: MinerU concatenated a perspective and
    the alexander article. The FIRST refs heading is the perspective's
    refs; its section extends through the alexander title + body. The
    extractor must skip refs headings followed by another `# Title`
    and pick the next one (alexander's actual refs)."""
    p = tmp_path / "alexander.md"
    p.write_text(
        "Some preamble.\n\n"
        "## References and Notes\n\n"
        "1. Perspective ref one.\n"
        "2. Perspective ref two.\n\n"
        "## Supporting Online Material\n\n"
        "# The Formation Conditions of Chondrules\n\n"
        "Body of the actual paper.\n\n"
        "References and Notes\n\n"
        "1. Alexander ref one.\n"
        "2. Alexander ref two.\n",
        encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    # Must NOT start with perspective refs.
    assert "Perspective ref one" not in out
    # Must contain alexander's refs.
    assert "Alexander ref one" in out
    assert "Alexander ref two" in out
    # Must NOT include the duplicated article body.
    assert "Body of the actual paper" not in out


def test_extract_mineru_refs_section_single_paper_unchanged(tmp_path):
    """Sanity: a paper with one refs heading + no `# Title` after it
    is unchanged by the contamination guard."""
    p = tmp_path / "single.md"
    p.write_text(
        "# Article Title\n\nBody.\n\n"
        "## References\n\n"
        "1. Ref one.\n"
        "2. Ref two.\n",
        encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    assert "1. Ref one" in out
    assert "2. Ref two" in out


def test_extract_mineru_refs_section_falls_back_when_all_followed_by_title(
        tmp_path):
    """Defensive: if every refs heading is followed by a `# Title`
    (unusual but possible if MinerU's order is reversed), fall back
    to the last refs heading. The section is then truncated at the
    next non-ref `#` / `##` heading so the trailing title doesn't
    leak through the swap (canup-style duplicate-back-matter bug
    fix)."""
    p = tmp_path / "weird.md"
    p.write_text(
        "## References\n\n"
        "1. First refs.\n\n"
        "# Some Title\n\n"
        "## References\n\n"
        "2. Second refs.\n\n"
        "# Another Title\n\nbody\n",
        encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    # Fallback picks the LAST refs heading.
    assert "2. Second refs" in out
    # New: trailing `# Another Title` is TRUNCATED (it's a non-ref
    # heading after the chosen refs section).
    assert "Another Title" not in out
    assert "body" not in out


class _FigBlock:
    """Stand-in for a parsed `_MBlock` with just the fields
    `_hybrid_fig_id_from_block` reads."""

    def __init__(self, text=None, figure_number=None):
        self.text = text
        self.figure_number = figure_number


def test_hybrid_fig_id_standard_caption_match():
    """Standard `FIG. N.` caption: leading-anchor match fires."""
    blk = _FigBlock(text="FIG. 5. Pressure-relative volume Hugoniot...")
    assert p2m._hybrid_fig_id_from_block(blk) == "5"


def test_hybrid_fig_id_panel_prefix_recovery():
    """Wackerle1962 pattern: MinerU's PaddleOCR concatenated the
    panel-b label fragment onto the front of Fig 9's caption text,
    yielding `(b) FIG. 9(a) and (b). ...`. The leading-anchor regex
    fails; the panel-prefix-strip fallback recovers id=9 so the
    image gets spliced."""
    blk = _FigBlock(text="(b) FIG. 9(a) and (b). Pressure vs relative "
                         "volume Hugoniot for fused quartz.")
    assert p2m._hybrid_fig_id_from_block(blk) == "9"


def test_hybrid_fig_id_panel_marker_variants():
    """The fallback handles the common panel-marker forms: `(a)`,
    `[b]`, `c.`, `d:`. All should strip cleanly and recover the
    fig id."""
    for prefix in ("(a) ", "[b] ", "c. ", "d: ", "(C) ", "[E] "):
        blk = _FigBlock(text=f"{prefix}Figure 7. Caption text.")
        assert p2m._hybrid_fig_id_from_block(blk) == "7", (
            f"failed on prefix {prefix!r}")


def test_hybrid_fig_id_panel_marker_only_returns_none():
    """A caption that's JUST the panel marker (no Fig N inside) must
    correctly return None -- the block is a true orphan (page-11 `(a)`
    in Wackerle1962) and should fall through to the EOD splice path,
    not falsely register as fig 'a'."""
    blk = _FigBlock(text="(a)")
    assert p2m._hybrid_fig_id_from_block(blk) is None
    blk = _FigBlock(text="(b) ")
    assert p2m._hybrid_fig_id_from_block(blk) is None


def test_hybrid_fig_id_no_false_positive_in_body_text():
    """Body text that mentions `Fig. 5` mid-sentence (cross-reference,
    not a caption) must NOT register as fig 5. The fallback only
    strips ONE panel-marker pattern; it doesn't search mid-string."""
    blk = _FigBlock(text="We observed the trend (see Fig. 5 below) in "
                         "the data.")
    # Doesn't start with Fig, doesn't start with a strippable panel
    # marker -> None. Crucial: don't fall through to a permissive search.
    assert p2m._hybrid_fig_id_from_block(blk) is None


def test_hybrid_fig_id_figure_number_field_wins():
    """When the block has `figure_number` set (populated by
    rescue_subpanel_groups on multi-panel primaries), use that
    regardless of caption text."""
    blk = _FigBlock(text="(a) random caption", figure_number=3)
    assert p2m._hybrid_fig_id_from_block(blk) == "3"


def test_extract_mineru_refs_section_truncates_canup_duplicate(tmp_path):
    """canup-style bug: MinerU's own .md duplicates the back-matter
    (## Refs / ## Acknowledgements / ## Methods / ## Refs again).
    The first refs heading is picked but its section must be
    truncated at the next non-ref heading (## Acknowledgements) so
    the swap doesn't append a duplicate Methods + second Refs to
    marker's body."""
    p = tmp_path / "canup_like.md"
    p.write_text(
        "# Lunar volatile depletion\n\n"
        "Body text of paper.\n\n"
        "## Methods\n\n"
        "First Methods content.\n\n"
        "## References\n\n"
        "1. Canup, R. M. (2014).\n"
        "2. Taylor (2006).\n\n"
        "## Acknowledgements\n\n"
        "We thank D. Stevenson.\n\n"
        "## Author contributions\n\n"
        "R.M.C. conceived the idea.\n\n"
        "## Methods\n\n"
        "DUPLICATE Methods content (MinerU's mis-classification).\n\n"
        "## References\n\n"
        "Second copy of refs (MinerU duplicate).\n",
        encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    # Refs entries preserved
    assert "1. Canup, R. M." in out
    assert "2. Taylor" in out
    # Duplicate back-matter is TRUNCATED -- key assertion of the fix
    assert "Acknowledgements" not in out
    assert "Author contributions" not in out
    assert "DUPLICATE Methods content" not in out
    assert "Second copy of refs" not in out


def test_extract_mineru_refs_section_continues_through_subref_headings(tmp_path):
    """A `## References` followed by another `## References` (sub-refs
    section or split refs) should NOT be truncated -- both are refs
    and belong together. Only NON-ref headings terminate the section."""
    p = tmp_path / "split_refs.md"
    p.write_text(
        "# Title\n\nBody.\n\n"
        "## References\n\n"
        "1. Smith.\n\n"
        "## References\n\n"
        "2. Jones (supplementary).\n",
        encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    assert "1. Smith" in out
    assert "2. Jones" in out


def test_extract_mineru_refs_section_case_insensitive(tmp_path):
    """ALL-CAPS `## REFERENCES` (Knudson-style) still matches."""
    p = tmp_path / "paper.md"
    p.write_text("# Title\n\nBody.\n\n## REFERENCES\n\n[1] Smith\n",
                 encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None
    assert "REFERENCES" in out
    assert "[1] Smith" in out


@pytest.mark.parametrize("heading_line", [
    "## References",                           # canup, canup2012, Knudson2013, Wackerle
    "### References",                          # lyzenga1983
    "#### References and Notes",               # cuk
    "#### REFERENCES AND NOTES",               # young
    "## **REFERENCES AND NOTES**",             # millot
    "#### **References**",                     # lyzenga1980
    "References and Notes",                    # MinerU heading-less for cuk
    "REFERENCES AND NOTES",                    # MinerU heading-less variant
    "**References**",                          # bold-only paragraph form
])
def test_refs_heading_re_matches_corpus_variants(heading_line, tmp_path):
    """The regex matches every heading style observed across the
    moon/chondrules/silica-shock hybrid corpus per
    docs/dev/HYBRID_REFERENCES_AUDIT_2026-05-12.md."""
    p = tmp_path / "paper.md"
    p.write_text(f"# Title\n\nBody.\n\n{heading_line}\n\n1. Ref one.\n",
                 encoding="utf-8")
    out = p2m._extract_mineru_refs_section(p)
    assert out is not None, f"Failed to match: {heading_line!r}"
    assert "1. Ref one." in out


def test_swap_hybrid_refs_with_mineru_basic(tmp_path):
    """Both sides have refs -> marker's refs section is replaced with
    MinerU's, marker's body before refs is preserved."""
    mineru_md = tmp_path / "paper.md"
    mineru_md.write_text(
        "# Title\n\n## References\n\n1. MinerU ref one.\n2. MinerU two.\n",
        encoding="utf-8")
    body = (
        "# Title\n\nIntro body.\n\n"
        "## References\n\n"
        "- Marker ref garbled.\n"
        "- Marker ref also.\n"
    )
    out, stats = p2m._swap_hybrid_refs_with_mineru(body, mineru_md)
    assert stats["mineru_refs_swapped"] is True
    # Marker's pre-refs body preserved.
    assert "Intro body." in out
    # Marker's refs replaced by MinerU's.
    assert "1. MinerU ref one." in out
    assert "2. MinerU two." in out
    assert "Marker ref garbled." not in out
    assert "Marker ref also." not in out


def test_swap_hybrid_refs_with_mineru_no_marker_heading(tmp_path):
    """If body has no `## References` heading (Elliott-style), body is
    returned unchanged."""
    mineru_md = tmp_path / "paper.md"
    mineru_md.write_text("## References\n\n1. MinerU ref.\n",
                         encoding="utf-8")
    body = "# Title\n\nBody with no refs heading.\n"
    out, stats = p2m._swap_hybrid_refs_with_mineru(body, mineru_md)
    assert stats["mineru_refs_swapped"] is False
    assert stats["reason"] == "no_marker_refs_heading"
    assert out == body


def test_swap_hybrid_refs_with_mineru_no_mineru_heading(tmp_path):
    """If MinerU has no refs heading (Wackerle case), marker's refs are
    kept and the swap is reported as not having happened."""
    mineru_md = tmp_path / "paper.md"
    mineru_md.write_text("# Title\n\nBody only, no refs.\n",
                         encoding="utf-8")
    body = (
        "# Title\n\nIntro.\n\n"
        "## References\n\n"
        "1. Marker has these.\n"
    )
    out, stats = p2m._swap_hybrid_refs_with_mineru(body, mineru_md)
    assert stats["mineru_refs_swapped"] is False
    assert stats["reason"] == "no_mineru_refs_heading"
    assert out == body
    assert "1. Marker has these." in out


def test_mineru_refs_swapped_surfaces_in_frontmatter():
    """The new counter shows up under rescues.hybrid_splice in both
    the YAML frontmatter and to_dict() outputs."""
    report = p2m.QualityReport()
    report.rescues["hybrid_splice"] = {
        "figures_spliced": 3,
        "mineru_refs_swapped": True,
    }
    yaml_text = "\n".join(report._rescues_yaml_lines())
    assert "hybrid_splice:" in yaml_text
    assert "mineru_refs_swapped" in yaml_text

    d = report.to_dict()
    assert (d["rescues"]["hybrid_splice"]
            ["mineru_refs_swapped"] is True)


# === table asset semantic-name rename (Phase 1) ============================
#
# Marker layout's process_tables saves table crops as
# `{prefix}table_p{page}_{idx}.jpg`. Hybrid layout used MinerU's
# content-hash filenames (`950a92fa....jpg`) which were unreadable and
# didn't pair with the per-table `.md` sidecar. Phase 1 renames the
# table JPG in assets/ to the marker convention at splice time.


def test_rename_table_asset_semantic_renames_file_and_block(tmp_path):
    """Happy path: MinerU's hash-named JPG exists on disk; the rename
    helper moves it to `{prefix}table_p{page}_{idx}.jpg` and updates
    block.image_path so downstream markdown emission uses the new name."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "deadbeef.jpg").write_bytes(b"\xff\xd8\xff")
    blk = _StubBlock(type="table", page_idx=2, image_path="deadbeef.jpg",
                     text="Table 1.", html="<table/>")
    p2m._rename_table_asset_semantic(blk, table_idx=1, assets_abs=assets,
                                     asset_prefix="")
    assert (assets / "table_p3_1.jpg").exists()
    assert not (assets / "deadbeef.jpg").exists()
    assert blk.image_path == "table_p3_1.jpg"


def test_rename_table_asset_semantic_honors_asset_prefix(tmp_path):
    """Supplement runs use asset_prefix='si_'. The rename target picks up
    the prefix so a main + SI run can share an assets dir."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "si_hashname.jpg").write_bytes(b"\xff\xd8\xff")
    blk = _StubBlock(type="table", page_idx=0, image_path="hashname.jpg",
                     text="Table 1.", html="<table/>")
    p2m._rename_table_asset_semantic(blk, table_idx=1, assets_abs=assets,
                                     asset_prefix="si_")
    assert (assets / "si_table_p1_1.jpg").exists()
    assert not (assets / "si_hashname.jpg").exists()
    assert blk.image_path == "table_p1_1.jpg"


def test_rename_table_asset_semantic_idempotent_when_already_renamed(tmp_path):
    """Re-run without --clean: source already renamed on a prior run.
    The helper bails silently and just points block.image_path at the
    already-existing target."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "table_p1_1.jpg").write_bytes(b"already-renamed")
    blk = _StubBlock(type="table", page_idx=0, image_path="hashname.jpg",
                     text="Table 1.", html="<table/>")
    p2m._rename_table_asset_semantic(blk, table_idx=1, assets_abs=assets,
                                     asset_prefix="")
    assert (assets / "table_p1_1.jpg").read_bytes() == b"already-renamed"
    assert blk.image_path == "table_p1_1.jpg"


def test_rename_table_asset_semantic_missing_source_leaves_block(tmp_path):
    """Source file missing AND target missing: nothing to rename. Helper
    must NOT crash and must leave block.image_path unchanged so the
    downstream image-link emission falls back gracefully."""
    assets = tmp_path / "assets"
    assets.mkdir()
    blk = _StubBlock(type="table", page_idx=0, image_path="ghost.jpg",
                     text="Table 1.", html="<table/>")
    p2m._rename_table_asset_semantic(blk, table_idx=1, assets_abs=assets,
                                     asset_prefix="")
    assert blk.image_path == "ghost.jpg"


def test_rename_table_asset_semantic_skips_when_no_page_idx(tmp_path):
    """Block with no page_idx can't form the semantic name; bail
    silently. (In practice MinerU always sets page_idx; this guards
    against synthetic blocks in test fixtures and refactors.)"""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "hashname.jpg").write_bytes(b"\xff\xd8\xff")
    blk = _StubBlock(type="table", page_idx=None,
                     image_path="hashname.jpg",
                     text="Table 1.", html="<table/>")
    p2m._rename_table_asset_semantic(blk, table_idx=1, assets_abs=assets,
                                     asset_prefix="")
    assert (assets / "hashname.jpg").exists()
    assert blk.image_path == "hashname.jpg"
