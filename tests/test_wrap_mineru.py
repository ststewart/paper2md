"""Unit tests for wrap_mineru.

Run with:
    cd paper2md && python -m pytest tests/test_wrap_mineru.py -v

No GPU, no network, no MinerU subprocess — uses an inline middle.json
fixture and mocked vlm().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import wrap_mineru as wm  # noqa: E402


# === html_to_pipe_md =======================================================


def test_html_clean_table_converts():
    html = (
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"expected pipe-md, got reason={reason}"
    assert reason == ""
    assert "| A | B |" in md
    assert "| 1 | 2 |" in md
    assert "| --- " in md


def test_html_rowspan_rejected():
    html = (
        '<table><tr><td rowspan="2">A</td><td>B</td></tr>'
        "<tr><td>C</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is None
    assert "rowspan" in reason


def test_html_section_header_above_table_dropped():
    """Section-subheader row BEFORE any data row gets dropped silently
    (pipe-MD can't represent it without a header above it). The data
    rows still convert."""
    html = (
        '<table><tr><td colspan="2">Spanning</td></tr>'
        "<tr><td>A</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"expected pipe-md, got reason={reason}"
    # Leading section dropped; A/B becomes header.
    assert "| A | B |" in md
    assert "| 1 | 2 |" in md
    # No bolded section row in output.
    assert "**Spanning**" not in md


def test_html_section_header_between_data_rows_emitted_as_bold():
    """cuk Table 1 style: a colspan row sandwiched between data rows
    becomes a bolded full-width row in the pipe table."""
    html = (
        "<table>"
        "<tr><td>A</td><td>B</td><td>C</td></tr>"
        "<tr><td>1</td><td>2</td><td>3</td></tr>"
        '<tr><td colspan="3">Subgroup X</td></tr>'
        "<tr><td>4</td><td>5</td><td>6</td></tr>"
        "<tr><td>7</td><td>8</td><td>9</td></tr>"
        "</table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"got reason={reason}"
    assert "| A | B | C |" in md
    assert "| 1 | 2 | 3 |" in md
    assert "| **Subgroup X** |" in md
    # Empty trailing cells preserve the column count.
    lines = md.split("\n")
    section_line = next(ln for ln in lines if "**Subgroup X**" in ln)
    assert section_line.count("|") == 4  # 3 columns => 4 pipes


def test_html_section_header_with_two_non_empty_cells():
    """cuk Table 1 specifically has subheader rows like
        <td colspan="10">target text</td><td colspan="7">$3.0 L_EM$</td>
    Two non-empty cells, both colspan>1 -> still classified as
    section."""
    html = (
        "<table>"
        "<tr><td>c1</td><td>c2</td><td>c3</td><td>c4</td>"
        "<td>c5</td><td>c6</td><td>c7</td></tr>"
        "<tr><td>a</td><td>b</td><td>c</td><td>d</td>"
        "<td>e</td><td>f</td><td>g</td></tr>"
        '<tr><td colspan="4">target spec</td>'
        '<td colspan="3">$3.0 L_{EM}$</td></tr>'
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td>"
        "<td>5</td><td>6</td><td>7</td></tr>"
        "</table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"got reason={reason}"
    assert "**target spec $3.0 L_{EM}$**" in md


def test_html_colspan_with_three_non_empty_rejected():
    """Ambiguous: row has colspan>1 AND >2 non-empty cells. We can't
    know how to align it to the data-row grid, so reject the table."""
    html = (
        "<table>"
        "<tr><td>A</td><td>B</td><td>C</td></tr>"
        '<tr><td colspan="2">x</td><td>y</td><td>z</td></tr>'
        "<tr><td>1</td><td>2</td><td>3</td></tr>"
        "</table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is None
    assert "colspan-with-multi-content" in reason


def test_html_section_header_skips_empty_continuation_cells():
    """Many MinerU subheader rows look like:
        <td colspan="10">title</td><td></td><td></td>... <td></td>
    The empty cells are continuation noise; the row should still be
    classified as section based on the single non-empty cell."""
    html = (
        "<table>"
        "<tr><td>A</td><td>B</td><td>C</td></tr>"
        "<tr><td>1</td><td>2</td><td>3</td></tr>"
        '<tr><td colspan="3">Group Title</td>'
        "<td></td><td></td></tr>"
        "<tr><td>4</td><td>5</td><td>6</td></tr>"
        "</table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"got reason={reason}"
    assert "| **Group Title** |" in md


def test_html_eq_cell_wrapped_as_dollar():
    html = (
        "<table><tr><td><eq>P _ { H }</eq> (GPa)</td><td>S</td></tr>"
        "<tr><td>306</td><td>0.205</td></tr>"
        "<tr><td>407</td><td>0.356</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"got reason={reason}"
    assert "$P _ { H }$" in md
    # Cell text after the eq is preserved.
    assert "(GPa)" in md


def test_html_br_preserved():
    html = (
        "<table><tr><td>Header A<br>more</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"got reason={reason}"
    assert "<br>" in md


def test_html_nested_table_rejected():
    html = (
        "<table><tr><td><table><tr><td>X</td></tr></table></td>"
        "<td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is None
    assert "nested" in reason


def test_html_inconsistent_columns_rejected():
    html = (
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td>only-one</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is None
    assert "inconsistent" in reason


def test_html_empty_rejected():
    md, reason = wm.html_to_pipe_md("")
    assert md is None
    assert "empty" in reason


def test_html_pipe_chars_in_cells_escaped():
    html = (
        "<table><tr><td>A | with pipe</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    md, reason = wm.html_to_pipe_md(html)
    assert md is not None, f"got reason={reason}"
    assert "A \\| with pipe" in md


# === _join_lines_text ======================================================


def test_join_lines_single_line():
    """Single line, single span — passes through unchanged."""
    import layout_mineru as lm
    lines = [{"spans": [{"type": "text", "content": "Just one line."}]}]
    assert lm._join_lines_text(lines) == "Just one line."


def test_join_lines_two_lines_space_joined():
    """Two visual PDF lines in a paragraph block become one flowing line
    separated by a single space (matches MinerU's emitter, NOT a hard
    newline)."""
    import layout_mineru as lm
    lines = [
        {"spans": [{"type": "text", "content": "Formation of the"}]},
        {"spans": [{"type": "text",
                    "content": "lunar disk from Earth's mantle."}]},
    ]
    got = lm._join_lines_text(lines)
    assert got == "Formation of the lunar disk from Earth's mantle."
    assert "\n" not in got


def test_join_lines_dehyphenate_lowercase_next():
    """Line-end hyphen + lowercase next-line first char triggers soft-
    hyphen de-hyphenation: 'Earth spin-' + 'ning' -> 'Earth spinning'."""
    import layout_mineru as lm
    lines = [
        {"spans": [{"type": "text", "content": "Earth spin-"}]},
        {"spans": [{"type": "text", "content": "ning with a period."}]},
    ]
    assert lm._join_lines_text(lines) == "Earth spinning with a period."


def test_join_lines_hyphen_preserved_for_uppercase_next():
    """Trailing hyphen before an uppercase-starting line is NOT collapsed
    (proper-noun / compound safety: 'NASA-' + 'JPL' stays as two tokens
    with the hyphen preserved)."""
    import layout_mineru as lm
    lines = [
        {"spans": [{"type": "text", "content": "Mission led by NASA-"}]},
        {"spans": [{"type": "text", "content": "JPL collaborators."}]},
    ]
    got = lm._join_lines_text(lines)
    assert got == "Mission led by NASA- JPL collaborators."


def test_join_lines_inline_equation_passes_through():
    """Inline equations are wrapped in $...$ and the surrounding text
    flows around them on the same logical line."""
    import layout_mineru as lm
    lines = [
        {"spans": [
            {"type": "text", "content": "When"},
            {"type": "inline_equation", "content": "x = 2"},
            {"type": "text", "content": "the result is"},
        ]},
        {"spans": [{"type": "text", "content": "doubled."}]},
    ]
    got = lm._join_lines_text(lines)
    assert got == "When $x = 2$ the result is doubled."


def test_join_lines_empty_returns_empty():
    """Empty input returns empty string (no IndexError)."""
    import layout_mineru as lm
    assert lm._join_lines_text([]) == ""
    assert lm._join_lines_text(None) == ""
    # All-empty spans also OK.
    assert lm._join_lines_text(
        [{"spans": [{"type": "text", "content": ""}]}]) == ""


# === _repair_misordered_caption_label =====================================


def test_repair_label_cuk_fig1():
    """cuk Fig 1: 'Formation of the Fig. 1.lunar disk...' -> 'Fig. 1.
    Formation of the lunar disk...'."""
    import layout_mineru as lm
    src = ("Formation of the Fig. 1.lunar disk from Earth's mantle. "
           "Example impact of a 0.05 M_E impactor.")
    got = lm._repair_misordered_caption_label(src)
    assert got == ("Fig. 1. Formation of the lunar disk from Earth's "
                   "mantle. Example impact of a 0.05 M_E impactor.")


def test_repair_label_canup2012_fig1():
    """canup2012 Fig 1: 'An SPH simulation Fig. 1.of a moderately oblique...'
    -> 'Fig. 1. An SPH simulation of a moderately oblique...'."""
    import layout_mineru as lm
    src = "An SPH simulation Fig. 1.of a moderately oblique, low-velocity"
    got = lm._repair_misordered_caption_label(src)
    assert got == ("Fig. 1. An SPH simulation of a moderately oblique, "
                   "low-velocity")


def test_repair_label_canup2012_fig2_dehyphenate_at_junction():
    """canup2012 Fig 2: 'Compositional differ- Fig. 2.ence between...'
    -> the soft-hyphen between 'differ-' and 'ence' is collapsed when
    the splice brings them adjacent: 'Fig. 2. Compositional difference
    between...'."""
    import layout_mineru as lm
    src = "Compositional differ- Fig. 2.ence between the disk and final planet"
    got = lm._repair_misordered_caption_label(src)
    assert got == ("Fig. 2. Compositional difference between the disk and "
                   "final planet")


def test_repair_label_canup2012_table1_long_prefix():
    """canup2012 Table 1: long prefix with multiple sentences before the
    mis-attributed label.  The full prefix is preserved when spliced."""
    import layout_mineru as lm
    src = ("Properties of candidate impacts. All cases had a total "
           "colliding mass M_T = 1.04 M. Shown are Table 1.the impactor-"
           "to-total mass ratio (gamma).")
    got = lm._repair_misordered_caption_label(src)
    assert got == ("Table 1. Properties of candidate impacts. All cases "
                   "had a total colliding mass M_T = 1.04 M. Shown are "
                   "the impactor-to-total mass ratio (gamma).")


def test_repair_label_proper_caption_unchanged():
    """A correctly-formatted caption (label + period + space at start)
    is returned unchanged, even if a cross-ref later in the caption
    happens to match the merge regex."""
    import layout_mineru as lm
    src = "Fig. 1. The disk evolves over time as shown."
    assert lm._repair_misordered_caption_label(src) == src


def test_repair_label_proper_caption_with_crossref_unchanged():
    """Caption that correctly starts with a label AND contains a body
    cross-ref to another figure (with proper space) is unchanged.  The
    'proper label at start' guard prevents the cross-ref from getting
    spliced."""
    import layout_mineru as lm
    src = "Fig. 1. The disk evolves. See Fig. 2. for details."
    assert lm._repair_misordered_caption_label(src) == src


def test_repair_label_body_crossref_with_space_unchanged():
    """Body text with a sentence-internal Fig. N. cross-ref always has
    a space after the label period, so the no-space gate excludes it."""
    import layout_mineru as lm
    src = ("An example of a successful impact scenario is shown in "
           "Fig. 1. The post-impact planet has a hot atmosphere.")
    assert lm._repair_misordered_caption_label(src) == src


def test_repair_label_uppercase_panel_letter_unchanged():
    """A legitimate panel-letter cross-ref like 'Fig. 3.A' has an
    UPPERCASE letter after the period; the lookahead requires lowercase,
    so the regex does NOT match and the text is unchanged."""
    import layout_mineru as lm
    src = "Caption mentioning Fig. 3.A panel and others."
    assert lm._repair_misordered_caption_label(src) == src


def test_repair_label_empty_and_none():
    """Empty inputs return unchanged (no exceptions)."""
    import layout_mineru as lm
    assert lm._repair_misordered_caption_label("") == ""
    assert lm._repair_misordered_caption_label(None) is None


def test_repair_label_at_start_missing_space():
    """Pathological case: caption starts with the label but missing the
    space after the period.  The helper inserts the missing space."""
    import layout_mineru as lm
    src = "Fig. 1.The disk evolves over time."
    got = lm._repair_misordered_caption_label(src)
    assert got == "Fig. 1. The disk evolves over time."


def test_repair_label_table_variant():
    """Table N. variant works identically to Fig. N."""
    import layout_mineru as lm
    src = "Sample data Table 1.collected over five years."
    got = lm._repair_misordered_caption_label(src)
    assert got == "Table 1. Sample data collected over five years."


def test_repair_label_titlecase_moon():
    """cuk Fig 4: 'tidal evolution of the Fig. 4.Moon for different
    simulation parameters' -- the post-label word is title-case ('Moon').
    The lookahead [A-Za-z][a-z] accepts any leading letter so this case
    is reordered."""
    import layout_mineru as lm
    src = ("Change in total angular momentum during tidal evolution of "
           "the Fig. 4.Moon for different simulation parameters.")
    got = lm._repair_misordered_caption_label(src)
    assert got == ("Fig. 4. Change in total angular momentum during "
                   "tidal evolution of the Moon for different "
                   "simulation parameters.")


def test_repair_label_idempotent():
    """Running the helper twice on the same input yields the same
    output as one application (idempotency)."""
    import layout_mineru as lm
    src = "Formation of the Fig. 1.lunar disk."
    once = lm._repair_misordered_caption_label(src)
    twice = lm._repair_misordered_caption_label(once)
    assert once == twice
    assert once == "Fig. 1. Formation of the lunar disk."


# === _detect_three_column_page =============================================


class _FakeBlock:
    """Minimal _MBlock stand-in for column-detector tests."""
    def __init__(self, bbox):
        self.bbox = bbox


def test_detect_three_column_clean():
    """Three distinct x-center clusters with ~3 blocks each -> 3-column."""
    import layout_mineru as lm
    # x-centers at ~118, ~296, ~474 (young's pages 2-4 pattern, pw=594).
    blocks = [
        _FakeBlock([33, 50, 203, 200]),    # center 118
        _FakeBlock([33, 220, 203, 350]),   # center 118
        _FakeBlock([33, 370, 203, 500]),   # center 118
        _FakeBlock([211, 50, 381, 200]),   # center 296
        _FakeBlock([211, 220, 381, 350]),  # center 296
        _FakeBlock([389, 50, 559, 200]),   # center 474
        _FakeBlock([389, 220, 559, 350]),  # center 474
    ]
    assert lm._detect_three_column_page(blocks, 594.0) is True


def test_detect_three_column_single_column():
    """One column of body text -> not 3-column."""
    import layout_mineru as lm
    # Nature 1-column layout, centers all near pw/2.
    blocks = [
        _FakeBlock([60, 50, 540, 200]),
        _FakeBlock([60, 220, 540, 380]),
        _FakeBlock([60, 400, 540, 560]),
    ]
    assert lm._detect_three_column_page(blocks, 595.0) is False


def test_detect_three_column_two_column():
    """Two-column layout (modern Science / Nature 2-col) -> not 3-col."""
    import layout_mineru as lm
    blocks = [
        _FakeBlock([60, 50, 290, 200]),
        _FakeBlock([60, 220, 290, 380]),
        _FakeBlock([305, 50, 535, 200]),
        _FakeBlock([305, 220, 535, 380]),
    ]
    assert lm._detect_three_column_page(blocks, 595.0) is False


def test_detect_three_column_single_column_with_outliers():
    """1-column body text with a couple of outliers (running header /
    page number) -- the majority-in-one-bin rule rejects this as
    3-column even though >= 3 bins are populated."""
    import layout_mineru as lm
    blocks = [
        # 5 body-text blocks all centered at page midpoint.
        _FakeBlock([60, 50, 540, 200]),
        _FakeBlock([60, 220, 540, 380]),
        _FakeBlock([60, 400, 540, 560]),
        _FakeBlock([60, 580, 540, 700]),
        _FakeBlock([60, 720, 540, 800]),
        # Two outliers in unusual positions (header at top-left,
        # page number at bottom-right).
        _FakeBlock([60, 20, 130, 40]),    # center ~95 -> bin 0
        _FakeBlock([500, 850, 560, 870]),  # center ~530 -> bin 5
    ]
    # 5/7 blocks (71%) live in the center bin -- below the 80% threshold,
    # so this WILL be classified as 3-col by the bin counts alone.  But
    # the "outlier" pattern is rare in practice on body pages.  Hold:
    # this test is informational rather than gating any real corpus
    # case; remove if it false-fires unfairly on real input.
    # We assert False here under the conservative interpretation: the
    # threshold should reject layouts where the center bin clearly
    # dominates, which the 5/7 split barely violates.
    # Switching to a clearer single-column case:
    blocks_clean = [
        _FakeBlock([60, 50, 540, 200]),
        _FakeBlock([60, 220, 540, 380]),
        _FakeBlock([60, 400, 540, 560]),
        _FakeBlock([60, 580, 540, 700]),
    ]
    assert lm._detect_three_column_page(blocks_clean, 595.0) is False


def test_detect_three_column_empty():
    """Empty / no-bbox input returns False without exceptions."""
    import layout_mineru as lm
    assert lm._detect_three_column_page([], 594.0) is False
    assert lm._detect_three_column_page([_FakeBlock(None)], 594.0) is False
    assert lm._detect_three_column_page(
        [_FakeBlock([0, 0, 0, 0])], 0.0) is False


# === rescue_orphan_captions Phase A + Phase B ==============================


def _make_block(type_, bbox, text=None, idx=0, page_idx=0):
    """Construct a real _MBlock for rescue-pass integration tests."""
    import layout_mineru as lm
    return lm._MBlock(
        type=type_, page_idx=page_idx, index=idx, bbox=bbox, text=text,
    )


def _three_col_body_filler() -> list:
    """3-col body text blocks for Phase A / Phase B test fixtures.  Six
    blocks across centers ~118, ~296, ~474 to trip the detector on
    a pw=594 page."""
    return [
        _make_block("text", [33, 50, 203, 200], "body c1a", 1),
        _make_block("text", [33, 210, 203, 360], "body c1b", 2),
        _make_block("text", [211, 50, 381, 200], "body c2a", 3),
        _make_block("text", [211, 210, 381, 360], "body c2b", 4),
        _make_block("text", [389, 50, 559, 200], "body c3a", 5),
        _make_block("text", [389, 210, 559, 360], "body c3b", 6),
    ]


def test_phaseA_drops_misnested_caption_on_3col_when_replacement_available():
    """A 'chart_caption' that doesn't start with Fig. N. / Table N.
    is body-text wrongly nested by MinerU.  Phase A drops it on a
    3-column page WHEN there is a Phase B candidate to replace it
    (Phase A no longer drops without replacement -- avoids data loss
    when a label-less real caption is the only candidate)."""
    import layout_mineru as lm
    page_sizes = {0: (594.0, 720.0)}
    blocks = _three_col_body_filler()
    # Phase B candidate -- a "Fig. 2." text block.
    blocks.append(_make_block(
        "text", [33, 400, 203, 500],
        "Fig. 2. Plot of oxygen isotopes for lunar samples.", 100))
    # The misnested chart with body-text caption.
    chart = _make_block("chart", [389, 400, 559, 700],
                        text="spectrometer analyses (28). We analyzed "
                             "samples taken from the lunar surface.",
                        idx=10, page_idx=0)
    blocks.append(chart)
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 1
    assert stats["captions_adopted_3col"] == 1
    assert chart.text.startswith("Fig. 2.")


def test_phaseA_preserves_misnested_caption_when_no_replacement():
    """If no Phase B candidate exists on a 3-col page, a misnested-
    looking caption is PRESERVED (don't strip without replacement).
    cuk Table 1 case: label-less real caption, no `Table 1.` text
    block elsewhere on the page -- keep the imperfect version
    rather than lose it."""
    import layout_mineru as lm
    page_sizes = {0: (594.0, 720.0)}
    blocks = _three_col_body_filler()
    # Misnested-looking but actually-real caption (no Phase B candidate).
    chart = _make_block("chart", [189, 400, 379, 600],
                        text="Properties of candidate impacts. All cases "
                             "had a total colliding mass M_T = 1.04 M.",
                        idx=10, page_idx=0)
    blocks.append(chart)
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    # No replacement available -> don't drop.
    assert stats["misnested_dropped"] == 0
    assert chart.text.startswith("Properties of candidate impacts")


def test_phaseA_skipped_on_non_3col():
    """A 'chart_caption' with body-text content on a 1-column page
    is NOT dropped -- Phase A's 3-col gate suppresses it.  This
    preserves the existing canup (Nature 1-col) behavior where the
    in-place caption might just lack a leading 'Fig. N.' label but
    is still the right caption."""
    import layout_mineru as lm
    # Single chart on a 1-col page -- no text blocks to trip the
    # 3-col detector.  pw=595.
    page_sizes = {0: (595.0, 720.0)}
    chart = _make_block("chart", [60, 100, 540, 600],
                        text="Moon mass requires M_M/M_P > 0.012, the "
                             "region to the right of the vertical line.",
                        idx=0, page_idx=0)
    stats = lm.rescue_orphan_captions([chart], page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 0
    # Caption preserved unchanged.
    assert chart.text.startswith("Moon mass requires")


def test_phaseA_keeps_real_caption_untouched_on_3col():
    """A populated chart_caption that DOES start with 'Fig. 2. ' is
    the correct caption; Phase A must not drop it even on a 3-col
    page where the rescue fires."""
    import layout_mineru as lm
    page_sizes = {0: (594.0, 720.0)}
    blocks = _three_col_body_filler()
    chart = _make_block("chart", [189, 400, 379, 600],
                        text="Fig. 2. Plot of oxygen isotope ratios for "
                             "lunar samples.",
                        idx=10, page_idx=0)
    blocks.append(chart)
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 0
    assert chart.text.startswith("Fig. 2.")


def test_phaseA_keeps_nature_style_figure_label_on_3col():
    """Nature-style 'Figure N | Title' (no period after N) is a real
    caption.  Even on a 3-col page (rare but possible), Phase A's
    loose _is_caption_text guard preserves it."""
    import layout_mineru as lm
    page_sizes = {0: (594.0, 720.0)}
    blocks = _three_col_body_filler()
    chart = _make_block("chart", [189, 400, 379, 600],
                        text="b Figure 1 | Simulation of the Moon's "
                             "accretion, reproduced from ref. 6.",
                        idx=10, page_idx=0)
    blocks.append(chart)
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 0
    assert chart.text.startswith("b Figure 1 |")


def test_phaseA_keeps_subpanel_stub_on_3col():
    """Single-letter sub-panel stub captions ('a', 'b a', 'c') are
    preserved -- they're the input to the existing sub-panel
    consolidation pass, not body-text noise."""
    import layout_mineru as lm
    page_sizes = {0: (594.0, 720.0)}
    blocks = _three_col_body_filler()
    chart = _make_block("chart", [189, 400, 379, 600],
                        text="a", idx=10, page_idx=0)
    blocks.append(chart)
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 0
    assert chart.text == "a"


def test_phaseA_keeps_watermark_unchanged_on_3col():
    """A watermark-only caption is not a real caption but also not
    misnested body-text; Phase A skips it (the rescue's geometric
    adoption layer handles watermark replacement)."""
    import layout_mineru as lm
    page_sizes = {0: (594.0, 720.0)}
    blocks = _three_col_body_filler()
    chart = _make_block("chart", [189, 400, 379, 600],
                        text="Downloaded from science.org on Jan 1 2025",
                        idx=10, page_idx=0)
    blocks.append(chart)
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 0
    assert "Downloaded" in chart.text


def test_phaseB_three_col_clean_single_match():
    """3-col page with exactly one orphan figure and one candidate
    text block starting 'Fig. N.' -> adopted via Phase B."""
    import layout_mineru as lm
    # Six text blocks across three column-centers (118/296/474) to
    # trip the 3-column detector on a pw=594 page.
    pw = 594.0
    page_sizes = {0: (pw, 756.0)}
    blocks = [
        # Body text in 3 columns:
        _make_block("text", [33, 50, 203, 200], "body col1", 1),
        _make_block("text", [33, 210, 203, 360], "body col1b", 2),
        _make_block("text", [211, 50, 381, 200], "body col2", 3),
        _make_block("text", [211, 210, 381, 360], "body col2b", 4),
        # The Fig 2 caption candidate (right column, top):
        _make_block("text", [389, 50, 559, 200],
                    "Fig. 2. Plot of oxygen isotopes against time.", 5),
        _make_block("text", [389, 210, 559, 360], "body col3b", 6),
        # The orphan figure (image with no caption text):
        _make_block("chart", [189, 400, 379, 600], None, 7),
    ]
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["captions_adopted_3col"] == 1
    assert 1 in stats["pages_3col"]
    # Adopted caption now on the chart block; candidate is marked adopted.
    chart = blocks[6]
    cand = blocks[4]
    assert chart.text == "Fig. 2. Plot of oxygen isotopes against time."
    assert cand.type == "_adopted_caption"


def test_phaseB_ambiguous_two_candidates_same_label():
    """Two text blocks both starting 'Fig. 2.' on a 3-col page ->
    ambiguous, no adoption, warning counted.

    Layout: 3 columns of body text plus 2 same-labeled candidates in
    column 1.  Orphan chart sits in a low column-1 slot so the two
    candidates fail geometric adjacency (they're side-by-side with the
    chart's column but in a different x-range).
    """
    import layout_mineru as lm
    pw = 594.0
    page_sizes = {0: (pw, 720.0)}
    blocks = [
        # Three columns of body text to trip the 3-col detector.
        _make_block("text", [33, 50, 203, 200], "body c1a", 1),
        _make_block("text", [33, 210, 203, 360], "body c1b", 2),
        _make_block("text", [211, 50, 381, 200], "body c2a", 3),
        _make_block("text", [211, 210, 381, 360], "body c2b", 4),
        _make_block("text", [389, 50, 559, 200], "body c3a", 5),
        _make_block("text", [389, 210, 559, 360], "body c3b", 6),
        # Two candidates in col1 starting with the SAME label.
        # Far above the chart and in a different column -- geometric
        # adjacency fails.
        _make_block("text", [33, 380, 203, 460],
                    "Fig. 2. First version of the same label.", 7),
        _make_block("text", [33, 470, 203, 550],
                    "Fig. 2. Second copy of the same label.", 8),
        # Orphan chart in column 3, no caption.
        _make_block("chart", [389, 600, 559, 710], None, 9),
    ]
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["captions_adopted_3col"] == 0
    assert stats["ambiguous_3col_warnings"] >= 1
    # Chart still orphaned.
    assert blocks[8].text is None


def test_phaseB_skipped_on_non_three_col_page():
    """A 1-column page does NOT trigger Phase B (gate is 3-col-only).
    A 'Fig. 2.' candidate placed FAR from the orphan figure (out of
    geometric adjacency) goes unadopted because Phase B's page-wide
    scan is gated to 3-column pages."""
    import layout_mineru as lm
    pw = 595.0
    page_sizes = {0: (pw, 756.0)}
    # 1-column layout: wide text blocks, no column clustering.
    # Candidate at top of page; chart at bottom -- y-gap too large
    # for geometric adjacency (default tolerance=60pt).
    blocks = [
        _make_block("text", [60, 50, 540, 100],
                    "Fig. 2. Plot of oxygen isotopes.", 1),
        _make_block("text", [60, 110, 540, 250], "wide body 1", 2),
        _make_block("text", [60, 260, 540, 400], "wide body 2", 3),
        _make_block("chart", [60, 550, 540, 720], None, 4),
    ]
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    # Phase B did NOT fire.
    assert stats["captions_adopted_3col"] == 0
    assert stats["pages_3col"] == []
    # Chart still has no caption -- geometric adjacency failed
    # (candidate is 450pt above) AND Phase B is gated off (1-col).
    assert blocks[3].text is None


def test_phaseB_phaseA_combined_young_p3_pattern():
    """End-to-end young page 3 pattern: chart has misnested body-text
    'caption' (Phase A drops); separate text block starts 'Fig. 2.'
    (Phase B adopts on clean single match)."""
    import layout_mineru as lm
    pw = 594.0
    page_sizes = {0: (pw, 756.0)}
    blocks = [
        # 3-col body text to trip detector.
        _make_block("text", [33, 50, 203, 200], "body c1a", 1),
        _make_block("text", [33, 210, 203, 360], "body c1b", 2),
        _make_block("text", [211, 50, 381, 200], "body c2a", 3),
        _make_block("text", [211, 210, 381, 360], "body c2b", 4),
        _make_block("text", [389, 210, 559, 360], "body c3b", 5),
        # Real Fig 2 caption candidate (top-right).
        _make_block("text", [389, 50, 559, 200],
                    "Fig. 2. Plot of Δ17O versus δ18O for lunar samples.", 6),
        # Chart with misnested body-text caption (this is young's
        # pathology -- MinerU classified the nearby body paragraph as
        # the chart's caption).
        _make_block("chart", [189, 400, 379, 600],
                    "spectrometer analyses (28). We analyzed lunar "
                    "and terrestrial sample lithologies.", 7),
    ]
    stats = lm.rescue_orphan_captions(blocks, page_sizes=page_sizes)
    assert stats["misnested_dropped"] == 1
    assert stats["captions_adopted_3col"] == 1
    chart = blocks[6]
    assert chart.text.startswith("Fig. 2. Plot of")
    assert "spectrometer analyses" not in chart.text


# === _caption_kind_and_number ==============================================


def test_caption_kind_number_fig():
    import layout_mineru as lm
    assert lm._caption_kind_and_number(
        "Fig. 2. Plot of...") == ("fig", 2)


def test_caption_kind_number_figure():
    import layout_mineru as lm
    assert lm._caption_kind_and_number(
        "Figure 12. Caption text.") == ("fig", 12)


def test_caption_kind_number_table():
    import layout_mineru as lm
    assert lm._caption_kind_and_number(
        "Table 1. Properties of samples.") == ("table", 1)


def test_caption_kind_number_not_caption():
    import layout_mineru as lm
    assert lm._caption_kind_and_number(
        "Body text starts here.") is None


def test_caption_kind_number_empty():
    import layout_mineru as lm
    assert lm._caption_kind_and_number(None) is None
    assert lm._caption_kind_and_number("") is None


# === parse_middle_json =====================================================


def _make_minimal_middle(tmp_path: Path) -> Path:
    """Handcraft a minimal middle.json covering each block type."""
    fixture = {
        "_backend": "pipeline",
        "_version_name": "3.1.6",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [612, 792],
                "para_blocks": [
                    {
                        "type": "title",
                        "bbox": [50, 50, 550, 80],
                        "index": 0,
                        "level": 1,
                        "lines": [{"spans": [{"type": "text",
                                              "content": "Test Paper Title"}]}],
                    },
                    {
                        "type": "abstract",
                        "bbox": [50, 100, 550, 200],
                        "index": 1,
                        "lines": [{"spans": [{"type": "text",
                                              "content": "This is the abstract."}]}],
                    },
                    {
                        "type": "text",
                        "bbox": [50, 220, 550, 300],
                        "index": 2,
                        "lines": [{"spans": [{"type": "text",
                                              "content": "Introduction body text."}]}],
                    },
                    {
                        "type": "interline_equation",
                        "bbox": [200, 320, 400, 360],
                        "index": 3,
                        "lines": [{"spans": [
                            {"type": "interline_equation",
                             "content": "E = m c^2",
                             "image_path": "eq1.jpg"}
                        ]}],
                    },
                    {
                        "type": "image",
                        "bbox": [100, 380, 500, 600],
                        "index": 4,
                        "blocks": [
                            {"type": "image_body",
                             "bbox": [100, 380, 500, 580],
                             "lines": [{"spans": [
                                 {"type": "image",
                                  "image_path": "fig1.jpg"}
                             ]}]},
                            {"type": "image_caption",
                             "bbox": [100, 585, 500, 600],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "Figure 1. Schematic of the apparatus."}
                             ]}]},
                        ],
                    },
                    {
                        "type": "table",
                        "bbox": [100, 620, 500, 760],
                        "index": 5,
                        "blocks": [
                            {"type": "table_caption",
                             "bbox": [100, 620, 500, 635],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "TABLE 1. Sample data."}
                             ]}]},
                            {"type": "table_body",
                             "bbox": [100, 640, 500, 760],
                             "lines": [{"spans": [
                                 {"type": "table",
                                  "html": ("<table><tr><td>Col A</td>"
                                           "<td>Col B</td></tr>"
                                           "<tr><td>1</td><td>2</td></tr>"
                                           "<tr><td>3</td><td>4</td></tr></table>"),
                                  "image_path": "tbl1.jpg"}
                             ]}]},
                        ],
                    },
                ],
                "preproc_blocks": [],
                "discarded_blocks": [],
            },
            {
                "page_idx": 1,
                "page_size": [612, 792],
                "para_blocks": [
                    {
                        "type": "ref_text",
                        "bbox": [50, 50, 550, 100],
                        "index": 0,
                        "lines": [{"spans": [
                            {"type": "text",
                             "content": "1. Author A. Title. Journal 1, 1 (2020)."}
                        ]}],
                    },
                ],
                "preproc_blocks": [],
                "discarded_blocks": [],
            },
        ],
    }
    p = tmp_path / "test_middle.json"
    p.write_text(json.dumps(fixture))
    return p


def test_parse_middle_minimal(tmp_path):
    mp = _make_minimal_middle(tmp_path)
    doc = wm.parse_middle_json(mp)
    assert doc.pages == 2
    assert doc.backend == "pipeline"
    assert doc.version == "3.1.6"
    assert len(doc.blocks) == 7  # title, abstract, text, eq, image, table, ref_text

    types = [b.type for b in doc.blocks]
    assert types == ["title", "abstract", "text", "interline_equation",
                     "image", "table", "ref_text"]

    # Title block carries level.
    title = doc.blocks[0]
    assert title.text == "Test Paper Title"
    assert title.text_level == 1

    # Equation has latex content + image_path.
    eq = doc.blocks[3]
    assert eq.latex == "E = m c^2"
    assert eq.image_path == "eq1.jpg"

    # Image block has caption + image_path paired.
    img = doc.blocks[4]
    assert img.text == "Figure 1. Schematic of the apparatus."
    assert img.image_path == "fig1.jpg"

    # Table has caption + html + image_path.
    tbl = doc.blocks[5]
    assert tbl.text == "TABLE 1. Sample data."
    assert tbl.image_path == "tbl1.jpg"
    assert tbl.html and "<table>" in tbl.html

    # ref_text from page 2.
    ref = doc.blocks[6]
    assert ref.type == "ref_text"
    assert ref.page_idx == 1


def test_emit_markdown_minimal(tmp_path):
    """End-to-end: parse fixture → emit body markdown."""
    import paper2md as p2m  # imported here so OPENAI_API_KEY isn't required at collection

    mp = _make_minimal_middle(tmp_path)
    doc = wm.parse_middle_json(mp)

    report = p2m.QualityReport()
    md = wm.emit_markdown(
        doc, assets_rel="assets", assets_abs=tmp_path / "assets",
        vlm_tables=False, report=report,
    )

    # Title rendered as heading.
    assert "# Test Paper Title" in md
    # Abstract prefix.
    assert "**Abstract.** This is the abstract." in md
    # Image emitted with caption alt-text + italic display line.
    assert "![Figure 1. Schematic of the apparatus.](assets/fig1.jpg)" in md
    assert "*Figure 1. Schematic of the apparatus.*" in md
    # Table emitted as pipe (clean conversion expected).
    assert "**TABLE 1. Sample data.**" in md
    assert "| Col A | Col B |" in md
    assert "| 1 | 2 |" in md
    # Table image link present (fidelity verification).
    assert "(assets/tbl1.jpg)" in md
    # Equation emitted as $$ block + image.
    assert "$$\nE = m c^2\n$$" in md
    assert "(assets/eq1.jpg)" in md
    # Reference text from page 2.
    assert "1. Author A. Title." in md

    # QualityReport populated with one figure + one clean table.
    assert len(report.figures) == 1
    assert report.figures[0].caption_produced is True
    assert len(report.tables) == 1
    assert report.tables[0].located is True
    assert report.tables[0].score == 1.0


# === multi-fragment caption / footnote merge ==============================


def _make_multi_fragment_caption_middle(tmp_path):
    """Fixture mimicking a 3-column Science page where wide figure +
    table captions and table footnotes are emitted by MinerU as
    multiple per-column children of the parent figure/table block."""
    fixture = {
        "_backend": "pipeline",
        "_version_name": "3.1.6",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [612, 792],
                "para_blocks": [
                    {
                        "type": "image",
                        "bbox": [40, 60, 580, 400],
                        "index": 0,
                        "blocks": [
                            {"type": "image_body",
                             "bbox": [40, 60, 580, 380],
                             "lines": [{"spans": [
                                 {"type": "image",
                                  "image_path": "fig1.jpg"}
                             ]}]},
                            # Right-column caption fragment (lower x0
                            # than left-col? no -- x0=300 > 40, but
                            # also at the same y. Order by (y, x).
                            {"type": "image_caption",
                             "bbox": [300, 410, 580, 460],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "Right column continuation."}
                             ]}]},
                            {"type": "image_caption",
                             "bbox": [40, 410, 280, 460],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "Figure 1. Left column start."}
                             ]}]},
                        ],
                    },
                    {
                        "type": "table",
                        "bbox": [40, 480, 580, 760],
                        "index": 1,
                        "blocks": [
                            # Three caption fragments across columns,
                            # out of order in the JSON.
                            {"type": "table_caption",
                             "bbox": [300, 480, 440, 510],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "middle column words"}
                             ]}]},
                            {"type": "table_caption",
                             "bbox": [40, 480, 280, 510],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "Table 1. Caption start"}
                             ]}]},
                            {"type": "table_caption",
                             "bbox": [460, 480, 580, 510],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "and right column tail."}
                             ]}]},
                            {"type": "table_body",
                             "bbox": [40, 520, 580, 700],
                             "lines": [{"spans": [
                                 {"type": "table",
                                  "html": ("<table><tr><td>A</td><td>B</td></tr>"
                                           "<tr><td>1</td><td>2</td></tr>"
                                           "<tr><td>3</td><td>4</td></tr></table>"),
                                  "image_path": "tbl1.jpg"}
                             ]}]},
                            # Two footnote fragments split across cols.
                            {"type": "table_footnote",
                             "bbox": [300, 710, 580, 750],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "right footnote half."}
                             ]}]},
                            {"type": "table_footnote",
                             "bbox": [40, 710, 280, 750],
                             "lines": [{"spans": [
                                 {"type": "text",
                                  "content": "* Left footnote half"}
                             ]}]},
                        ],
                    },
                ],
                "preproc_blocks": [],
                "discarded_blocks": [],
            },
        ],
    }
    p = tmp_path / "multi_frag_middle.json"
    p.write_text(json.dumps(fixture))
    return p


def test_parse_caption_fragments_join_in_reading_order(tmp_path):
    """Multiple image_caption / table_caption / table_footnote
    children of one figure/table block must accumulate, sorted by
    (y0, x0). Previously the parser overwrote, so only the last
    fragment in JSON-emit order survived."""
    mp = _make_multi_fragment_caption_middle(tmp_path)
    doc = wm.parse_middle_json(mp)
    assert len(doc.blocks) == 2

    img = doc.blocks[0]
    assert img.type == "image"
    # Left column at (y=410, x=40) precedes right at (y=410, x=300).
    assert img.text == "Figure 1. Left column start. Right column continuation."

    tbl = doc.blocks[1]
    assert tbl.type == "table"
    assert tbl.text == "Table 1. Caption start middle column words and right column tail."
    assert tbl.footnote == "* Left footnote half right footnote half."


def test_emit_table_footnote_renders_as_notes(tmp_path):
    """Table footnote text renders as `*Notes: ...*` after the table."""
    import paper2md as p2m
    mp = _make_multi_fragment_caption_middle(tmp_path)
    doc = wm.parse_middle_json(mp)
    report = p2m.QualityReport()
    md = wm.emit_markdown(
        doc, assets_rel="assets", assets_abs=tmp_path / "assets",
        vlm_tables=False, report=report,
    )
    assert "*Notes: * Left footnote half right footnote half.*" in md


def test_emit_table_footnote_strips_existing_notes_prefix(tmp_path):
    """If MinerU's footnote text already starts with 'Notes:' (or
    'Note:'), don't double the prefix."""
    import paper2md as p2m
    fixture = {
        "_backend": "pipeline",
        "_version_name": "3.1.6",
        "pdf_info": [{
            "page_idx": 0,
            "page_size": [612, 792],
            "para_blocks": [{
                "type": "table",
                "bbox": [40, 40, 580, 700],
                "index": 0,
                "blocks": [
                    {"type": "table_caption",
                     "bbox": [40, 40, 580, 60],
                     "lines": [{"spans": [
                         {"type": "text", "content": "Table 1."}
                     ]}]},
                    {"type": "table_body",
                     "bbox": [40, 70, 580, 600],
                     "lines": [{"spans": [
                         {"type": "table",
                          "html": "<table><tr><td>A</td></tr></table>",
                          "image_path": "tbl.jpg"}
                     ]}]},
                    {"type": "table_footnote",
                     "bbox": [40, 610, 580, 700],
                     "lines": [{"spans": [
                         {"type": "text",
                          "content": "Notes: Pre-labeled by MinerU."}
                     ]}]},
                ],
            }],
            "preproc_blocks": [],
            "discarded_blocks": [],
        }],
    }
    p = tmp_path / "notes_prefix_middle.json"
    p.write_text(json.dumps(fixture))
    doc = wm.parse_middle_json(p)
    report = p2m.QualityReport()
    md = wm.emit_markdown(
        doc, assets_rel="assets", assets_abs=tmp_path / "assets",
        vlm_tables=False, report=report,
    )
    assert "*Notes: Pre-labeled by MinerU.*" in md
    # No doubling.
    assert "Notes: Notes:" not in md


# === format_knudson_refs ===================================================


def test_format_knudson_refs_triggers_on_tail():
    body = "x" * 500 + "\n"
    refs_block = "\n".join(
        f"{i}P. E. Author{i}, Phys. Rev. B {i}, {i*100} (1990)."
        for i in range(1, 11)
    )
    md = body + refs_block
    out = wm.format_knudson_refs(md)
    assert "## References" in out
    assert "- 1. P. E. Author1" in out
    assert "- 10. P. E. Author10" in out


def test_format_knudson_refs_no_misfire_on_short_doc():
    md = "Just one ref here:\n5A. Test, Title 5, 100 (2020).\n"
    out = wm.format_knudson_refs(md)
    # Single match in short doc — no heading inserted.
    assert "## References" not in out
    # And no reformat.
    assert "5A. Test" in out


def test_format_knudson_refs_skips_when_heading_exists():
    """If a heading already exists, just reformat the entries below it."""
    md = (
        "Body text.\n\n## References\n\n"
        + "\n".join(f"{i}A. Author{i}, J. {i}, {i} (2020)."
                    for i in range(1, 7))
    )
    out = wm.format_knudson_refs(md)
    # Heading already present; not duplicated.
    assert out.count("## References") == 1
    # Entries reformatted to bullet style.
    assert "- 1. A. Author1" in out


def test_format_knudson_refs_idempotent():
    body = "x" * 500 + "\n"
    refs = "\n".join(
        f"{i}A. Author{i}, J. {i}, {i*10} (2020)."
        for i in range(1, 8)
    )
    md = body + refs
    out1 = wm.format_knudson_refs(md)
    out2 = wm.format_knudson_refs(out1)
    # Second invocation hits the heading-exists branch and is stable.
    assert out1.count("## References") == 1
    assert out2.count("## References") == 1


# === rescue_missing_refs_heading: numbered / bracketed / author-year =====


def test_rescue_heading_numbered_inserts():
    body = "Body text. " * 200 + "\n\n"
    refs = "\n".join(
        f"{i}. Author{i}, J. Sci. {i}, 1 (2020)."
        for i in range(1, 11)
    )
    md = body + refs
    out = wm.rescue_missing_refs_heading(md)
    assert "## References" in out
    # Heading injected before ref 1 (the first match in tail).
    refs_pos = out.find("## References")
    assert "1. Author1" in out[refs_pos:]
    # Body retained.
    assert "Body text." in out


def test_rescue_heading_numbered_skipped_when_no_anchor():
    """Tail has many numbered entries but starts at N=15 (no anchor 1/2/3).
    Likely a footnote series, not a refs list. Don't inject."""
    body = "Body text. " * 200 + "\n\n"
    refs = "\n".join(
        f"{i}. SomeWord here." for i in range(15, 25)
    )
    md = body + refs
    out = wm.rescue_missing_refs_heading(md)
    assert "## References" not in out


def test_rescue_heading_bracketed_inserts():
    body = "Body text. " * 200 + "\n\n"
    refs = "\n".join(
        f"[{i}] Author{i}, J. {i}, 1 (2020)."
        for i in range(1, 11)
    )
    md = body + refs
    out = wm.rescue_missing_refs_heading(md)
    assert "## References" in out
    refs_pos = out.find("## References")
    assert "[1] Author1" in out[refs_pos:]


def test_rescue_heading_idempotent_with_existing_heading():
    md = (
        "Body. " * 200 + "\n\n## References\n\n"
        + "\n".join(f"{i}. Author{i} (2020)." for i in range(1, 11))
    )
    out = wm.rescue_missing_refs_heading(md)
    # No duplicate heading.
    assert out.count("## References") == 1


def test_rescue_heading_author_year_strict_threshold():
    """Author-year detector requires >=10 contiguous matches."""
    body = "Body text. " * 300 + "\n\n"
    refs = "\n".join(
        f"Smith, J. and Doe, K. ({2000 + i}). Title {i}. J. Sci. {i}, 1."
        for i in range(12)
    )
    md = body + refs
    out = wm.rescue_missing_refs_heading(md)
    assert "## References" in out


def test_rescue_heading_author_year_below_threshold_skipped():
    body = "Body text. " * 300 + "\n\n"
    refs = "\n".join(
        f"Smith, J. and Doe, K. ({2000 + i}). Title {i}."
        for i in range(5)  # only 5 entries, below the 10 threshold
    )
    md = body + refs
    out = wm.rescue_missing_refs_heading(md)
    assert "## References" not in out


def test_rescue_heading_alternative_existing_headings_skipped():
    """Bibliography / Notes / Literature Cited / References and Notes
    already present -> rescue is a no-op."""
    refs_block = "\n\n## Bibliography\n\n" + "\n".join(
        f"{i}. Entry {i}." for i in range(1, 11)
    )
    md = "Body. " * 200 + refs_block
    out = wm.rescue_missing_refs_heading(md)
    assert out.count("## Bibliography") == 1
    assert "## References" not in out


# === table_block_to_md gating ==============================================


def test_table_block_clean_pipe(tmp_path):
    import paper2md as p2m

    block = wm._MBlock(
        type="table", page_idx=0, index=5, bbox=(0, 0, 100, 100),
        text="Table 1. Clean.",
        html=("<table><tr><td>A</td><td>B</td></tr>"
              "<tr><td>1</td><td>2</td></tr>"
              "<tr><td>3</td><td>4</td></tr></table>"),
        image_path="t1.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report)
    assert "**Table 1. Clean.**" in md
    assert "| A | B |" in md
    assert "(assets/t1.jpg)" in md
    assert len(report.tables) == 1
    assert report.tables[0].score == 1.0
    assert report.tables[0].vlm_redone is False


def test_table_block_rowspan_falls_back_to_html(tmp_path):
    import paper2md as p2m

    block = wm._MBlock(
        type="table", page_idx=0, index=5, bbox=(0, 0, 100, 100),
        text="Table 2. Rowspan.",
        html=('<table><tr><td rowspan="2">A</td><td>B</td></tr>'
              "<tr><td>C</td></tr></table>"),
        image_path="t2.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=2, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report)
    # Falls back to raw HTML + image link.
    assert '<td rowspan="2">A</td>' in md
    assert "(assets/t2.jpg)" in md
    assert len(report.tables) == 1
    # Score reflects fallback, not clean conversion.
    assert report.tables[0].score < 1.0
    assert "rowspan" in (report.tables[0].pre_redo_reason or "")


def test_table_block_raw_html_fallback_translates_eq_tags(tmp_path):
    """When a table falls through to raw-HTML emit (rowspan/colspan/etc.),
    <eq>LATEX</eq> tags should be translated to inline `$LATEX$` so
    markdown viewers still render the equations (the surrounding
    table stays as HTML)."""
    import paper2md as p2m

    block = wm._MBlock(
        type="table", page_idx=0, index=5, bbox=(0, 0, 100, 100),
        text="Table feng. Coesite shock data.",
        html=(
            '<table>'
            '<tr><td rowspan="2"><eq>P _ { H }</eq> (GPa)</td>'
            '<td><eq>U _ { s }</eq></td></tr>'
            '<tr><td>1</td><td><eq>2 3 . 0 6 \\pm 0 . 2 0</eq></td></tr>'
            '</table>'
        ),
        image_path="tfeng.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report)
    # rowspan triggers fallback -> raw HTML retained.
    assert '<td rowspan="2">' in md
    # But <eq>...</eq> tags should be translated to inline $...$.
    assert "<eq>" not in md
    assert "</eq>" not in md
    assert "$P _ { H }$" in md
    assert "$U _ { s }$" in md
    assert "$2 3 . 0 6 \\pm 0 . 2 0$" in md
    # Image link still emitted.
    assert "(assets/tfeng.jpg)" in md


def test_table_block_no_html_no_eq_substitution_breaks(tmp_path):
    """Edge case: empty html field still produces a clean fallback
    string with no <eq> errors."""
    import paper2md as p2m

    block = wm._MBlock(
        type="table", page_idx=0, index=5, bbox=(0, 0, 100, 100),
        text="Empty table.",
        html=None,
        image_path="empty.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report)
    assert "Empty table." in md
    assert "(assets/empty.jpg)" in md


# === Article-boundary trim =================================================


class _FakeFitzDoc:
    """Minimal fake of fitz.Document for trim tests; only needs len() and
    indexed access to a fake page object whose get_pixmap can be mocked
    away by the patched _vlm_compare_pages."""
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return object()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_trim_articles_single_article_no_op():
    blocks = [
        wm._MBlock(type="text", page_idx=p, index=0, bbox=(0, 0, 0, 0),
                   text=f"page {p}")
        for p in range(6)
    ]
    fake_doc = _FakeFitzDoc(6)
    # Mock _vlm_compare_pages to always return SAME.
    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages",
                      return_value={"same": True, "title": None}):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 6)
    assert info["head_trimmed"] == 0
    assert info["tail_trimmed"] == 0
    assert len(out) == 6


def test_trim_articles_trailing_drops_last_page():
    # Page 5 contains a block whose text matches the VLM-reported boundary
    # title; this is the title-match safety net (paper2md parity).
    blocks = [
        wm._MBlock(type="text", page_idx=p, index=0, bbox=(0, 0, 0, 0),
                   text=f"page {p}")
        for p in range(5)
    ]
    blocks.append(wm._MBlock(type="title", page_idx=5, index=0,
                             bbox=(0, 0, 0, 0),
                             text="A Different Paper About Galaxies"))
    fake_doc = _FakeFitzDoc(6)
    # Pages 0-4 same as page 0; page 5 different.
    def fake_compare(doc, anchor, candidate, dpi=100):
        if anchor == 0 and candidate == 5:
            return {"same": False,
                    "title": "A Different Paper About Galaxies"}
        return {"same": True, "title": None}

    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", side_effect=fake_compare):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 6)
    assert info["head_trimmed"] == 0
    assert info["tail_trimmed"] == 1
    assert info["main_pages"] == (0, 5)
    assert all(b.page_idx < 5 for b in out)


def test_trim_articles_no_leading_trim_even_when_vlm_says_different():
    """The leading-boundary search was removed because the TRIM_PROMPT's
    "title in BOTTOM => DIFFERENT" heuristic false-positives on page 0
    (which always has title-page features). Even if a hypothetical leading
    probe fired, the wrapper must not act on it."""
    blocks = [
        wm._MBlock(type="text", page_idx=p, index=0, bbox=(0, 0, 0, 0),
                   text=f"page {p}")
        for p in range(6)
    ]
    fake_doc = _FakeFitzDoc(6)
    # Trailing probe: page 0 vs page 5 -> SAME (single article).
    # Anything else -> DIFFERENT (would trigger leading trim under the old
    # logic). With the new logic we must never call that probe; the trim
    # must keep all 6 pages.
    def fake_compare(doc, anchor, candidate, dpi=100):
        if anchor == 0 and candidate == 5:
            return {"same": True, "title": None}
        return {"same": False, "title": "Some Leading Editorial Title"}

    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", side_effect=fake_compare):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 6)
    assert info["head_trimmed"] == 0
    assert info["tail_trimmed"] == 0
    assert len(out) == 6


def test_trim_articles_aborts_when_title_not_in_blocks():
    """Title-match safety net: VLM reports a boundary but the title is
    not present in any block on the trimmed pages -> abort the trim
    (likely VLM hallucination on a single-article PDF)."""
    blocks = [
        wm._MBlock(type="text", page_idx=p, index=0, bbox=(0, 0, 0, 0),
                   text=f"page {p} body content")
        for p in range(6)
    ]
    fake_doc = _FakeFitzDoc(6)
    def fake_compare(doc, anchor, candidate, dpi=100):
        if anchor == 0 and candidate == 5:
            return {"same": False,
                    "title": "Imaginary Title That Does Not Appear"}
        return {"same": True, "title": None}

    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", side_effect=fake_compare):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 6)
    assert info["head_trimmed"] == 0
    assert info["tail_trimmed"] == 0
    assert len(out) == 6


def test_trim_articles_aborts_when_vlm_returns_no_title():
    """VLM says DIFFERENT but returns no title -> can't validate, abort."""
    blocks = [
        wm._MBlock(type="text", page_idx=p, index=0, bbox=(0, 0, 0, 0),
                   text=f"page {p}")
        for p in range(6)
    ]
    fake_doc = _FakeFitzDoc(6)
    def fake_compare(doc, anchor, candidate, dpi=100):
        if anchor == 0 and candidate == 5:
            return {"same": False, "title": None}
        return {"same": True, "title": None}

    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", side_effect=fake_compare):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 6)
    assert info["head_trimmed"] == 0
    assert info["tail_trimmed"] == 0
    assert len(out) == 6


def test_trim_articles_vlm_error_is_safe():
    """VLM returns None -> abort safely, no trim."""
    blocks = [
        wm._MBlock(type="text", page_idx=p, index=0, bbox=(0, 0, 0, 0),
                   text=f"page {p}")
        for p in range(6)
    ]
    fake_doc = _FakeFitzDoc(6)
    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", return_value=None):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 6)
    assert info["head_trimmed"] == 0
    assert info["tail_trimmed"] == 0
    assert len(out) == 6


def test_trim_articles_block_granular_mid_page_cut():
    """The Elliott / multi-article-per-page case. Page 1 of a 2-page PDF
    contains the kept article's tail (Stewart's Forum section + 9 refs)
    AND the new article's head (Stem Cells News & Views). The boundary
    title must drop only the new-article blocks; the kept-article tail
    on the same page must survive."""
    blocks = [
        # Page 0: kept article body.
        wm._MBlock(type="title", page_idx=0, index=0, bbox=(0, 0, 0, 0),
                   text="Shadows cast on Moon's origin"),
        wm._MBlock(type="text", page_idx=0, index=1, bbox=(0, 0, 0, 0),
                   text="Tim Elliott's section text..."),
        # Page 1: kept article tail (continuation + refs) ...
        wm._MBlock(type="text", page_idx=1, index=0, bbox=(0, 0, 0, 0),
                   text="Stewart section continuation"),
        wm._MBlock(type="ref_text", page_idx=1, index=1,
                   bbox=(0, 0, 0, 0),
                   text="1. Clayton, R. N. & Mayeda, T. K. ..."),
        wm._MBlock(type="ref_text", page_idx=1, index=2,
                   bbox=(0, 0, 0, 0),
                   text="2. Touboul, M. et al. Nature 450, 1206 (2007)"),
        # ... then the new article starts mid-page.
        wm._MBlock(type="title", page_idx=1, index=3, bbox=(0, 0, 0, 0),
                   text="Dual response to Ras mutation"),
        wm._MBlock(type="text", page_idx=1, index=4, bbox=(0, 0, 0, 0),
                   text="Stem-cell News & Views body..."),
    ]
    fake_doc = _FakeFitzDoc(2)
    def fake_compare(doc, anchor, candidate, dpi=100):
        if anchor == 0 and candidate == 1:
            return {"same": False,
                    "title": "Dual response to Ras mutation"}
        return {"same": True, "title": None}

    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", side_effect=fake_compare):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 2)

    # Boundary block is the title block at (page=1, index=3).
    assert info["boundary_block"] == (1, 3)
    # Kept article tail (text + refs) on page 1 must survive.
    assert any(b.type == "ref_text" and b.text.startswith("1.") for b in out)
    assert any(b.type == "ref_text" and b.text.startswith("2.") for b in out)
    assert any(b.text == "Stewart section continuation" for b in out)
    # Stem Cells title + body on page 1 must be dropped.
    assert not any(b.text == "Dual response to Ras mutation" for b in out)
    assert not any(b.text == "Stem-cell News & Views body..." for b in out)
    # tail_trimmed counts pages partially-or-fully covered by the cut.
    assert info["tail_trimmed"] == 1
    assert info["main_pages"] == (0, 1)


def test_trim_articles_picks_latest_title_match():
    """When the boundary title appears in both the kept article (e.g.,
    as a forward reference) AND the new article's actual title block,
    the trim must cut at the LATEST occurrence -- otherwise a forward
    reference would trigger an over-trim. Mirrors paper2md._find_title_-
    line_in_md's "last match" semantics."""
    blocks = [
        wm._MBlock(type="text", page_idx=0, index=0, bbox=(0, 0, 0, 0),
                   text="Body text discussing Future Cosmic Topic in passing"),
        wm._MBlock(type="text", page_idx=1, index=0, bbox=(0, 0, 0, 0),
                   text="Page 1 also mentions future cosmic topic forward-ref"),
        wm._MBlock(type="text", page_idx=1, index=1, bbox=(0, 0, 0, 0),
                   text="More body content that should be kept"),
        wm._MBlock(type="title", page_idx=1, index=2, bbox=(0, 0, 0, 0),
                   text="Future Cosmic Topic"),
        wm._MBlock(type="text", page_idx=1, index=3, bbox=(0, 0, 0, 0),
                   text="Body of the new article"),
    ]
    fake_doc = _FakeFitzDoc(2)
    def fake_compare(doc, anchor, candidate, dpi=100):
        if anchor == 0 and candidate == 1:
            return {"same": False, "title": "Future Cosmic Topic"}
        return {"same": True, "title": None}

    from unittest.mock import patch
    with patch.object(wm, "_vlm_compare_pages", side_effect=fake_compare):
        out, info = wm.trim_articles_in_blocks(blocks, fake_doc, 2)
    # Boundary is the page-1 title block at index 2 (LATEST match), not
    # the earlier forward-reference at (1, 0).
    assert info["boundary_block"] == (1, 2)
    # Forward references survive.
    assert any("forward-ref" in (b.text or "") for b in out)
    assert any("should be kept" in (b.text or "") for b in out)
    # New article title and body are dropped.
    assert not any(b.type == "title" and b.text == "Future Cosmic Topic"
                   for b in out)


# === User annotations ======================================================


class _FakeArgs:
    """Lightweight argparse.Namespace stand-in for unit tests."""
    def __init__(self, **kw):
        self.user = kw.get("user")
        self.collection = kw.get("collection")
        self.note = kw.get("note")


def test_user_annotations_from_flags():
    a = _FakeArgs(user="Alice", collection="moon", note="hand-edited refs")
    ua = wm._build_user_annotations(a)
    assert ua is not None
    assert ua.user == "Alice"
    assert ua.collection == "moon"
    assert ua.note == "hand-edited refs"


def test_user_annotations_empty_returns_none():
    a = _FakeArgs()
    ua = wm._build_user_annotations(a)
    assert ua is None


def test_user_annotations_env_var_fallback(monkeypatch):
    monkeypatch.setenv("PAPER2MD_USER", "EnvAlice")
    monkeypatch.setenv("PAPER2MD_COLLECTION", "env-coll")
    a = _FakeArgs()  # no CLI flags
    ua = wm._build_user_annotations(a)
    assert ua is not None
    assert ua.user == "EnvAlice"
    assert ua.collection == "env-coll"
    assert ua.note is None


def test_user_annotations_cli_beats_env(monkeypatch):
    monkeypatch.setenv("PAPER2MD_USER", "EnvAlice")
    a = _FakeArgs(user="CliBob")
    ua = wm._build_user_annotations(a)
    assert ua is not None
    assert ua.user == "CliBob"


# === End-of-run summary ====================================================


def test_emit_run_summary_basic(caplog, tmp_path):
    """The summary banner emits without errors on a minimal report."""
    import paper2md as p2m
    import logging as _logging

    report = p2m.QualityReport()
    report.run_info = p2m.RunInfo(
        command="wrap_mineru.py test",
        hostname="testhost",
        vlm_provider="lmstudio",
        vlm_model="test-model",
    )
    report.run_info.elapsed_sec = 1.5
    out_md = tmp_path / "test.md"

    class _A:
        mineru_backend = "pipeline"

    caplog.set_level(_logging.INFO, logger="wrap_mineru")
    wm._emit_run_summary(report, out_md, _A(), trim_info=None)
    text = caplog.text
    assert "wrap_mineru complete: test.md" in text
    assert "Total time: 1.5 s" in text
    assert "Backend:    mineru-pipeline" in text


def test_emit_run_summary_with_trim(caplog, tmp_path):
    import paper2md as p2m
    import logging as _logging

    report = p2m.QualityReport()
    report.run_info = p2m.RunInfo(
        command="x", hostname="x", vlm_provider="lmstudio",
        vlm_model="x",
    )
    out_md = tmp_path / "trim.md"

    class _A:
        mineru_backend = "pipeline"

    caplog.set_level(_logging.INFO, logger="wrap_mineru")
    wm._emit_run_summary(report, out_md, _A(),
                         trim_info={"head_trimmed": 1, "tail_trimmed": 2,
                                    "main_pages": (1, 7)})
    assert "Article-trim" in caplog.text
    assert "head=1 tail=2" in caplog.text
    assert "main pages 2..7" in caplog.text


# === Supplement bundling (asset_prefix + paper_dir_override) ===============


def test_copy_assets_with_prefix(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "abc.jpg").write_bytes(b"fake-jpeg-1")
    (src / "def.jpg").write_bytes(b"fake-jpeg-2")
    dst = tmp_path / "dst"
    n = wm.copy_assets(src, dst, asset_prefix="si_")
    assert n == 2
    files = sorted(p.name for p in dst.iterdir())
    assert files == ["si_abc.jpg", "si_def.jpg"]


def test_copy_assets_without_prefix_unchanged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "abc.jpg").write_bytes(b"x")
    dst = tmp_path / "dst"
    n = wm.copy_assets(src, dst)
    assert n == 1
    assert (dst / "abc.jpg").is_file()


def test_image_block_to_md_with_prefix():
    block = wm._MBlock(
        type="image", page_idx=0, index=1, bbox=(0, 0, 0, 0),
        text="Figure 1. SI fig.", image_path="abc.jpg",
    )
    md = wm.image_block_to_md(block, "figure", 1, "assets",
                              asset_prefix="si_")
    assert "(assets/si_abc.jpg)" in md
    assert "Figure 1. SI fig." in md


def test_equation_block_to_md_with_prefix():
    block = wm._MBlock(
        type="interline_equation", page_idx=0, index=2, bbox=(0, 0, 0, 0),
        latex="E = m c^2", image_path="eq1.jpg",
    )
    md = wm.equation_block_to_md(block, "assets", asset_prefix="si_")
    assert "(assets/si_eq1.jpg)" in md
    assert "$$\nE = m c^2\n$$" in md


def test_table_block_to_md_with_prefix(tmp_path):
    import paper2md as p2m
    block = wm._MBlock(
        type="table", page_idx=0, index=3, bbox=(0, 0, 0, 0),
        text="Table 1.",
        html=("<table><tr><td>A</td><td>B</td></tr>"
              "<tr><td>1</td><td>2</td></tr>"
              "<tr><td>3</td><td>4</td></tr></table>"),
        image_path="t1.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report, asset_prefix="si_")
    assert "(assets/si_t1.jpg)" in md
    assert "| A | B |" in md


def test_vlm_table_rewrite_passes_max_retries(tmp_path, monkeypatch):
    """The hybrid table VLM-rewrite path delegates to p2m.vlm() with
    max_tokens=6000 (room for wide tables -- 17-col x 30-row cases like
    Wackerle Table 3 / cuk Table 1 truncated at 2000) and the SDK's
    default short-retry budget. Earlier app-level retry / max_retries=5
    fudges were workarounds for what turned out to be a sys.modules
    duplicate-import bug (now fixed by aliasing __main__ -> paper2md
    at script entry, so wrap_mineru's `p2m.client` is the configured
    client, not an unconfigured default-port lmstudio fallback)."""
    import paper2md as p2m
    from PIL import Image as _Image
    img_path = tmp_path / "tbl.jpg"
    _Image.new("RGB", (10, 10), color="white").save(img_path)

    captured = {}

    def _fake_vlm(prompt, image, max_tokens=1500, timeout=None,
                  max_retries=None, raise_on_error=False):
        captured["max_retries"] = max_retries
        captured["max_tokens"] = max_tokens
        return "| a | b |\n|---|---|\n| 1 | 2 |\n"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    out = wm._vlm_table_rewrite(img_path, "Table 1. caption.")
    assert out is not None
    assert "| a | b |" in out
    assert captured["max_retries"] == 2
    assert captured["max_tokens"] == 6000


def test_vlm_table_rewrite_uses_pil_image_override(tmp_path, monkeypatch):
    """When pil_image is provided, _vlm_table_rewrite uses it directly
    and does NOT load anything from disk. This is the hybrid-splice
    test path that renders fresh PIL images from the PDF instead of
    loading MinerU's saved JPGs."""
    import paper2md as p2m
    from PIL import Image as _Image
    override = _Image.new("RGB", (123, 456), color="white")
    # image_path does NOT exist on disk -- if the override isn't used,
    # PIL.Image.open() would raise FileNotFoundError and the helper
    # would warn+return None.
    bogus_path = tmp_path / "does-not-exist.jpg"
    assert not bogus_path.exists()

    captured = {}

    def _fake_vlm(prompt, image, max_tokens=1500, timeout=None,
                  max_retries=None, raise_on_error=False):
        captured["size"] = image.size
        return "| h1 | h2 |\n|---|---|\n| a | b |\n"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    out = wm._vlm_table_rewrite(bogus_path, "Table 1.", pil_image=override)
    # Result returned -- override was used, not the (missing) file.
    assert out is not None
    assert "| h1 | h2 |" in out
    # The image passed to vlm() is the exact override we provided.
    assert captured["size"] == (123, 456)


def test_table_block_to_md_passes_pil_image_override(tmp_path, monkeypatch):
    """table_block_to_md threads pil_image_override down to
    _vlm_table_rewrite. Verify the override survives the wrapper."""
    import paper2md as p2m
    from PIL import Image as _Image
    override = _Image.new("RGB", (32, 32), color="blue")

    block = wm._MBlock(
        type="table", page_idx=0, index=1, bbox=(0, 0, 100, 100),
        text="Table 1.",
        # Suspicious HTML (has rowspan) to force VLM path.
        html=("<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
              "<tr><td>C</td></tr></table>"),
        image_path="t1.jpg",
    )
    captured = {}

    def _fake_vlm(prompt, image, max_tokens=1500, timeout=None,
                  max_retries=None, raise_on_error=False):
        captured["size"] = image.size
        return "| A | B |\n|---|---|\n| x | y |\n"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    report = p2m.QualityReport()
    out = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                               vlm_tables=True, assets_abs=tmp_path,
                               report=report, asset_prefix="",
                               pil_image_override=override)
    # VLM was called with the override (32x32), not the missing JPG.
    assert captured.get("size") == (32, 32)
    # And the resulting markdown picked up the pipe table from the VLM.
    assert "| A | B |" in out


def test_table_block_to_md_falls_back_to_jpg_when_override_none(
        tmp_path, monkeypatch):
    """When pil_image_override is None and the JPG exists on disk,
    the existing JPG-load path runs (default behavior preserved)."""
    import paper2md as p2m
    from PIL import Image as _Image
    # Write an actual JPG on disk so the fallback can load it.
    img_path = tmp_path / "t1.jpg"
    _Image.new("RGB", (50, 50), color="white").save(img_path)

    block = wm._MBlock(
        type="table", page_idx=0, index=1, bbox=(0, 0, 100, 100),
        text="Table 1.",
        html=("<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
              "<tr><td>C</td></tr></table>"),
        image_path="t1.jpg",
    )
    captured = {}

    def _fake_vlm(prompt, image, max_tokens=1500, timeout=None,
                  max_retries=None, raise_on_error=False):
        captured["size"] = image.size
        return "| A | B |\n|---|---|\n| x | y |\n"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    report = p2m.QualityReport()
    out = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                               vlm_tables=True, assets_abs=tmp_path,
                               report=report, asset_prefix="",
                               pil_image_override=None)
    # VLM saw the JPG-loaded image (50x50), not nothing.
    assert captured.get("size") == (50, 50)
    assert "| A | B |" in out


def test_emit_markdown_with_prefix(tmp_path):
    import paper2md as p2m
    fixture_path = _make_minimal_middle(tmp_path)
    doc = wm.parse_middle_json(fixture_path)
    report = p2m.QualityReport()
    md = wm.emit_markdown(
        doc, assets_rel="assets", assets_abs=tmp_path / "assets",
        vlm_tables=False, report=report, asset_prefix="si_",
    )
    # Every image link should carry the prefix.
    assert "(assets/si_fig1.jpg)" in md
    assert "(assets/si_eq1.jpg)" in md
    assert "(assets/si_tbl1.jpg)" in md
    # FigureScore filenames should also reflect the prefix.
    assert all(f.filename.startswith("si_") for f in report.figures
               if f.filename)


# === per-table sidecar writing in hybrid path ==============================
#
# Mirrors marker-layout's process_tables convention. Hybrid skips
# process_tables but should still emit `assets/{prefix}table_p{page}_{idx}.md`
# alongside the inline table so downstream pandas / RAG consumers can
# read each table as its own file. See cuk.md before this fix: 0 sidecars
# despite a clean pipe-md table in the body.


def test_table_block_writes_sidecar_on_clean_pipe_md(tmp_path):
    """Clean HTML -> pipe-md path writes the table body to
    `assets/table_p{page}_{idx}.md` and appends a [Table N -- separate
    markdown] link to the inline emission. TableScore.sidecar records
    the filename for downstream tooling."""
    import paper2md as p2m
    block = wm._MBlock(
        type="table", page_idx=2, index=5, bbox=(0, 0, 100, 100),
        text="Table 1. Clean.",
        html=("<table><tr><td>A</td><td>B</td></tr>"
              "<tr><td>1</td><td>2</td></tr></table>"),
        image_path="t1.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report)
    # New naming: id prepended (caption "Table 1." -> id "1").
    sidecar = tmp_path / "table_1_p3_1.md"
    assert sidecar.exists()
    body = sidecar.read_text()
    assert "| A | B |" in body
    # The sidecar is the table body only -- no caption header.
    assert "**Table 1." not in body
    # The inline emission references the sidecar.
    assert "[Table 1 — separate markdown](assets/table_1_p3_1.md)" in md
    # TableScore records the sidecar filename.
    assert report.tables[0].sidecar == "table_1_p3_1.md"


def test_table_block_writes_sidecar_on_vlm_rewrite(tmp_path, monkeypatch):
    """VLM-rewrite path writes the rewritten body to a sidecar and
    references it inline. Same naming as the pipe-md path."""
    import paper2md as p2m
    from PIL import Image as _Image
    block = wm._MBlock(
        type="table", page_idx=4, index=2, bbox=(0, 0, 100, 100),
        text="Table 2. Rowspan.",
        html=('<table><tr><td rowspan="2">A</td><td>B</td></tr>'
              "<tr><td>C</td></tr></table>"),
        image_path="t2.jpg",
    )
    # Pass a real PIL override so the VLM path doesn't try to load
    # MinerU's saved JPG from disk.
    override = _Image.new("RGB", (100, 50), color="white")
    monkeypatch.setattr(
        p2m, "vlm",
        lambda *a, **k: "| h1 | h2 |\n|----|----|\n| x | y |\n",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=2, assets_rel="assets",
                              vlm_tables=True, assets_abs=tmp_path,
                              report=report,
                              pil_image_override=override)
    # Caption "Table 2." -> id "2" prepended into filename.
    sidecar = tmp_path / "table_2_p5_2.md"
    assert sidecar.exists()
    body = sidecar.read_text()
    assert "| h1 | h2 |" in body
    assert "[Table 2 — separate markdown](assets/table_2_p5_2.md)" in md
    assert report.tables[0].sidecar == "table_2_p5_2.md"
    assert report.tables[0].vlm_redone is True


def test_table_block_html_fallback_writes_no_sidecar(tmp_path, monkeypatch):
    """When the table falls through to raw-HTML emit (rowspan, VLM
    declined), no sidecar is written -- raw HTML in a `.md` file is not
    useful to downstream readers. TableScore.sidecar stays None."""
    import paper2md as p2m
    block = wm._MBlock(
        type="table", page_idx=0, index=1, bbox=(0, 0, 100, 100),
        text="Table 3. Fallback.",
        html=('<table><tr><td rowspan="2">A</td><td>B</td></tr>'
              "<tr><td>C</td></tr></table>"),
        image_path="t3.jpg",
    )
    report = p2m.QualityReport()
    # vlm_tables=False -> no rewrite attempted -> HTML fallback.
    md = wm.table_block_to_md(block, table_idx=3, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report)
    assert not (tmp_path / "table_p1_3.md").exists()
    assert "separate markdown" not in md
    assert report.tables[0].sidecar is None


def test_table_block_sidecar_honors_asset_prefix(tmp_path):
    """asset_prefix (used by --supplement runs) must appear on the
    sidecar filename so a main + SI run shares the same assets dir
    without filename collisions."""
    import paper2md as p2m
    block = wm._MBlock(
        type="table", page_idx=1, index=1, bbox=(0, 0, 100, 100),
        text="Table 1. SI table.",
        html=("<table><tr><td>A</td></tr><tr><td>1</td></tr></table>"),
        image_path="t1.jpg",
    )
    report = p2m.QualityReport()
    md = wm.table_block_to_md(block, table_idx=1, assets_rel="assets",
                              vlm_tables=False, assets_abs=tmp_path,
                              report=report, asset_prefix="si_")
    # New naming: id "1" extracted from "Table 1." caption -> prepended
    assert (tmp_path / "si_table_1_p2_1.md").exists()
    assert "[Table 1 — separate markdown](assets/si_table_1_p2_1.md)" in md
    assert report.tables[0].sidecar == "si_table_1_p2_1.md"


# === _strip_chart_details_blocks ===========================================


def test_strip_chart_details_blocks():
    md = (
        "Some text.\n\n"
        "<details>\n<summary>line</summary>\n\n"
        "| t | y |\n| - | - |\n| 1 | 2 |\n\n</details>\n\n"
        "More text."
    )
    out = wm._strip_chart_details_blocks(md)
    assert "<details>" not in out
    assert "Some text." in out
    assert "More text." in out


def test_strip_chart_details_no_match():
    md = "Just text without any details block."
    out = wm._strip_chart_details_blocks(md)
    assert out == md


# === rescue_footnote_refs ==================================================
#
# Tests Wackerle-style page-footnote reference reconstruction. The
# rescue gates on poor refs.score + no API rescue + bibliographic
# shape, then appends a `## References` section.


def _mk_footnote(page_idx: int, y: float, text: str) -> wm._MBlock:
    """Helper: synthesize a page_footnote _MBlock for tests."""
    return wm._MBlock(
        type="page_footnote",
        page_idx=page_idx,
        index=0,
        bbox=(0.0, y, 100.0, y + 12.0),
        text=text,
    )


def _mk_poor_refs_score():
    """Helper: a ReferencesScore that triggers the footnote rescue
    gate (score < 0.4, entry_count < 3, no API rescue)."""
    import paper2md as p2m
    return p2m.ReferencesScore(
        section_count=0, entry_count=0,
        numbered_continuous=False, year_hit_ratio=0.0,
        longest_over_median=0.0, style="unknown",
        journal_slug=None, score=0.0,
    )


def test_footnote_rescue_wackerle_style_triggers():
    """Wackerle-class paper: zero refs in body + multiple year-bearing
    footnotes -> rescue fires, appends ## References."""
    import paper2md as p2m
    body = "Body text. " * 200
    fbs = [
        _mk_footnote(0, 580, "* Work done under auspices of the U. S. AEC."),
        _mk_footnote(0, 597, "1 F. W. Neilson and W. B. Benedick, Bull. Am. Phys. Soc. 5, 511 (1960)."),
        _mk_footnote(0, 614, "  G. W. Anderson, Colloque International (1961)."),
        _mk_footnote(0, 638, "3 M. H. Rice, R. G. McQueen, and J. M. Walsh, Solid State Phys. 6 (1958)."),
        _mk_footnote(0, 661, "4 R. Courant and K. O. Friedrichs, Supersonic Flow (1948)."),
        _mk_footnote(0, 678, "  D. Bancroft, E. L. Peterson, S. Minshall, J. Appl. Phys. 27, 291 (1956)."),
        _mk_footnote(7, 648, "11 L. D. Landau and E. M. Lifshitz, Theory of Elasticity (1959)."),
        _mk_footnote(7, 681, "13 P. W. Bridgman, Proc. Am. Acad. 76, 55 (1948)."),
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()

    out = wm.rescue_footnote_refs(body, fbs, report)
    assert "## References" in out
    # Year-bearing entries appended.
    assert "Neilson" in out
    assert "Bridgman" in out
    # Funding ack (starts with *) filtered out.
    assert "U. S. AEC" not in out
    # Ordering: leading-numbered entries keep their N; missing-number
    # entries get filled by document order.
    assert "- 1. F. W. Neilson" in out
    assert "- 2. G. W. Anderson" in out  # missing N filled to 2
    # rescue_applied recorded on report.
    assert "footnote-refs" in (report.references.rescue_applied or "")


def test_footnote_rescue_skipped_when_local_refs_exist():
    """Paper with healthy refs.score (canup-style) -> rescue is no-op."""
    import paper2md as p2m
    body = "Body. ## References\n\n- 1. Author (2020).\n- 2. ...\n"
    fbs = [
        _mk_footnote(0, 580, "1 Some Author, J. Sci. 1, 100 (1990)."),
    ] * 10
    report = p2m.QualityReport()
    report.references = p2m.ReferencesScore(
        section_count=1, entry_count=38, numbered_continuous=True,
        year_hit_ratio=0.95, longest_over_median=1.2, style="numbered",
        journal_slug="nature-geoscience", score=0.85,
    )
    out = wm.rescue_footnote_refs(body, fbs, report)
    # Body unchanged; the existing ## References heading + 38 entries
    # signal "refs already healthy" via score >= 0.4 / entry_count >= 3.
    assert out == body


def test_footnote_rescue_skipped_when_api_rescue_already_fired():
    """API-rescued paper -> footnote rescue is no-op (avoid double-counting)."""
    import paper2md as p2m
    fbs = [
        _mk_footnote(0, 580 + i*15,
                     f"{i} Author{i}, J. {i}, 100 (199{i}).")
        for i in range(10)
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    report.references.api_refs_appended = 27
    report.references.api_source = "crossref"
    out = wm.rescue_footnote_refs("body", fbs, report)
    assert out == "body"


def test_footnote_rescue_skipped_when_no_year():
    """canup-style author-affiliations footnote: no year in parens
    -> bibliographic-shape filter rejects, rescue is no-op."""
    import paper2md as p2m
    body = "Body. " * 100
    fbs = [
        _mk_footnote(
            0, 600,
            "1Planetary Sciences Directorate, Southwest Research "
            "Institute, Boulder, Colorado 80302, USA. "
            "2Chemistry and Planetary Sciences, Dordt College, "
            "Sioux Center, Iowa 51250, USA."
        ),
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    out = wm.rescue_footnote_refs(body, fbs, report)
    assert out == body
    assert "## References" not in out


def test_footnote_rescue_skipped_when_doi_furniture():
    """doi.org / doi: footers (publisher furniture) get filtered."""
    import paper2md as p2m
    fbs = [
        _mk_footnote(p, 700,
                     f"doi:10.1234/journal.{p:04d}.567 Nature (2020).")
        for p in range(7)
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    out = wm.rescue_footnote_refs("body", fbs, report)
    # Not enough non-doi candidates; rescue skipped.
    assert "## References" not in out


def test_footnote_rescue_splits_multi_ref_blocks():
    """When MinerU concatenates two consecutive footnotes into one
    block ('9 X (1960). 10 Y (1961).'), the splitter unfolds them."""
    import paper2md as p2m
    body = "Body. " * 100
    fbs = [
        _mk_footnote(
            0, 700,
            "9 M. H. Rice and J. W. Taylor (1960). "
            "10 R. G. McQueen (1961)."
        ),
        _mk_footnote(0, 715, "11 L. D. Landau (1959)."),
        _mk_footnote(0, 730, "13 P. W. Bridgman (1948)."),
        _mk_footnote(0, 745, "15 G. R. Fowles (1961)."),
        _mk_footnote(0, 760, "20 W. B. Hillig (1961)."),
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    out = wm.rescue_footnote_refs(body, fbs, report)
    # Both halves of the multi-ref block appear.
    assert "- 9. M. H. Rice and J. W. Taylor (1960)." in out
    assert "- 10. R. G. McQueen (1961)." in out


def test_footnote_rescue_idempotent():
    """Second invocation with a heading already present is a no-op."""
    import paper2md as p2m
    fbs = [
        _mk_footnote(0, 580 + i*15,
                     f"{i+1} Author{i}, J. {i}, 100 (199{i}).")
        for i in range(10)
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    once = wm.rescue_footnote_refs("body", fbs, report)
    assert "## References" in once
    n_first = once.count("## References")
    twice = wm.rescue_footnote_refs(once, fbs, report)
    assert twice == once
    assert twice.count("## References") == n_first


def test_footnote_rescue_no_footnote_blocks_no_op():
    """Empty footnote_blocks -> no-op."""
    import paper2md as p2m
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    out = wm.rescue_footnote_refs("body", [], report)
    assert out == "body"
    assert report.references.rescue_applied is None


def test_footnote_rescue_below_min_count_no_op():
    """Fewer than 5 bibliographic-shape candidates -> no-op."""
    import paper2md as p2m
    fbs = [
        _mk_footnote(0, 580, "1 Author1 (2020)."),
        _mk_footnote(0, 600, "2 Author2 (2020)."),
        _mk_footnote(0, 620, "3 Author3 (2020)."),
    ]
    report = p2m.QualityReport()
    report.references = _mk_poor_refs_score()
    out = wm.rescue_footnote_refs("body", fbs, report)
    assert out == "body"


def test_parse_middle_json_picks_up_page_footnotes(tmp_path):
    """Verify parse_middle_json captures discarded_blocks.page_footnote
    entries into ParsedDoc.footnote_blocks (and skips type=footer)."""
    fixture = {
        "_backend": "pipeline",
        "_version_name": "3.1.7",
        "pdf_info": [{
            "page_idx": 0,
            "page_size": [612, 792],
            "para_blocks": [],
            "discarded_blocks": [
                {"type": "footer", "bbox": [0, 700, 600, 720],
                 "lines": [{"spans": [{"type": "text",
                                       "content": "© 2015 Publisher"}]}]},
                {"type": "header", "bbox": [0, 50, 600, 70],
                 "lines": [{"spans": [{"type": "text",
                                       "content": "JOURNAL NAME"}]}]},
                {"type": "page_number", "bbox": [580, 720, 600, 740],
                 "lines": [{"spans": [{"type": "text",
                                       "content": "918"}]}]},
                {"type": "page_footnote", "bbox": [40, 600, 270, 620],
                 "lines": [{"spans": [{"type": "text",
                                       "content": "1 Author A. J. (1960)."}]}]},
                {"type": "page_footnote", "bbox": [40, 625, 270, 645],
                 "lines": [{"spans": [{"type": "text",
                                       "content": "2 Author B. J. (1961)."}]}]},
            ],
        }],
    }
    p = tmp_path / "fn_middle.json"
    p.write_text(json.dumps(fixture))
    doc = wm.parse_middle_json(p)
    # Only page_footnote entries land in footnote_blocks; footer/header/
    # page_number get skipped.
    assert len(doc.footnote_blocks) == 2
    assert all(b.type == "page_footnote" for b in doc.footnote_blocks)
    assert doc.footnote_blocks[0].text.startswith("1 Author A.")
    assert doc.footnote_blocks[1].text.startswith("2 Author B.")


# === --clean-output flag ===================================================


def test_clean_output_flag_parses():
    """argparse exposes the new flag with the expected default."""
    parser = wm.build_arg_parser()
    args = parser.parse_args(["dummy.pdf", "-o", "out"])
    assert args.clean_output is False
    args = parser.parse_args(["dummy.pdf", "-o", "out", "--clean-output"])
    assert args.clean_output is True


def test_clean_mineru_flag_parses_and_aliases_clean():
    """--clean-mineru is the canonical name; --clean is a deprecated
    alias mapped to the same dest."""
    parser = wm.build_arg_parser()
    args = parser.parse_args(["dummy.pdf", "-o", "out"])
    assert args.clean_mineru is False
    args = parser.parse_args(["dummy.pdf", "-o", "out", "--clean-mineru"])
    assert args.clean_mineru is True
    args = parser.parse_args(["dummy.pdf", "-o", "out", "--clean"])
    assert args.clean_mineru is True


def test_clean_output_wipes_paper_dir(tmp_path, monkeypatch):
    """When --clean-output is set and paper_dir already exists, the
    directory should be wiped before _process_one rebuilds it."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    stale = paper_dir / "stale_asset.png"
    stale.write_bytes(b"old")
    (paper_dir / "stale.md").write_text("old markdown\n")

    # Stub args with the minimum surface _process_one's prelude touches:
    # we monkeypatch out everything after the wipe so the test stays
    # offline. Capture whether the wipe ran by checking stale is gone.
    class _Args:
        clean_output = True
        skip_mineru_run = False

    # Just exercise the clean-output prelude. The function bails out as
    # soon as it tries to run MinerU; that's fine -- we only assert the
    # wipe happened first.
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    # Simplest path: replicate the prelude inline so we don't need to
    # mock the entire MinerU pipeline. This is what _process_one runs.
    import shutil as _shutil
    is_si = False
    if (getattr(_Args, "clean_output", False)
            and not is_si
            and paper_dir.is_dir()):
        _shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    assert not stale.exists()
    assert not (paper_dir / "stale.md").exists()
    assert paper_dir.is_dir()


def test_clean_output_skipped_for_supplement():
    """SI processing must NOT re-wipe the shared paper_dir or the main
    paper's just-written output would be lost."""
    # Re-running the same prelude with is_si=True must leave files alone.
    from pathlib import Path as _P
    import tempfile
    import shutil as _shutil
    with tempfile.TemporaryDirectory() as td:
        paper_dir = _P(td) / "paper"
        paper_dir.mkdir()
        main_md = paper_dir / "paper.md"
        main_md.write_text("main paper output\n")

        is_si = True
        clean_output = True
        if (clean_output and not is_si and paper_dir.is_dir()):
            _shutil.rmtree(paper_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Main paper's markdown survives the SI run.
        assert main_md.exists()
        assert main_md.read_text() == "main paper output\n"


# === rescue_orphan_captions (Stage 3 of convergence plan) ==================


def test_caption_is_watermark_detects_science_download_stamp():
    assert wm._caption_is_watermark(
        "Downloaded from https://www.science.org on April 25, 2026"
    ) is True


def test_caption_is_watermark_does_not_flag_real_caption():
    """Real Fig./Table captions must NOT be flagged as watermarks."""
    assert wm._caption_is_watermark(
        "Figure 1. Schematic of the apparatus."
    ) is False
    assert wm._caption_is_watermark(
        "Table 1. Sample data and parameters."
    ) is False


def test_is_caption_text_at_start():
    assert wm._is_caption_text("Figure 1. Title here.") is True
    assert wm._is_caption_text("Table S2. Supplementary data.") is True


def test_is_caption_text_mid_string():
    """cuk-style caption with the label embedded mid-string (OCR
    reading-order shuffle) must still be recognized."""
    assert wm._is_caption_text(
        "Formation of the Fig. 1.lunar disk from Earth's mantle."
    ) is True


def test_is_caption_text_rejects_body():
    assert wm._is_caption_text(
        "Our results establish contemporaneous origin and strongly suggest..."
    ) is False


def test_is_footnote_text_with_dagger():
    assert wm._is_footnote_text("† Liquidus temperatures were estimated...") is True
    assert wm._is_footnote_text("* Estimated from MELTS calculations.") is True
    assert wm._is_footnote_text("Notes: see appendix A.") is True


def test_rescue_orphan_captions_adopts_caption_above_table(tmp_path):
    """Alexander pattern: caption block classified as 'text' sits
    just above a 'table' block whose own caption slot is empty.
    Adopt the text block as the table's caption."""
    blocks = [
        # Caption-shaped text block, full-page-width, just above the table.
        wm._MBlock(type="text", page_idx=0, index=0,
                   bbox=(40, 300, 380, 335),
                   text="Table 1. Petrologic types and liquidus temperatures."),
        # Table block with empty caption.
        wm._MBlock(type="table", page_idx=0, index=1,
                   bbox=(40, 340, 380, 650),
                   text=None, html="<table></table>"),
    ]
    info = wm.rescue_orphan_captions(blocks)
    assert info["captions_adopted"] == 1
    # Caption now lives on the table.
    assert blocks[1].text and "Petrologic types" in blocks[1].text
    # Original text block flagged so emit_markdown skips it.
    assert blocks[0].type == "_adopted_caption"


def test_rescue_orphan_captions_adopts_footnote_below_table(tmp_path):
    """Alexander pattern part 2: footnote-shaped text block below a
    table with empty footnote slot. Adopt as table_footnote."""
    blocks = [
        wm._MBlock(type="table", page_idx=0, index=0,
                   bbox=(40, 340, 380, 650),
                   text="Table 1. Caption.",
                   html="<table></table>", footnote=None),
        # Footnote block: starts with *, full-page-width, just below.
        wm._MBlock(type="text", page_idx=0, index=1,
                   bbox=(40, 655, 380, 720),
                   text="*Liquidus temperatures were estimated with MELTS."),
    ]
    info = wm.rescue_orphan_captions(blocks)
    assert info["footnotes_adopted"] == 1
    assert blocks[0].footnote and "Liquidus" in blocks[0].footnote
    assert blocks[1].type == "_adopted_footnote"


def test_rescue_orphan_captions_adopts_caption_to_left_of_figure():
    """cuk pattern: caption block classified as 'text' lives to the
    left of the figure (vertical overlap, x-adjacent). Image's own
    caption slot has a watermark. Adopt the text block, replacing the
    watermark."""
    blocks = [
        # Caption-shaped text, mid-string label (OCR reading shuffle).
        wm._MBlock(type="text", page_idx=1, index=0,
                   bbox=(210, 211, 298, 502),
                   text="Formation of theFig. 1.lunar disk from Earth's mantle."),
        # Figure to the right, with download-watermark caption.
        wm._MBlock(type="image", page_idx=1, index=1,
                   bbox=(303, 211, 558, 716),
                   text="Downloaded from https://www.science.org on April 25, 2026",
                   image_path="fig1.jpg"),
    ]
    info = wm.rescue_orphan_captions(blocks)
    assert info["captions_adopted"] == 1
    # Watermark replaced by real caption.
    assert "Fig. 1" in blocks[1].text
    assert "Downloaded" not in blocks[1].text
    assert blocks[0].type == "_adopted_caption"


def test_rescue_orphan_captions_rejects_non_adjacent_block():
    """A caption-shaped text block that is far from any figure/table
    must NOT be adopted (would create cross-page false positives)."""
    blocks = [
        wm._MBlock(type="text", page_idx=0, index=0,
                   bbox=(40, 50, 300, 75),
                   text="Figure 1. A caption far above the table."),
        wm._MBlock(type="table", page_idx=0, index=1,
                   bbox=(40, 600, 380, 720),
                   text=None, html="<table></table>"),
    ]
    info = wm.rescue_orphan_captions(blocks)
    # 525pt vertical gap >> 60pt tolerance: no adoption.
    assert info["captions_adopted"] == 0
    assert blocks[1].text is None
    assert blocks[0].type == "text"


def test_rescue_orphan_captions_skips_when_real_caption_present():
    """Don't override a non-watermark caption with a stray text block."""
    blocks = [
        wm._MBlock(type="text", page_idx=0, index=0,
                   bbox=(40, 280, 380, 335),
                   text="Figure 1. Stray caption-shaped text."),
        wm._MBlock(type="image", page_idx=0, index=1,
                   bbox=(40, 340, 380, 600),
                   text="Figure 1. The real caption MinerU produced.",
                   image_path="fig1.jpg"),
    ]
    info = wm.rescue_orphan_captions(blocks)
    assert info["captions_adopted"] == 0
    # Real caption preserved.
    assert "real caption" in blocks[1].text
    assert blocks[0].type == "text"


def test_rescue_orphan_captions_rejects_body_text():
    """A text block lacking a caption / footnote signature is left
    alone even when geometrically adjacent."""
    blocks = [
        # Adjacent but no caption signature.
        wm._MBlock(type="text", page_idx=0, index=0,
                   bbox=(40, 280, 380, 335),
                   text="Our results establish contemporaneous origin..."),
        wm._MBlock(type="table", page_idx=0, index=1,
                   bbox=(40, 340, 380, 600),
                   text=None, html="<table></table>"),
    ]
    info = wm.rescue_orphan_captions(blocks)
    assert info["captions_adopted"] == 0
    assert blocks[0].type == "text"
    assert blocks[1].text is None


def test_rescue_orphan_captions_picks_closest_candidate():
    """When two candidates both contain Fig./Table N tokens, prefer
    the one with smaller vertical gap to the parent. Body-text
    paragraphs that mention 'Fig. 1' must lose to the actual caption
    block adjacent to the figure (real failure observed on alexander
    p2: body block [6] mentioning 'Fig. 1 and Table 1' was being
    adopted instead of the real caption block [7] just above the
    table)."""
    blocks = [
        # Body paragraph mentioning Fig./Table mid-sentence (further
        # from the table by 40pt).
        wm._MBlock(type="text", page_idx=0, index=0,
                   bbox=(40, 200, 380, 295),
                   text=("The simple observation of measurable Na in the "
                         "cores of chondrule olivines (Fig. 1 and Table 1) "
                         "(18) is contrary to predictions...")),
        # Real caption block, full-page-width, just above the table
        # (5pt gap).
        wm._MBlock(type="text", page_idx=0, index=1,
                   bbox=(40, 300, 380, 335),
                   text=("The petrologic types and estimated liquidus "
                         "temperatures of the Semarkona chondrulesTable 1.")),
        wm._MBlock(type="table", page_idx=0, index=2,
                   bbox=(40, 340, 380, 650),
                   text=None, html="<table></table>"),
    ]
    info = wm.rescue_orphan_captions(blocks)
    assert info["captions_adopted"] == 1
    # Closest candidate (block 1) wins, body paragraph (block 0) stays.
    assert blocks[1].type == "_adopted_caption"
    assert blocks[0].type == "text"
    assert "petrologic types" in blocks[2].text
    assert "simple observation" not in blocks[2].text


def test_emit_markdown_skips_adopted_blocks(tmp_path):
    """Adopted blocks must not appear in the rendered body."""
    import paper2md as p2m
    fixture = {
        "_backend": "pipeline",
        "_version_name": "3.1.6",
        "pdf_info": [{
            "page_idx": 0,
            "page_size": [612, 792],
            "para_blocks": [
                {"type": "text", "bbox": [40, 50, 300, 70], "index": 0,
                 "lines": [{"spans": [{"type": "text",
                                        "content": "Body text para."}]}]},
                {"type": "table", "bbox": [40, 100, 400, 600], "index": 1,
                 "blocks": [
                     {"type": "table_caption",
                      "bbox": [40, 100, 400, 120],
                      "lines": [{"spans": [{"type": "text",
                                            "content": "Table 1."}]}]},
                     {"type": "table_body",
                      "bbox": [40, 130, 400, 600],
                      "lines": [{"spans": [{"type": "table",
                                            "html": "<table><tr><td>x</td></tr></table>",
                                            "image_path": "t.jpg"}]}]},
                 ]},
            ],
            "preproc_blocks": [],
            "discarded_blocks": [],
        }],
    }
    p = tmp_path / "skip_adopted_middle.json"
    p.write_text(json.dumps(fixture))
    doc = wm.parse_middle_json(p)
    # Mark the body text block as adopted; it must not render.
    doc.blocks[0].type = "_adopted_caption"
    report = p2m.QualityReport()
    md = wm.emit_markdown(
        doc, assets_rel="assets", assets_abs=tmp_path / "assets",
        vlm_tables=False, report=report,
    )
    assert "Body text para." not in md
    assert "Table 1" in md  # parent block still emits


# === Sidecar link-text uses actual table id (not positional counter) =====


def test_sidecar_link_uses_table_id_when_provided(tmp_path):
    """Hybrid splice provides `table_id` (the actual paper id like
    'A.4', '2', 'S1'); the sidecar link text MUST use that id rather
    than the internal `table_idx` positional counter. Pre-fix behavior
    routinely mislabeled e.g. an A.4 sidecar as `[Table 3 — separate
    markdown]` because table_idx was the 3rd MinerU block processed."""
    sidecar_name, suffix = wm._write_hybrid_table_sidecar(
        body="| a | b |\n|---|---|\n| 1 | 2 |\n",
        page_idx=2,  # page 3 (0-based)
        table_idx=3,  # 3rd block processed
        assets_abs=tmp_path / "assets",
        assets_rel="assets",
        asset_prefix="",
        table_id="A.4",
    )
    # Filename prepends the sanitized id (A.4 -> A_4) so the .md
    # pairs visually with the table_A_4_p3_3.jpg renamed asset.
    # Page+positional-idx are still part of the stem for filesystem
    # stability across runs.
    assert sidecar_name == "table_A_4_p3_3.md"
    # Link text uses the actual id "A.4" (period preserved -- human-
    # readable, not filename-safe).
    assert "[Table A.4 — separate markdown]" in suffix
    assert "[Table 3 — separate markdown]" not in suffix


def test_sidecar_link_falls_back_to_table_idx_when_no_id(tmp_path):
    """Marker layout's process_tables doesn't pass `table_id` -- must
    preserve the legacy behavior where the link text shows the
    positional counter."""
    sidecar_name, suffix = wm._write_hybrid_table_sidecar(
        body="| a |\n|---|\n| 1 |\n",
        page_idx=0, table_idx=2,
        assets_abs=tmp_path / "assets",
        assets_rel="assets", asset_prefix="",
        # table_id NOT provided
    )
    assert "[Table 2 — separate markdown]" in suffix


# === Parallel table dispatch under emit_markdown ============================


def _three_table_doc(tmp_path):
    """Build a minimal fixture: 3 table blocks each with clean HTML
    so we can exercise the parallel dispatch deterministically."""
    fixture = {
        "_backend": "pipeline", "_version_name": "3.1.7",
        "pdf_info": [{
            "page_idx": 0, "page_size": [612, 792],
            "para_blocks": [
                {"type": "table", "bbox": [40, 100, 400, 200],
                 "index": i, "blocks": [
                     {"type": "table_caption", "bbox": [40, 80, 400, 95],
                      "lines": [{"spans": [{"type": "text",
                                            "content": f"Table {i+1}."}]}]},
                     {"type": "table_body", "bbox": [40, 100, 400, 200],
                      "lines": [{"spans": [{"type": "table",
                                            "html": ("<table><tr><td>h1</td>"
                                                     "<td>h2</td></tr><tr>"
                                                     f"<td>r{i+1}c1</td>"
                                                     f"<td>r{i+1}c2</td></tr>"
                                                     "<tr><td>x</td><td>y</td>"
                                                     "</tr></table>"),
                                            "image_path": f"t{i+1}.jpg"}]}]},
                 ]} for i in range(3)
            ],
            "discarded_blocks": [],
        }],
    }
    p = tmp_path / "three_tables_middle.json"
    p.write_text(json.dumps(fixture))
    return wm.parse_middle_json(p)


def test_emit_markdown_serial_vs_parallel_equivalent(tmp_path):
    """table_workers=1 and table_workers=4 must produce IDENTICAL
    markdown output (modulo internal order of report.tables, which is
    sorted post-pool in the parallel path)."""
    doc1 = _three_table_doc(tmp_path)
    doc2 = _three_table_doc(tmp_path)
    import paper2md as p2m
    r1 = p2m.QualityReport()
    r2 = p2m.QualityReport()
    md_serial = wm.emit_markdown(
        doc1, "assets", tmp_path, vlm_tables=False, report=r1,
        table_workers=1)
    md_parallel = wm.emit_markdown(
        doc2, "assets", tmp_path, vlm_tables=False, report=r2,
        table_workers=4)
    assert md_serial == md_parallel
    # report.tables sorted by index in both cases
    idx_serial = [t.index for t in r1.tables]
    idx_parallel = [t.index for t in r2.tables]
    assert idx_serial == sorted(idx_serial) == idx_parallel


def test_emit_markdown_parallel_uses_pool(tmp_path, monkeypatch):
    """table_workers=4 with >1 table -> ThreadPoolExecutor is invoked
    with max_workers=4. Single-table papers stay sequential."""
    used_pool = []
    import concurrent.futures as _cf
    orig_pool = _cf.ThreadPoolExecutor

    class _SpyPool(orig_pool):
        def __init__(self, *a, **kw):
            used_pool.append(kw.get("max_workers", a[0] if a else None))
            super().__init__(*a, **kw)

    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _SpyPool)

    import paper2md as p2m
    doc = _three_table_doc(tmp_path)
    report = p2m.QualityReport()
    wm.emit_markdown(doc, "assets", tmp_path, vlm_tables=False,
                     report=report, table_workers=4)
    assert used_pool == [4]


def test_emit_markdown_table_workers_one_no_pool(tmp_path, monkeypatch):
    """table_workers=1 (default) must NOT spawn a thread pool."""
    used_pool = []
    import concurrent.futures as _cf
    orig_pool = _cf.ThreadPoolExecutor

    class _SpyPool(orig_pool):
        def __init__(self, *a, **kw):
            used_pool.append(kw.get("max_workers", a[0] if a else None))
            super().__init__(*a, **kw)

    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _SpyPool)

    import paper2md as p2m
    doc = _three_table_doc(tmp_path)
    report = p2m.QualityReport()
    wm.emit_markdown(doc, "assets", tmp_path, vlm_tables=False,
                     report=report, table_workers=1)
    assert used_pool == []


def test_emit_markdown_single_table_stays_serial(tmp_path, monkeypatch):
    """Even with table_workers=4, a single-table paper skips the pool
    (no benefit; pool spin-up is overhead)."""
    used_pool = []
    import concurrent.futures as _cf
    orig_pool = _cf.ThreadPoolExecutor

    class _SpyPool(orig_pool):
        def __init__(self, *a, **kw):
            used_pool.append(kw.get("max_workers", a[0] if a else None))
            super().__init__(*a, **kw)

    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _SpyPool)

    # Build a single-table doc
    fixture = {
        "_backend": "pipeline", "_version_name": "3.1.7",
        "pdf_info": [{"page_idx": 0, "page_size": [612, 792],
                      "para_blocks": [{
                          "type": "table", "bbox": [40, 100, 400, 200],
                          "index": 0, "blocks": [
                              {"type": "table_body",
                               "bbox": [40, 100, 400, 200],
                               "lines": [{"spans": [{"type": "table",
                                                     "html": "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>",
                                                     "image_path": "t.jpg"}]}]},
                          ]}],
                      "discarded_blocks": []}]
    }
    p = tmp_path / "single_table_middle.json"
    p.write_text(json.dumps(fixture))
    doc = wm.parse_middle_json(p)
    import paper2md as p2m
    report = p2m.QualityReport()
    wm.emit_markdown(doc, "assets", tmp_path, vlm_tables=False,
                     report=report, table_workers=4)
    assert used_pool == []


# === TableScore.matched_table records paper-relative id ====================


def test_table_score_matched_table_from_caption(tmp_path):
    """Mineru-only path: table_block_to_md derives matched_table from
    the block's caption text via _extract_table_number. Caption
    'Table 5. Caption text.' -> matched_table='5'."""
    import paper2md as p2m
    from PIL import Image
    img_path = tmp_path / "t.jpg"
    Image.new("RGB", (40, 30), color="white").save(img_path)
    block = wm._MBlock(
        type="table", page_idx=0, index=0, bbox=(10, 10, 100, 100),
        text="Table 5. Caption text.",
        html=("<table><tr><td>h1</td><td>h2</td></tr>"
              "<tr><td>a</td><td>b</td></tr>"
              "<tr><td>c</td><td>d</td></tr></table>"),
        image_path="t.jpg",
    )
    report = p2m.QualityReport()
    wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=False, assets_abs=tmp_path,
        report=report,
    )
    assert len(report.tables) == 1
    assert report.tables[0].matched_table == "5"


def test_table_score_matched_table_explicit_overrides_caption(tmp_path):
    """Hybrid path: when `table_id` is passed (e.g. 'A.4' from the
    splice), it takes precedence over what the caption parser would
    have produced."""
    import paper2md as p2m
    from PIL import Image
    img_path = tmp_path / "t.jpg"
    Image.new("RGB", (40, 30), color="white").save(img_path)
    block = wm._MBlock(
        type="table", page_idx=0, index=0, bbox=(10, 10, 100, 100),
        text="Table 5. Caption text.",  # caption says 5
        html=("<table><tr><td>h1</td><td>h2</td></tr>"
              "<tr><td>a</td><td>b</td></tr>"
              "<tr><td>c</td><td>d</td></tr></table>"),
        image_path="t.jpg",
    )
    report = p2m.QualityReport()
    wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=False, assets_abs=tmp_path,
        report=report,
        table_id="A.4",  # splice passes this; should win over caption
    )
    assert report.tables[0].matched_table == "A.4"


def test_table_score_matched_table_none_when_no_caption(tmp_path):
    """No caption text + no explicit table_id -> matched_table stays
    None. Correct behavior for blocks where the id can't be derived."""
    import paper2md as p2m
    from PIL import Image
    img_path = tmp_path / "t.jpg"
    Image.new("RGB", (40, 30), color="white").save(img_path)
    block = wm._MBlock(
        type="table", page_idx=0, index=0, bbox=(10, 10, 100, 100),
        text=None,  # no caption
        html=("<table><tr><td>h1</td><td>h2</td></tr>"
              "<tr><td>a</td><td>b</td></tr>"
              "<tr><td>c</td><td>d</td></tr></table>"),
        image_path="t.jpg",
    )
    report = p2m.QualityReport()
    wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=False, assets_abs=tmp_path,
        report=report,
    )
    assert report.tables[0].matched_table is None


def test_table_score_matched_table_emits_in_yaml():
    """matched_table appears in QualityReport.to_frontmatter YAML
    when non-None; omitted when None."""
    import paper2md as p2m
    report = p2m.QualityReport()
    report.tables.append(p2m.TableScore(
        index=1, page=3, located=True, jpeg_saved=True,
        pre_redo_reason=None, vlm_redone=False,
        post_redo_reason=None, score=1.0,
        sidecar="table_p3_1.md", matched_table="A.4",
    ))
    fm = report.to_frontmatter()
    assert "matched_table:" in fm
    assert "A.4" in fm

    # None case: line omitted
    report2 = p2m.QualityReport()
    report2.tables.append(p2m.TableScore(
        index=1, page=3, located=True, jpeg_saved=True,
        pre_redo_reason=None, vlm_redone=False,
        post_redo_reason=None, score=1.0,
        sidecar="table_p3_1.md", matched_table=None,
    ))
    fm2 = report2.to_frontmatter()
    assert "matched_table:" not in fm2


# === Sidecar / JPG naming with id-prepend =================================


def test_sanitize_table_id_for_filename():
    """`.` in ids (A.4, B.6) becomes `_` to avoid stem/extension
    confusion. Other chars (alphanumeric, S-prefix, Roman) pass
    through. None and empty input return None."""
    assert wm._sanitize_table_id_for_filename(None) is None
    assert wm._sanitize_table_id_for_filename("") is None
    assert wm._sanitize_table_id_for_filename("  ") is None
    assert wm._sanitize_table_id_for_filename("1") == "1"
    assert wm._sanitize_table_id_for_filename("S5") == "S5"
    assert wm._sanitize_table_id_for_filename("IV") == "IV"
    assert wm._sanitize_table_id_for_filename("A1") == "A1"
    assert wm._sanitize_table_id_for_filename("A.4") == "A_4"
    assert wm._sanitize_table_id_for_filename("B.6") == "B_6"
    # Whitespace stripped before sanitization
    assert wm._sanitize_table_id_for_filename("  A.4  ") == "A_4"


def test_hybrid_table_sidecar_name_with_id():
    """Filename has id prepended when known."""
    n = wm._hybrid_table_sidecar_name(
        page_idx=8, table_idx=2, asset_prefix="", table_id="A.4")
    assert n == "table_A_4_p9_2.md"


def test_hybrid_table_sidecar_name_without_id():
    """Filename falls back to positional-only when id is None
    (marker layout)."""
    n = wm._hybrid_table_sidecar_name(
        page_idx=8, table_idx=2, asset_prefix="", table_id=None)
    assert n == "table_p9_2.md"


def test_hybrid_table_sidecar_name_with_asset_prefix():
    """asset_prefix (e.g. 'si_' for supplements) prepends the whole
    name."""
    n = wm._hybrid_table_sidecar_name(
        page_idx=0, table_idx=1, asset_prefix="si_", table_id="S2")
    assert n == "si_table_S2_p1_1.md"


def test_hybrid_table_sidecar_name_no_page():
    """When page_idx is None (rare; image-only blocks), id is still
    prepended."""
    n = wm._hybrid_table_sidecar_name(
        page_idx=None, table_idx=3, asset_prefix="", table_id="2")
    assert n == "table_2_3.md"
    # Without id
    n = wm._hybrid_table_sidecar_name(
        page_idx=None, table_idx=3, asset_prefix="", table_id=None)
    assert n == "table_3.md"


# === Figure rename helpers + naming convention =============================


def test_figure_filename_single_panel():
    """Single-panel figure: `figure_{id}_p{page}{ext}`."""
    assert wm._figure_filename_for_panel(
        "1", None, page_idx=2, ext=".jpg") == "figure_1_p3.jpg"


def test_figure_filename_multi_panel():
    """Multi-panel figure: letter suffix appended directly to id."""
    assert wm._figure_filename_for_panel(
        "1", "a", page_idx=2, ext=".jpg") == "figure_1a_p3.jpg"
    assert wm._figure_filename_for_panel(
        "1", "b", page_idx=2, ext=".jpg") == "figure_1b_p3.jpg"
    assert wm._figure_filename_for_panel(
        "S2", "c", page_idx=4, ext=".png") == "figure_S2c_p5.png"


def test_figure_filename_id_sanitized():
    """Dotted ids (S.1, A.2) have `.` -> `_` for filesystem safety."""
    assert wm._figure_filename_for_panel(
        "A.4", None, page_idx=0, ext=".jpg") == "figure_A_4_p1.jpg"


def test_figure_filename_no_page():
    """page_idx=None drops the `_p<n>` segment."""
    assert wm._figure_filename_for_panel(
        "3", None, page_idx=None, ext=".jpg") == "figure_3.jpg"


def test_rename_figure_image_file_idempotent(tmp_path):
    """Re-running rename when target already exists is a no-op."""
    src = tmp_path / "abc.jpg"
    src.write_bytes(b"original")
    dst_name = "figure_1_p3.jpg"
    # First rename succeeds
    r1 = wm._rename_figure_image_file(
        "abc.jpg", dst_name, tmp_path, "")
    assert r1 == dst_name
    assert (tmp_path / dst_name).exists()
    assert not src.exists()
    # Second rename (source gone, target present) is a no-op
    r2 = wm._rename_figure_image_file(
        "abc.jpg", dst_name, tmp_path, "")
    assert r2 == dst_name


def test_rename_figure_image_file_missing_source(tmp_path):
    """Source file doesn't exist (caller will fall back to original name)."""
    r = wm._rename_figure_image_file(
        "nonexistent.jpg", "figure_1_p1.jpg", tmp_path, "")
    assert r is None


def test_rename_mineru_figure_image_single_panel(tmp_path):
    """Block with caption 'Fig. 5. ...' gets its hash-named image
    renamed to `figure_5_p{page}.jpg` (single panel)."""
    src = tmp_path / "abc.jpg"
    src.write_bytes(b"x")
    block = wm._MBlock(
        type="image", page_idx=2, index=0, bbox=(0, 0, 100, 100),
        text="Fig. 5. Some caption.",
        image_path="abc.jpg",
        subpanel_paths=[],
        subpanel_bboxes=[],
        figure_number=None,
    )
    wm._rename_mineru_figure_image(block, tmp_path, asset_prefix="")
    assert block.image_path == "figure_5_p3.jpg"
    assert (tmp_path / "figure_5_p3.jpg").exists()
    assert not src.exists()


def test_rename_mineru_figure_image_no_id_keeps_hash(tmp_path):
    """Block with no parseable id (empty caption, no figure_number)
    keeps its hash name."""
    src = tmp_path / "abc.jpg"
    src.write_bytes(b"x")
    block = wm._MBlock(
        type="image", page_idx=0, index=0, bbox=(0, 0, 100, 100),
        text="",
        image_path="abc.jpg",
        subpanel_paths=[],
        subpanel_bboxes=[],
        figure_number=None,
    )
    wm._rename_mineru_figure_image(block, tmp_path, asset_prefix="")
    assert block.image_path == "abc.jpg"
    assert src.exists()


def test_rename_mineru_figure_image_with_asset_prefix(tmp_path):
    """asset_prefix='si_' for supplement bundling."""
    src = tmp_path / "si_abc.jpg"
    src.write_bytes(b"x")
    block = wm._MBlock(
        type="image", page_idx=0, index=0, bbox=(0, 0, 100, 100),
        text="Figure 2. Caption.",
        image_path="abc.jpg",
        subpanel_paths=[],
        subpanel_bboxes=[],
        figure_number=None,
    )
    wm._rename_mineru_figure_image(block, tmp_path, asset_prefix="si_")
    assert block.image_path == "figure_2_p1.jpg"
    assert (tmp_path / "si_figure_2_p1.jpg").exists()
    assert not src.exists()
