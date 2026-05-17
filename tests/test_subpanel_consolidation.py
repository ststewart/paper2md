"""Sub-panel consolidation tests.

Mirrors the design section split:

  Section 1: caption predicates (pure string functions, no _MBlock)
  Section 2: rescue_subpanel_groups (operates on _MBlock list)
  Section 3: rendering (image_block_to_md with subpanel_paths)
  Section 4: end-to-end through layout_mineru -> emit_markdown

Scope (per Sarah, 2026-05-10):
  - Pattern A (canup-style real-caption-with-letter + stub-letter siblings)
  - Pattern C (luo-style stub 'Figure N' siblings)
  - Pattern B (cross-figure absorption) is NOT fixed but MUST be guarded
    against -- a primary with 2+ figure anchors should NOT consolidate.
  - Pattern D (multi-page tables) explicitly out of scope.
  - Tables (any pattern) explicitly out of scope.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import layout_mineru as lm  # noqa: E402


# === Section 1 — caption predicates ========================================


class TestIsFigureCaptionPrimary:
    def test_canup_letter_prefix_pipe_separator(self):
        assert lm._is_figure_caption_primary(
            "b Figure 1 | Simulation of the Moon's accretion, "
            "reproduced from ref. 6. a, Moon mass versus time."
        )

    def test_jacquet_period_after_fig(self):
        assert lm._is_figure_caption_primary(
            "Fig. 1. Abundance patterns for elements arranged in "
            "order of increasing volatility for different types of "
            "chondrules."
        )

    def test_luo_period_after_full_figure(self):
        assert lm._is_figure_caption_primary(
            "Figure 4. Representative time-resolved radiance "
            "profiles recorded by the six-channel pyrometer."
        )

    def test_root_caps_period(self):
        assert lm._is_figure_caption_primary(
            "FIG. 4. DFT-MD calculated Hugoniot temperatures from "
            "this work and Refs (15)."
        )

    def test_rejects_stub_letter(self):
        assert not lm._is_figure_caption_primary("a")

    def test_rejects_figure_label_no_body(self):
        assert not lm._is_figure_caption_primary("Figure 7")

    def test_rejects_empty(self):
        assert not lm._is_figure_caption_primary("")

    def test_rejects_none(self):
        assert not lm._is_figure_caption_primary(None)

    def test_rejects_arbitrary_body_text(self):
        assert not lm._is_figure_caption_primary(
            "The Moon's accretion proceeds in three phases.")

    def test_feng_parens_panel_prefix(self):
        """feng's J. Geophys. Res. style: `(c) Figure 1. body...`
        Parenthesized panel-letter prefix observed in 3 of feng's 5
        multi-panel figures."""
        assert lm._is_figure_caption_primary(
            "(c) Figure 1. Experimental configuration and optical "
            "diagnostic patterns."
        )

    def test_parens_with_whitespace(self):
        assert lm._is_figure_caption_primary(
            "(b) Figure 2. Hugoniot relations of coesite."
        )

    def test_ocampo_closing_paren_only_prefix(self):
        """ocampo's AGU style: `b) Figure 1. body...` (closing paren
        only, no opening). Three of ocampo's panels use this form."""
        assert lm._is_figure_caption_primary(
            "b) Figure 1. (a) Schematic illustration of the "
            "experimental configuration."
        )


class TestIsFigureCaptionStub:
    def test_single_letter(self):
        assert lm._is_figure_caption_stub("a")

    def test_single_letter_with_whitespace(self):
        assert lm._is_figure_caption_stub("  c  ")

    def test_multiple_letters(self):
        assert lm._is_figure_caption_stub("b a")

    def test_figure_label_only(self):
        assert lm._is_figure_caption_stub("Figure 1")

    def test_fig_label_only(self):
        assert lm._is_figure_caption_stub("Fig. 7")

    def test_caps_label_only(self):
        assert lm._is_figure_caption_stub("FIG. 2")

    def test_empty(self):
        assert lm._is_figure_caption_stub("")

    def test_whitespace_only(self):
        assert lm._is_figure_caption_stub("   \n  ")

    def test_none(self):
        assert lm._is_figure_caption_stub(None)

    def test_rejects_real_caption(self):
        assert not lm._is_figure_caption_stub(
            "Figure 1 | Simulation of the Moon's accretion."
        )

    def test_rejects_arbitrary_body_text(self):
        assert not lm._is_figure_caption_stub(
            "The Moon's accretion proceeds in three phases."
        )

    def test_parens_panel_letter_stub(self):
        """feng panel-only stub: `(a)` or `(b)` on its own line."""
        assert lm._is_figure_caption_stub("(a)")
        assert lm._is_figure_caption_stub("(b)")
        assert lm._is_figure_caption_stub("  (c)  ")

    def test_closing_paren_panel_stub(self):
        """ocampo's AGU style: `a)` and `b)` as panel-only stubs."""
        assert lm._is_figure_caption_stub("a)")
        assert lm._is_figure_caption_stub("b)")
        assert lm._is_figure_caption_stub("  c)  ")

    def test_closing_paren_multi_letter_stub(self):
        """ocampo blk[51]: `a b)` -- two panel labels in one stub."""
        assert lm._is_figure_caption_stub("a b)")
        assert lm._is_figure_caption_stub("a b c)")


class TestStripLeadingPanelLetter:
    def test_letter_then_figure_pipe(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "b Figure 1 | text body")
        assert cleaned == "Figure 1 | text body"
        assert letter == "b"

    def test_letter_then_fig_period(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "a Fig. 5. text body")
        assert cleaned == "Fig. 5. text body"
        assert letter == "a"

    def test_no_leading_letter(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "Figure 1 | text body")
        assert cleaned == "Figure 1 | text body"
        assert letter is None

    def test_uppercase_letter_preserved_as_lower(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "B Figure 1 | text body")
        assert cleaned == "Figure 1 | text body"
        assert letter == "b"

    def test_letter_not_followed_by_figure_label_left_alone(self):
        # 'a Moon' should NOT be stripped (would mangle real text)
        cleaned, letter = lm._strip_leading_panel_letter(
            "a Moon mass versus time.")
        assert cleaned == "a Moon mass versus time."
        assert letter is None

    def test_empty(self):
        cleaned, letter = lm._strip_leading_panel_letter("")
        assert cleaned == ""
        assert letter is None

    def test_none(self):
        cleaned, letter = lm._strip_leading_panel_letter(None)
        assert cleaned == ""
        assert letter is None

    def test_parens_letter_then_figure_period(self):
        """feng: `(c) Figure 1. body...` -> letter='c', clean caption."""
        cleaned, letter = lm._strip_leading_panel_letter(
            "(c) Figure 1. Experimental configuration text.")
        assert cleaned == "Figure 1. Experimental configuration text."
        assert letter == "c"

    def test_parens_letter_uppercase_lowered(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "(B) Fig. 2. body text.")
        assert cleaned == "Fig. 2. body text."
        assert letter == "b"

    def test_parens_with_internal_whitespace(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "( a ) Figure 5. body text.")
        assert cleaned == "Figure 5. body text."
        assert letter == "a"

    def test_closing_paren_letter_then_figure(self):
        """ocampo: `b) Figure 1. body...` -> letter='b'."""
        cleaned, letter = lm._strip_leading_panel_letter(
            "b) Figure 1. (a) Schematic illustration.")
        assert cleaned == "Figure 1. (a) Schematic illustration."
        assert letter == "b"

    def test_closing_paren_uppercase_lowered(self):
        cleaned, letter = lm._strip_leading_panel_letter(
            "A) Fig. 2. body text.")
        assert cleaned == "Fig. 2. body text."
        assert letter == "a"


class TestExtractFigureNumber:
    def test_parens_panel_prefix(self):
        """feng: `(c) Figure 1. body` -> 1."""
        assert lm._extract_figure_number(
            "(c) Figure 1. Experimental configuration.") == 1

    def test_bare_letter_panel_prefix(self):
        """canup: `b Figure 1 | text` -> 1."""
        assert lm._extract_figure_number(
            "b Figure 1 | Simulation of the Moon's accretion.") == 1

    def test_no_panel_prefix(self):
        assert lm._extract_figure_number(
            "Figure 3 | Clump formation temperature.") == 3

    def test_fig_caps(self):
        assert lm._extract_figure_number(
            "FIG. 4. DFT-MD calculated Hugoniot.") == 4

    def test_closing_paren_prefix(self):
        """ocampo: `b) Figure 1. body` -> 1."""
        assert lm._extract_figure_number(
            "b) Figure 1. (a) Schematic illustration.") == 1


class TestCountFigureAnchors:
    def test_jacquet_two_headers(self):
        # The cross-figure-absorption case: two figure HEADERS (each
        # at a sentence boundary). Drives the Pattern B guard.
        assert lm._count_figure_anchors(
            "Fig. 6. Rolling averages of Na/Al text. "
            "Fig. 7. Rolling averages of K/Al."
        ) == 2

    def test_root_two_headers(self):
        assert lm._count_figure_anchors(
            "FIG. 2. Fused silica Hugoniot data text. "
            "FIG. 3. Fused silica Hugoniot data."
        ) == 2

    def test_single_header(self):
        assert lm._count_figure_anchors(
            "Figure 1 | Simulation of the Moon's accretion."
        ) == 1

    def test_cross_reference_not_counted(self):
        """canup Fig 3 caption mentions 'Fig. 1' inside the body --
        that's a cross-reference, NOT a header. Should count 1."""
        assert lm._count_figure_anchors(
            "Figure 3 | Clump formation temperature. a, Inner disk "
            "surface density versus time from the Fig. 1 simulation. "
            "b, Mid-plane temperature."
        ) == 1

    def test_multiple_cross_references(self):
        """Multiple Fig N cross-references inside body shouldn't
        trigger the Pattern B guard."""
        assert lm._count_figure_anchors(
            "Figure 5 | Mesostasis Na/Al vs SiO2 contents. Open "
            "circles correspond to Fig. 2 chondrites; symbols match "
            "Fig. 3 conventions."
        ) == 1

    def test_no_anchors(self):
        assert lm._count_figure_anchors("a") == 0
        assert lm._count_figure_anchors("") == 0
        assert lm._count_figure_anchors(None) == 0


class TestCountPanelReferences:
    def test_canup_fig1_two_panels(self):
        assert lm._count_panel_references(
            "Figure 1 | Simulation of the Moon's accretion. "
            "a, Moon mass versus time. b, Fate of inner disk clumps."
        ) == 2

    def test_canup_fig3_three_panels(self):
        assert lm._count_panel_references(
            "Figure 3 | Clump formation temperature and volatile "
            "content. a, Inner disk surface density. b, Mid-plane "
            "temperature. c, The Moon's mass."
        ) == 3

    def test_merrill_no_panels(self):
        """Merrill 1922 caption: single-image figure, no panel refs.
        Drives the Merrill-plate guard."""
        assert lm._count_panel_references(
            "Fig. 2.—One of the same chondrules broken and showing "
            "radial structure."
        ) == 0

    def test_luo_no_panels(self):
        """luo Fig 4 caption is single-panel with stub 'Figure 4'
        sibling. The numbered stub adopts via Pass 2 (figure-number
        match), so panel-references count of 0 is fine here."""
        assert lm._count_panel_references(
            "Figure 4. Representative time-resolved radiance "
            "profiles recorded by the six-channel pyrometer."
        ) == 0

    def test_period_separator(self):
        assert lm._count_panel_references(
            "Figure 1. a. body. b. body."
        ) == 2

    def test_dedupe_repeated_letter(self):
        """Same panel letter mentioned twice still counts once."""
        assert lm._count_panel_references(
            "Figure 1. a, body. ... compared to a, the new model."
        ) == 1

    def test_empty_or_none(self):
        assert lm._count_panel_references("") == 0
        assert lm._count_panel_references(None) == 0


class TestExtractFigureNumber:
    def test_canup_letter_prefix(self):
        assert lm._extract_figure_number(
            "b Figure 1 | Simulation of the Moon's accretion."
        ) == 1

    def test_jacquet_period_form(self):
        assert lm._extract_figure_number(
            "Fig. 1. Abundance patterns for elements."
        ) == 1

    def test_caps_form(self):
        assert lm._extract_figure_number(
            "FIG. 6 that this border portion contains enclosures."
        ) == 6

    def test_stub_figure_n(self):
        assert lm._extract_figure_number("FIG. 5") == 5
        assert lm._extract_figure_number("Figure 14") == 14

    def test_letter_only_returns_none(self):
        assert lm._extract_figure_number("a") is None
        assert lm._extract_figure_number("b a") is None

    def test_empty_returns_none(self):
        assert lm._extract_figure_number("") is None
        assert lm._extract_figure_number(None) is None

    def test_body_text_returns_none(self):
        assert lm._extract_figure_number(
            "The Moon's accretion proceeds in three phases.") is None

    def test_two_digit_number(self):
        assert lm._extract_figure_number("Figure 14") == 14
        assert lm._extract_figure_number("FIG. 100 some body text.") == 100


class TestSubpanelLetterAssignments:
    def test_no_extracted_letter_assigns_by_reading_order(self):
        labels = lm._subpanel_letter_assignments(None, 3)
        assert labels == ["a", "b", "c"]

    def test_extracted_letter_placed_first_others_skip_it(self):
        # primary's caption was 'b Figure 1 | ...'; primary renders
        # first. So labels[0] should be 'b', remaining filled with
        # a, c, d (skipping b).
        labels = lm._subpanel_letter_assignments("b", 3)
        assert labels == ["b", "a", "c"]

    def test_extracted_letter_a_no_skip_needed(self):
        labels = lm._subpanel_letter_assignments("a", 3)
        assert labels == ["a", "b", "c"]

    def test_single_panel(self):
        # Edge case: shouldn't happen (no consolidation triggers on
        # 1 panel), but the function should still do something sane.
        assert lm._subpanel_letter_assignments(None, 1) == ["a"]
        assert lm._subpanel_letter_assignments("c", 1) == ["c"]


# === Section 2 — group reconstruction ======================================


def _make_image_block(page_idx: int, index: int, bbox: tuple,
                     text, image_path):
    """Synthetic _MBlock helper."""
    return lm._MBlock(
        type="image",
        page_idx=page_idx,
        index=index,
        bbox=bbox,
        text=text,
        image_path=image_path,
    )


class TestRescueSubpanelGroups:
    def test_canup_pattern_a_consolidates(self):
        """Canup-style: 1 primary with letter prefix + 2 stubs ->
        primary keeps its image, gets subpanel_paths from the 2 stubs,
        siblings marked _adopted_subpanel.

        Caption has panel refs (a, b) so the Merrill-plate guard
        allows up to 2 letter-stubs."""
        primary = _make_image_block(
            page_idx=0, index=2,
            bbox=(100, 200, 500, 400),
            text="b Figure 1 | Simulation of the Moon's accretion. "
                 "a, Moon mass versus time. b, Fate of inner disk clumps.",
            image_path="primary.jpg",
        )
        stub_letter = _make_image_block(
            page_idx=0, index=1,
            bbox=(100, 100, 500, 199),  # above primary
            text="a",
            image_path="panel_a.jpg",
        )
        stub_fig_n = _make_image_block(
            page_idx=0, index=3,
            bbox=(100, 410, 500, 600),  # below primary
            text="Figure 1",
            image_path="another.jpg",
        )
        blocks = [stub_letter, primary, stub_fig_n]
        info = lm.rescue_subpanel_groups(blocks)
        assert info["groups"] == 1
        assert info["panels_adopted"] == 2
        # Primary preserved, caption stripped, letter extracted.
        assert primary.text.startswith("Figure 1 |")
        assert primary.panel_letter == "b"
        # Subpanels in reading order (top-to-bottom): panel_a (y=100)
        # before another (y=410).
        assert primary.subpanel_paths == ["panel_a.jpg", "another.jpg"]
        # Stubs marked adopted.
        assert stub_letter.type == "_adopted_subpanel"
        assert stub_fig_n.type == "_adopted_subpanel"

    def test_luo_pattern_c_no_letter_prefix(self):
        """Luo-style: 1 primary (no letter prefix) + 1 stub
        'Figure 4'. Primary's panel_letter is None, subpanel adopted."""
        primary = _make_image_block(
            page_idx=2, index=4,
            bbox=(50, 100, 550, 300),
            text="Figure 4. Representative time-resolved radiance "
                 "profiles recorded by the six-channel pyrometer.",
            image_path="primary.jpg",
        )
        stub = _make_image_block(
            page_idx=2, index=5,
            bbox=(50, 350, 550, 500),
            text="Figure 4",
            image_path="stub.jpg",
        )
        blocks = [primary, stub]
        info = lm.rescue_subpanel_groups(blocks)
        assert info["groups"] == 1
        assert info["panels_adopted"] == 1
        assert primary.panel_letter is None
        assert primary.subpanel_paths == ["stub.jpg"]
        assert stub.type == "_adopted_subpanel"

    def test_pattern_b_guard_2plus_anchors_skipped(self):
        """Pattern B: primary's caption has absorbed two figure
        captions. Skip consolidation to avoid making this worse."""
        primary = _make_image_block(
            page_idx=5, index=10,
            bbox=(50, 100, 550, 300),
            text="Fig. 6. Rolling averages of Na/Al for mesostasis "
                 "data. Fig. 7. Rolling averages of K/Al for mesostasis "
                 "data.",
            image_path="combined.jpg",
        )
        stub = _make_image_block(
            page_idx=5, index=11,
            bbox=(50, 350, 550, 500),
            text="Figure 7",
            image_path="fig7.jpg",
        )
        blocks = [primary, stub]
        info = lm.rescue_subpanel_groups(blocks)
        assert info["groups"] == 0
        assert primary.subpanel_paths == []
        assert primary.panel_letter is None
        # Stub NOT marked adopted; remains as its own image block.
        assert stub.type == "image"

    def test_no_primary_no_op(self):
        """Page with only stubs (no primary) -- nothing to consolidate
        under. Don't touch them."""
        s1 = _make_image_block(0, 1, (100, 100, 500, 200),
                              "a", "a.jpg")
        s2 = _make_image_block(0, 2, (100, 250, 500, 350),
                              "Figure 1", "b.jpg")
        blocks = [s1, s2]
        info = lm.rescue_subpanel_groups(blocks)
        assert info["groups"] == 0
        assert s1.type == "image"
        assert s2.type == "image"

    def test_multiple_primaries_chunked_by_reading_order(self):
        """Page with 2 primaries + interleaved stubs (canup page 3
        pattern: Fig 2 + Fig 3 share the page, each with their own
        sub-panels). Reading-order chunking via Pass 1: each primary
        drains the buffer of pending letter-stubs since the last
        primary. Captions include 'a, body. b, body.' panel refs so
        the Merrill-plate guard allows the adoptions."""
        # Canup page 3 ordering:
        #   stub "a" -> primary "b Figure 2 | ..." -> stub "b a"
        #   -> stub "c" -> primary "Figure 3 | ..."
        s1 = _make_image_block(0, 1, (100, 100, 500, 150),
                              "a", "s1.jpg")
        p1 = _make_image_block(
            0, 2, (100, 200, 500, 300),
            "b Figure 2 | Melt-vapour equilibria. a, Vapour density. "
            "b, Observed silicate Moon abundances.",
            "p1.jpg")
        s2 = _make_image_block(0, 3, (100, 350, 500, 400),
                              "b a", "s2.jpg")
        s3 = _make_image_block(0, 4, (100, 410, 500, 460),
                              "c", "s3.jpg")
        p2 = _make_image_block(
            0, 5, (100, 500, 500, 600),
            "Figure 3 | Clump formation. a, Inner disk surface "
            "density. b, Mid-plane temperature. c, Moon's mass.",
            "p2.jpg")
        blocks = [s1, p1, s2, s3, p2]
        info = lm.rescue_subpanel_groups(blocks)
        assert info["groups"] == 2
        assert info["panels_adopted"] == 3
        # Fig 2 picks up "a" (the one stub preceding it).
        assert p1.panel_letter == "b"
        assert p1.figure_number == 2
        assert p1.subpanel_paths == ["s1.jpg"]
        # Fig 3 picks up "b a" + "c" (the two stubs between p1 and p2).
        assert p2.panel_letter is None
        assert p2.figure_number == 3
        assert p2.subpanel_paths == ["s2.jpg", "s3.jpg"]
        assert s1.type == "_adopted_subpanel"
        assert s2.type == "_adopted_subpanel"
        assert s3.type == "_adopted_subpanel"


    def test_two_primaries_no_stubs_no_op(self):
        """Two primaries with no stubs between them: nothing to
        consolidate -- both primaries kept as single-panel figures."""
        p1 = _make_image_block(
            0, 1, (100, 100, 500, 200),
            "Figure 1 | The first figure caption with body.", "p1.jpg")
        p2 = _make_image_block(
            0, 2, (100, 250, 500, 350),
            "Figure 2 | The second figure caption with body.", "p2.jpg")
        info = lm.rescue_subpanel_groups([p1, p2])
        assert info["groups"] == 0
        assert p1.subpanel_paths == []
        assert p2.subpanel_paths == []


    def test_trailing_stub_attaches_to_last_primary(self):
        """Stub appearing AFTER the last primary on a page (caption-
        above layout fallback): attaches to the last primary."""
        p = _make_image_block(
            0, 1, (100, 100, 500, 200),
            "Figure 1 | Caption-above-panels layout. a, panel a body. "
            "b, panel b body.",
            "p.jpg")
        s = _make_image_block(0, 2, (100, 250, 500, 350),
                             "a", "s.jpg")
        info = lm.rescue_subpanel_groups([p, s])
        assert info["groups"] == 1
        assert p.subpanel_paths == ["s.jpg"]
        assert s.type == "_adopted_subpanel"

    def test_single_image_no_op(self):
        """Page with one image block (no siblings): nothing to consolidate."""
        b = _make_image_block(
            0, 1, (100, 100, 500, 300),
            "Figure 1 | One figure on this page.", "p.jpg")
        info = lm.rescue_subpanel_groups([b])
        assert info["groups"] == 0
        assert b.subpanel_paths == []

    def test_table_blocks_ignored(self):
        """Tables out of scope: a table primary + image stub doesn't
        consolidate. Both kept as-is."""
        t = lm._MBlock(
            type="table", page_idx=0, index=1,
            bbox=(100, 100, 500, 200),
            text="TABLE V: Fused Silica Hugoniot Data with body.",
            image_path="t1.jpg",
        )
        i_stub = _make_image_block(0, 2, (100, 250, 500, 350),
                                  "a", "i1.jpg")
        info = lm.rescue_subpanel_groups([t, i_stub])
        assert info["groups"] == 0
        assert t.type == "table"
        assert i_stub.type == "image"

    def test_subpanels_preserve_minerus_index_order(self):
        """Stub image_paths land in primary.subpanel_paths in the
        order MinerU emitted them (the input list is sorted by
        page_idx + intra-page index in parse_middle_json, which is
        MinerU's reading order). Reading-order chunking doesn't
        re-sort; it preserves whatever the input order was."""
        # MinerU intra-page index order: top stub, middle stub,
        # primary, bottom stub.  Trailing stub attaches to the
        # primary (caption-above fallback for the trailing 'c').
        s_top = _make_image_block(0, 1, (100, 50, 500, 100),
                                 "a", "top.jpg")
        s_mid = _make_image_block(0, 2, (100, 200, 500, 250),
                                 "b", "mid.jpg")
        primary = _make_image_block(
            0, 3, (100, 300, 500, 400),
            "Figure 3 | three-panel figure caption. a, panel a body. "
            "b, panel b body. c, panel c body.",
            "primary.jpg")
        s_bot = _make_image_block(0, 4, (100, 500, 500, 600),
                                 "c", "bot.jpg")
        blocks = [s_top, s_mid, primary, s_bot]
        lm.rescue_subpanel_groups(blocks)
        # Order: stubs preceding primary first (a, b), then trailing
        # stub (c) appended at the end.
        assert primary.subpanel_paths == ["top.jpg", "mid.jpg", "bot.jpg"]

    def test_merrill_pattern_numbered_stub_not_adopted_to_different_primary(self):
        """Merrill 1922 pattern: 'FIG. 5' stub on the same page as
        'FIG. 6 <body>' primary. They are SEPARATE figures (each Fig
        gets one image + caption).  Stub MUST NOT be adopted as a
        panel of FIG. 6 -- different figure numbers."""
        fig5_stub = _make_image_block(
            5, 5, (100, 100, 500, 200),
            "FIG. 5", "fig5.jpg")
        fig6_primary = _make_image_block(
            5, 8, (100, 250, 500, 350),
            "FIG. 6 that this border portion often contains "
            "enclosures of the surrounding base material.",
            "fig6.jpg")
        info = lm.rescue_subpanel_groups([fig5_stub, fig6_primary])
        # No consolidation: stub's figure number (5) doesn't match
        # primary's figure number (6).
        assert info["groups"] == 0
        assert fig6_primary.subpanel_paths == []
        assert fig5_stub.type == "image"  # not adopted

    def test_numbered_stub_adopted_when_number_matches(self):
        """jacquet pattern: stub 'Figure 1' WITH the same number as
        primary 'Fig. 1. Abundance patterns...' IS adopted."""
        stub = _make_image_block(
            2, 1, (100, 100, 500, 200),
            "Figure 1", "stub.jpg")
        primary = _make_image_block(
            2, 2, (100, 250, 500, 350),
            "Fig. 1. Abundance patterns for elements arranged in "
            "order of increasing volatility.",
            "primary.jpg")
        info = lm.rescue_subpanel_groups([stub, primary])
        assert info["groups"] == 1
        assert primary.subpanel_paths == ["stub.jpg"]
        assert stub.type == "_adopted_subpanel"
        assert primary.figure_number == 1

    def test_letter_only_stub_flows_to_next_primary(self):
        """Letter-only stubs ('a', 'b') don't have figure numbers and
        flow to the NEXT primary in reading order (caption-below-
        panels = canonical layout). Each primary drains the buffer
        of pending letter-stubs at the moment it appears."""
        stub_a = _make_image_block(
            3, 1, (100, 100, 500, 150), "a", "a.jpg")
        primary1 = _make_image_block(
            3, 2, (100, 200, 500, 300),
            "b Figure 2 | First multi-panel figure. a, body of a. "
            "b, body of b.",
            "p1.jpg")
        stub_c = _make_image_block(
            3, 3, (100, 400, 500, 450), "c", "c.jpg")
        primary2 = _make_image_block(
            3, 4, (100, 500, 500, 600),
            "Figure 3 | Second multi-panel figure. a, body of a. "
            "b, body of b. c, body of c.",
            "p2.jpg")
        blocks = [stub_a, primary1, stub_c, primary2]
        info = lm.rescue_subpanel_groups(blocks)
        # 'a' precedes primary1 -> primary1.
        # 'c' precedes primary2 -> primary2.
        assert info["groups"] == 2
        assert primary1.subpanel_paths == ["a.jpg"]
        assert primary1.figure_number == 2
        assert primary2.subpanel_paths == ["c.jpg"]
        assert primary2.figure_number == 3

    def test_mixed_numbered_and_letter_stubs(self):
        """Page with one numbered stub matching the primary AND
        letter-only stubs: numbered stub adopts via figure-number
        match, letter stubs flow by reading order."""
        stub_a = _make_image_block(
            4, 1, (100, 100, 500, 150), "a", "a.jpg")
        primary = _make_image_block(
            4, 2, (100, 200, 500, 300),
            "b Figure 5 | Multi-panel mesostasis. a, panel a body. "
            "b, panel b body.",
            "p.jpg")
        stub_fig5 = _make_image_block(
            4, 3, (100, 400, 500, 450),
            "Figure 5", "fig5_stub.jpg")
        info = lm.rescue_subpanel_groups([stub_a, primary, stub_fig5])
        assert info["groups"] == 1
        # Both adopted: 'a' via reading order, 'Figure 5' via number match.
        assert primary.subpanel_paths == ["a.jpg", "fig5_stub.jpg"]
        assert primary.figure_number == 5

    def test_merrill_plate_guard_drops_excess_letter_stubs(self):
        """Merrill 1922 page-14 plate: 10 empty-caption stubs + 1
        primary 'Fig. 2.—One of the same chondrules...' (0 panel
        references in caption). Pass 1 would adopt all 10 stubs in
        reading order; the panel-reference cap drops them all
        (n_letter_stubs=10 > n_refs=0). Result: no consolidation."""
        # 10 empty-caption stubs flowing toward a no-panel-refs primary.
        stubs = [
            _make_image_block(13, i, (100, 50 + i*40, 500, 80 + i*40),
                             "", f"chondrule_{i}.jpg")
            for i in range(1, 11)
        ]
        primary = _make_image_block(
            13, 16, (100, 800, 500, 900),
            "Fig. 2.—One of the same chondrules broken and showing "
            "radial structure.",
            "fig2_image.jpg")
        info = lm.rescue_subpanel_groups(stubs + [primary])
        assert info["groups"] == 0
        assert primary.subpanel_paths == []
        # Stubs not adopted.
        for s in stubs:
            assert s.type == "image", \
                f"stub {s.image_path} should NOT be adopted"

    def test_canup_within_panel_cap(self):
        """canup Fig 1: 1 letter-stub + primary with 2 panel refs.
        n_letter_stubs (1) <= n_refs (2) -> consolidate."""
        stub = _make_image_block(
            1, 1, (100, 100, 500, 200), "", "stub.jpg")
        primary = _make_image_block(
            1, 2, (100, 250, 500, 400),
            "b Figure 1 | Simulation of the Moon's accretion. "
            "a, Moon mass versus time. b, Fate of inner disk clumps.",
            "primary.jpg")
        info = lm.rescue_subpanel_groups([stub, primary])
        assert info["groups"] == 1
        assert primary.subpanel_paths == ["stub.jpg"]
        assert primary.figure_number == 1

    def test_luo_numbered_stub_not_capped(self):
        """luo Fig 4: 0 panel refs in caption + 1 numbered stub
        ('Figure 4'). The cap is on Pass 1 (letter-stubs) only;
        Pass 2 numbered-stub adoption proceeds regardless."""
        primary = _make_image_block(
            2, 4, (100, 100, 500, 200),
            "Figure 4. Representative time-resolved radiance "
            "profiles recorded by the six-channel pyrometer.",
            "p.jpg")
        stub = _make_image_block(
            2, 5, (100, 250, 500, 350),
            "Figure 4", "s.jpg")
        info = lm.rescue_subpanel_groups([primary, stub])
        # 0 panel refs in caption is OK because the stub is numbered;
        # Pass 2 adopts it via figure-number match (4=4).
        assert info["groups"] == 1
        assert primary.subpanel_paths == ["s.jpg"]

    def test_idempotent_second_run_no_op(self):
        """Running the rescue twice on the same blocks doesn't
        re-consolidate (already-adopted blocks have type
        _adopted_subpanel which fails the primary AND stub regex)."""
        primary = _make_image_block(
            0, 2, (100, 200, 500, 400),
            "Figure 1 | text body of the caption. a, panel a body.",
            "p.jpg")
        stub = _make_image_block(0, 1, (100, 100, 500, 199),
                                "a", "s.jpg")
        info1 = lm.rescue_subpanel_groups([stub, primary])
        assert info1["groups"] == 1
        info2 = lm.rescue_subpanel_groups([stub, primary])
        assert info2["groups"] == 0
        # Primary still has its subpanel_paths from the first run.
        assert primary.subpanel_paths == ["s.jpg"]
        # Stub still adopted.
        assert stub.type == "_adopted_subpanel"

    def test_pages_independent(self):
        """Per-page grouping: a primary on page 0 and a stub on
        page 1 should NOT be consolidated."""
        p0 = _make_image_block(
            0, 1, (100, 100, 500, 300),
            "Figure 1 | Page 0 figure caption body.", "p0.jpg")
        s1 = _make_image_block(1, 1, (100, 100, 500, 300),
                              "a", "s1.jpg")
        info = lm.rescue_subpanel_groups([p0, s1])
        assert info["groups"] == 0
        assert p0.subpanel_paths == []
        assert s1.type == "image"


# === Section 3 — rendering =================================================


class TestImageBlockToMdMultiPanel:
    def test_two_panels_with_letter_prefix(self):
        """Primary with subpanel_paths=[X] and panel_letter='b' -> emits
        'panel b' (primary, first) then 'panel a' (stub, by reading
        order skipping 'b')."""
        from wrap_mineru import image_block_to_md
        primary = lm._MBlock(
            type="image", page_idx=0, index=1, bbox=(0, 0, 0, 0),
            text="Figure 1 | Multi-panel caption body text here.",
            image_path="p.jpg",
        )
        primary.subpanel_paths = ["s.jpg"]
        primary.panel_letter = "b"
        out = image_block_to_md(primary, "figure", 1, "assets")
        assert "![Figure 1 panel b](assets/p.jpg)" in out
        assert "![Figure 1 panel a](assets/s.jpg)" in out
        # Order: primary first (its letter is 'b' but it's still
        # rendered first in the output stream).
        b_idx = out.index("panel b")
        a_idx = out.index("panel a")
        assert b_idx < a_idx
        # Single shared italic caption line.
        assert out.count("*Figure 1 |") == 1

    def test_three_panels_no_letter_prefix(self):
        """primary.panel_letter=None: assign a, b, c by reading order."""
        from wrap_mineru import image_block_to_md
        primary = lm._MBlock(
            type="image", page_idx=0, index=1, bbox=(0, 0, 0, 0),
            text="Figure 3 | Three-panel figure caption body text.",
            image_path="primary.jpg",
        )
        primary.subpanel_paths = ["panel2.jpg", "panel3.jpg"]
        primary.panel_letter = None
        out = image_block_to_md(primary, "figure", 3, "assets")
        assert "![Figure 3 panel a](assets/primary.jpg)" in out
        assert "![Figure 3 panel b](assets/panel2.jpg)" in out
        assert "![Figure 3 panel c](assets/panel3.jpg)" in out
        assert out.count("*Figure 3 |") == 1

    def test_single_panel_unchanged(self):
        """Block with no subpanel_paths uses the existing single-image
        emission path (regression check)."""
        from wrap_mineru import image_block_to_md
        b = lm._MBlock(
            type="image", page_idx=0, index=1, bbox=(0, 0, 0, 0),
            text="Figure 1 | Simple single-panel figure.",
            image_path="single.jpg",
        )
        out = image_block_to_md(b, "figure", 1, "assets")
        # Existing format: '![<caption>](assets/...)\n\n*<caption>*'
        assert out.startswith(
            "![Figure 1 | Simple single-panel figure.](assets/single.jpg)")
        assert "*Figure 1 | Simple single-panel figure.*" in out
