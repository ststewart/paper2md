"""Tests for the figure/table reference-vs-insertion auditor."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import audit_md  # noqa: E402


def test_balanced_paper_no_mismatch(tmp_path):
    """Body cites Fig 1 and Fig 2; markdown has 2 figure insertions
    + 1 table caption + 1 Table 1 cite -> no mismatch flags."""
    md = tmp_path / "paper.md"
    md.write_text(
        "# Paper\n\n"
        "We see this trend in Fig. 1 and discuss it further "
        "in Figure 2. See also Table 1 for the numbers.\n\n"
        "![Figure 1 | Caption](assets/figure_1.jpg)\n\n"
        "Fig. 1. Caption of figure 1.\n\n"
        "![Figure 2 | Caption](assets/figure_2.jpg)\n\n"
        "Fig. 2. Caption of figure 2.\n\n"
        "**Table 1.** Data summary.\n\n"
        "| col1 | col2 |\n|---|---|\n| a | b |\n"
    )
    r = audit_md.audit_md(md)
    assert sorted(r["figures_referenced"]) == ["1", "2"]
    assert r["figures_inserted"] == 2
    assert sorted(r["tables_referenced"]) == ["1"]
    assert sorted(r["tables_inserted"]) == ["1"]
    assert r["figure_mismatch"] is False
    assert r["table_mismatch"] is False


def test_dropped_figure_flagged(tmp_path):
    """Body cites Fig 1, 2, 3 but only 2 image insertions -> flag."""
    md = tmp_path / "paper.md"
    md.write_text(
        "Discussion of Fig. 1, Fig. 2, and Fig. 3.\n\n"
        "![](assets/figure_1.jpg)\n\n"
        "![](assets/figure_2.jpg)\n"
    )
    r = audit_md.audit_md(md)
    assert sorted(r["figures_referenced"]) == ["1", "2", "3"]
    assert r["figures_inserted"] == 2
    assert r["figure_mismatch"] is True


def test_dropped_table_flagged(tmp_path):
    """Body cites Table 1-3, only 1 table caption / sidecar -> flag."""
    md = tmp_path / "paper.md"
    md.write_text(
        "Results in Table 1, Table 2, and Table 3.\n\n"
        "**Table 1.** First table.\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    r = audit_md.audit_md(md)
    assert sorted(r["tables_referenced"]) == ["1", "2", "3"]
    assert sorted(r["tables_inserted"]) == ["1"]
    assert r["table_mismatch"] is True


def test_extra_insertions_not_flagged(tmp_path):
    """More insertions than refs is benign (uncited figures happen);
    NOT flagged."""
    md = tmp_path / "paper.md"
    md.write_text(
        "Just mention Fig. 1.\n\n"
        "![](assets/figure_1.jpg)\n\n"
        "![](assets/figure_2.jpg)\n\n"
        "![](assets/figure_3.jpg)\n"
    )
    r = audit_md.audit_md(md)
    assert r["figures_inserted"] == 3
    assert r["figures_referenced"] == ["1"]
    assert r["figure_mismatch"] is False


def test_table_images_dont_count_as_figures(tmp_path):
    """`table_p3_1.jpg` is a TABLE image -- must NOT inflate the
    figure insertion count."""
    md = tmp_path / "paper.md"
    md.write_text(
        "See Fig. 1 and Table 1.\n\n"
        "![](assets/figure_1.jpg)\n\n"
        "**Table 1.** Data.\n\n"
        "![](assets/table_p3_1.jpg)\n"
        "[Table 1 — separate markdown](assets/table_p3_1.md)\n"
    )
    r = audit_md.audit_md(md)
    assert r["figures_inserted"] == 1  # NOT 2 (table image excluded)
    assert sorted(r["tables_inserted"]) == ["1"]


def test_dedup_by_figure_id(tmp_path):
    """A figure cited 4 times counts once."""
    md = tmp_path / "paper.md"
    md.write_text("Fig. 1 here. Fig. 1 there. Figure 1 again. Fig. 1.\n\n"
                  "![](assets/figure_1.jpg)\n")
    r = audit_md.audit_md(md)
    assert r["figures_referenced"] == ["1"]


def test_supplementary_figures_recognized(tmp_path):
    """Fig. S1, Fig. A1, Extended Data Fig. 1 -- distinct ids."""
    md = tmp_path / "paper.md"
    md.write_text(
        "Cited: Fig. 1, Fig. S2, Extended Data Fig. 3, Fig. A1.\n\n"
        "![](assets/figure_1.jpg)\n"
    )
    r = audit_md.audit_md(md)
    refs = set(r["figures_referenced"])
    assert "1" in refs
    assert "S2" in refs
    assert "3" in refs    # Extended Data Fig. 3
    assert "A1" in refs


def test_frontmatter_stripped(tmp_path):
    """Figures mentioned in YAML frontmatter must NOT count as body
    refs (would skew the heuristic on every paper)."""
    md = tmp_path / "paper.md"
    md.write_text(
        "---\n"
        "title: Paper that talks about Fig. 99 in metadata\n"
        "---\n\n"
        "Body cites Fig. 1.\n\n"
        "![](assets/figure_1.jpg)\n"
    )
    r = audit_md.audit_md(md)
    assert r["figures_referenced"] == ["1"]
    assert "99" not in r["figures_referenced"]


def test_si_sibling_filters_s_prefix_refs(tmp_path):
    """When `<stem>_sm.md` sibling exists, S-prefix refs in the main
    are SKIPPED — those figs/tables live in the SI, not the main.
    Pre-refinement, main cites Fig. S1 -> spurious mismatch flag."""
    main = tmp_path / "paper.md"
    si = tmp_path / "paper_sm.md"
    main.write_text(
        "We cite Fig. 1 and Fig. 2 in the body; Fig. S1 and Table S1 "
        "are described in the supplement.\n\n"
        "![](assets/figure_1.jpg)\n"
        "![](assets/figure_2.jpg)\n"
    )
    si.write_text("# SI\n![](assets/figure_s1.jpg)\n")
    r = audit_md.audit_md(main)
    assert sorted(r["figures_referenced"]) == ["1", "2"]
    assert r["figure_mismatch"] is False
    assert r["tables_referenced"] == []
    assert r["table_mismatch"] is False


def test_si_file_itself_keeps_s_prefix_refs(tmp_path):
    """When auditing the SI file directly, S-prefix refs ARE the SI's
    own figures and must remain in the count."""
    main = tmp_path / "paper.md"
    si = tmp_path / "paper_sm.md"
    main.write_text("Main body.\n")
    si.write_text(
        "Cite Fig. S1 here.\n\n![](assets/figure_s1.jpg)\n"
    )
    r = audit_md.audit_md(si)
    assert "S1" in r["figures_referenced"]
    assert r["figure_mismatch"] is False  # 1 ref, 1 insert


def test_no_si_sibling_keeps_s_prefix_refs(tmp_path):
    """A standalone paper with no SI sibling: S-prefix refs are kept
    (the splice could plausibly have inserted them in-place)."""
    md = tmp_path / "paper.md"
    md.write_text(
        "Body cites Fig. 1 and Fig. S1.\n\n"
        "![](assets/figure_1.jpg)\n"
    )
    r = audit_md.audit_md(md)
    assert sorted(r["figures_referenced"]) == ["1", "S1"]
    assert r["figure_mismatch"] is True  # S1 was a real drop


def test_roman_table_ids_deduped_against_arabic(tmp_path):
    """Refs to `Table V` and `Table 5` are the same table. Mismatch
    check uses canonical (arabic) form to avoid double-counting.
    (Mimics the Knudson2013 case from the corpus run.)"""
    md = tmp_path / "paper.md"
    md.write_text(
        "Discussion: results in Table V, Table III, Table II, Table I.\n\n"
        "**Table 1.** First.\n\n"
        "**Table 2.** Second.\n\n"
        "**Table 3.** Third.\n\n"
        "**Table 5.** Fifth.\n"
    )
    r = audit_md.audit_md(md)
    # Both caption-lines AND body cross-refs are caught by _TBL_REF_RE,
    # so refs is the union: arabic 1,2,3,5 (from captions) + roman
    # I,II,III,V (from body).
    assert sorted(r["tables_referenced"]) == [
        "1", "2", "3", "5", "I", "II", "III", "V"]
    assert sorted(r["tables_inserted"]) == ["1", "2", "3", "5"]
    # Canonical refs collapse to {1, 2, 3, 5} -- same as inserts,
    # so NO mismatch. (Pre-refinement this would have flagged.)
    assert r["table_mismatch"] is False


def test_roman_to_int_helper():
    """Spot-check the roman conversion."""
    assert audit_md._roman_to_int("I") == 1
    assert audit_md._roman_to_int("IV") == 4
    assert audit_md._roman_to_int("V") == 5
    assert audit_md._roman_to_int("IX") == 9
    assert audit_md._roman_to_int("XIII") == 13
    assert audit_md._roman_to_int("") is None
    assert audit_md._roman_to_int("ZZ") is None


def test_canonical_preserves_s_prefix_arabic():
    """`S5` is already canonical; don't try to re-interpret the S
    as a roman literal."""
    assert audit_md._canonical_tbl_id("S5") == "S5"
    assert audit_md._canonical_tbl_id("V") == "5"
    assert audit_md._canonical_tbl_id("5") == "5"
    assert audit_md._canonical_tbl_id("A1") == "A1"  # not roman


def test_no_figures_no_tables_no_flag(tmp_path):
    """Theory paper with no figs or tables -- no mismatch flagged
    (both counts are 0, refs <= inserts trivially)."""
    md = tmp_path / "paper.md"
    md.write_text("# Pure theory\n\nWe prove that p = np.\n")
    r = audit_md.audit_md(md)
    assert r["figure_mismatch"] is False
    assert r["table_mismatch"] is False
    assert r["figures_inserted"] == 0
    assert r["figures_referenced"] == []
