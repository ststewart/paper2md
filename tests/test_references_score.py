"""Unit tests for paper2md.score_references — the references-section
quality validator added in Phase 1 of journal-aware reconstruction.

Run with:
    cd paper2md && python -m pytest tests/test_references_score.py -v

The validator is a pure function of the markdown body: positive samples
are well-formed References sections in different journal styles, negative
samples are the failure modes the user reported in
collections/refs-score.txt (refs run together, garbled, missing entries,
out-of-order, last-ref-into-acks).
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paper2md as p  # noqa: E402


# ---------------------------------------------------------------------------
# Section detection (heading + trailing-cluster fallback)
# ---------------------------------------------------------------------------

def test_detect_refs_section_with_heading():
    md = dedent("""
        # Body
        Lorem ipsum.

        ## References
        - 1. Author A. *Journal*. 2020.
        - 2. Author B. *Journal*. 2021.
    """).strip()
    sc, span = p._detect_refs_section(md)
    assert sc == 1
    assert span is not None
    body = md[span[0]:span[1]]
    assert "Author A" in body
    assert "Author B" in body


def test_detect_refs_section_no_heading_aps_style():
    """APS PRL/PRB papers emit references without a `## References`
    heading — just a bulleted [N] cluster at the end. The fallback
    finds the cluster; section_count stays 0."""
    md = dedent("""
        # Some paper

        body text body text body text.

        - [1] R. J. Hemley, in *Physics Meets Mineralogy* (Cambridge, 2000).
        - [2] R. M. Canup, Icarus 168, 433 (2004).
        - [3] M. J. Walter and R. G. Tronnes, EPSL 225, 253 (2004).
        - [4] Q. Williams and E. J. Garnero, Science 273, 1528 (1996).
        - [5] R. J. Hemley *et al.*, Phys. Rev. Lett. 57, 747 (1986).
        - [6] B. Mysen, Phys. Earth Planet. Inter. 107, 23 (1998).
    """).strip()
    sc, span = p._detect_refs_section(md)
    assert sc == 0
    assert span is not None
    body = md[span[0]:span[1]]
    assert "[1]" in body
    assert "[6]" in body


def test_detect_refs_section_two_headings_picks_last():
    """When merge_reference_sections didn't run, two References headings
    might still be present. Detect both, span the LAST."""
    md = dedent("""
        ## References
        - 1. First-section ref. 2018.
        - 2. Another. 2019.

        # Methods

        body.

        ## References
        - 1. Methods ref one. 2020.
        - 2. Methods ref two. 2021.
    """).strip()
    sc, span = p._detect_refs_section(md)
    assert sc == 2
    body = md[span[0]:span[1]]
    assert "Methods ref one" in body


def test_detect_refs_section_none():
    md = "# Just a title\n\nSome body text. No references at all.\n"
    sc, span = p._detect_refs_section(md)
    assert sc == 0
    assert span is None


# ---------------------------------------------------------------------------
# Style detection + entry splitting
# ---------------------------------------------------------------------------

def test_classify_numbered():
    body = "- 1. Author A. 2020.\n- 2. Author B. 2021.\n"
    assert p._classify_ref_style(body) == "numbered"


def test_classify_bracketed_numbered():
    body = "- [1] Author A. 2020.\n- [2] Author B. 2021.\n"
    assert p._classify_ref_style(body) == "numbered"


def test_classify_author_year():
    body = "- Smith, J. K., and Doe, A. B. (2020). *Title*.\n- Jones, M. N. (2021). *X*.\n"
    assert p._classify_ref_style(body) == "author-year"


def test_classify_unknown():
    body = "Just a paragraph. No bulleted refs here.\n"
    assert p._classify_ref_style(body) == "unknown"


def test_split_numbered_entries_period_style():
    body = "- 1. Author A. 2020.\n- 2. Author B. 2021.\n- 3. Author C. 2022.\n"
    entries = p._split_ref_entries(body, "numbered")
    assert len(entries) == 3
    assert entries[0].startswith("- 1.")
    assert entries[2].startswith("- 3.")


def test_split_numbered_entries_bracketed_style():
    body = "- [1] Author A. 2020.\n- [2] Author B. 2021.\n- [3] Author C. 2022.\n"
    entries = p._split_ref_entries(body, "numbered")
    assert len(entries) == 3


def test_numbered_continuous_yes():
    entries = ["- 1. A.", "- 2. B.", "- 3. C."]
    assert p._numbered_continuous(entries) is True


def test_numbered_continuous_gap():
    entries = ["- 1. A.", "- 3. C.", "- 4. D."]
    assert p._numbered_continuous(entries) is False


def test_numbered_continuous_bracketed():
    entries = ["- [1] A.", "- [2] B.", "- [3] C."]
    assert p._numbered_continuous(entries) is True


# ---------------------------------------------------------------------------
# Composite score: positive (well-formed) vs negative (broken) cases
# ---------------------------------------------------------------------------

# Eight well-formed entries, numbered 1..8, every entry has a year.
_GOOD_NUMBERED_BODY = "\n".join(
    f"- {i}. Author {chr(64+i)} et al., *Journal*, **{i}**, 100 ({1990 + i})."
    for i in range(1, 9)
)
_GOOD_NUMBERED = (
    "# Body text\n\nblah blah.\n\n## References\n\n" + _GOOD_NUMBERED_BODY + "\n"
)


def test_score_well_formed_numbered():
    s = p.score_references(_GOOD_NUMBERED, journal_slug="science")
    assert s.section_count == 1
    assert s.entry_count == 8
    assert s.numbered_continuous is True
    assert s.year_hit_ratio == 1.0
    assert s.style == "numbered"
    assert s.score >= 0.85
    assert s.journal_slug == "science"


def test_score_well_formed_aps_no_heading():
    """APS PRL/PRB style: bulleted [N] cluster at end of doc, no heading.
    Should still score well — section_count==0 just costs us 0.20."""
    body = "\n".join(
        f"- [{i}] Author {chr(64+i)} et al., Phys. Rev. Lett. **{i}**, 100 ({1990 + i})."
        for i in range(1, 9)
    )
    md = "# Body text\n\nbody.\n\n" + body + "\n"
    s = p.score_references(md, journal_slug="aps-prl")
    assert s.entry_count == 8
    assert s.style == "numbered"
    assert s.numbered_continuous is True
    assert s.section_count == 0  # no heading
    assert 0.7 <= s.score <= 0.85  # docked for missing heading


def test_score_run_together_one_mega_entry():
    """The PNAS Merrill / Boslough failure mode: refs all run together
    into a single multi-paragraph entry. Detector finds 1 entry (or
    none) — score stays low."""
    md = dedent("""
        ## References

        - 1. Smith J 1920 PNAS X 1 1; Jones M 1921 PNAS X 2 2; Brown K 1922 PNAS X 3 3; Davis L 1923 PNAS X 4 4; Evans N 1924 PNAS X 5 5; Foster O 1925 PNAS X 6 6; Garcia P 1926 PNAS X 7 7; Hill Q 1927 PNAS X 8 8.
    """).strip()
    s = p.score_references(md, journal_slug="pnas")
    assert s.entry_count <= 2
    assert s.score < 0.6


def test_score_last_entry_into_acknowledgments():
    """The millot/tracy failure mode: last reference bleeds into the
    Acknowledgments paragraph. last_well_bounded check fires → score
    docks 0.10, AND longest_over_median rises sharply."""
    body = "\n".join(
        f"- {i}. Author {chr(64+i)}, *J*, **{i}**, 1 ({1990 + i})."
        for i in range(1, 8)
    )
    body += (
        "\n- 8. Author H, *J*, **8**, 1 (1998). "
        "Acknowledgments. We thank the OMEGA staff "
        + "and many others " * 50
        + "for invaluable assistance."
    )
    md = "## References\n\n" + body + "\n"
    s = p.score_references(md, journal_slug="aps-prl")
    assert s.entry_count == 8
    assert s.longest_over_median > 5.0
    assert s.score < 0.85


def test_score_missing_section():
    """No References heading and no trailing numbered cluster: score 0."""
    md = "# Just title\n\nA short paper with no bibliography at all.\n"
    s = p.score_references(md, journal_slug=None)
    assert s.section_count == 0
    assert s.entry_count == 0
    assert s.score == 0.0


def test_score_garbled_no_year_hits():
    """The Elliott / young failure mode: refs section is technically
    present but entries are truncated / garbled so years are missing."""
    md = dedent("""
        ## References

        - 1. Author A,
        - 2. Author B,
        - 3. Author C,
        - 4. Author D,
    """).strip()
    s = p.score_references(md, journal_slug="science")
    assert s.year_hit_ratio == 0.0
    # Numbered+continuous still helps but missing years caps the score.
    assert s.score < 0.7


def test_score_carries_journal_slug():
    s = p.score_references(_GOOD_NUMBERED, journal_slug="elsevier-icarus")
    assert s.journal_slug == "elsevier-icarus"
    s2 = p.score_references(_GOOD_NUMBERED, journal_slug=None)
    assert s2.journal_slug is None


def test_score_is_deterministic():
    """Same input -> same score. Phase 1 must be stable across re-runs."""
    s1 = p.score_references(_GOOD_NUMBERED, journal_slug="aps-prl")
    s2 = p.score_references(_GOOD_NUMBERED, journal_slug="aps-prl")
    assert s1 == s2
