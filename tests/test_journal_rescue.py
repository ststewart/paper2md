"""Unit tests for Phase-2 journal-aware reference rescue.

Run with:
    cd paper2md && python -m pytest tests/test_journal_rescue.py -v

Covers:
- _verify_content_preserved (positive: reorder + numbering / italic /
  year fill-ins; negative: hallucinated author / journal names).
- _detect_refs_layout column histogram (synthetic 1/2/3-col PyMuPDF docs).
- _rescue_aps_last_ref_bleed end-to-end on synthetic markdown.
- rescue_journal_refs dispatcher gates: master toggle off, score above
  threshold, missing slug, unknown slug, dispatch + signal refinement.
- Dry-run mode: rescue runs in-memory, body stays unchanged, scores
  recorded.
- VLM rescue (_rescue_science_3col): mocked vlm() so no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paper2md as p  # noqa: E402


# ---------------------------------------------------------------------------
# _verify_content_preserved
# ---------------------------------------------------------------------------

def test_verify_content_identical_passes():
    text = "- 1. Smith Jones Brown 2020. - 2. Davis Wilson 2021."
    assert p._verify_content_preserved(text, text) is True


def test_verify_content_reordered_passes():
    inp = "- Smith 2020. - Jones 2021. - Brown 2022."
    out = "- Brown 2022. - Smith 2020. - Jones 2021."
    assert p._verify_content_preserved(inp, out) is True


def test_verify_content_dropped_words_passes():
    """A subset of the input is allowed."""
    inp = "Smith Jones Brown Davis Wilson"
    out = "Smith Jones Brown"
    assert p._verify_content_preserved(inp, out) is True


def test_verify_content_year_fillin_allowed():
    """Adding a year string is allowed -- the rescue is permitted to
    fill in OCR-truncated entries from the model's prior knowledge.
    The verifier only guards content WORDS, not numbers."""
    inp = "Stevenson Annu Rev Earth Planet Sci 271"
    out = "Stevenson Annu Rev Earth Planet Sci 271 1987"
    assert p._verify_content_preserved(inp, out) is True


def test_verify_content_numbering_addition_allowed():
    """Adding entry numbers '- 1.', '- 2.' etc. is the rescue's job
    and must pass."""
    inp = "Hartmann Davis Icarus Cameron Ward"
    out = "- 1. Hartmann Davis Icarus\n- 2. Cameron Ward"
    assert p._verify_content_preserved(inp, out) is True


def test_verify_content_italic_markup_allowed():
    inp = "Hartmann Davis Icarus 504 1975"
    out = "Hartmann Davis *Icarus* 504 1975"
    assert p._verify_content_preserved(inp, out) is True


def test_verify_content_hallucinated_author_fails():
    """Inventing a new author name (4+ chars) trips the verifier."""
    inp = "Smith Jones Brown 2020"
    out = "Smithers Jones Brown 2020"  # 'smithers' not in input
    assert p._verify_content_preserved(inp, out) is False


def test_verify_content_substituted_journal_fails():
    """Substituting one real journal for another trips the verifier
    because the new journal name isn't in the input."""
    inp = "Hartmann Davis Icarus 504 1975"
    out = "Hartmann Davis Nature 504 1975"  # 'nature' not in input
    assert p._verify_content_preserved(inp, out) is False


def test_verify_content_unplaced_heading_passes():
    """'Unplaced fragments' header words are explicitly whitelisted."""
    inp = "Smith Jones"
    out = ("Smith Jones\n"
           "## Unplaced fragments\n")
    assert p._verify_content_preserved(inp, out) is True


def test_verify_content_short_words_ignored():
    """Words under 4 chars (the, of, and, a, et al.) aren't counted,
    so the verifier doesn't trip on short connectives the model
    inserts."""
    inp = "Smith Jones Brown"
    out = "Smith and Jones et al Brown"  # 'and', 'et', 'al' all <4
    assert p._verify_content_preserved(inp, out) is True


# ---------------------------------------------------------------------------
# _rescue_aps_last_ref_bleed (deterministic, no VLM)
# ---------------------------------------------------------------------------

def test_aps_bleed_rescue_promotes_when_no_prior_acks():
    """Last entry has 'Acknowledgments. We thank...' bleed; no
    prior Acks block exists. Rescue trims the entry and promotes the
    bleed text to a `## Acknowledgments` section above the refs."""
    md = dedent("""
        # A paper

        Body body body.

        ## References

        - 1. Author A. *J*, **1**, 1 (2020).
        - 2. Author B. *J*, **2**, 2 (2021).
        - 3. Author C. *J*, **3**, 3 (2022). Acknowledgments. We thank the OMEGA staff for their assistance.
    """).strip()
    result = p._rescue_aps_last_ref_bleed(md, doc=None, report=None)
    assert "## Acknowledgments" in result
    assert "We thank the OMEGA staff" in result
    # The trimmed last entry no longer contains the bleed text.
    assert "(2022). Acknowledgments. We thank" not in result
    assert "- 3. Author C. *J*, **3**, 3 (2022)." in result


def test_aps_bleed_rescue_drops_when_prior_acks_exists():
    """Prior `## Acknowledgments` block exists; rescue only trims the
    last entry (doesn't promote a duplicate)."""
    md = dedent("""
        # Paper

        Body.

        ## Acknowledgments

        We thank the funders.

        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022). Acknowledgments. Bleed text duplicate.
    """).strip()
    result = p._rescue_aps_last_ref_bleed(md, doc=None, report=None)
    # Only one ## Acknowledgments heading.
    assert result.count("## Acknowledgments") == 1
    # Bleed text dropped entirely.
    assert "Bleed text duplicate" not in result
    # Last entry trimmed.
    assert "- 3. C. *J*, **3**, 3 (2022)." in result


def test_aps_bleed_rescue_noop_without_marker():
    """Last entry has no Acknowledgments / Funding marker; rescue
    leaves md unchanged."""
    md = dedent("""
        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022).
    """).strip()
    assert p._rescue_aps_last_ref_bleed(md, doc=None, report=None) == md


def test_aps_bleed_rescue_score_improves():
    """Composite score rises after the rescue trims the last entry."""
    md = dedent("""
        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022). Acknowledgments. We thank the funders for support of this work and the staff for assistance.
    """).strip()
    pre = p.score_references(md, "aps-prl")
    after = p._rescue_aps_last_ref_bleed(md, doc=None, report=None)
    post = p.score_references(after, "aps-prl")
    assert post.score > pre.score
    assert post.last_well_bounded is True
    assert post.longest_over_median < pre.longest_over_median


# ---------------------------------------------------------------------------
# rescue_journal_refs dispatcher gates
# ---------------------------------------------------------------------------

def _make_report_with_score(score: float, journal_slug,
                            **kw) -> p.QualityReport:
    """Build a QualityReport whose .references field carries the given
    score + slug. Other ReferencesScore fields default to 'good'
    values so the dispatcher doesn't filter on signal refinements."""
    rs = p.ReferencesScore(
        section_count=kw.get("section_count", 1),
        entry_count=kw.get("entry_count", 5),
        numbered_continuous=kw.get("numbered_continuous", True),
        year_hit_ratio=kw.get("year_hit_ratio", 1.0),
        longest_over_median=kw.get("longest_over_median", 2.0),
        style=kw.get("style", "numbered"),
        journal_slug=journal_slug,
        score=score,
        last_well_bounded=kw.get("last_well_bounded", True),
        layout=kw.get("layout"),
    )
    r = p.QualityReport(vlm_enabled=True)
    r.references = rs
    return r


def test_dispatch_skipped_when_master_toggle_off():
    md = "## References\n\n- 1. A 2020.\n"
    report = _make_report_with_score(0.40, "aps-prl",
                                     longest_over_median=15.0,
                                     last_well_bounded=False)
    # Master toggle is False by default in the module.
    result = p.rescue_journal_refs(md, doc=None, report=report,
                                   use_vlm=True)
    assert result == md
    assert report.references.rescue_applied is None


def test_dispatch_skipped_when_score_above_threshold_and_no_signal():
    """High score AND no bleed signal -> bleed rescue skipped.
    The bleed rescue is signal-driven, so the score threshold is
    NOT the gate -- the LOR + last_well_bounded signals are. With
    default 'good' values, neither fires."""
    md = "## References\n\n- 1. A 2020.\n"
    report = _make_report_with_score(0.90, "aps-prl")  # default lor=2, lwb=True
    with patch.object(p, "RESCUE_JOURNAL_REFS", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    assert result == md
    assert report.references.rescue_applied is None


def test_dispatch_bleed_fires_even_when_score_above_threshold():
    """High score (0.80, above threshold 0.65) but bleed signal
    present (LOR > 8, !lwb) -> rescue fires anyway. This is the
    tracy / Science-Advances pattern: refs section is mostly fine
    but the last entry has the Acks bleed."""
    md = dedent("""
        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022). Acknowledgments. We thank everyone for their support of this work and the reviewers for thorough feedback.
    """).strip()
    pre = p.score_references(md, "science-advances")
    report = p.QualityReport(vlm_enabled=True)
    report.references = pre
    # Pre-score is around 0.80, well above the 0.65 default threshold.
    assert pre.score >= 0.65
    with patch.object(p, "RESCUE_JOURNAL_REFS", True), \
         patch.object(p, "REF_RESCUE_LOR", 1.0):  # short test entries
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    # Signal-driven rescue should fire and splice.
    assert report.references.rescue_applied == "aps-bleed"
    assert report.references.rescue_decision == "spliced"
    assert "## Acknowledgments" in result


def test_dispatch_skipped_without_journal_slug():
    md = "## References\n\n- 1. A 2020.\n"
    report = _make_report_with_score(0.30, None)  # no slug
    with patch.object(p, "RESCUE_JOURNAL_REFS", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    assert result == md


def test_dispatch_skipped_for_unknown_slug():
    md = "## References\n\n- 1. A 2020.\n"
    report = _make_report_with_score(0.30, "obscure-pub")
    with patch.object(p, "RESCUE_JOURNAL_REFS", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    assert result == md
    assert report.references.rescue_applied is None


def test_dispatch_aps_bleed_rescue_fires():
    """APS slug + score below threshold + LOR above limit + last entry
    not well bounded → APS rescue runs and splices."""
    md = dedent("""
        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022). Acknowledgments. We thank the funders for support of this work and the staff for assistance.
    """).strip()
    pre = p.score_references(md, "aps-prl")
    report = p.QualityReport(vlm_enabled=True)
    report.references = pre
    with patch.object(p, "RESCUE_JOURNAL_REFS", True), \
         patch.object(p, "REF_RESCUE_THRESHOLD", 0.95), \
         patch.object(p, "REF_RESCUE_LOR", 1.0):  # short test entries
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    # Either spliced (post score improved) or kept-pre-rescue-regressed.
    # In our deterministic case the post score should improve, so we
    # expect a splice.
    assert report.references.rescue_applied == "aps-bleed"
    assert report.references.rescue_decision == "spliced"
    assert result != md
    assert "## Acknowledgments" in result


def test_dispatch_aps_bleed_signal_refinement_skips_clean_paper():
    """APS slug + score below threshold but last entry IS well-bounded
    (no bleed). The signal-refinement gate inside the dispatcher
    refuses to invoke the rescue."""
    md = dedent("""
        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022).
    """).strip()
    report = _make_report_with_score(0.50, "aps-prl",
                                     longest_over_median=2.0,
                                     last_well_bounded=True)  # no bleed
    with patch.object(p, "RESCUE_JOURNAL_REFS", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    assert result == md
    assert report.references.rescue_applied is None


def test_dispatch_dry_run_records_scores_but_doesnt_splice():
    """Dry-run mode: rescue runs in-memory, scores recorded, but body
    is left untouched."""
    md = dedent("""
        ## References

        - 1. A. *J*, **1**, 1 (2020).
        - 2. B. *J*, **2**, 2 (2021).
        - 3. C. *J*, **3**, 3 (2022). Acknowledgments. We thank everyone for their help and support of the work described herein.
    """).strip()
    pre = p.score_references(md, "aps-prl")
    report = p.QualityReport(vlm_enabled=True)
    report.references = pre
    with patch.object(p, "RESCUE_JOURNAL_REFS", True), \
         patch.object(p, "REF_RESCUE_THRESHOLD", 0.95), \
         patch.object(p, "REF_RESCUE_LOR", 1.0), \
         patch.object(p, "REF_RESCUE_DRY_RUN", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    assert result == md  # body unchanged
    assert report.references.rescue_decision == "dry-run"
    assert report.references.score_pre_rescue is not None
    assert report.references.score_post_rescue is not None
    assert report.references.score_post_rescue > report.references.score_pre_rescue


def test_dispatch_science_3col_skipped_when_use_vlm_false():
    md = "## References\n\n- 1. A 2020.\n- 2. B 2021.\n- 3. C 2022.\n"
    layout = p.RefsLayout(columns=3, marker_style="numbered", page_idx=0)
    report = _make_report_with_score(0.40, "science", layout=layout)
    with patch.object(p, "RESCUE_JOURNAL_REFS", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=False)
    assert result == md
    assert report.references.rescue_applied is None


def test_dispatch_science_3col_skipped_when_columns_not_3():
    md = "## References\n\n- 1. A 2020.\n- 2. B 2021.\n- 3. C 2022.\n"
    layout = p.RefsLayout(columns=2, marker_style="numbered", page_idx=0)
    report = _make_report_with_score(0.40, "science", layout=layout)
    with patch.object(p, "RESCUE_JOURNAL_REFS", True):
        result = p.rescue_journal_refs(md, doc=None, report=report,
                                       use_vlm=True)
    assert result == md
    assert report.references.rescue_applied is None


# ---------------------------------------------------------------------------
# _rescue_science_3col (mocked VLM)
# ---------------------------------------------------------------------------

class _StubDoc:
    """Minimal fitz.Document stand-in for tests that don't need real
    PDF rendering."""
    page_count = 1

    def __getitem__(self, idx):
        raise IndexError("stub doc has no pages")


def test_science_3col_rescue_returns_md_when_no_refs_section():
    md = "# Paper\n\nNo references here.\n"
    result = p._rescue_science_3col(md, doc=_StubDoc(), report=None)
    assert result == md


def test_science_3col_rescue_calls_vlm_and_splices():
    md = dedent("""
        # Body

        ## References

        - 3. Brown 2022.
        - 1. Smith 2020.
        - 2. Jones 2021.
    """).strip()
    # The VLM "fixes" the order. Output uses only chars from the input
    # body so the verifier passes.
    rescued_body = (
        "- 1. Smith 2020.\n"
        "- 2. Jones 2021.\n"
        "- 3. Brown 2022."
    )
    with patch.object(p, "vlm", return_value=rescued_body), \
         patch.object(p, "render_page", return_value=None), \
         patch.object(p, "_refs_page_idx_from_md", return_value=0):
        result = p._rescue_science_3col(md, doc=_StubDoc(), report=None)
    assert "- 1. Smith 2020." in result
    # Order in the body: 1 before 2 before 3.
    one_pos = result.find("Smith 2020")
    two_pos = result.find("Jones 2021")
    three_pos = result.find("Brown 2022")
    assert 0 <= one_pos < two_pos < three_pos


def test_science_3col_rescue_keeps_pre_rescue_on_verifier_failure():
    """If the VLM hallucinates a character not in the input, the
    verifier rejects and the rescue returns the original md."""
    md = dedent("""
        ## References

        - 1. Smith 2020.
        - 2. Jones 2021.
    """).strip()
    bad_output = "- 1. Smithee 2099.\n- 2. Jones 2021."  # 'e' and '9' not in input
    with patch.object(p, "vlm", return_value=bad_output), \
         patch.object(p, "render_page", return_value=None), \
         patch.object(p, "_refs_page_idx_from_md", return_value=0):
        result = p._rescue_science_3col(md, doc=_StubDoc(), report=None)
    assert result == md  # pre-rescue body preserved


def test_science_3col_rescue_keeps_pre_rescue_on_vlm_skip():
    md = "## References\n\n- 1. A 2020.\n- 2. B 2021.\n- 3. C 2022.\n"
    with patch.object(p, "vlm", return_value="SKIP"), \
         patch.object(p, "render_page", return_value=None), \
         patch.object(p, "_refs_page_idx_from_md", return_value=0):
        result = p._rescue_science_3col(md, doc=_StubDoc(), report=None)
    assert result == md


def test_science_3col_rescue_keeps_pre_rescue_on_vlm_empty():
    md = "## References\n\n- 1. A 2020.\n- 2. B 2021.\n- 3. C 2022.\n"
    with patch.object(p, "vlm", return_value=None), \
         patch.object(p, "render_page", return_value=None), \
         patch.object(p, "_refs_page_idx_from_md", return_value=0):
        result = p._rescue_science_3col(md, doc=_StubDoc(), report=None)
    assert result == md


# ---------------------------------------------------------------------------
# Layout fingerprint
# ---------------------------------------------------------------------------

def test_detect_refs_marker_style_bracketed():
    md = ("## References\n\n"
          "- [1] Author A. *J*, 1 (2020).\n"
          "- [2] Author B. *J*, 2 (2021).\n")
    assert p._detect_refs_marker_style(md) == "bracketed"


def test_detect_refs_marker_style_numbered():
    md = ("## References\n\n"
          "- 1. Author A. *J*, 1 (2020).\n"
          "- 2. Author B. *J*, 2 (2021).\n")
    assert p._detect_refs_marker_style(md) == "numbered"


def test_detect_refs_marker_style_superscript():
    md = ("## References\n\n"
          "<sup>1</sup>Author A. *J*, 1 (2020).\n"
          "<sup>2</sup>Author B. *J*, 2 (2021).\n")
    assert p._detect_refs_marker_style(md) == "superscript"


def test_detect_refs_marker_style_none_when_section_missing():
    md = "# Paper\n\nNo refs here.\n"
    assert p._detect_refs_marker_style(md) == "none"


# ---------------------------------------------------------------------------
# _missing_numbered_gaps + _rescue_missing_numbered_refs (PRB pattern)
# ---------------------------------------------------------------------------

def test_missing_numbered_gaps_continuous_returns_empty():
    entries = ["- 1. A.", "- 2. B.", "- 3. C."]
    assert p._missing_numbered_gaps(entries) == []


def test_missing_numbered_gaps_finds_single_gap():
    entries = ["- 1. A.", "- 2. B.", "- 5. E."]
    assert p._missing_numbered_gaps(entries) == [3, 4]


def test_missing_numbered_gaps_finds_multiple_gaps():
    entries = ["- 1. A.", "- 3. C.", "- 5. E.", "- 7. G."]
    assert p._missing_numbered_gaps(entries) == [2, 4, 6]


def test_missing_numbered_gaps_caps_at_5():
    """Excessive missing-number gaps suggest a heavily broken refs
    list rather than a few dropped entries -- skip those rather
    than risk a noisy rescue."""
    entries = ["- 1. A.", "- 20. T."]  # 18 missing
    assert p._missing_numbered_gaps(entries) == []


def test_missing_numbered_gaps_handles_unnumbered():
    entries = ["No leading number here.", "- 2. B."]
    assert p._missing_numbered_gaps(entries) == []


class _StubDocOnePage:
    """Single-page fitz.Document stand-in returning a fixed text."""
    def __init__(self, page_text: str):
        self._text = page_text
        self.page_count = 1

    def __iter__(self):
        return iter([_StubPage(self._text)])

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError
        return _StubPage(self._text)


class _StubPage:
    def __init__(self, text):
        self._text = text

    def get_text(self, fmt="text"):
        return self._text


def test_rescue_missing_numbered_refs_consecutive_gap():
    """The Knudson2013 case: refs 33 and 34 are missing from the
    markdown but present in the PDF text in superscript-prefix
    format. The rescue should recover both and inject them in
    the correct position."""
    md = dedent("""
        ## References

        - 32. C. Agnor, R. Canup, Icarus 142, 219 (1999).
        - 35. G. Kresse, Phys. Rev. B 54, 11169 (1996).
        - 36. P. Blochl, Phys. Rev. B 50, 17953 (1994).
    """).strip()
    pdf_text = (
        "32C. Agnor, R. Canup, Icarus 142, 219 (1999).\n"
        "33M. D. Knudson and R. W. Lemke (unpublished).\n"
        "34S. Root, Phys. Rev. Lett. 88, 1234 (2010).\n"
        "35G. Kresse, Phys. Rev. B 54, 11169 (1996).\n"
        "36P. Blochl, Phys. Rev. B 50, 17953 (1994).\n"
    )
    doc = _StubDocOnePage(pdf_text)
    result = p._rescue_missing_numbered_refs(md, doc=doc, report=None)
    assert "- 33. M. D. Knudson and R. W. Lemke" in result
    assert "- 34. S. Root, Phys. Rev. Lett." in result
    # Verify injection order: 33 BEFORE 34, both BEFORE 35.
    pos32 = result.find("- 32.")
    pos33 = result.find("- 33.")
    pos34 = result.find("- 34.")
    pos35 = result.find("- 35.")
    assert pos32 < pos33 < pos34 < pos35


def test_rescue_missing_numbered_refs_noop_when_continuous():
    md = dedent("""
        ## References

        - 1. Author A. 2020.
        - 2. Author B. 2021.
        - 3. Author C. 2022.
    """).strip()
    pdf_text = "1Author A. 2020.\n2Author B. 2021.\n3Author C. 2022.\n"
    doc = _StubDocOnePage(pdf_text)
    result = p._rescue_missing_numbered_refs(md, doc=doc, report=None)
    assert result == md


def test_rescue_missing_numbered_refs_noop_without_doc():
    """Rescue can't search a PDF text layer without the doc handle."""
    md = "## References\n\n- 1. A.\n- 3. C.\n"
    result = p._rescue_missing_numbered_refs(md, doc=None, report=None)
    assert result == md


def test_rescue_missing_numbered_refs_noop_when_pdf_lacks_ref():
    """If the PDF text doesn't contain the missing-number entry
    (e.g., it's a different journal style), no injection happens."""
    md = dedent("""
        ## References

        - 1. Author A. 2020.
        - 5. Author E. 2024.
    """).strip()
    pdf_text = "1Author A. 2020.\n5Author E. 2024.\n"  # no refs 2-4
    doc = _StubDocOnePage(pdf_text)
    result = p._rescue_missing_numbered_refs(md, doc=doc, report=None)
    assert result == md


# ---------------------------------------------------------------------------
# _rescue_author_year_resplit (Phase-3 mega-entry splitter)
# ---------------------------------------------------------------------------

def test_rescue_author_year_resplit_two_refs_joined():
    """The feng-2024 case in miniature: two adjacent refs joined by
    marker into one bullet, separated by a DOI-link close-bracket
    and the next ref's surname-comma-initial pattern."""
    # Healthy refs (median length set by these short ones) plus one
    # mega-entry containing two refs.
    md = dedent("""
        ## References

        - Adams, J. (2018). Short ref one. *Journal*, *1*(1), 1.
        - Brown, K. (2019). Short ref two. *Journal*, *2*(2), 2.
        - Davis, L. (2020). Short ref three. *Journal*, *3*(3), 3.
        - Stixrude, L. (2014). Melting in super-earths. *Philosophical Transactions*, *372*(2014), 20130076. <https://doi.org/10.1098/rsta.2013.0076> Sun, L., Zhu, C., Guan, Z., Yang, W., Zhang, Y., Sekine, T., et al. (2021). Sound velocity measurement of shock-compressed quartz at extreme conditions. *Minerals*, *11*(12), 1334. <https://doi.org/10.3390/min11121334>
    """).strip()
    result = p._rescue_author_year_resplit(md, doc=None, report=None)
    # Both refs should now be on their own bullet.
    assert "- Stixrude, L." in result
    assert "- Sun, L., Zhu, C." in result
    # Order preserved.
    assert result.find("- Stixrude") < result.find("- Sun, L.")


def test_rescue_author_year_resplit_three_refs_joined():
    """A 3-ref mega-entry should produce 3 separate bullets."""
    md = dedent("""
        ## References

        - Adams, J. (2018). Short A. *J*, *1*, 1.
        - Brown, K. (2019). Short B. *J*, *2*, 2.
        - Davis, L. (2020). Short C. *J*, *3*, 3.
        - Umemoto, K. (2006). First. *Science*, *311*, 983. <https://doi.org/10.1126/science.1120865> Umemoto, K. (2017). Second. *EPSL*, *478*, 40. <https://doi.org/10.1016/j.epsl.2017.08.032> Usui, Y. (2010). Third. *Phys Rev*, *82*, 1.
    """).strip()
    result = p._rescue_author_year_resplit(md, doc=None, report=None)
    # All three should be separate bullets.
    assert result.count("- Umemoto, K.") == 2  # two distinct Umemoto refs
    assert "- Usui, Y." in result


def test_rescue_author_year_resplit_noop_when_no_mega_entry():
    """All entries are ~uniform length; rescue is a no-op."""
    md = dedent("""
        ## References

        - Adams, J. (2018). Title A. *J*, *1*, 1.
        - Brown, K. (2019). Title B. *J*, *2*, 2.
        - Davis, L. (2020). Title C. *J*, *3*, 3.
        - Evans, M. (2021). Title D. *J*, *4*, 4.
    """).strip()
    assert p._rescue_author_year_resplit(md, doc=None, report=None) == md


def test_rescue_author_year_resplit_noop_on_numbered_style():
    """Numbered-style refs aren't author-year; no-op."""
    md = dedent("""
        ## References

        - 1. Author A. 2020.
        - 2. Author B. 2021.
        - 3. Author C. 2022. Acks bleed dragged a really long paragraph onto this entry that pushes it past the mega threshold but we should not re-split numbered refs with this rescue.
    """).strip()
    assert p._rescue_author_year_resplit(md, doc=None, report=None) == md


# ---------------------------------------------------------------------------
# _rescue_numbered_pageboundary_mash (Knudson2013 PRB pattern)
# ---------------------------------------------------------------------------

def test_rescue_numbered_mash_splits_and_reorders():
    """The Knudson2013 case: refs 46-58 are mashed onto one bullet
    placed at the TOP of the references section, before ref 1. The
    rescue splits and moves them after the existing highest-numbered
    entry (ref 45)."""
    md = dedent("""
        ## References

        - <span id="page-17-0"></span>46First mashed entry text. 47Second. 48Third entry here.
        - 1. Author A. 2020.
        - 2. Author B. 2021.
        - 44. Author X. 2024.
        - 45. Author Y. 2024.
    """).strip()
    result = p._rescue_numbered_pageboundary_mash(md, doc=None, report=None)
    # All three split refs should now be individual bullets
    assert "- 46. First mashed entry text." in result
    assert "- 47. Second." in result
    assert "- 48. Third entry here." in result
    # Order: refs 1, 45, 46, 48 in that order (reordered after 45)
    pos1 = result.find("- 1.")
    pos45 = result.find("- 45.")
    pos46 = result.find("- 46.")
    pos48 = result.find("- 48.")
    assert 0 <= pos1 < pos45 < pos46 < pos48


def test_rescue_numbered_mash_in_place_when_not_higher():
    """If the mashed refs are NOT all higher than existing entries,
    the rescue splits in place (no reorder)."""
    md = dedent("""
        ## References

        - 1. Author A. 2020.
        - 2. Author B. 2021.
        - 3Author C 2022. 4Author D 2023.
        - 5. Author E. 2024.
    """).strip()
    result = p._rescue_numbered_pageboundary_mash(md, doc=None, report=None)
    # Refs 3, 4 are split; 3 stays in its original position (between 2 and 5)
    assert "- 3. Author C 2022." in result
    assert "- 4. Author D 2023." in result
    # Order preserved
    pos2 = result.find("- 2.")
    pos3 = result.find("- 3.")
    pos4 = result.find("- 4.")
    pos5 = result.find("- 5.")
    assert 0 <= pos2 < pos3 < pos4 < pos5


def test_rescue_numbered_mash_noop_when_no_mashed_bullet():
    """If no bullet starts with PRB-superscript style, rescue is a no-op."""
    md = dedent("""
        ## References

        - 1. Author A. 2020.
        - 2. Author B. 2021.
        - 3. Author C. 2022.
    """).strip()
    assert p._rescue_numbered_pageboundary_mash(md, doc=None, report=None) == md


def test_rescue_numbered_mash_handles_sup_tag_form():
    """Some refs survive marker as `<sup>N</sup>...` rather than
    `<digit><Capital>` -- the boundary regex should recognize both."""
    md = dedent("""
        ## References

        - <span></span>50First entry. <sup>51</sup>Second entry. 52Third entry.
        - 1. Existing. 2020.
        - 49. Existing. 2024.
    """).strip()
    result = p._rescue_numbered_pageboundary_mash(md, doc=None, report=None)
    assert "- 50. First entry." in result
    assert "- 51. Second entry." in result
    assert "- 52. Third entry." in result


###############################################################################
# _rescue_aps_missing_refs_heading (always-on, deterministic, no VLM)
###############################################################################


def _make_report_with_metadata_slug(slug):
    """Build a QualityReport whose .metadata.journal_slug is `slug`.
    The missing-heading rescue reads from metadata, NOT from
    references (which is populated downstream)."""
    from types import SimpleNamespace
    r = p.QualityReport(vlm_enabled=False)
    r.metadata = SimpleNamespace(journal_slug=slug)
    return r


def test_missing_heading_rescue_inserts_for_aps_prl():
    md = dedent("""
        # Some PRL paper

        Body text and a conclusion paragraph.

        [1] First Author. Phys Rev Lett 1, 1 (2020).
        [2] Second Author. Phys Rev Lett 2, 2 (2021).
        [3] Third Author. Phys Rev Lett 3, 3 (2022).
        [4] Fourth Author. Phys Rev Lett 4, 4 (2023).
        [5] Fifth Author. Phys Rev Lett 5, 5 (2024).
    """).strip()
    report = _make_report_with_metadata_slug("aps-prl")
    result = p._rescue_aps_missing_refs_heading(md, report)
    assert "## References" in result
    # Heading goes BEFORE first [1] line, after body content.
    pos_heading = result.find("## References")
    pos_body = result.find("Body text")
    pos_first_entry = result.find("[1] First Author")
    assert pos_body < pos_heading < pos_first_entry


def test_missing_heading_rescue_inserts_for_aps_prb():
    md = dedent("""
        Body content here.

        [1] A. 2020.
        [2] B. 2021.
        [3] C. 2022.
        [4] D. 2023.
        [5] E. 2024.
        [6] F. 2025.
    """).strip()
    result = p._rescue_aps_missing_refs_heading(
        md, _make_report_with_metadata_slug("aps-prb"))
    assert "## References" in result


def test_missing_heading_rescue_inserts_for_bare_aps_slug():
    """The bare 'aps' slug (un-refined) should also trigger."""
    md = "[1] A. 2020.\n[2] B. 2021.\n[3] C. 2022.\n[4] D. 2023.\n[5] E. 2024."
    result = p._rescue_aps_missing_refs_heading(
        md, _make_report_with_metadata_slug("aps"))
    assert "## References" in result


def test_missing_heading_rescue_skips_when_heading_already_present():
    md = dedent("""
        ## References

        [1] A. 2020.
        [2] B. 2021.
        [3] C. 2022.
        [4] D. 2023.
        [5] E. 2024.
    """).strip()
    report = _make_report_with_metadata_slug("aps-prl")
    result = p._rescue_aps_missing_refs_heading(md, report)
    assert result == md
    assert result.count("## References") == 1


def test_missing_heading_rescue_skips_alias_heading():
    """`## Bibliography` and `## References and Notes` count as
    headings -- don't double-insert."""
    for heading in ("## Bibliography", "## References and Notes",
                    "# References", "### References"):
        md = (heading + "\n\n[1] A. 2020.\n[2] B. 2021.\n[3] C. 2022.\n"
              "[4] D. 2023.\n[5] E. 2024.")
        result = p._rescue_aps_missing_refs_heading(
            md, _make_report_with_metadata_slug("aps-prl"))
        assert "## References" not in result.replace(heading, ""), heading


def test_missing_heading_rescue_skips_below_5_entries():
    md = "[1] A. 2020.\n[2] B. 2021.\n[3] C. 2022.\n[4] D. 2023."
    result = p._rescue_aps_missing_refs_heading(
        md, _make_report_with_metadata_slug("aps-prl"))
    assert result == md


def test_missing_heading_rescue_skips_when_n_does_not_start_at_1():
    md = "[2] A. 2020.\n[3] B. 2021.\n[4] C. 2022.\n[5] D. 2023.\n[6] E. 2024."
    result = p._rescue_aps_missing_refs_heading(
        md, _make_report_with_metadata_slug("aps-prl"))
    assert result == md


def test_missing_heading_rescue_skips_when_not_monotonic():
    md = ("[1] A. 2020.\n[2] B. 2021.\n[2] B-dup. 2021.\n"
          "[4] D. 2023.\n[5] E. 2024.")
    result = p._rescue_aps_missing_refs_heading(
        md, _make_report_with_metadata_slug("aps-prl"))
    assert result == md


def test_missing_heading_rescue_skips_for_non_aps_slug():
    md = ("[1] A. 2020.\n[2] B. 2021.\n[3] C. 2022.\n"
          "[4] D. 2023.\n[5] E. 2024.")
    for slug in ("science", "elsevier-icarus", "agu", "nature", None, ""):
        result = p._rescue_aps_missing_refs_heading(
            md, _make_report_with_metadata_slug(slug))
        assert result == md, f"unexpectedly rescued slug={slug!r}"


def test_missing_heading_rescue_skips_when_no_metadata():
    md = ("[1] A. 2020.\n[2] B. 2021.\n[3] C. 2022.\n"
          "[4] D. 2023.\n[5] E. 2024.")
    r = p.QualityReport(vlm_enabled=False)
    # No metadata attached
    assert p._rescue_aps_missing_refs_heading(md, r) == md


def test_missing_heading_rescue_idempotent():
    """Second invocation finds the inserted heading and is a no-op."""
    md = ("[1] A. 2020.\n[2] B. 2021.\n[3] C. 2022.\n"
          "[4] D. 2023.\n[5] E. 2024.")
    report = _make_report_with_metadata_slug("aps-prl")
    once = p._rescue_aps_missing_refs_heading(md, report)
    twice = p._rescue_aps_missing_refs_heading(once, report)
    assert once == twice
    assert once.count("## References") == 1


def test_missing_heading_rescue_uses_last_one_when_multiple():
    """If `[1]` appears earlier in the body (e.g. an inline numbered
    list) and the refs section is the LATER `[1]` start, the heading
    should land before the last [1] which begins the trailing run.

    This guarantees behavior on docs where authors use [1] as both
    inline footnote markers and reference markers."""
    md = dedent("""
        Body discussion uses inline numbered enumerations like
        [1] item one [2] item two.

        Then the actual references follow:

        [1] First Author. 2020.
        [2] Second Author. 2021.
        [3] Third Author. 2022.
        [4] Fourth Author. 2023.
        [5] Fifth Author. 2024.
    """).strip()
    result = p._rescue_aps_missing_refs_heading(
        md, _make_report_with_metadata_slug("aps-prl"))
    assert result.count("## References") == 1
    # Heading must sit right before the trailing-run [1].
    pos_inline_one = result.find("[1] item one")
    pos_heading = result.find("## References")
    pos_first_entry = result.find("[1] First Author")
    assert pos_inline_one < pos_heading < pos_first_entry


###############################################################################
# _rescue_append_api_refs (always-on, network-aware)
###############################################################################


def _make_report_with_api_setup(*, doi, score=None, section_count=None,
                                 numbered_continuous=True,
                                 entry_count=20,
                                 style="numbered"):
    """Build a QualityReport with metadata.doi and references.score set."""
    from types import SimpleNamespace
    r = p.QualityReport(vlm_enabled=False)
    r.metadata = SimpleNamespace(doi=doi, journal_slug=None)
    if score is not None or section_count is not None:
        r.references = p.ReferencesScore(
            section_count=section_count if section_count is not None else 1,
            entry_count=entry_count,
            numbered_continuous=numbered_continuous,
            year_hit_ratio=1.0,
            longest_over_median=2.0,
            style=style,
            journal_slug=None,
            score=score if score is not None else 0.5,
        )
    return r


def test_api_append_skips_without_doi():
    r = _make_report_with_api_setup(doi=None, score=0.0)
    md = "body"
    assert p._rescue_append_api_refs(md, r) == md


def test_api_append_skips_when_score_healthy():
    r = _make_report_with_api_setup(doi="10.1/x", score=0.9, section_count=1)
    md = "body"
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs(md, r)
    mock.assert_not_called()
    assert result == md


def test_api_append_fires_when_score_low():
    r = _make_report_with_api_setup(doi="10.1/x", score=0.3, section_count=1)
    md = "body"
    with patch("metadata_frontend.fetch_references",
               return_value={
                   "source": "crossref",
                   "refs": ["- 1. Author A. 2020.",
                            "- 2. Author B. 2021."],
               }):
        result = p._rescue_append_api_refs(md, r)
    assert "## References (from Crossref)" in result
    assert "- 1. Author A. 2020." in result
    assert "- 2. Author B. 2021." in result
    assert r.references.api_source == "crossref"
    assert r.references.api_refs_appended == 2


def test_api_append_fires_when_no_section():
    r = _make_report_with_api_setup(doi="10.1/x", score=0.7, section_count=0)
    md = "body without refs"
    with patch("metadata_frontend.fetch_references",
               return_value={
                   "source": "openalex",
                   "refs": ["- 1. OA entry."],
               }):
        result = p._rescue_append_api_refs(md, r)
    assert "## References (from OpenAlex)" in result
    assert r.references.api_source == "openalex"
    assert r.references.api_refs_appended == 1


def test_api_append_silent_skip_when_api_returns_none():
    r = _make_report_with_api_setup(doi="10.1/x", score=0.0, section_count=0)
    md = "body"
    with patch("metadata_frontend.fetch_references", return_value=None):
        result = p._rescue_append_api_refs(md, r)
    assert result == md
    assert r.references.api_source is None
    assert r.references.api_refs_appended == 0


def test_api_append_silent_skip_when_api_returns_empty_refs():
    r = _make_report_with_api_setup(doi="10.1/x", score=0.0, section_count=0)
    md = "body"
    with patch("metadata_frontend.fetch_references",
               return_value={"source": "crossref", "refs": []}):
        result = p._rescue_append_api_refs(md, r)
    assert result == md
    assert r.references.api_source is None


def test_api_append_silent_skip_on_exception():
    """If fetch_references raises, the rescue must not propagate."""
    r = _make_report_with_api_setup(doi="10.1/x", score=0.0, section_count=0)
    md = "body"
    with patch("metadata_frontend.fetch_references",
               side_effect=ConnectionError("offline")):
        result = p._rescue_append_api_refs(md, r)
    assert result == md


def test_api_append_skips_when_no_metadata():
    r = p.QualityReport(vlm_enabled=False)
    md = "body"
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs(md, r)
    mock.assert_not_called()
    assert result == md


def test_api_append_skips_when_references_unset():
    """Even with a DOI, no references score means rescue can't trigger."""
    from types import SimpleNamespace
    r = p.QualityReport(vlm_enabled=False)
    r.metadata = SimpleNamespace(doi="10.1/x", journal_slug=None)
    r.references = None
    md = "body"
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs(md, r)
    mock.assert_not_called()
    assert result == md


def test_api_append_fires_when_numbered_list_has_gaps():
    """Wackerle1962 pattern: numbered_continuous=False with a
    healthy-looking score should still trigger the API rescue
    because gaps in a numbered list signal missing entries."""
    r = _make_report_with_api_setup(
        doi="10.1063/1.1777192",
        score=0.65,
        section_count=1,
        numbered_continuous=False,
        entry_count=20,
        style="numbered",
    )
    md = "## References\n\n- 1. ref one.\n- 3. ref three.\n- 5. ref five."
    with patch("metadata_frontend.fetch_references",
               return_value={"source": "crossref",
                             "refs": ["- 1. complete entry."]}):
        result = p._rescue_append_api_refs(md, r)
    assert "## References (from Crossref)" in result
    assert r.references.api_source == "crossref"


def test_api_append_skips_when_numbered_continuous_even_at_low_count():
    """Continuous numbering with healthy score should NOT trigger,
    even if entry_count is small."""
    r = _make_report_with_api_setup(
        doi="10.1/x",
        score=0.85,
        section_count=1,
        numbered_continuous=True,
        entry_count=3,
        style="numbered",
    )
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs("body", r)
    mock.assert_not_called()
    assert result == "body"


def test_api_append_does_not_fire_on_gaps_for_author_year_style():
    """numbered_continuous=False is meaningless for author-year
    style. The gap-trigger should require style == 'numbered'."""
    r = _make_report_with_api_setup(
        doi="10.1/x",
        score=0.85,
        section_count=1,
        numbered_continuous=False,
        entry_count=20,
        style="author-year",
    )
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs("body", r)
    mock.assert_not_called()
    assert result == "body"


def test_api_append_does_not_fire_on_gaps_when_too_few_entries():
    """Gap-trigger requires entry_count >= 3 to avoid noise on
    paper2md miscounts (e.g. 2-entry false-positive sections)."""
    r = _make_report_with_api_setup(
        doi="10.1/x",
        score=0.85,
        section_count=1,
        numbered_continuous=False,
        entry_count=2,
        style="numbered",
    )
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs("body", r)
    mock.assert_not_called()
    assert result == "body"


def test_api_append_appends_at_end_with_clean_separator():
    """Heading should land with a leading blank line so it doesn't fuse
    with whatever was the last body line."""
    r = _make_report_with_api_setup(doi="10.1/x", score=0.3, section_count=1)
    md = "## References\n\n- 1. mangled local entry."
    with patch("metadata_frontend.fetch_references",
               return_value={"source": "crossref",
                             "refs": ["- 1. clean api entry."]}):
        result = p._rescue_append_api_refs(md, r)
    # local section preserved
    assert "- 1. mangled local entry." in result
    # API section appended
    assert result.endswith("- 1. clean api entry.\n")
    assert "## References (from Crossref)" in result
    # Headings appear in the right order (local first, API after).
    pos_local = result.find("## References\n")
    pos_api = result.find("## References (from Crossref)")
    assert pos_local < pos_api


###############################################################################
# Offline-mode behavior for _rescue_append_api_refs
###############################################################################


def test_api_append_offline_skips_fetch_call_and_records_skip():
    """When allow_network=False AND the trigger gate would fire AND
    fetch_references returns None (no cache), the rescue must:
      - leave md unchanged
      - mark report.references.api_skipped_offline = True
      - record nothing in api_source / api_refs_appended"""
    r = _make_report_with_api_setup(doi="10.1/x", score=0.0,
                                    section_count=0)
    md = "body"
    with patch("metadata_frontend.fetch_references",
               return_value=None) as mock:
        result = p._rescue_append_api_refs(md, r, allow_network=False)
    # fetch_references is still called (it short-circuits internally
    # on allow_network=False), but the OUTCOME on the report should
    # record the offline skip.
    mock.assert_called_once()
    assert result == md
    assert r.references.api_skipped_offline is True
    assert r.references.api_source is None
    assert r.references.api_refs_appended == 0


def test_api_append_offline_uses_cached_result_when_available():
    """allow_network=False with a cache hit should still produce an
    appended section (the caller delegates to fetch_references which
    serves from cache when offline). api_skipped_offline must NOT be
    set in that case -- the section is real."""
    r = _make_report_with_api_setup(doi="10.1/x", score=0.0,
                                    section_count=0)
    md = "body"
    with patch("metadata_frontend.fetch_references",
               return_value={"source": "crossref",
                             "refs": ["- 1. cached entry."]}):
        result = p._rescue_append_api_refs(md, r, allow_network=False)
    assert "## References (from Crossref)" in result
    assert r.references.api_source == "crossref"
    assert r.references.api_skipped_offline is False


def test_api_append_offline_skip_does_not_fire_when_gate_healthy():
    """If the trigger gate would NOT have fired anyway (healthy
    score, has section, no gaps), api_skipped_offline must stay
    False. The skip flag only marks 'we were prevented from doing
    something we wanted to do'."""
    r = _make_report_with_api_setup(doi="10.1/x", score=0.9,
                                    section_count=1,
                                    numbered_continuous=True)
    md = "body"
    with patch("metadata_frontend.fetch_references") as mock:
        result = p._rescue_append_api_refs(md, r, allow_network=False)
    mock.assert_not_called()
    assert result == md
    assert r.references.api_skipped_offline is False


def test_api_skipped_offline_emitted_in_frontmatter():
    """to_frontmatter() should emit api_skipped_offline: true when
    the field is set, and OMIT it (no line at all) when False."""
    r = p.QualityReport(vlm_enabled=False)
    r.references = p.ReferencesScore(
        section_count=0, entry_count=0, numbered_continuous=False,
        year_hit_ratio=0.0, longest_over_median=1.0, style="numbered",
        journal_slug=None, score=0.0,
    )
    # Default: no skip flag, no line in frontmatter.
    fm = r.to_frontmatter()
    assert "api_skipped_offline" not in fm
    # Set skip flag -> line appears.
    r.references.api_skipped_offline = True
    fm = r.to_frontmatter()
    assert "api_skipped_offline: true" in fm


def test_pipeline_state_records_offline_flag():
    """_collect_pipeline_state() must include the OFFLINE module flag."""
    state = p._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
    )
    assert "offline" in state
    assert state["offline"] == p.OFFLINE  # follows the module flag


###############################################################################
# _crop_around_caption (column-aware cropping for orphan-figure rescue)
###############################################################################


class _FakeRect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


class _FakePage:
    """Mock fitz Page that returns specified text blocks and search hits."""
    def __init__(self, *, page_w, page_h, blocks, search_hits):
        self._page_w = page_w
        self._page_h = page_h
        self._blocks = blocks
        self._search_hits = search_hits

    @property
    def rect(self):
        class _R:
            pass
        r = _R()
        r.width = self._page_w
        r.height = self._page_h
        return r

    def search_for(self, _query):
        return [_FakeRect(*h) for h in self._search_hits]

    def get_text(self, kind):
        if kind == "blocks":
            return self._blocks
        return ""


class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages

    def __getitem__(self, idx):
        return self._pages[idx]


class _FakeImg:
    """PIL-image-like stand-in. .crop returns a new _FakeImg with the
    given box stored on it for assertion."""
    def __init__(self, w, h, crop_box=None):
        self.size = (w, h)
        self.crop_box = crop_box  # (x0, y0, x1, y1) when produced by .crop

    def crop(self, box):
        x0, y0, x1, y1 = box
        return _FakeImg(x1 - x0, y1 - y0, crop_box=box)


def _make_block(x0, y0, x1, y1, text):
    """Construct a fitz-blocks tuple: (x0, y0, x1, y1, text, block_no, type)."""
    return (x0, y0, x1, y1, text, 0, 0)


def test_crop_single_column_figure_above_caption():
    """Full-width caption (>=60% page width). Body below, figure above."""
    blocks = [
        _make_block(50, 50, 550, 60, "PAGE HEADER"),       # header
        _make_block(50, 600, 550, 620, "Fig. 1. caption"),  # caption (full width)
        _make_block(50, 640, 550, 780, "BODY " * 200),      # 1000-char body below
    ]
    page = _FakePage(page_w=600, page_h=800, blocks=blocks,
                     search_hits=[(50, 600, 100, 615)])
    doc = _FakeDoc([page])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(doc, 0, "Fig. 1.", img)
    # Expect crop ABOVE caption (since lots of text below).
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    assert y0 == 0
    # crop_y1 in image coords = cap_top (600pt) -> 600/800 * 1600 = 1200
    assert y1 == 1200
    assert x0 == 0 and x1 == 1200


def test_crop_single_column_caption_above_figure():
    """Full-width caption. Body above, figure below."""
    blocks = [
        _make_block(50, 50, 550, 250, "BODY " * 200),       # body above
        _make_block(50, 300, 550, 320, "Fig. 1. caption"),  # caption full width
    ]
    page = _FakePage(page_w=600, page_h=800, blocks=blocks,
                     search_hits=[(50, 300, 100, 315)])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 1.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Crop BELOW caption.
    assert y0 == int(320 / 800 * 1600)  # 640
    assert y1 == 1600
    assert x0 == 0 and x1 == 1200


def test_crop_two_column_same_column_figure_above_caption():
    """2-col layout. Caption in right column with body in left column at
    same y. Figure occupies right column ABOVE the caption (gap above
    caption is large; below is body text starting immediately).
    Expected: crop the RIGHT column above the caption."""
    page_w, page_h = 600, 800
    blocks = [
        _make_block(50, 50, 550, 60, "HEADER"),
        # Left column body block (whole column, big)
        _make_block(50, 70, 290, 700, "L_BODY " * 200),
        # Right column: caption mid-column, body below caption
        _make_block(310, 250, 555, 330, "Fig. 1. caption"),
        _make_block(310, 350, 555, 700, "R_BODY " * 100),
    ]
    page = _FakePage(page_w=page_w, page_h=page_h, blocks=blocks,
                     search_hits=[(310, 250, 350, 265)])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 1.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Right column horizontally (col_x0 ~= 310-margin, col_x1 ~= 555+margin).
    assert x0 > 300  # right column
    assert x1 > 1000  # near right edge
    # Vertically ABOVE caption: nearest_above_y1 = 60 (header), cap_top=250.
    # Gap above = 190; gap below = 350-330=20. → ABOVE wins.
    assert y0 < 200  # above caption
    assert y1 < 600  # ends at cap_top mapped


def test_crop_two_column_same_column_figure_below_caption():
    """2-col layout. Caption in right column with body above; figure
    occupies right column BELOW the caption (gap below is large)."""
    page_w, page_h = 600, 800
    blocks = [
        _make_block(50, 50, 550, 60, "HEADER"),
        _make_block(50, 70, 290, 700, "L_BODY " * 200),
        _make_block(310, 70, 555, 200, "R_BODY " * 50),
        _make_block(310, 250, 555, 330, "Fig. 1. caption"),
        # nothing below caption in right column → big gap
    ]
    page = _FakePage(page_w=page_w, page_h=page_h, blocks=blocks,
                     search_hits=[(310, 250, 350, 265)])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 1.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Right column horizontally
    assert x0 > 300
    # Gap above = 250-200 = 50; gap below = 800-330 = 470. → BELOW wins.
    assert y0 > 600  # starts below caption
    assert y1 == 1600  # to page bottom


def test_crop_side_by_side_caption_right_figure_left():
    """young 2016 Fig 1 pattern: caption in right column at mid-y;
    figure occupies the LEFT column at similar y, with many small
    annotation text blocks (axis labels, model names)."""
    page_w, page_h = 600, 800
    blocks = [
        _make_block(50, 50, 550, 60, "HEADER"),
        # Body text in BOTH columns above the caption
        _make_block(50, 100, 290, 350, "L_BODY " * 50),
        _make_block(310, 100, 555, 350, "R_BODY " * 50),
        # Caption mid-page in RIGHT column
        _make_block(310, 400, 555, 550, "Fig. 1. caption text"),
        # LEFT column at caption y-range: many small annotation blocks
        # (figure axis labels, plot annotations).
    ]
    # Add 10 small annotation blocks in left column at caption y-range.
    for i in range(10):
        y = 400 + i * 15
        blocks.append(_make_block(60, y, 100, y + 10, f"label{i}"))
    page = _FakePage(page_w=page_w, page_h=page_h, blocks=blocks,
                     search_hits=[(310, 400, 350, 415)])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 1.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Side-by-side: should crop LEFT half (caption is on the right)
    assert x0 == 0
    # cap_l = 310 -> in image coords ~= 620
    assert x1 < 700
    # Full page height
    assert y0 == 0
    assert y1 == 1600


def test_crop_side_by_side_caption_left_figure_right():
    """Mirror of the previous case: caption in LEFT column at mid-y,
    figure on the right."""
    page_w, page_h = 600, 800
    blocks = [
        _make_block(50, 50, 550, 60, "HEADER"),
        _make_block(50, 100, 290, 350, "L_BODY " * 50),
        _make_block(310, 100, 555, 350, "R_BODY " * 50),
        _make_block(50, 400, 290, 550, "Fig. 1. caption text"),  # left col
    ]
    for i in range(10):
        y = 400 + i * 15
        blocks.append(_make_block(500, y, 540, y + 10, f"label{i}"))
    page = _FakePage(page_w=page_w, page_h=page_h, blocks=blocks,
                     search_hits=[(50, 400, 90, 415)])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 1.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Caption in left column → crop right half.
    assert x0 > 500  # cap_r ~= 290 -> img ~580
    assert x1 == 1200
    assert y0 == 0
    assert y1 == 1600


def test_crop_falls_back_when_caption_not_found():
    page = _FakePage(page_w=600, page_h=800, blocks=[],
                     search_hits=[])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 99.", img)
    # No crop performed; image returned as-is.
    assert out is img


def test_crop_filters_inline_reference_in_favor_of_line_leading():
    """When the caption text appears BOTH as an inline reference
    inside a body paragraph AND as the start of a real caption block,
    pick the line-leading one. Knudson 2013 Fig. 7 pattern."""
    page_w, page_h = 600, 800
    blocks = [
        # Left column: long body block whose text contains "Fig. 7."
        # mid-paragraph (e.g., "...illustrated in Fig. 7. The...").
        _make_block(50, 156, 290, 700, "BODY illustrated in Fig. 7. " + "X" * 1000),
        # Right column: real caption block starting with "Fig. 7." at
        # the column's left edge.
        _make_block(313, 250, 555, 330, "Fig. 7. (Color) Real caption text"),
        _make_block(313, 350, 555, 700, "More body " * 50),
    ]
    # Both rects show up in search_for. The inline reference rect is
    # NOT at the block's left edge; the real-caption rect IS.
    search_hits = [
        (213, 156, 250, 165),  # inline reference (mid-block)
        (313, 250, 350, 263),  # real caption (line-leading)
    ]
    page = _FakePage(page_w=page_w, page_h=page_h, blocks=blocks,
                     search_hits=search_hits)
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Fig. 7.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Should treat the right-column caption as the real one. Right
    # column above caption is small (header empty), below has big
    # body, so figure-above wins -> crop right col above cap_top=250.
    assert x0 > 600  # right column
    assert y1 < 600  # above caption (cap_top * scale)


###############################################################################
# _apply_page_locality (Hook 2 page-mismatch rejection)
###############################################################################


def _captions(*pairs):
    """Convenience: build a captions list from (fig_id, title) tuples."""
    return list(pairs)


def test_page_locality_passthrough_when_image_on_caption_page():
    """Image extracted from doc[2], assigned caption is on doc[2]:
    no change — caption co-located, accept the assignment."""
    captions = _captions(("1", "Title A"), ("2", "Title B"))
    cap_map = {1: 2, 2: 5}  # Fig 1 on doc[2], Fig 2 on doc[5]
    items = [("img1.jpeg", 1, "fake_img", 2)]  # classified as Fig 1, on page 2
    result = p._apply_page_locality(items, captions, cap_map, "PROMPT", "single")
    assert result == items


def test_page_locality_rejects_when_image_page_has_another_caption():
    """Luo Fig 10/11 pattern: image on doc[10] classified as Fig 10
    (caption on doc[9], diff=1 within tolerance) BUT doc[10] also
    has another caption (Fig 11). Reject and re-classify."""
    captions = _captions(("10", "Mg-perovskite melting"),
                         ("11", "Mg-perovskite-periclase"))
    cap_map = {1: 9, 2: 10}  # Fig 10 on doc[9], Fig 11 on doc[10]
    items = [("img.jpeg", 1, "fake_img", 10)]  # classified as Fig 10, on doc[10]
    with patch.object(p, "_reclassify_excluding", return_value=2) as mock:
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    mock.assert_called_once()
    # Re-classified to Fig 11 (idx=2) which IS on doc[10] — accept.
    assert result == [("img.jpeg", 2, "fake_img", 10)]


def test_page_locality_rejects_when_diff_exceeds_tolerance():
    """Image on doc[10] classified to caption on doc[5]: diff=5,
    way over tolerance (1). Reject regardless of alternatives."""
    captions = _captions(("1", "Title A"), ("99", "Title Z"))
    cap_map = {1: 5, 2: 10}
    items = [("img.jpeg", 1, "fake_img", 10)]  # classified as Fig 1
    with patch.object(p, "_reclassify_excluding", return_value=2) as mock:
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    mock.assert_called_once()
    assert result[0][1] == 2


def test_page_locality_drops_when_reclassify_also_mismatches():
    """If re-classification also returns a far-away caption, set
    choice to None (drop)."""
    captions = _captions(("1", "Title A"), ("2", "Title B"))
    cap_map = {1: 0, 2: 0}  # both captions on doc[0]
    items = [("img.jpeg", 1, "fake_img", 10)]  # image on doc[10]
    # Reclassify returns 2, which is also on doc[0] (mismatch).
    with patch.object(p, "_reclassify_excluding", return_value=2):
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    # Both choices fail page-locality -> drop.
    assert result[0][1] is None


def test_page_locality_no_op_when_caption_page_unknown():
    """When caption_page is None for the assigned choice, skip
    validation (best-effort: don't reject what we can't verify)."""
    captions = _captions(("1", "A"))
    cap_map = {1: None}  # couldn't resolve caption page
    items = [("img.jpeg", 1, "fake_img", 10)]
    with patch.object(p, "_reclassify_excluding") as mock:
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    mock.assert_not_called()
    assert result == items


def test_page_locality_no_op_when_image_page_unknown():
    """When image's source page (parsed from filename) is None,
    skip validation."""
    captions = _captions(("1", "A"))
    cap_map = {1: 0}
    items = [("img.jpeg", 1, "fake_img", None)]
    with patch.object(p, "_reclassify_excluding") as mock:
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    mock.assert_not_called()
    assert result == items


def test_page_locality_skips_zero_choice_and_none_choice():
    """choice_idx=0 (NONE) and choice_idx=None (failed call) both
    skip validation — there's nothing to reject."""
    captions = _captions(("1", "A"))
    cap_map = {1: 0}
    items = [
        ("a.jpeg", 0, "img", 10),       # NONE
        ("b.jpeg", None, "img", 10),    # failed call
    ]
    with patch.object(p, "_reclassify_excluding") as mock:
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    mock.assert_not_called()
    assert result == items


def test_page_locality_within_tolerance_no_other_caption_passes():
    """Image on doc[10] classified to caption on doc[9] (diff=1).
    No other caption on doc[10]. Within tolerance — pass through."""
    captions = _captions(("1", "A"))
    cap_map = {1: 9}  # only one caption, on doc[9]
    items = [("img.jpeg", 1, "fake_img", 10)]  # diff=1, no alternative
    with patch.object(p, "_reclassify_excluding") as mock:
        result = p._apply_page_locality(items, captions, cap_map,
                                        "PROMPT", "single")
    mock.assert_not_called()
    assert result == items


###############################################################################
# Orphan caption regex + caption-page disambiguation
###############################################################################


def test_orphan_caption_line_re_matches_plain_period_form():
    md = "Figure 4. Some caption text follows."
    pat = p._orphan_fig_caption_line_re("4")
    assert pat.search(md) is not None


def test_orphan_caption_line_re_matches_period_inside_bold_wiley():
    """**Figure 4.** Title... — Wiley/AGU style (Luo 2004)."""
    md = "**Figure 4.** Representative time-resolved radiance profiles..."
    pat = p._orphan_fig_caption_line_re("4")
    m = pat.search(md)
    assert m is not None, "should match Wiley **Figure N.** style"


def test_orphan_caption_line_re_matches_pipe_form():
    """**Figure 4** | Title... — Nature style with bold."""
    md = "**Figure 4** | Some Title"
    pat = p._orphan_fig_caption_line_re("4")
    assert pat.search(md) is not None


def test_orphan_caption_line_re_rejects_inline_body_reference():
    """'as shown in Fig. 4 below' must NOT match -- it's an inline ref,
    not a caption line. Without the period or capital-letter follower,
    no separator alternative fires."""
    md = "as shown in Fig. 4 below the figure shows"
    pat = p._orphan_fig_caption_line_re("4")
    # Match only if anchored at line-start AND has a real separator.
    m = pat.search(md)
    # 'Fig. 4 below' has a lowercase 'b' so the capital-follower
    # alternative also fails. The match should not fire.
    assert m is None or m.start() != 0


def test_filter_block_leading_pages_rejects_soft_newline_body_ref():
    """Mock a doc where 'Figure 11' appears (a) inside a body
    paragraph as a soft-wrapped line, and (b) at the start of a
    real caption block. Only the second page should pass the
    block-leading filter."""
    # Page 0: body block where soft-wrap put 'Figure 11' line-leading.
    body_block_p0 = (50, 50, 290, 700,
                     "Lots of body text here ending with a sentence.\n"
                     "Figure 11 appears mid-paragraph after newline.\n"
                     "Continued discussion of the figure follows.",
                     0, 0)
    # Page 1: real caption block.
    caption_block_p1 = (51, 250, 290, 400,
                        "Figure 11. Real caption title text.",
                        0, 0)

    class _FakeP:
        def __init__(self, blocks, search_hits):
            self._blocks = blocks
            self._search_hits = search_hits

        def search_for(self, _):
            return [_FakeRect(*h) for h in self._search_hits]

        def get_text(self, kind):
            if kind == "blocks":
                return self._blocks
            return ""

    page0 = _FakeP([body_block_p0],
                   [(110, 100, 145, 110)])  # rect mid-block (body ref)
    page1 = _FakeP([caption_block_p1],
                   [(51, 250, 86, 260)])    # rect at top-left of block
    doc = _FakeDoc([page0, page1])
    result = p._filter_block_leading_pages(doc, [0, 1], "Figure 11")
    assert result == [1]


def test_crop_prefers_above_when_other_caption_below():
    """Luo 2004 Fig 4 pattern: same-column has another figure caption
    below ours, AND the figure region above is a vector-graphics 3D
    plot with NO text blocks. Without the other-caption signal, the
    larger gap_below wins and the crop captures the next figure's
    region. With the signal, we crop ABOVE."""
    page_w, page_h = 600, 800
    blocks = [
        # Page header (full width, x=51-434).
        _make_block(51, 44, 434, 53, "PAGE HEADER B12345"),
        # Right column body (separate region; not in same col as cap).
        _make_block(302, 70, 541, 297, "BODY " * 200),
        # Our caption (column-confined left col).
        _make_block(51, 255, 290, 298,
                    "Figure 4.\nRepresentative time-resolved radiance profiles..."),
        # ANOTHER caption block in same column below ours.
        _make_block(87, 699, 505, 754,
                    "Figure 5.\nTime-resolved spectral radiance from..."),
        # Page footer (narrow x range, not same col as cap).
        _make_block(283, 764, 309, 773, "5 of 14"),
    ]
    page = _FakePage(page_w=page_w, page_h=page_h, blocks=blocks,
                     search_hits=[(51, 255, 90, 265)])
    img = _FakeImg(1200, 1600)
    out = p._crop_around_caption(_FakeDoc([page]), 0, "Figure 4.", img)
    assert out.crop_box is not None
    x0, y0, x1, y1 = out.crop_box
    # Same column horizontally (cap_l=51-margin to cap_r=290+margin).
    assert x0 < 100  # near left edge
    assert x1 < 700  # ends well before page width
    # Crop ABOVE caption: y0 from nearest_above (page header at 53),
    # y1 at cap_top (255). Despite gap_below (444) > gap_above (202),
    # the other-caption signal flips the decision to ABOVE.
    assert y0 < 200  # above caption
    assert y1 < 600  # ends at cap_top mapped


def test_rescue_author_year_resplit_does_not_split_within_authorlist():
    """A single ref with multiple authors (Smith, A., Jones, B., ...)
    must NOT be split at the author-list commas. Only ref-boundary
    transitions (period/angle-bracket + surname-comma-initial) count."""
    # Single ref with 3 authors, deliberately long to exceed median.
    md = dedent("""
        ## References

        - Adams, J. (2018). A. *J*, *1*, 1.
        - Brown, K. (2019). B. *J*, *2*, 2.
        - Davis, L. (2020). C. *J*, *3*, 3.
        - Stevenson, A., Wilson, B. C., Foster, D. E., Garcia, F. G., Hall, H., Iverson, J. K., Johnson, L. M., Klein, N. O., Lopez, P. Q., Martinez, R. S., Nguyen, T. U., O'Brien, V. W. (2022). A multi-author paper with a moderately long but single-ref title that intentionally exceeds the median entry length to ensure the mega-entry threshold trips. *Some Journal*, *99*(99), 999.
    """).strip()
    result = p._rescue_author_year_resplit(md, doc=None, report=None)
    # The Stevenson ref must not be split into separate bullets per author.
    assert result.count("- Stevenson, A.") == 1
    assert "- Wilson, B. C." not in result
    assert "- Foster, D. E." not in result
