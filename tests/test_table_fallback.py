"""Unit tests for the page-image fallback in paper2md.process_tables.

When PyMuPDF (and optionally TATR) fails to locate a markdown table, the
fallback now searches for the table's caption ("Table II") in PyMuPDF's
per-page text, renders that page, and asks the VLM to extract the named
table from the image. These tests cover the three meaningful branches:

  1. caption found on a page -> page-image VLM call -> sidecar written
  2. no caption found near the table -> no VLM call -> no sidecar
  3. caption found, VLM returns SKIP -> no sidecar, marker markdown kept
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paper2md  # noqa: E402


# ---------------------------------------------------------------------------
# Caption extraction from markdown context
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("md_excerpt, expected", [
    # Caption directly above the table (Nature-style)
    ("Table II.\n| a | b |\n|---|---|\n| 1 | 2 |\n",
     "Table II"),
    # Caption directly below the table (Elsevier-style)
    ("| a | b |\n|---|---|\n| 1 | 2 |\nTable 3. Some title\n",
     "Table 3"),
    # Caption embedded with surrounding body text above
    ("Earlier sentence. Table IV. Description follows.\n\n"
     "| a | b |\n|---|---|\n| 1 | 2 |\n",
     "Table IV"),
    # No caption nearby
    ("Body text only.\n| a | b |\n|---|---|\n| 1 | 2 |\n",
     None),
    # Multiple captions: closest wins
    ("Table I. blah blah blah.\nLong paragraph filler. Table V. Closer one.\n"
     "| a | b |\n|---|---|\n| 1 | 2 |\n",
     "Table V"),
])
def test_caption_for_table_match(md_excerpt, expected):
    m = paper2md.TABLE_RE.search(md_excerpt)
    assert m is not None, "test fixture must contain a markdown table"
    got = paper2md._caption_for_table_match(md_excerpt, m)
    assert got == expected


# ---------------------------------------------------------------------------
# Caption-to-page resolution against a fake fitz.Document
# ---------------------------------------------------------------------------

class _FakePage:
    def __init__(self, text: str, caption_rects: Optional[dict] = None):
        self._text = text
        # Stand-in for fitz.Page.rect; needed by process_tables when a
        # located bbox flows through the full-page detection check.
        self.rect = SimpleNamespace(width=600.0, height=800.0)
        # Optional: per-caption rect lists, used by
        # _pick_candidate_near_caption when multiple candidate bboxes
        # need to be disambiguated against the caption's vertical
        # position. Each rect is a SimpleNamespace with y0/y1 floats.
        self._caption_rects = caption_rects or {}

    def get_text(self, _kind: str) -> str:
        return self._text

    def search_for(self, needle: str):
        """Mimic fitz.Page.search_for: return a list of rect-like objects
        for occurrences of `needle` on the page. The test sets up the
        mapping in the constructor; missing keys return [] (caption not
        found, the sentinel that drives the fallback branch in
        _pick_candidate_near_caption)."""
        return list(self._caption_rects.get(needle, []))


class _FakeDoc:
    def __init__(self, page_texts: list[str],
                 caption_rects: Optional[list] = None):
        """page_texts: per-page text. caption_rects: optional per-page
        dict mapping caption string -> list of rect-likes (objects with
        y0/y1) returned by _FakePage.search_for. None means the
        per-page page has no rects registered (search_for returns [])."""
        rects_per_page = caption_rects or [None] * len(page_texts)
        self._pages = [_FakePage(t, rects)
                       for t, rects in zip(page_texts, rects_per_page)]

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def __getitem__(self, i: int) -> _FakePage:
        return self._pages[i]


def test_find_caption_page_line_leading():
    doc = _FakeDoc([
        "Body text mentioning Table II inline.\n",   # page 0: cross-ref only
        "Table II. The real caption is here.\n"     # page 1: real caption
        "More text below.\n",
    ])
    assert paper2md._find_caption_page(doc, "Table II") == 1


def test_find_caption_page_inline_fallback():
    """No line-leading match anywhere; fall back to the first inline match."""
    doc = _FakeDoc([
        "Nothing relevant.\n",
        "  We refer to Table III throughout this paragraph.\n",
        "  More refs to Table III later on.\n",
    ])
    assert paper2md._find_caption_page(doc, "Table III") == 1


def test_find_caption_page_no_match():
    doc = _FakeDoc(["foo\n", "bar\n", "baz\n"])
    assert paper2md._find_caption_page(doc, "Table V") is None


def test_find_caption_page_prefers_line_leading_over_inline():
    """Pages 0/1 mention the caption inline; page 2 has the real line-leading
    caption. Returns page 2 even though earlier pages match."""
    doc = _FakeDoc([
        "Body text with Table IV reference.\n",
        "Another page with Table IV inline mention.\n",
        "Table IV. The actual caption.\n",
    ])
    assert paper2md._find_caption_page(doc, "Table IV") == 2


# ---------------------------------------------------------------------------
# Integration: process_tables() with mocked VLM and render_page
# ---------------------------------------------------------------------------

def _build_md(caption: Optional[str] = "Table II") -> str:
    """Build a markdown blob containing one matchable table with optional
    caption above it. Used to exercise the unlocated-table branch of
    process_tables()."""
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    if caption:
        return f"{caption}.\n{table}\n"
    return f"Body text only.\n{table}\n"


def _process_one(md: str, page_text_for_caption: Optional[str],
                 vlm_response: Optional[str], tmp_path: Path):
    """Run process_tables() on a single-table markdown blob. Mocks the
    table-finder (always misses), the doc (one page with optional caption
    text), render_page (returns a sentinel image), and vlm (returns
    `vlm_response`). Returns (transformed_md, report, vlm_was_called)."""
    page_text = page_text_for_caption or "no caption here\n"
    doc = _FakeDoc([page_text])

    # finder.candidates yields nothing -> find_pdf_table returns None.
    finder = MagicMock()
    finder.candidates = MagicMock(return_value=iter([]))

    report = paper2md.QualityReport(vlm_enabled=True)
    assets = tmp_path / "assets"
    assets.mkdir()

    vlm_calls = []

    def fake_vlm(prompt, image, max_tokens=1500):
        vlm_calls.append((prompt, image))
        return vlm_response

    fake_image = object()  # sentinel; vlm is mocked, image content irrelevant

    with patch.object(paper2md, "find_pdf_table", return_value=None), \
         patch.object(paper2md, "render_page", return_value=fake_image), \
         patch.object(paper2md, "vlm", side_effect=fake_vlm):
        out_md = paper2md.process_tables(md, doc, finder, assets, report,
                                       use_vlm=True, vlm_rewrite_tables=True)
    return out_md, report, vlm_calls


def test_unlocated_caption_found_writes_sidecar(tmp_path):
    md = _build_md("Table II")
    out_md, report, calls = _process_one(
        md,
        page_text_for_caption="Table II. The real caption.\n",
        vlm_response="| col1 | col2 |\n|------|------|\n| a | b |\n",
        tmp_path=tmp_path,
    )
    # VLM was called with the page-image prompt (not text cleanup)
    assert len(calls) == 1
    prompt, _ = calls[0]
    assert "'Table II'" in prompt  # caption substituted into the prompt
    # Sidecar written with new naming pattern
    assert (tmp_path / "assets" / "table_page1_1.md").exists()
    # Quality report records page=1, located=false, vlm_redone=true
    assert len(report.tables) == 1
    ts = report.tables[0]
    assert ts.located is False
    assert ts.vlm_redone is True
    assert ts.page == 1
    assert ts.sidecar == "table_page1_1.md"
    # Linked into the body
    assert "[Table 1 — separate markdown](assets/table_page1_1.md)" in out_md


def test_unlocated_no_caption_skips_vlm(tmp_path):
    md = _build_md(caption=None)
    out_md, report, calls = _process_one(
        md,
        page_text_for_caption="irrelevant page text",
        vlm_response="should not be called",
        tmp_path=tmp_path,
    )
    # No VLM call: caption couldn't be extracted from markdown context
    assert calls == []
    # No sidecar written
    assert not list((tmp_path / "assets").iterdir())
    # Score reflects no rewrite
    ts = report.tables[0]
    assert ts.vlm_redone is False
    assert ts.located is False
    assert ts.sidecar is None


def test_unlocated_vlm_returns_skip_no_sidecar(tmp_path):
    md = _build_md("Table IV")
    out_md, report, calls = _process_one(
        md,
        page_text_for_caption="Table IV. Real caption.\n",
        vlm_response="SKIP",
        tmp_path=tmp_path,
    )
    # VLM was called, but it declined
    assert len(calls) == 1
    # No sidecar
    assert not list((tmp_path / "assets").iterdir())
    ts = report.tables[0]
    assert ts.vlm_redone is False
    assert ts.sidecar is None


def test_unlocated_vlm_returns_none_no_sidecar(tmp_path):
    """Network/parse failure path: vlm() returns None on errors. Same
    treatment as SKIP -- no sidecar, no rewrite."""
    md = _build_md("Table V")
    out_md, report, calls = _process_one(
        md,
        page_text_for_caption="Table V. Real caption.\n",
        vlm_response=None,
        tmp_path=tmp_path,
    )
    assert len(calls) == 1
    assert not list((tmp_path / "assets").iterdir())
    assert report.tables[0].vlm_redone is False


# ---------------------------------------------------------------------------
# Caption-page bypass (TODO 1a)
# ---------------------------------------------------------------------------
#
# When the markdown table has a caption AND the selected --table-finder
# reports exactly one bbox on the caption's page, the bypass claims that
# bbox without going through find_pdf_table's anchor-substring search.
# These tests gate the three branches: claims-single, claims-closest-
# below when multiple bboxes (via _pick_candidate_near_caption), and
# defers-on-zero (the last is already covered by
# test_unlocated_caption_found_writes_sidecar but we add an explicit
# bypass-aware version that also asserts find_pdf_table WAS called).

def _process_with_finder(md: str, page_texts: list,
                         finder_cands: dict,
                         vlm_response: Optional[str],
                         tmp_path: Path,
                         fpt_return=None,
                         caption_rects: Optional[list] = None):
    """Run process_tables() with a finder whose candidates(doc, page_idx)
    returns finder_cands.get(page_idx, []). Mocks find_pdf_table to
    return fpt_return (default None). Tracks find_pdf_table call count
    so tests can assert whether the bypass short-circuited it.
    `caption_rects` is an optional per-page list of dicts mapping
    caption strings to rect-likes for _FakePage.search_for; needed by
    multi-bbox bypass tests that exercise _pick_candidate_near_caption."""
    doc = _FakeDoc(page_texts, caption_rects=caption_rects)

    finder = MagicMock()
    finder.candidates = MagicMock(
        side_effect=lambda d, page_idx: finder_cands.get(page_idx, []))

    report = paper2md.QualityReport(vlm_enabled=True)
    assets = tmp_path / "assets"
    assets.mkdir()

    fake_image = MagicMock()  # has .save() so the JPEG path doesn't crash

    fpt_calls = []
    def fake_fpt(d, table_md, f):
        fpt_calls.append(table_md)
        return fpt_return

    with patch.object(paper2md, "find_pdf_table", side_effect=fake_fpt), \
         patch.object(paper2md, "render_crop", return_value=fake_image), \
         patch.object(paper2md, "render_page", return_value=fake_image), \
         patch.object(paper2md, "vlm", return_value=vlm_response):
        out_md = paper2md.process_tables(md, doc, finder, assets, report,
                                       use_vlm=True, vlm_rewrite_tables=True)
    return out_md, report, fpt_calls


def test_caption_page_bypass_claims_single_bbox(tmp_path):
    """Caption found + exactly one bbox on caption page -> bypass claims
    the bbox, find_pdf_table is NOT called, sidecar uses the
    high-confidence `table_p{P}_{N}.md` naming with a JPEG saved."""
    md = _build_md("Table II")
    out_md, report, fpt_calls = _process_with_finder(
        md,
        page_texts=["Table II. Real caption.\n"],
        finder_cands={0: [((10.0, 20.0, 100.0, 80.0), "anchor text")]},
        vlm_response="| col1 |\n|------|\n| val |\n",
        tmp_path=tmp_path,
    )
    # Bypass took effect -> find_pdf_table never called
    assert fpt_calls == [], "bypass should short-circuit find_pdf_table"
    # Located naming on the sidecar (the .jpg comes from render_crop's
    # mock and isn't materialized on disk in unit tests; jpeg_saved on
    # the report is the authoritative signal)
    assert (tmp_path / "assets" / "table_p1_1.md").exists()
    ts = report.tables[0]
    assert ts.located is True
    assert ts.vlm_redone is True
    assert ts.page == 1
    assert ts.jpeg_saved is True
    assert ts.sidecar == "table_p1_1.md"


def test_caption_page_bypass_picks_closest_below_when_multiple_bboxes(tmp_path):
    """Caption found + 2 bboxes on caption page -> bypass disambiguates
    via _pick_candidate_near_caption, claiming the bbox closest to (and
    below) the caption's vertical position. find_pdf_table is NOT
    called; sidecar uses the high-confidence `table_p{P}_{N}.md`
    naming with a JPEG saved.

    Setup: caption rect at y0=10/y1=15 (cap_bottom=15). bbox A starts
    at y0=20 (5pt below caption); bbox B starts at y0=200 (185pt
    below). The disambiguator should claim A."""
    md = _build_md("Table II")
    cap_rect = SimpleNamespace(y0=10.0, y1=15.0)
    out_md, report, fpt_calls = _process_with_finder(
        md,
        page_texts=["Table II. Real caption.\n"],
        finder_cands={0: [
            ((10.0, 20.0, 100.0, 80.0), "anchor a"),
            ((10.0, 200.0, 100.0, 260.0), "anchor b"),
        ]},
        caption_rects=[{"Table II": [cap_rect]}],
        vlm_response="| col |\n|---|\n| v |\n",
        tmp_path=tmp_path,
    )
    # Bypass took effect -> find_pdf_table never called
    assert fpt_calls == [], "bypass should claim a bbox via _pick_candidate_near_caption"
    # Located naming on the sidecar; jpeg_saved flag set on the report
    assert (tmp_path / "assets" / "table_p1_1.md").exists()
    ts = report.tables[0]
    assert ts.located is True
    assert ts.vlm_redone is True
    assert ts.page == 1
    assert ts.jpeg_saved is True
    assert ts.sidecar == "table_p1_1.md"


def test_caption_page_bypass_falls_back_to_first_when_caption_unsearchable(tmp_path):
    """Caption found + 2 bboxes on caption page + caption text not
    findable via page.search_for (rect-set empty) -> bypass still
    claims a bbox via the first-candidate fallback in
    _pick_candidate_near_caption. find_pdf_table is NOT called."""
    md = _build_md("Table II")
    out_md, report, fpt_calls = _process_with_finder(
        md,
        page_texts=["Table II. Real caption.\n"],
        finder_cands={0: [
            ((10.0, 20.0, 100.0, 80.0), "anchor a"),
            ((10.0, 200.0, 100.0, 260.0), "anchor b"),
        ]},
        caption_rects=[{}],  # search_for returns []
        vlm_response="| col |\n|---|\n| v |\n",
        tmp_path=tmp_path,
    )
    assert fpt_calls == [], "bypass should still claim via the first-candidate fallback"
    assert (tmp_path / "assets" / "table_p1_1.md").exists()
    ts = report.tables[0]
    assert ts.located is True
    assert ts.page == 1
    assert ts.sidecar == "table_p1_1.md"


def test_caption_page_bypass_skipped_when_no_caption(tmp_path):
    """Markdown has no caption near the table -> bypass guard fails
    (caption is None), find_pdf_table is consulted as before. Even
    with finder candidates available, the bypass must not fire when
    there's no caption to anchor to."""
    md = _build_md(caption=None)
    out_md, report, fpt_calls = _process_with_finder(
        md,
        page_texts=["Some unrelated body text.\n"],
        finder_cands={0: [((10.0, 20.0, 100.0, 80.0), "irrelevant")]},
        vlm_response="this should not be called",
        tmp_path=tmp_path,
    )
    # No caption -> bypass skipped, find_pdf_table called
    assert len(fpt_calls) == 1
    # find_pdf_table returned None and caption is None -> no fallback,
    # no sidecar, marker markdown stays
    assert not list((tmp_path / "assets").iterdir())
    ts = report.tables[0]
    assert ts.located is False
    assert ts.vlm_redone is False
    assert ts.sidecar is None


# ---------------------------------------------------------------------------
# Detector-text mode (default; previously gated by --no-vlm-tables)
# ---------------------------------------------------------------------------

def _process_detector_text(md: str, page_texts: list,
                           finder_cands: dict,
                           tmp_path: Path):
    """Run process_tables() with vlm_rewrite_tables=False (the default
    detector-text mode). find_pdf_table is mocked to return None so all
    routing goes through the bypass (or not-located paths). vlm() is
    mocked to fail loudly if called -- the detector-text mode must not
    call the VLM for tables."""
    doc = _FakeDoc(page_texts)
    finder = MagicMock()
    finder.candidates = MagicMock(
        side_effect=lambda d, page_idx: finder_cands.get(page_idx, []))
    report = paper2md.QualityReport(vlm_enabled=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    fake_image = MagicMock()

    def fail_if_called(*a, **kw):
        raise AssertionError("vlm() must not be called when "
                             "vlm_rewrite_tables=False")

    with patch.object(paper2md, "find_pdf_table", return_value=None), \
         patch.object(paper2md, "render_crop", return_value=fake_image), \
         patch.object(paper2md, "render_page", return_value=fake_image), \
         patch.object(paper2md, "vlm", side_effect=fail_if_called):
        out_md = paper2md.process_tables(md, doc, finder, assets, report,
                                       use_vlm=True,
                                       vlm_rewrite_tables=False)
    return out_md, report


def test_detector_text_uses_detector_markdown_when_shaped(tmp_path):
    """Located + detector returns markdown-shaped text -> body becomes
    the detector's markdown directly. No VLM call. Sidecar uses the
    high-confidence `table_p{P}_{N}.md` naming. Bypass also fires here
    because there's exactly one bbox on the caption page."""
    md = _build_md("Table II")
    detector_md = "| Hdr1 | Hdr2 |\n|------|------|\n| 11 | 22 |\n"
    out_md, report = _process_detector_text(
        md,
        page_texts=["Table II. Real caption.\n"],
        finder_cands={0: [((10.0, 20.0, 100.0, 80.0), detector_md)]},
        tmp_path=tmp_path,
    )
    # Sidecar written with the detector's markdown content
    sidecar = tmp_path / "assets" / "table_p1_1.md"
    assert sidecar.exists()
    assert "Hdr1" in sidecar.read_text()
    assert "Hdr2" in sidecar.read_text()
    ts = report.tables[0]
    assert ts.located is True
    assert ts.vlm_redone is True  # body was rewritten (from detector)
    assert ts.page == 1
    assert ts.sidecar == "table_p1_1.md"


def test_detector_text_keeps_marker_when_text_not_markdown(tmp_path):
    """Located + detector text is raw text (not markdown-shaped) ->
    body stays as marker's; no sidecar; no VLM call."""
    md = _build_md("Table II")
    raw_text = "Shot type 12.7 5.89 0.66 ..."  # space-joined, not | markdown
    out_md, report = _process_detector_text(
        md,
        page_texts=["Table II. Real caption.\n"],
        finder_cands={0: [((10.0, 20.0, 100.0, 80.0), raw_text)]},
        tmp_path=tmp_path,
    )
    # Located -> JPEG was saved (rendered crop), but no .md sidecar
    # because we didn't rewrite the body.
    assert not (tmp_path / "assets" / "table_p1_1.md").exists()
    ts = report.tables[0]
    assert ts.located is True
    assert ts.vlm_redone is False
    assert ts.sidecar is None


def test_detector_text_skips_page_image_fallback(tmp_path):
    """Not located AND vlm_rewrite_tables=False -> the page-image
    fallback must not run (it requires the VLM). Marker body stays;
    no sidecar."""
    md = _build_md("Table II")
    out_md, report = _process_detector_text(
        md,
        page_texts=["Table II. Real caption.\n"],
        finder_cands={},  # no bboxes on any page
        tmp_path=tmp_path,
    )
    assert not list((tmp_path / "assets").iterdir())
    ts = report.tables[0]
    assert ts.located is False
    assert ts.vlm_redone is False
    assert ts.sidecar is None


# ---------------------------------------------------------------------------
# RunInfo / `run:` front-matter block
# ---------------------------------------------------------------------------

def test_run_info_emits_yaml_block_between_copyright_and_quality():
    """RunInfo should render as a `run:` block between any `copyright:`
    block and the `quality:` block. Without RunInfo, no `run:` line
    appears -- backwards-compatible with existing front-matter consumers."""
    ri = paper2md.RunInfo(
        command="paper2md.py paper.pdf --table-finder docling",
        hostname="dgx-spark-test",
        vlm_provider="vllm",
        vlm_model="Qwen/Qwen3-VL-32B-Instruct",
        elapsed_sec=42.5,
    )
    report = paper2md.QualityReport(vlm_enabled=True, run_info=ri)
    report.pages.append(paper2md.PageScore(
        page=1, char_density=0.012,
        sparse=False, rescued=False,
    ))
    fm = report.to_frontmatter()
    assert "run:" in fm
    assert "command:" in fm
    assert "hostname:" in fm
    assert "dgx-spark-test" in fm
    assert "elapsed_sec: 42.5" in fm
    assert "vlm_provider: vllm" in fm
    assert "Qwen/Qwen3-VL-32B-Instruct" in fm
    # Order: run: must precede quality:
    assert fm.index("run:") < fm.index("quality:")


def test_run_info_vlm_disclosure_fields_emit(tmp_path):
    """The VLM-disclosure fields `vlm_system_prompt` and
    `vlm_inference_params` (added for publication reproducibility, see
    USAGE.md §18) emit in both the YAML frontmatter and the
    .meta.json dict so a downstream `jq '.run.vlm_*'` extracts them."""
    ri = paper2md.RunInfo(
        command="paper2md.py paper.pdf",
        hostname="test-host",
        vlm_provider="vllm",
        vlm_model="Qwen/Qwen3-VL-32B-Instruct",
        vlm_endpoint="http://localhost:8000/v1/",
        vlm_system_prompt="You are a precise document digitizer.",
        vlm_inference_params=(
            "server-side defaults; paper2md does not override "
            "temperature, top_p, or seed."),
        elapsed_sec=10.0,
    )
    report = paper2md.QualityReport(vlm_enabled=True, run_info=ri)
    report.pages.append(paper2md.PageScore(
        page=1, char_density=0.012, sparse=False, rescued=False))

    fm = report.to_frontmatter()
    assert "vlm_system_prompt:" in fm
    assert "You are a precise document digitizer." in fm
    assert "vlm_inference_params:" in fm
    assert "server-side defaults" in fm

    d = ri.to_dict()
    assert "vlm_system_prompt" in d
    assert "vlm_inference_params" in d
    assert d["vlm_system_prompt"] == "You are a precise document digitizer."


def test_run_info_disclosure_fields_optional():
    """The new fields are optional -- empty string -> omitted from
    YAML (backward-compat with manifests/tests that don't set them)."""
    ri = paper2md.RunInfo(
        command="x", hostname="h",
        vlm_provider="vllm", vlm_model="m",
        elapsed_sec=1.0,
    )
    yaml_lines = ri.to_yaml_lines()
    yaml_text = "\n".join(yaml_lines)
    assert "vlm_system_prompt:" not in yaml_text
    assert "vlm_inference_params:" not in yaml_text


def test_run_info_omitted_when_unset():
    """No run_info -> no `run:` block in front-matter. Preserves
    backward-compat with consumers that expect just `quality:`."""
    report = paper2md.QualityReport(vlm_enabled=True)
    report.pages.append(paper2md.PageScore(
        page=1, char_density=0.012,
        sparse=False, rescued=False,
    ))
    fm = report.to_frontmatter()
    assert "run:" not in fm
    assert "quality:" in fm


# ---------------------------------------------------------------------------
# .meta.json sidecar (structured frontmatter mirror)
# ---------------------------------------------------------------------------

def test_quality_report_to_dict_round_trip(tmp_path):
    """QualityReport.to_dict produces a JSON-serialisable mirror of the YAML
    frontmatter; _write_meta_json lands the file next to the .md."""
    import json
    from metadata_frontend import ArticleMetadata
    ri = paper2md.RunInfo(
        command="paper2md.py p.pdf",
        hostname="host", vlm_provider="vllm",
        vlm_model="Qwen/Qwen3-VL-32B-Instruct",
        elapsed_sec=12.3,
    )
    meta = ArticleMetadata(
        doi="10.1073/pnas.6.8.449", year=1920,
        license="public-domain-us", safe_to_distribute="true",
        confidence="medium",
    )
    report = paper2md.QualityReport(
        vlm_enabled=True, run_info=ri, metadata=meta,
    )
    report.tables.append(paper2md.TableScore(
        index=1, page=2, located=True, jpeg_saved=True,
        pre_redo_reason=None, vlm_redone=True,
        post_redo_reason=None, score=1.0, sidecar="assets/table_p2_1.md",
    ))
    # rescued=True so the pages block emits -- otherwise the gate
    # introduced for Hook 3 audit-trail trimming would suppress it.
    report.pages.append(paper2md.PageScore(
        page=1, char_density=0.012,
        sparse=False, rescued=True,
    ))

    d = report.to_dict()
    assert d["quality"]["grade"] == "A"
    assert d["quality"]["vlm_enabled"] is True
    assert d["quality"]["tables"][0]["sidecar"] == "assets/table_p2_1.md"
    assert d["quality"]["pages"][0]["page"] == 1
    assert d["copyright"]["doi"] == "10.1073/pnas.6.8.449"
    assert d["copyright"]["year"] == 1920
    assert d["copyright"]["license"] == "public-domain-us"
    assert d["run"]["hostname"] == "host"
    assert d["run"]["elapsed_sec"] == 12.3

    # Pure-stdlib JSON round-trip: must not raise.
    text = json.dumps(d)
    round_tripped = json.loads(text)
    assert round_tripped["copyright"]["year"] == 1920

    # _write_meta_json drops the file at the right path.
    out_path = paper2md._write_meta_json(tmp_path, "paper", report)
    assert out_path == tmp_path / "paper.meta.json"
    on_disk = json.loads(out_path.read_text())
    assert on_disk["copyright"]["license"] == "public-domain-us"


def test_edits_block_omitted_when_empty():
    """A QualityReport without an EditsAnnotation (or with one that's
    is_empty) does NOT emit an `edits:` block in the frontmatter or
    to_dict output."""
    import paper2md
    # No edits at all
    r = paper2md.QualityReport(vlm_enabled=True)
    assert "edits:" not in r.to_frontmatter()
    assert "edits" not in r.to_dict()
    # is_empty (no edited flag, no note)
    r.edits = paper2md.EditsAnnotation()
    assert "edits:" not in r.to_frontmatter()
    assert "edits" not in r.to_dict()


def test_edits_block_emitted_when_set():
    """User-set edited=true with a single-line note round-trips
    to YAML and JSON as a peer block alongside `quality:`."""
    import paper2md
    r = paper2md.QualityReport(vlm_enabled=True)
    r.edits = paper2md.EditsAnnotation(
        edited=True,
        note="Manually inserted refs 33-34 from PDF text layer",
    )
    fm = r.to_frontmatter()
    assert "edits:" in fm
    assert "edited: true" in fm
    assert "Manually inserted refs 33-34" in fm
    d = r.to_dict()
    assert d["edits"]["edited"] is True
    assert "33-34" in d["edits"]["note"]


def test_edits_block_multiline_note_uses_block_scalar():
    """Multi-line notes get YAML block-scalar form (`|`) so newlines
    survive parse without escape-soup."""
    import paper2md
    r = paper2md.QualityReport(vlm_enabled=True)
    r.edits = paper2md.EditsAnnotation(
        edited=True,
        note="line one\nline two\nline three",
    )
    fm = r.to_frontmatter()
    assert "note: |" in fm
    # All three lines indented under note:
    assert "    line one" in fm
    assert "    line two" in fm
    assert "    line three" in fm


def test_edits_block_only_note_without_edited_flag():
    """A user who set the note but left edited=False (default) is
    still treated as not-empty -- the note alone signals presence
    of edits even without the flag."""
    import paper2md
    e = paper2md.EditsAnnotation(note="some edit happened")
    assert not e.is_empty()
    r = paper2md.QualityReport(vlm_enabled=True, edits=e)
    fm = r.to_frontmatter()
    assert "edits:" in fm
    assert "edited: false" in fm
    assert "some edit happened" in fm


def test_pages_block_omitted_when_no_rescues():
    """The per-page audit trail is gated on any-rescued: when no page
    triggered Hook 3 (--rescue-sparse-pages, opt-in and rarely used),
    the `pages:` block must NOT appear in YAML or `.meta.json`. This
    trims ~N lines from frontmatter on every paper (one entry per
    PDF page) for a feature most users never enable."""
    report = paper2md.QualityReport(vlm_enabled=False)
    # All non-rescued -- the typical state for the default run.
    for i in range(1, 11):
        report.pages.append(paper2md.PageScore(
            page=i, char_density=0.012, sparse=False, rescued=False))

    fm = report.to_frontmatter()
    assert "  pages:" not in fm, \
        "non-rescued pages must not emit YAML block"

    d = report.to_dict()
    assert d["quality"]["pages"] == [], \
        "non-rescued pages -> empty list in .meta.json"


def test_pages_block_emits_when_any_rescued():
    """As soon as Hook 3 rescues even one page, the full per-page
    audit trail emits so the user can see what was sparse-but-not-
    rescued alongside what was. Non-rescued context is informative
    when investigating why the rescue did or didn't fire."""
    report = paper2md.QualityReport(vlm_enabled=True)
    report.pages.append(paper2md.PageScore(
        page=1, char_density=0.04, sparse=False, rescued=False))
    report.pages.append(paper2md.PageScore(
        page=2, char_density=0.001, sparse=True, rescued=True))
    report.pages.append(paper2md.PageScore(
        page=3, char_density=0.05, sparse=False, rescued=False))

    fm = report.to_frontmatter()
    assert "  pages:" in fm
    # ALL pages emit when any rescued -- context matters for the audit.
    assert "page: 1" in fm and "page: 2" in fm and "page: 3" in fm
    assert "rescued: true" in fm and "rescued: false" in fm

    d = report.to_dict()
    pages = d["quality"]["pages"]
    assert len(pages) == 3
    assert {p["page"] for p in pages} == {1, 2, 3}
    assert sum(1 for p in pages if p["rescued"]) == 1


def test_quality_report_to_dict_no_optional_blocks():
    """Without metadata or run_info, to_dict returns only the quality block.
    Mirrors to_frontmatter's behavior."""
    report = paper2md.QualityReport(vlm_enabled=False)
    # rescued=True so the per-page block emits -- as of 2026-05-17,
    # the pages block is gated on any-rescued (audit trail for Hook 3,
    # opt-in --rescue-sparse-pages). Non-rescued pages produce no
    # `pages` entries because no consumer reads the diagnostic.
    report.pages.append(paper2md.PageScore(
        page=1, char_density=0.01,
        sparse=False, rescued=True,
    ))
    d = report.to_dict()
    assert "copyright" not in d
    assert "run" not in d
    assert d["quality"]["vlm_enabled"] is False
    # Pages are emitted as diagnostics (page, char_density, sparse,
    # rescued) but no longer carry a per-page score -- char_density
    # is a content-type proxy, not a quality signal, and was dropped
    # from QualityReport.overall in 2026-05.
    assert d["quality"]["pages"][0]["page"] == 1
    assert d["quality"]["pages"][0]["char_density"] == 0.01
    assert "score" not in d["quality"]["pages"][0]


# ---------------------------------------------------------------------------
# Concurrent table VLM dispatch (--table-workers, TODO 1b)
# ---------------------------------------------------------------------------

def test_table_workers_multi_table_concurrent_preserves_order(tmp_path):
    """3 markdown tables with table_workers=4. The phase-2 dispatcher
    runs them concurrently; phase-3 finalize must still produce
    sidecars + report entries in original (i=1,2,3) order."""
    md = (
        "Table I.\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "Table II.\n| c | d |\n|---|---|\n| 3 | 4 |\n\n"
        "Table III.\n| e | f |\n|---|---|\n| 5 | 6 |\n"
    )
    doc = _FakeDoc([
        "Table I. caption page 0\n",
        "Table II. caption page 1\n",
        "Table III. caption page 2\n",
    ])
    # One bbox per page so the bypass fires for every table.
    finder = MagicMock()
    def cands(d, page_idx):
        return [((10.0, 20.0, 100.0, 80.0), f"docling md p{page_idx}")]
    finder.candidates = MagicMock(side_effect=cands)

    report = paper2md.QualityReport(vlm_enabled=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    fake_image = MagicMock()

    # Each VLM call returns a unique table tagged by call-count so we
    # can verify no result got mis-routed to the wrong task.
    call_counter = {"n": 0}
    def fake_vlm(prompt, image, max_tokens=1500):
        call_counter["n"] += 1
        n = call_counter["n"]
        return f"| col{n} |\n|------|\n| val{n} |"

    with patch.object(paper2md, "find_pdf_table", return_value=None), \
         patch.object(paper2md, "render_crop", return_value=fake_image), \
         patch.object(paper2md, "render_page", return_value=fake_image), \
         patch.object(paper2md, "vlm", side_effect=fake_vlm):
        out_md = paper2md.process_tables(md, doc, finder, assets, report,
                                       use_vlm=True, vlm_rewrite_tables=True,
                                       table_workers=4)

    # All three tables produced sidecars with the located naming.
    assert (assets / "table_p1_1.md").exists()
    assert (assets / "table_p2_2.md").exists()
    assert (assets / "table_p3_3.md").exists()
    # report.tables in original index order, regardless of completion order
    assert [t.index for t in report.tables] == [1, 2, 3]
    assert [t.page for t in report.tables] == [1, 2, 3]
    assert all(t.located and t.vlm_redone for t in report.tables)
    # Each sidecar got a unique col label (proves no mix-up between tasks)
    contents = sorted(p.read_text() for p in assets.glob("table_p*_*.md"))
    cols = [c.split("\n")[0] for c in contents]
    assert len(set(cols)) == 3, "each task must have received its own VLM result"


# ---------------------------------------------------------------------------
# bbox expansion for table footnotes
# ---------------------------------------------------------------------------

def _fake_page(width: float, height: float, blocks: list):
    """Build a minimal stand-in for a fitz.Page for _expand_bbox_for_footnotes.
    blocks is a list of (x0, y0, x1, y1, text)."""
    return SimpleNamespace(
        rect=SimpleNamespace(width=width, height=height),
        get_text=lambda kind: blocks if kind == "blocks" else "",
    )


def test_expand_bbox_letter_footnotes_below():
    """Letter-prefixed (`a`, `b`, ...) blocks immediately below the
    table get pulled into the bbox."""
    page = _fake_page(600, 800, blocks=[
        (50, 100, 550, 300, "BODY"),
        # Footnote block, top within 30pt of bbox bottom
        (50, 320, 550, 400,
         "a Note about the first column.\nb Note about the second."),
        # Far below: should NOT be included
        (50, 700, 550, 750, "Body paragraph"),
    ])
    bbox = (50, 100, 550, 300)  # table covers blocks[0] region
    new_bbox = paper2md._expand_bbox_for_footnotes(page, bbox)
    assert new_bbox[0] == 50 and new_bbox[1] == 100 and new_bbox[2] == 550
    assert new_bbox[3] == 400, f"expected y1=400 (incl. footnotes), got {new_bbox[3]}"


def test_expand_bbox_no_footnote_block_unchanged():
    """When no marker-prefixed block sits below the table, bbox is
    unchanged (returns the original coordinates as floats)."""
    page = _fake_page(600, 800, blocks=[
        (50, 100, 550, 300, "BODY"),
        # Block below but starts with regular text, NOT a footnote marker
        (50, 320, 550, 400, "The next paragraph begins here..."),
    ])
    bbox = (50, 100, 550, 300)
    new_bbox = paper2md._expand_bbox_for_footnotes(page, bbox)
    assert new_bbox == (50.0, 100.0, 550.0, 300.0)


def test_expand_bbox_symbol_footnotes_below():
    """Symbol footnotes (`*`, dagger) are also recognised."""
    page = _fake_page(600, 800, blocks=[
        (50, 100, 550, 300, "BODY"),
        (50, 315, 550, 380, "* First note.\n† Dagger note."),
    ])
    bbox = (50, 100, 550, 300)
    new_bbox = paper2md._expand_bbox_for_footnotes(page, bbox)
    assert new_bbox[3] == 380


def test_expand_bbox_capped_at_25_percent_page_height():
    """An anomalously long footnote block can't extend the bbox more
    than 25% of page height (defensive cap against runaway capture)."""
    page = _fake_page(600, 800, blocks=[
        (50, 100, 550, 300, "BODY"),
        # Footnote-marked block that, if accepted whole, would extend
        # the bbox by 250pt -- but 25% of 800 = 200pt, so we cap.
        (50, 320, 550, 700, "a Long note " * 50),
    ])
    bbox = (50, 100, 550, 300)
    new_bbox = paper2md._expand_bbox_for_footnotes(page, bbox)
    # cap: original_y1 (300) + max_extension (200) = 500
    assert new_bbox[3] <= 500.0


def test_expand_bbox_stops_at_first_non_footnote_block():
    """If the first block below the table is a body paragraph, we stop
    there even if a later block has footnote markers."""
    page = _fake_page(600, 800, blocks=[
        (50, 100, 550, 300, "BODY"),
        (50, 315, 550, 360, "Following paragraph in the body."),
        # Footnote-shaped block, but blocked by the body block above
        (50, 365, 550, 420, "a Should not be captured."),
    ])
    bbox = (50, 100, 550, 300)
    new_bbox = paper2md._expand_bbox_for_footnotes(page, bbox)
    assert new_bbox[3] == 300.0


# ---------------------------------------------------------------------------
# Span-anchored reference normalisation pre-pass
# ---------------------------------------------------------------------------

def test_normalise_refs_rebullets_plain_span_lines():
    """Near-contiguous run of plain-<span> ref lines (no bullet) gets
    re-bulletted so fix_reference_numbering can see them. Both purely
    consecutive lines and lines separated by single blank lines (the
    real Marker output shape on Icarus) must trigger."""
    # Consecutive plain-span lines.
    md1 = (
        "Other text.\n"
        '<span id="page-15-1"></span>Alexander, C.M.O., 1994. Title 1.\n'
        '<span id="page-15-2"></span>Alexander, C.M.O., 1995. Title 2.\n'
        "More text.\n"
    )
    out1, c1 = paper2md.normalise_span_anchored_refs(md1)
    assert c1["bulletted"] == 2
    assert "- <span id=\"page-15-1\"></span>" in out1

    # Plain-span lines with a single blank line between (Marker's
    # actual Icarus shape).
    md2 = (
        "## References\n\n"
        '<span id="page-15-1"></span>Alexander, C.M.O., 1994. Title 1.\n'
        "\n"
        '<span id="page-15-2"></span>Alexander, C.M.O., 1995. Title 2.\n'
        "\n"
        '<span id="page-15-3"></span>Bauer, J., 1996. Title 3.\n'
    )
    out2, c2 = paper2md.normalise_span_anchored_refs(md2)
    assert c2["bulletted"] == 3
    assert "- <span id=\"page-15-1\"></span>" in out2
    assert "- <span id=\"page-15-2\"></span>" in out2
    assert "- <span id=\"page-15-3\"></span>" in out2


def test_normalise_refs_rejoins_split_doi_links():
    """`[http://dx.doi.org/](u1) [10.x/y](u2)` becomes a single autolink."""
    md = (
        "Smith, A., 1980. Some article. Earth Planet. Sci. Lett. "
        r"[http://dx.doi.org/](https://example.com/) [10.1016/0012-821X\(80\)90038-2]"
        "(https://example.com/landing).\n"
    )
    out, counts = paper2md.normalise_span_anchored_refs(md)
    assert counts["dois_rejoined"] == 1
    assert "<https://doi.org/10.1016/0012-821X(80)90038-2>" in out
    assert "[http://dx.doi.org/]" not in out


def test_normalise_refs_strips_refhub_link_wrappers():
    """`[Title](http://refhub.elsevier.com/...)` becomes plain `Title`.
    The URL contains balanced inner parens (`S0019-1035(26)00036-9/sbXX`);
    the regex must consume the whole URL, not stop at the first inner `)`,
    or a trailing fragment like `00036-9/sb17)` is left in the output."""
    md = (
        "Reference with [Cathodoluminescence](http://refhub.elsevier.com/"
        "S0019-1035(26)00036-9/sb17) embedded.\n"
    )
    out, counts = paper2md.normalise_span_anchored_refs(md)
    assert counts["refhub_stripped"] == 1
    assert "Cathodoluminescence embedded." in out
    assert "refhub.elsevier.com" not in out
    assert "[Cathodoluminescence]" not in out
    # Crucial leftover-fragment guard: prior naive [^)]+ regex left
    # `00036-9/sb17)` behind because parens in the URL aren't escaped.
    assert "00036-9/sb17" not in out


def test_normalise_refs_idempotent_on_clean_input():
    """Already-clean references unchanged; counts all zero."""
    md = (
        '- <span id="page-12-1"></span> 1. Smith, A., 1980. Title.\n'
        '- <span id="page-12-2"></span> 2. Jones, B., 1985. Other.\n'
    )
    out, counts = paper2md.normalise_span_anchored_refs(md)
    assert out == md
    assert counts == {"bulletted": 0, "dois_rejoined": 0, "refhub_stripped": 0}


def test_normalise_refs_singleton_span_left_alone():
    """A single plain-span line (not a contiguous run) is left alone --
    avoids mass-bulleting incidental anchor spans."""
    md = (
        "Body text.\n"
        '<span id="page-3-7"></span>An incidental anchor with text.\n'
        "More body.\n"
    )
    out, counts = paper2md.normalise_span_anchored_refs(md)
    assert counts["bulletted"] == 0
    assert out == md
