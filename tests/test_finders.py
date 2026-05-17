"""Unit tests for DoclingFinder.

Mocks DocumentConverter so tests don't load any real docling models.
Covers:
  1. BoundingBox in BOTTOMLEFT origin gets flipped using the actual
     fitz page height.
  2. BoundingBox in TOPLEFT origin passes through unchanged.
  3. Per-PDF cache: a second `candidates(...)` call on a different
     page does NOT trigger a second .convert().
  4. Tables on multiple pages each land in the right per-page bucket.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paper2md  # noqa: E402

# CoordOrigin is part of docling-core; we import it through the same path
# the production code uses so the test exercises the real enum identity.
from docling_core.types.doc.base import CoordOrigin  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: synthesize a docling-shaped result without docling models
# ---------------------------------------------------------------------------

def _bb(l, t, r, b, origin):
    return SimpleNamespace(l=l, t=t, r=r, b=b, coord_origin=origin)


def _table(page_no: int, bbox, md: str = "| a | b |\n|---|---|\n| 1 | 2 |"):
    """Build an object whose `.prov[0]` and `.export_to_markdown(doc)`
    match the surface DoclingFinder reads."""
    prov = SimpleNamespace(page_no=page_no, bbox=bbox)
    return SimpleNamespace(
        prov=[prov],
        export_to_markdown=MagicMock(return_value=md),
    )


def _result(*tables):
    document = SimpleNamespace(tables=list(tables))
    return SimpleNamespace(document=document)


class _FakePage:
    def __init__(self, height: float):
        self.rect = SimpleNamespace(height=height)


class _FakeDoc:
    """fitz-shaped: indexable by page_idx, exposes .name."""
    def __init__(self, name: str, page_heights: list[float]):
        self.name = name
        self._pages = [_FakePage(h) for h in page_heights]

    def __getitem__(self, i: int) -> _FakePage:
        return self._pages[i]

    @property
    def page_count(self) -> int:
        return len(self._pages)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bottomleft_bbox_is_flipped_with_actual_page_height():
    """A BOTTOMLEFT bbox at (l, t=669.5, r, b=490) on a 740-pt-tall page
    must come out as (l, top=70.5, r, bottom=250.0) in TOPLEFT/PyMuPDF
    coordinates."""
    finder = paper2md.DoclingFinder.__new__(paper2md.DoclingFinder)
    finder.converter = MagicMock()
    finder._cache = {}
    finder.converter.convert = MagicMock(return_value=_result(
        _table(page_no=3, bbox=_bb(42.0, 669.5, 508.0, 490.0,
                                   CoordOrigin.BOTTOMLEFT)),
    ))
    doc = _FakeDoc("/tmp/fake.pdf", page_heights=[740.0] * 3)

    cands = finder.candidates(doc, page_idx=2)  # docling page_no=3 -> idx 2
    assert len(cands) == 1
    bbox, _text = cands[0]
    assert bbox == pytest.approx((42.0, 70.5, 508.0, 250.0))


def test_topleft_bbox_passes_through():
    """If docling reports TOPLEFT (e.g. for image-based PDFs), the bbox
    is taken as-is without any height flip."""
    finder = paper2md.DoclingFinder.__new__(paper2md.DoclingFinder)
    finder.converter = MagicMock()
    finder._cache = {}
    finder.converter.convert = MagicMock(return_value=_result(
        _table(page_no=1, bbox=_bb(10.0, 20.0, 110.0, 80.0,
                                   CoordOrigin.TOPLEFT)),
    ))
    doc = _FakeDoc("/tmp/fake.pdf", page_heights=[792.0])

    cands = finder.candidates(doc, page_idx=0)
    assert len(cands) == 1
    bbox, _text = cands[0]
    assert bbox == (10.0, 20.0, 110.0, 80.0)


def test_per_pdf_cache_avoids_reconvert():
    """Two queries on different pages of the same PDF must trigger
    exactly ONE .convert() call. The cache key is doc.name."""
    finder = paper2md.DoclingFinder.__new__(paper2md.DoclingFinder)
    finder.converter = MagicMock()
    finder._cache = {}
    finder.converter.convert = MagicMock(return_value=_result(
        _table(page_no=1, bbox=_bb(0, 50, 100, 10,
                                   CoordOrigin.BOTTOMLEFT)),
        _table(page_no=2, bbox=_bb(0, 50, 100, 10,
                                   CoordOrigin.BOTTOMLEFT)),
    ))
    doc = _FakeDoc("/tmp/fake.pdf", page_heights=[100.0, 100.0])

    finder.candidates(doc, page_idx=0)
    finder.candidates(doc, page_idx=1)
    finder.candidates(doc, page_idx=0)
    assert finder.converter.convert.call_count == 1


def test_multipage_tables_routed_to_correct_pages():
    finder = paper2md.DoclingFinder.__new__(paper2md.DoclingFinder)
    finder.converter = MagicMock()
    finder._cache = {}
    finder.converter.convert = MagicMock(return_value=_result(
        _table(page_no=1, bbox=_bb(0, 100, 50, 50,
                                   CoordOrigin.BOTTOMLEFT), md="A"),
        _table(page_no=3, bbox=_bb(0, 100, 50, 50,
                                   CoordOrigin.BOTTOMLEFT), md="C"),
    ))
    doc = _FakeDoc("/tmp/fake.pdf", page_heights=[200.0, 200.0, 200.0])

    p0 = finder.candidates(doc, page_idx=0)
    p1 = finder.candidates(doc, page_idx=1)
    p2 = finder.candidates(doc, page_idx=2)
    assert len(p0) == 1 and p0[0][1] == "A"
    assert p1 == []
    assert len(p2) == 1 and p2[0][1] == "C"


def test_tables_without_provenance_are_skipped():
    """Defensive: if docling returns a table without prov info, ignore it
    rather than raising."""
    finder = paper2md.DoclingFinder.__new__(paper2md.DoclingFinder)
    finder.converter = MagicMock()
    finder._cache = {}
    no_prov = SimpleNamespace(
        prov=[],
        export_to_markdown=MagicMock(return_value="X"),
    )
    finder.converter.convert = MagicMock(
        return_value=SimpleNamespace(
            document=SimpleNamespace(tables=[no_prov]),
        )
    )
    doc = _FakeDoc("/tmp/fake.pdf", page_heights=[100.0])
    assert finder.candidates(doc, page_idx=0) == []


def test_export_to_markdown_failure_keeps_empty_text():
    """If a table's markdown export raises, the candidate is still
    emitted with empty text -- the bbox is what find_pdf_table cares
    about and an empty text simply fails the substring match."""
    finder = paper2md.DoclingFinder.__new__(paper2md.DoclingFinder)
    finder.converter = MagicMock()
    finder._cache = {}
    bad_export = MagicMock(side_effect=RuntimeError("boom"))
    tbl = SimpleNamespace(
        prov=[SimpleNamespace(
            page_no=1,
            bbox=_bb(0, 80, 50, 20, CoordOrigin.BOTTOMLEFT),
        )],
        export_to_markdown=bad_export,
    )
    finder.converter.convert = MagicMock(
        return_value=SimpleNamespace(
            document=SimpleNamespace(tables=[tbl]),
        )
    )
    doc = _FakeDoc("/tmp/fake.pdf", page_heights=[100.0])
    cands = finder.candidates(doc, page_idx=0)
    assert len(cands) == 1
    bbox, text = cands[0]
    assert text == ""


# ---------------------------------------------------------------------------
# build_table_finder dispatch
# ---------------------------------------------------------------------------

def test_build_table_finder_docling():
    """`build_table_finder('docling')` should return a DoclingFinder
    without invoking the actual docling import / model load. We patch
    DoclingFinder.__init__ to avoid the real conversion machinery."""
    with patch.object(paper2md.DoclingFinder, "__init__",
                      return_value=None):
        f = paper2md.build_table_finder("docling")
    assert isinstance(f, paper2md.DoclingFinder)


def test_build_table_finder_unknown_raises():
    with pytest.raises(ValueError, match="unknown table finder"):
        paper2md.build_table_finder("nonsense")


def test_docling_finder_helpful_error_when_uninstalled():
    """When docling isn't installed, DoclingFinder() raises a clear
    RuntimeError pointing the user at `pip install docling==2.92.0`
    rather than a bare ImportError. Drives the v0.3.x Option-B
    contract (docling is optional; install on demand)."""
    # Simulate `docling` missing from sys.modules + import failure.
    # Easiest: force the inner import to fail by patching importlib so
    # `from docling.document_converter import DocumentConverter` raises.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(RuntimeError) as ei:
            paper2md.DoclingFinder()
    msg = str(ei.value)
    assert "docling is not installed" in msg
    assert "pip install docling" in msg
    # Suggest --table-finder alternatives so user can recover without
    # installing docling.
    assert "pymupdf" in msg
