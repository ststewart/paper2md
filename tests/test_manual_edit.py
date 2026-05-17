"""Tests for the user-driven manual replacement workflow.

Covers:
  - do_replace_table: VLM stub call, sidecar write, image copy,
    frontmatter `edits:` entry, paste hint with caption line.
  - do_replace_fig: in-place asset overwrite, .orig archive,
    frontmatter entry.
  - Idempotency: re-running with same kind+id replaces (not appends).
  - Other frontmatter keys preserved across edit.
  - do_revert: figure restores .orig and drops entry; table just
    drops entry.
  - Error paths: missing image, missing paper, missing asset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import manual_edit  # noqa: E402
import yaml  # noqa: E402


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _write_paper(paper_dir: Path, *, fm_extra: str = "",
                 body: str = "") -> Path:
    """Build a minimal paper2md-shaped output: paper.md + assets/."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    assets = paper_dir / "assets"
    assets.mkdir(exist_ok=True)
    fm = "---\ncopyright:\n  doi: 10.1/test\nquality:\n  overall: 0.9\n"
    if fm_extra:
        fm += fm_extra
    fm += "---\n"
    paper = paper_dir / "paper.md"
    paper.write_text(fm + (body or "# Body\n\n"))
    return paper


def _stub_vlm_table(image_path: Path,
                    caption=None) -> str:
    """Test stub: returns a fixed pipe-md table without calling the VLM."""
    return "| col1 | col2 |\n|---|---|\n| a | b |\n"


def _parse_frontmatter(md_path: Path) -> dict:
    text = md_path.read_text()
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end])


# ---------------------------------------------------------------------
# Table replacement
# ---------------------------------------------------------------------


def test_replace_table_writes_sidecar_and_edits(tmp_path):
    paper = _write_paper(
        tmp_path,
        body="# Paper\n\n**Table 1.** Caption.\n\n"
             "| broken | content |\n|---|---|\n| x | y |\n",
    )
    img = tmp_path / "crop.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    result = manual_edit.do_replace_table(
        paper, "1", img, note="splice garbled",
        vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "test-model",
    )

    # Sidecar + image both landed in assets/
    sidecar = paper.parent / "assets" / "user_edit_table_1.md"
    assert sidecar.exists()
    assert "| col1 | col2 |" in sidecar.read_text()
    assert (paper.parent / "assets" / "user_edit_table_1.png").exists()

    # Frontmatter has the edit
    fm = _parse_frontmatter(paper)
    assert isinstance(fm.get("edits"), list)
    assert len(fm["edits"]) == 1
    e = fm["edits"][0]
    assert e["kind"] == "table_replacement"
    assert e["id"] == "1"
    assert e["image"] == "assets/user_edit_table_1.png"
    assert e["sidecar"] == "assets/user_edit_table_1.md"
    assert e["vlm_model"] == "test-model"
    assert e["note"] == "splice garbled"
    assert "image_sha256" in e and len(e["image_sha256"]) == 64

    # Paste hint locates the caption line
    assert result["caption_line"] is not None
    assert "Table 1" in result["paste_hint"]
    assert "user_edit_table_1.md" in result["paste_hint"]

    # Body NOT modified -- garbled table still present
    body = paper.read_text().split("\n---\n", 1)[1]
    assert "broken" in body


def test_replace_table_idempotent_replaces_not_appends(tmp_path):
    """Running twice with same id REPLACES the prior entry; one entry
    in `edits:` not two."""
    paper = _write_paper(tmp_path, body="**Table 1.** Cap.\n")
    img = tmp_path / "crop.png"
    img.write_bytes(b"fake1")

    manual_edit.do_replace_table(
        paper, "1", img, vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "m")

    img.write_bytes(b"fake2-different-bytes")  # different sha
    manual_edit.do_replace_table(
        paper, "1", img, note="second pass",
        vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "m")

    fm = _parse_frontmatter(paper)
    assert len(fm["edits"]) == 1
    assert fm["edits"][0]["note"] == "second pass"


def test_replace_table_preserves_other_frontmatter_keys(tmp_path):
    paper = _write_paper(
        tmp_path,
        fm_extra="run:\n  paper2md_version: 0.4.0\n",
        body="**Table 1.** Cap.\n",
    )
    img = tmp_path / "crop.png"
    img.write_bytes(b"fake")
    manual_edit.do_replace_table(
        paper, "1", img, vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "m")

    fm = _parse_frontmatter(paper)
    assert fm["copyright"]["doi"] == "10.1/test"
    assert fm["quality"]["overall"] == 0.9
    assert fm["run"]["paper2md_version"] == "0.4.0"
    assert isinstance(fm["edits"], list)


def test_replace_table_vlm_empty_raises(tmp_path):
    paper = _write_paper(tmp_path, body="**Table 1.** Cap.\n")
    img = tmp_path / "crop.png"
    img.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="VLM returned empty"):
        manual_edit.do_replace_table(
            paper, "1", img,
            vlm_callback=lambda *a, **kw: None,
            vlm_model_name=lambda: "m")


def test_replace_table_missing_image_raises(tmp_path):
    paper = _write_paper(tmp_path)
    with pytest.raises(FileNotFoundError, match="image not found"):
        manual_edit.do_replace_table(
            paper, "1", tmp_path / "no-such.png",
            vlm_callback=_stub_vlm_table,
            vlm_model_name=lambda: "m")


# ---------------------------------------------------------------------
# Figure replacement
# ---------------------------------------------------------------------


def test_replace_fig_overwrites_asset_in_place(tmp_path):
    """Existing figure asset is overwritten; original archived as .orig.
    Body unchanged."""
    paper = _write_paper(
        tmp_path,
        body="**Fig. 3.** Caption.\n\n![](assets/figure_3.jpg)\n",
    )
    orig_asset = paper.parent / "assets" / "figure_3.jpg"
    orig_asset.write_bytes(b"OLD")
    new_img = tmp_path / "fixed.jpg"
    new_img.write_bytes(b"NEW-CORRECT")

    result = manual_edit.do_replace_fig(paper, "3", new_img)

    assert orig_asset.read_bytes() == b"NEW-CORRECT"
    archive = paper.parent / "assets" / "figure_3.jpg.orig"
    assert archive.exists()
    assert archive.read_bytes() == b"OLD"
    assert result["archived_original"] == archive

    fm = _parse_frontmatter(paper)
    assert len(fm["edits"]) == 1
    e = fm["edits"][0]
    assert e["kind"] == "figure_replacement"
    assert e["id"] == "3"
    assert e["image"] == "assets/figure_3.jpg"
    assert e["original_archived_as"] == "assets/figure_3.jpg.orig"


def test_replace_fig_locates_asset_via_caption_proximity(tmp_path):
    """Asset name doesn't match the `figure_<id>` convention (e.g. a
    MinerU hash). The caption-proximity scan still finds it."""
    paper = _write_paper(
        tmp_path,
        body="**Fig. 2.** Caption.\n\n![](assets/abc123hash.jpg)\n",
    )
    hash_asset = paper.parent / "assets" / "abc123hash.jpg"
    hash_asset.write_bytes(b"OLD")
    new_img = tmp_path / "fix.jpg"
    new_img.write_bytes(b"NEW")

    manual_edit.do_replace_fig(paper, "2", new_img)
    assert hash_asset.read_bytes() == b"NEW"
    assert (paper.parent / "assets" / "abc123hash.jpg.orig").exists()


def test_replace_fig_idempotent_keeps_first_orig(tmp_path):
    """Second run does NOT clobber the original .orig (loses the true
    original). The `edits:` entry replaces, .orig is preserved."""
    paper = _write_paper(
        tmp_path,
        body="**Fig. 1.** Cap.\n\n![](assets/figure_1.png)\n",
    )
    asset = paper.parent / "assets" / "figure_1.png"
    asset.write_bytes(b"TRUE-ORIGINAL")

    fix1 = tmp_path / "fix1.png"
    fix1.write_bytes(b"first-replacement")
    manual_edit.do_replace_fig(paper, "1", fix1)

    fix2 = tmp_path / "fix2.png"
    fix2.write_bytes(b"second-replacement")
    manual_edit.do_replace_fig(paper, "1", fix2)

    orig = paper.parent / "assets" / "figure_1.png.orig"
    assert orig.read_bytes() == b"TRUE-ORIGINAL"
    assert asset.read_bytes() == b"second-replacement"

    fm = _parse_frontmatter(paper)
    assert len(fm["edits"]) == 1


def test_replace_fig_missing_asset_raises(tmp_path):
    paper = _write_paper(
        tmp_path,
        body="**Fig. 5.** Mentioned but no insertion.\n",
    )
    img = tmp_path / "fix.png"
    img.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="could not locate"):
        manual_edit.do_replace_fig(paper, "5", img)


# ---------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------


def test_revert_figure_restores_orig_and_drops_entry(tmp_path):
    paper = _write_paper(
        tmp_path,
        body="**Fig. 1.** Cap.\n\n![](assets/figure_1.png)\n",
    )
    asset = paper.parent / "assets" / "figure_1.png"
    asset.write_bytes(b"TRUE")

    fix = tmp_path / "wrong.png"
    fix.write_bytes(b"WRONG")
    manual_edit.do_replace_fig(paper, "1", fix)
    assert asset.read_bytes() == b"WRONG"

    result = manual_edit.do_revert(paper, "figure", "1")

    assert asset.read_bytes() == b"TRUE"
    assert not (paper.parent / "assets" / "figure_1.png.orig").exists()
    assert result["restored"] == asset

    fm = _parse_frontmatter(paper)
    assert "edits" not in fm or fm.get("edits") in (None, [])


def test_revert_table_drops_entry_only(tmp_path):
    """Table revert is metadata-only: sidecar stays on disk."""
    paper = _write_paper(tmp_path, body="**Table 1.** Cap.\n")
    img = tmp_path / "crop.png"
    img.write_bytes(b"fake")
    manual_edit.do_replace_table(
        paper, "1", img, vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "m")
    sidecar = paper.parent / "assets" / "user_edit_table_1.md"
    assert sidecar.exists()

    manual_edit.do_revert(paper, "table", "1")
    fm = _parse_frontmatter(paper)
    assert "edits" not in fm or fm.get("edits") in (None, [])
    # Sidecar preserved
    assert sidecar.exists()


def test_revert_unknown_raises(tmp_path):
    paper = _write_paper(tmp_path)
    with pytest.raises(LookupError, match="no recorded edit"):
        manual_edit.do_revert(paper, "table", "42")


def test_revert_kept_entries_survive(tmp_path):
    """Reverting one of multiple edits keeps the others intact."""
    paper = _write_paper(tmp_path, body="**Table 1.** A.\n**Table 2.** B.\n")
    img = tmp_path / "crop.png"
    img.write_bytes(b"x")
    manual_edit.do_replace_table(
        paper, "1", img, vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "m")
    manual_edit.do_replace_table(
        paper, "2", img, vlm_callback=_stub_vlm_table,
        vlm_model_name=lambda: "m")

    manual_edit.do_revert(paper, "table", "1")

    fm = _parse_frontmatter(paper)
    assert len(fm["edits"]) == 1
    assert fm["edits"][0]["id"] == "2"


# ---------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------


def test_extract_existing_edits_handles_no_block():
    fm = "---\ncopyright:\n  doi: 10.1/x\n---\n"
    assert manual_edit._extract_existing_edits(fm) == []


def test_extract_existing_edits_parses_block():
    fm = (
        "---\n"
        "copyright:\n  doi: 10.1/x\n"
        "edits:\n"
        "  - kind: table_replacement\n"
        "    id: '1'\n"
        "    note: hi\n"
        "---\n"
    )
    out = manual_edit._extract_existing_edits(fm)
    assert len(out) == 1
    assert out[0]["kind"] == "table_replacement"
    assert out[0]["note"] == "hi"


def test_find_caption_line():
    body = "intro\n\n**Table 1.** Description.\n\n| a | b |\n"
    assert manual_edit._find_caption_line(body, "table", "1") == 3
    assert manual_edit._find_caption_line(body, "figure", "1") is None
