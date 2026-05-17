"""Tests for the mineru_recovery module.

Strategy: build minimal fake `<paper-dir>/mineru/<stem>_middle.json`
fixtures that mimic MinerU 3.1.7's schema (the only piece we care
about: a `pdf_info[*].para_blocks` list with `type=table` blocks
carrying a caption text + html + image_path). Real middle.json files
are 100s of KB; the fixtures here stay surgical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import manual_edit  # noqa: E402
import mineru_recovery  # noqa: E402


# ---------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------


def _build_paper(tmp_path, body: str = "",
                 vlm_rewrite_tables: bool = True,
                 fm_extra: str = "") -> Path:
    """Build a minimal paper2md-shaped output with a configurable
    `run.pipeline.vlm_rewrite_tables` setting."""
    paper_dir = tmp_path / "paper_out"
    paper_dir.mkdir()
    (paper_dir / "assets").mkdir()
    (paper_dir / "mineru").mkdir()
    fm = (
        "---\n"
        "copyright:\n  doi: 10.1/test\n"
        "run:\n"
        "  paper2md_version: 0.4.0\n"
        "  pipeline:\n"
        f"    vlm_rewrite_tables: {str(vlm_rewrite_tables).lower()}\n"
    )
    if fm_extra:
        fm += fm_extra
    fm += "quality:\n  overall: 0.9\n---\n"
    paper = paper_dir / "paper.md"
    paper.write_text(fm + body)
    return paper


def _build_middle_json(paper_md: Path,
                       table_blocks: list[dict]) -> Path:
    """Write `mineru/<stem>_middle.json` with the given table blocks.

    Each `table_blocks[i]` is a dict with optional keys:
      id (str), html (str), caption (str), image_filename (str),
      page_idx (int), bbox (list).
    Captions auto-generate as "Table {id}. Caption text." if absent.
    """
    mineru_dir = paper_md.parent / "mineru"
    images_dir = mineru_dir / "images"
    images_dir.mkdir(exist_ok=True)
    para_blocks = []
    for i, t in enumerate(table_blocks):
        id_ = t.get("id", str(i + 1))
        caption = t.get("caption", f"Table {id_}. Caption text.")
        html = t.get("html", "<table><tr><td>a</td><td>b</td></tr></table>")
        img_name = t.get("image_filename", f"tbl_{id_}.jpg")
        img_path = images_dir / img_name
        img_path.write_bytes(b"FAKE-JPG-BYTES-FOR-TEST")
        para_blocks.append({
            "type": "table",
            "bbox": t.get("bbox", [100, 100, 500, 300]),
            "index": i,
            "score": 0.95,
            "blocks": [
                {"type": "table_caption", "bbox": [100, 80, 500, 95],
                 "index": i, "score": 0.95,
                 "lines": [{"bbox": [100, 80, 500, 95],
                            "spans": [{"type": "text",
                                       "content": caption,
                                       "bbox": [100, 80, 500, 95]}]}]},
                {"type": "table_body", "bbox": t.get("bbox", [100, 100, 500, 300]),
                 "index": i, "score": 0.95,
                 "lines": [{"bbox": [100, 100, 500, 300],
                            "spans": [{"type": "table", "html": html,
                                       "image_path": f"images/{img_name}",
                                       "bbox": [100, 100, 500, 300]}]}]},
            ],
        })
    middle = {
        "_backend": "pipeline",
        "_version_name": "3.1.7",
        "pdf_info": [{
            "page_idx": table_blocks[0].get("page_idx", 0) if table_blocks else 0,
            "page_size": [612, 792],
            "para_blocks": para_blocks,
            "discarded_blocks": [],
        }],
    }
    middle_path = mineru_dir / f"{paper_md.stem}_middle.json"
    middle_path.write_text(json.dumps(middle))
    return middle_path


def _stub_vlm(image_path, caption=None):
    return "| recovered | header |\n|---|---|\n| x | y |\n"


def _parse_fm(md_path: Path) -> dict:
    text = md_path.read_text()
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end])


# ---------------------------------------------------------------------
# find_recoverable_tables
# ---------------------------------------------------------------------


def test_find_recoverable_matches_missing_ref(tmp_path):
    """Body cites Table 1 + Table 2 but only Table 1 is inserted.
    middle.json has both. Result: just Table 2 (the missing one)."""
    paper = _build_paper(
        tmp_path,
        body="See Table 1 and Table 2.\n\n"
             "**Table 1.** First.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    )
    _build_middle_json(paper, [
        {"id": "1", "caption": "Table 1. First."},
        {"id": "2", "caption": "Table 2. Second."},
    ])
    items = mineru_recovery.find_recoverable_tables(paper)
    assert len(items) == 1
    assert items[0]["id"] == "2"
    assert items[0]["canonical_id"] == "2"
    assert items[0]["html"].startswith("<table>")
    assert items[0]["image_path"].exists()


def test_find_recoverable_no_middle_json(tmp_path):
    """When no mineru/ subdir, return [] cleanly (not an error)."""
    paper = _build_paper(tmp_path, body="See Table 1.\n")
    # Don't build middle.json. Remove the dir to make it explicit.
    (paper.parent / "mineru").rmdir()
    items = mineru_recovery.find_recoverable_tables(paper)
    assert items == []


def test_find_recoverable_ref_not_in_mineru(tmp_path):
    """Body cites Table 5, but middle.json only has Table 1.
    Table 5 is unrecoverable -> excluded from result."""
    paper = _build_paper(
        tmp_path,
        body="See Table 1 and Table 5.\n\n**Table 1.** First.\n",
    )
    _build_middle_json(paper, [{"id": "1"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    assert items == []


def test_find_recoverable_id_from_html_when_caption_missing(tmp_path):
    """Real-world case (feng paper): MinerU classifies the table fine
    but doesn't classify the caption block -- the `Table N` label is
    embedded as the first content row of the HTML instead. The
    fallback HTML id-extractor still matches it."""
    paper = _build_paper(
        tmp_path,
        body="As shown in Table 1, results are key.\n",
    )
    # Build middle.json with NO caption block, but Table 1 in HTML
    mineru_dir = paper.parent / "mineru"
    images_dir = mineru_dir / "images"
    images_dir.mkdir(exist_ok=True)
    img = images_dir / "x.jpg"
    img.write_bytes(b"FAKE")
    middle = {
        "_backend": "pipeline", "_version_name": "3.1.7",
        "pdf_info": [{
            "page_idx": 2, "page_size": [612, 792],
            "para_blocks": [{
                "type": "table", "bbox": [100, 100, 500, 300],
                "index": 0, "score": 0.95,
                "blocks": [
                    # NOTE: no table_caption child here
                    {"type": "table_body",
                     "bbox": [100, 100, 500, 300],
                     "index": 0, "score": 0.95,
                     "lines": [{"bbox": [100, 100, 500, 300],
                                "spans": [{"type": "table",
                                           "html": '<table><tr><td colspan="6">Table 1</td></tr><tr><td>x</td></tr></table>',
                                           "image_path": "images/x.jpg",
                                           "bbox": [100, 100, 500, 300]}]}]},
                ],
            }],
            "discarded_blocks": [],
        }],
    }
    (mineru_dir / f"{paper.stem}_middle.json").write_text(json.dumps(middle))

    items = mineru_recovery.find_recoverable_tables(paper)
    assert len(items) == 1
    assert items[0]["canonical_id"] == "1"


def test_find_recoverable_canonical_roman_match(tmp_path):
    """Body cites Table V; middle.json has Table 5. The canonical-id
    bridge matches them. (Mimics Knudson2013-style mixed numbering.)"""
    paper = _build_paper(
        tmp_path,
        body="See Table V for results.\n",
    )
    _build_middle_json(paper, [{"id": "5", "caption": "Table 5. Roman."}])
    items = mineru_recovery.find_recoverable_tables(paper)
    assert len(items) == 1
    assert items[0]["canonical_id"] == "5"


# ---------------------------------------------------------------------
# detect_original_vlm_mode
# ---------------------------------------------------------------------


def test_detect_vlm_mode_true(tmp_path):
    paper = _build_paper(tmp_path, vlm_rewrite_tables=True)
    assert mineru_recovery.detect_original_vlm_mode(paper) is True


def test_detect_vlm_mode_false(tmp_path):
    paper = _build_paper(tmp_path, vlm_rewrite_tables=False)
    assert mineru_recovery.detect_original_vlm_mode(paper) is False


def test_detect_vlm_mode_missing_field(tmp_path):
    """No `run.pipeline.vlm_rewrite_tables` -> False (safe default)."""
    paper_dir = tmp_path / "p"
    paper_dir.mkdir()
    paper = paper_dir / "paper.md"
    paper.write_text("---\nquality:\n  overall: 1.0\n---\n# body\n")
    assert mineru_recovery.detect_original_vlm_mode(paper) is False


# ---------------------------------------------------------------------
# stage_one_table
# ---------------------------------------------------------------------


def test_stage_vlm_path_writes_sidecar(tmp_path):
    """Body references Table 1 but the splice dropped it -- no caption
    line in the body. MinerU has it; recovery stages from MinerU."""
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 confirm the trend.\n",
    )
    _build_middle_json(paper, [{"id": "1"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    result = mineru_recovery.stage_one_table(
        paper, items[0], use_vlm=True,
        vlm_callback=_stub_vlm,
        vlm_model_name=lambda: "test-model",
    )
    sidecar = paper.parent / "assets" / "user_edit_table_1.md"
    asset = paper.parent / "assets" / "user_edit_table_1.jpg"
    assert sidecar.exists()
    assert asset.exists()
    assert "recovered | header" in sidecar.read_text()

    fm = _parse_fm(paper)
    assert isinstance(fm.get("pending_recoveries"), list)
    e = fm["pending_recoveries"][0]
    assert e["kind"] == "table_replacement"
    assert e["id"] == "1"
    assert e["source"] == "vlm_rewrite"
    assert e["vlm_model"] == "test-model"
    assert e["mineru_page_idx"] == 0
    assert result["source"] == "vlm_rewrite"


def test_stage_html_path_uses_pipe_md(tmp_path):
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 are key.\n",
    )
    _build_middle_json(paper, [{"id": "1",
                                "html": "<table><tr><td>x</td><td>y</td></tr></table>"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    result = mineru_recovery.stage_one_table(
        paper, items[0], use_vlm=False,
        vlm_callback=lambda *a, **kw: pytest.fail("VLM should not run"),
    )
    sidecar = (paper.parent / "assets" / "user_edit_table_1.md").read_text()
    assert "x" in sidecar  # html converted to pipe-md
    assert result["source"] in ("mineru_html", "mineru_html_raw")
    fm = _parse_fm(paper)
    assert fm["pending_recoveries"][0]["source"] == result["source"]
    assert fm["pending_recoveries"][0]["vlm_model"] is None


def test_stage_vlm_empty_falls_back_to_html(tmp_path):
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 confirm.\n",
    )
    _build_middle_json(paper, [{"id": "1"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    result = mineru_recovery.stage_one_table(
        paper, items[0], use_vlm=True,
        vlm_callback=lambda *a, **kw: None,
        vlm_model_name=lambda: "test-model",
    )
    # Source ends with `_fallback`
    assert result["source"].endswith("_fallback")
    fm = _parse_fm(paper)
    assert fm["pending_recoveries"][0]["source"] == result["source"]


def test_stage_idempotent_by_id(tmp_path):
    """Re-staging the same table id REPLACES the pending entry; one
    entry, not two."""
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 confirm.\n",
    )
    _build_middle_json(paper, [{"id": "1"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    mineru_recovery.stage_one_table(
        paper, items[0], use_vlm=False)
    mineru_recovery.stage_one_table(
        paper, items[0], use_vlm=False)
    fm = _parse_fm(paper)
    assert len(fm["pending_recoveries"]) == 1


def test_stage_refuses_if_already_confirmed(tmp_path):
    """If `edits:` already has a `recovered_from: mineru` entry for
    the same id, refuse -- the user has presumably pasted; running
    again would clobber their work silently."""
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 confirm.\n",
    )
    _build_middle_json(paper, [{"id": "1"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    mineru_recovery.stage_one_table(paper, items[0], use_vlm=False)
    mineru_recovery.confirm_recovery(paper, "table", "1")
    # Now another recovery attempt for Table 1 must error.
    with pytest.raises(RuntimeError, match="already confirmed"):
        mineru_recovery.stage_one_table(paper, items[0], use_vlm=False)


# ---------------------------------------------------------------------
# confirm_recovery
# ---------------------------------------------------------------------


def test_confirm_recovery_moves_pending_to_edits(tmp_path):
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 confirm.\n",
    )
    _build_middle_json(paper, [{"id": "1"}])
    items = mineru_recovery.find_recoverable_tables(paper)
    mineru_recovery.stage_one_table(paper, items[0], use_vlm=False)

    result = mineru_recovery.confirm_recovery(paper, "table", "1")
    fm = _parse_fm(paper)
    # pending_recoveries is empty / absent
    assert not fm.get("pending_recoveries")
    # edits has the promoted entry
    assert len(fm["edits"]) == 1
    e = fm["edits"][0]
    assert e["kind"] == "table_replacement"
    assert e["id"] == "1"
    assert e["recovered_from"] == "mineru"
    assert "confirmed_at" in e
    assert e["source"]  # preserved from staging
    assert result["entry"]["recovered_from"] == "mineru"


def test_confirm_nonexistent_raises(tmp_path):
    paper = _build_paper(tmp_path)
    with pytest.raises(LookupError, match="no pending recovery"):
        mineru_recovery.confirm_recovery(paper, "table", "99")


# ---------------------------------------------------------------------
# do_recover_from_mineru (orchestrator)
# ---------------------------------------------------------------------


def test_orchestrator_recovers_all_missing(tmp_path):
    paper = _build_paper(
        tmp_path,
        body="See Table 1 and Table 2.\n\n"
             "**Table 1.** First.\n\n| a |\n|---|\n| 1 |\n",
        vlm_rewrite_tables=False,  # disable VLM -> HTML path
    )
    _build_middle_json(paper, [
        {"id": "1"},
        {"id": "2"},
    ])
    result = mineru_recovery.do_recover_from_mineru(paper)
    assert result["use_vlm"] is False
    assert result["vlm_origin"] == "original"
    # Only Table 2 needed recovery
    assert len(result["recovered"]) == 1
    assert result["recovered"][0]["id"] == "2"
    fm = _parse_fm(paper)
    assert len(fm["pending_recoveries"]) == 1


def test_orchestrator_vlm_override(tmp_path, monkeypatch):
    """--vlm forces VLM even when the original run was --no-vlm-tables."""
    paper = _build_paper(
        tmp_path,
        body="The results in Table 1 confirm.\n",  # ref + caption only
        vlm_rewrite_tables=False,
    )
    # Body cites Table 2 too -- the missing one
    paper.write_text(paper.read_text().replace(
        "See Table 1.", "See Table 1 and Table 2."))
    _build_middle_json(paper, [{"id": "1"}, {"id": "2"}])
    # Inject a VLM stub by monkeypatching the module callback default
    monkeypatch.setattr(manual_edit, "_real_vlm_table_call",
                        lambda img, cap=None: "| z |\n|---|\n| 1 |\n")
    monkeypatch.setattr(manual_edit, "_real_vlm_model_name",
                        lambda: "stub-vlm")
    result = mineru_recovery.do_recover_from_mineru(
        paper, vlm_override=True)
    assert result["use_vlm"] is True
    assert result["vlm_origin"] == "override"
    fm = _parse_fm(paper)
    e = fm["pending_recoveries"][0]
    assert e["source"] == "vlm_rewrite"
    assert e["vlm_model"] == "stub-vlm"


def test_orchestrator_no_recoverable(tmp_path):
    """Body cites Table 5 but middle.json doesn't have it. Result:
    nothing recovered, clean summary."""
    paper = _build_paper(tmp_path, body="See Table 5.\n")
    _build_middle_json(paper, [{"id": "1"}])  # Table 5 not present
    result = mineru_recovery.do_recover_from_mineru(paper)
    assert result["recovered"] == []
    assert "No recoverable" in result["summary"]


def test_orchestrator_handles_missing_image_gracefully(tmp_path):
    """If MinerU's image is referenced but missing on disk, record an
    error per-item rather than aborting the whole run."""
    paper = _build_paper(
        tmp_path,
        body="See Table 1 and Table 2.\n",
        vlm_rewrite_tables=False,
    )
    _build_middle_json(paper, [
        {"id": "1", "image_filename": "tbl1.jpg"},
        {"id": "2", "image_filename": "tbl2.jpg"},
    ])
    # Delete one of the images on disk before staging
    (paper.parent / "mineru" / "images" / "tbl2.jpg").unlink()
    result = mineru_recovery.do_recover_from_mineru(paper)
    by_id = {r["id"]: r for r in result["recovered"]}
    # Table 1 staged fine; Table 2 errored
    assert "error" not in by_id["1"]
    assert "error" in by_id["2"]
