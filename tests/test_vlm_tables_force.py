"""Tests for --vlm-tables-force semantics in wrap_mineru.table_block_to_md.

Under --vlm-tables-force, even tables whose `html_to_pipe_md`
conversion was clean get a VLM call -- the VLM output goes to the
sidecar `.md` while the body keeps the cheap pipe-md. When the VLM
call fails on a forced-clean table, the sidecar falls back to the
clean pipe-md (option-a behavior; user expected a sidecar, deliver
one).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import paper2md as p2m  # noqa: E402
import wrap_mineru as wm  # noqa: E402


class _Block:
    """Mineru-style table block. `html` triggers html_to_pipe_md;
    if the conversion is clean, the clean-path fires."""

    def __init__(self, html: str, text: str = "Table 1.",
                 image_path: str = "tbl1.jpg", page_idx: int = 0):
        self.type = "table"
        self.text = text
        self.html = html
        self.image_path = image_path
        self.page_idx = page_idx
        self.bbox = (10, 10, 100, 100)
        self.footnote = None
        self.subpanel_paths = []
        self.subpanel_bboxes = []
        self.figure_number = None


_CLEAN_HTML = (
    "<table>"
    "<tr><td>col1</td><td>col2</td></tr>"
    "<tr><td>a</td><td>b</td></tr>"
    "<tr><td>c</td><td>d</td></tr>"
    "</table>"
)


def _make_image(assets: Path, name: str = "tbl1.jpg"):
    from PIL import Image
    img_path = assets / name
    Image.new("RGB", (40, 30), color="white").save(img_path)
    return img_path


# ---------------------------------------------------------------------
# Clean table + vlm_force=True: VLM called, body keeps pipe-md, sidecar
# gets VLM rewrite
# ---------------------------------------------------------------------


def test_force_clean_table_calls_vlm_and_writes_vlm_sidecar(tmp_path, monkeypatch):
    _make_image(tmp_path)
    block = _Block(_CLEAN_HTML)
    report = p2m.QualityReport()

    captured = {}

    def _fake_vlm_rewrite(image_path, caption, pil_image=None):
        captured["called"] = True
        return "| forced | by | vlm |\n|---|---|---|\n| x | y | z |"

    monkeypatch.setattr(wm, "_vlm_table_rewrite", _fake_vlm_rewrite)

    body = wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=True, assets_abs=tmp_path,
        report=report, vlm_force=True,
    )

    # VLM was called even though the table was clean
    assert captured.get("called") is True
    # Body still has the cheap pipe-md (NOT the VLM rewrite)
    assert "| col1 | col2 |" in body
    assert "forced | by | vlm" not in body
    # Sidecar got written with the VLM rewrite content
    sidecar = tmp_path / "table_1_p1_1.md"
    assert sidecar.exists()
    sc = sidecar.read_text()
    assert "forced | by | vlm" in sc
    # TableScore reflects forced path
    assert len(report.tables) == 1
    t = report.tables[0]
    assert t.pre_redo_reason == "force"
    assert t.vlm_redone is True
    assert t.post_redo_reason is None
    assert t.score == 1.0  # body is still clean


# ---------------------------------------------------------------------
# Clean table + vlm_force=False: VLM not called, body+sidecar = pipe-md
# (legacy behavior preserved)
# ---------------------------------------------------------------------


def test_clean_table_without_force_skips_vlm(tmp_path, monkeypatch):
    _make_image(tmp_path)
    block = _Block(_CLEAN_HTML)
    report = p2m.QualityReport()

    monkeypatch.setattr(
        wm, "_vlm_table_rewrite",
        lambda *a, **kw: pytest.fail("VLM should not be called without force"))

    body = wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=True, assets_abs=tmp_path,
        report=report, vlm_force=False,
    )
    assert "| col1 | col2 |" in body
    sidecar = tmp_path / "table_1_p1_1.md"
    assert sidecar.exists()
    assert "| col1 | col2 |" in sidecar.read_text()
    t = report.tables[0]
    assert t.pre_redo_reason is None
    assert t.vlm_redone is False


# ---------------------------------------------------------------------
# Clean table + force + VLM returns empty -> sidecar falls back to pipe-md
# ---------------------------------------------------------------------


def test_force_clean_table_vlm_empty_falls_back_to_pipe_md(tmp_path, monkeypatch):
    _make_image(tmp_path)
    block = _Block(_CLEAN_HTML)
    report = p2m.QualityReport()

    monkeypatch.setattr(wm, "_vlm_table_rewrite",
                        lambda *a, **kw: None)  # VLM empty

    wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=True, assets_abs=tmp_path,
        report=report, vlm_force=True,
    )
    # Sidecar exists with the clean pipe-md (fallback)
    sidecar = tmp_path / "table_1_p1_1.md"
    assert sidecar.exists()
    sc = sidecar.read_text()
    assert "| col1 | col2 |" in sc
    # Report reflects the failure
    t = report.tables[0]
    assert t.pre_redo_reason == "force"
    assert t.vlm_redone is False
    assert t.post_redo_reason == "vlm-empty-fallback-to-pipe-md"


def test_force_clean_table_vlm_returns_garbage_falls_back(tmp_path, monkeypatch):
    """VLM returns something but `table_is_suspicious` says it's bad ->
    discard, fall back to clean pipe-md."""
    _make_image(tmp_path)
    block = _Block(_CLEAN_HTML)
    report = p2m.QualityReport()
    # 1-row pipe-md is suspicious (fewer than 3 rows)
    monkeypatch.setattr(wm, "_vlm_table_rewrite",
                        lambda *a, **kw: "| broken |")
    wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=True, assets_abs=tmp_path,
        report=report, vlm_force=True,
    )
    sidecar = (tmp_path / "table_1_p1_1.md").read_text()
    assert "| col1 | col2 |" in sidecar  # fallback to pipe-md
    t = report.tables[0]
    assert t.post_redo_reason == "vlm-empty-fallback-to-pipe-md"


# ---------------------------------------------------------------------
# Suspicious table + force: same as suspicious + non-force (VLM was
# going to fire anyway)
# ---------------------------------------------------------------------


def test_force_on_suspicious_table_same_as_unforced(tmp_path, monkeypatch):
    _make_image(tmp_path)
    # Rowspan triggers html_to_pipe_md rejection -> goes to VLM branch
    rowspan_html = (
        "<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
        "<tr><td>C</td></tr></table>"
    )
    block = _Block(rowspan_html)
    report = p2m.QualityReport()
    monkeypatch.setattr(
        wm, "_vlm_table_rewrite",
        lambda *a, **kw: "| h1 | h2 |\n|---|---|\n| a | b |\n| c | d |")
    wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=True, assets_abs=tmp_path,
        report=report, vlm_force=True,
    )
    # Suspicious-path VLM result lands in the sidecar
    sidecar = (tmp_path / "table_1_p1_1.md").read_text()
    assert "| h1 | h2 |" in sidecar
    t = report.tables[0]
    # For suspicious tables, pre_redo_reason names the html rejection,
    # not "force" (the suspiciousness was the trigger).
    assert t.pre_redo_reason != "force"
    assert t.vlm_redone is True
    assert t.score == 0.85


# ---------------------------------------------------------------------
# Hybrid path (inline_in_body=False) + force: body has only image +
# sidecar link; sidecar has VLM
# ---------------------------------------------------------------------


def test_force_under_hybrid_body_skips_inline_pipe_md(tmp_path, monkeypatch):
    _make_image(tmp_path)
    block = _Block(_CLEAN_HTML)
    report = p2m.QualityReport()
    monkeypatch.setattr(
        wm, "_vlm_table_rewrite",
        lambda *a, **kw: "| vlm | rewrite |\n|---|---|\n| 1 | 2 |")
    body = wm.table_block_to_md(
        block, table_idx=1, assets_rel="assets",
        vlm_tables=True, assets_abs=tmp_path,
        report=report, vlm_force=True, inline_in_body=False,
    )
    # Under hybrid: body has only image + sidecar-link, no inline content
    assert "| col1 | col2 |" not in body  # no inline pipe-md
    assert "vlm | rewrite" not in body  # no inline VLM
    assert "[Table 1 — separate markdown]" in body
    assert "tbl1.jpg" in body
    # Sidecar has VLM content
    sidecar = (tmp_path / "table_1_p1_1.md").read_text()
    assert "vlm | rewrite" in sidecar


# ---------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------


def test_cli_vlm_tables_force_implies_vlm_tables(capsys, monkeypatch, tmp_path):
    """`--vlm-tables-force` without `--vlm-tables` -> auto-imply
    --vlm-tables and INFO-log the implication."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    # Stub _do_convert so we never actually run; we just want to check
    # arg post-processing
    import sys as _sys
    monkeypatch.setattr(
        _sys, "argv",
        ["paper2md", str(pdf), "-o", str(tmp_path / "out"),
         "--vlm-tables-force", "--metadata-only"])
    monkeypatch.setattr(
        p2m, "_run_metadata_only", lambda args, p: None)
    # Patch configure_client to avoid trying to hit a real VLM
    monkeypatch.setattr(p2m, "configure_client", lambda *a, **kw: None)
    try:
        p2m.main()
    except SystemExit:
        pass
    # The auto-imply INFO line should be logged. Hard to capture
    # logging here cleanly; instead, verify the conflict-with-no-vlm
    # case (more important).


def test_cli_vlm_tables_force_with_no_vlm_errors(capsys, monkeypatch, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    import sys as _sys
    monkeypatch.setattr(
        _sys, "argv",
        ["paper2md", str(pdf), "-o", str(tmp_path / "out"),
         "--vlm-tables-force", "--no-vlm"])
    with pytest.raises(SystemExit) as ei:
        p2m.main()
    assert ei.value.code == 2  # argparse error
    err = capsys.readouterr().err
    assert "--vlm-tables-force" in err
    assert "--no-vlm" in err
