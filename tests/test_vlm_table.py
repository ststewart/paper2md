"""Tests for the standalone vlm-table tool."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import paper2md as p2m  # noqa: E402
import vlm_table  # noqa: E402


def _make_image(tmp_path: Path, name: str = "table.png") -> Path:
    """Write a tiny valid PNG so PIL.Image.open succeeds."""
    p = tmp_path / name
    Image.new("RGB", (40, 30), color="white").save(p)
    return p


# ---------------------------------------------------------------------
# transcribe_table_image() core
# ---------------------------------------------------------------------


def test_md_format_uses_table_prompt(tmp_path, monkeypatch):
    """--format md (default) routes through paper2md.TABLE_PROMPT."""
    captured = {}

    def _fake_vlm(prompt, img, **kwargs):
        captured["prompt"] = prompt
        return "| a | b |\n|---|---|\n| 1 | 2 |"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    img = _make_image(tmp_path)

    out = vlm_table.transcribe_table_image(img, fmt="md")
    assert captured["prompt"] == p2m.TABLE_PROMPT
    assert "| a | b |" in out


def test_csv_format_uses_csv_prompt(tmp_path, monkeypatch):
    """--format csv routes through TABLE_PROMPT_CSV (different prompt)."""
    captured = {}

    def _fake_vlm(prompt, img, **kwargs):
        captured["prompt"] = prompt
        return "a,b\n1,2\n"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    img = _make_image(tmp_path)

    out = vlm_table.transcribe_table_image(img, fmt="csv")
    assert captured["prompt"] == vlm_table.TABLE_PROMPT_CSV
    assert "RFC 4180" in captured["prompt"]
    assert "a,b" in out


def test_caption_prepended_to_prompt(tmp_path, monkeypatch):
    """--caption text is prepended as 'The table caption is: ...' to
    give the VLM disambiguating context."""
    captured = {}

    def _fake_vlm(prompt, img, **kwargs):
        captured["prompt"] = prompt
        return "x,y\n"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    img = _make_image(tmp_path)

    vlm_table.transcribe_table_image(
        img, fmt="csv", caption="Hugoniot data for coesite")
    assert captured["prompt"].startswith(
        "The table caption is: 'Hugoniot data for coesite'.")


def test_empty_vlm_response_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(p2m, "vlm", lambda *a, **kw: None)
    img = _make_image(tmp_path)
    with pytest.raises(RuntimeError, match="empty response"):
        vlm_table.transcribe_table_image(img, fmt="md")


def test_invalid_format_raises(tmp_path):
    img = _make_image(tmp_path)
    with pytest.raises(ValueError, match="fmt must be"):
        vlm_table.transcribe_table_image(img, fmt="json")


def test_code_fence_stripped(tmp_path, monkeypatch):
    """The model sometimes wraps output in ``` fences despite the
    prompt's 'no fences' instruction. Strip them so downstream
    consumers get pure pipe-md / CSV."""
    monkeypatch.setattr(
        p2m, "vlm",
        lambda *a, **kw: "```csv\na,b\n1,2\n```",
    )
    img = _make_image(tmp_path)
    out = vlm_table.transcribe_table_image(img, fmt="csv")
    assert "```" not in out
    assert "a,b" in out


# ---------------------------------------------------------------------
# CLI (main())
# ---------------------------------------------------------------------


def test_cli_writes_to_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p2m, "vlm", lambda *a, **kw: "| h |\n|---|\n| v |")
    monkeypatch.setattr(p2m, "configure_client", lambda *a, **kw: None)
    img = _make_image(tmp_path)

    rc = vlm_table.main([str(img)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| h |" in out


def test_cli_writes_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(p2m, "vlm", lambda *a, **kw: "x,y\n1,2")
    monkeypatch.setattr(p2m, "configure_client", lambda *a, **kw: None)
    img = _make_image(tmp_path)
    out_path = tmp_path / "result.csv"

    rc = vlm_table.main([str(img), "-f", "csv", "-o", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert "x,y" in out_path.read_text()


def test_cli_missing_image_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p2m, "configure_client", lambda *a, **kw: None)
    with pytest.raises(SystemExit) as e:
        vlm_table.main([str(tmp_path / "nope.png")])
    assert e.value.code == 2  # argparse error
    err = capsys.readouterr().err
    assert "image not found" in err


def test_cli_vlm_empty_returns_2(tmp_path, monkeypatch):
    """Exit code 2 when VLM returns empty -- distinguishable from
    argparse errors (also 2 but with stderr msg) and success (0)."""
    monkeypatch.setattr(p2m, "vlm", lambda *a, **kw: None)
    monkeypatch.setattr(p2m, "configure_client", lambda *a, **kw: None)
    img = _make_image(tmp_path)
    rc = vlm_table.main([str(img)])
    assert rc == 2


def test_cli_max_tokens_threaded_through(tmp_path, monkeypatch):
    captured = {}

    def _fake_vlm(prompt, img, max_tokens=1500, **kwargs):
        captured["max_tokens"] = max_tokens
        return "| a |\n|---|\n| 1 |"

    monkeypatch.setattr(p2m, "vlm", _fake_vlm)
    monkeypatch.setattr(p2m, "configure_client", lambda *a, **kw: None)
    img = _make_image(tmp_path)

    vlm_table.main([str(img), "--max-tokens", "9000"])
    assert captured["max_tokens"] == 9000
