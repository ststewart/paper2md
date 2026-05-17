"""Phase B + memory-check tests.

Phase B: a supplement (SI) processing failure must NOT discard the
successful main-paper output. The batch manifest reflects `status=ok`
on the main with separate `si_status=error` + `si_error*` fields.

Memory check: `layout_mineru._check_memory_before_mineru` emits a clear
WARN when /proc/meminfo says available memory is below 8 GiB, so the
user knows to drop vLLM `--gpu-memory-utilization` before debugging a
CUDA OOM that hasn't happened yet.
"""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import paper2md as p2m  # noqa: E402
import layout_mineru as lm  # noqa: E402


# === Phase B: SI failure isolated from main =============================


class _Args:
    """Minimal stub for the argparse Namespace passed to run_batch.
    Only the fields _process_one reads."""
    def __init__(self, out: Path):
        self.out = out
        self.force = True
        self.hdf5 = False
        self.quality_threshold = None
        self.paper_timeout = 0  # disable timeout for tests
        self.workers = 1


def _fake_do_convert_factory(behaviors: dict):
    """Build a do_convert stub keyed on the PDF stem. `behaviors` maps
    stem -> ("ok", md_path) or ("raise", exc). Returns (md_path, report)
    or raises."""
    def _fake(pdf_path, paper_dir, asset_prefix=""):
        stem = pdf_path.stem
        action = behaviors[stem]
        if action[0] == "raise":
            raise action[1]
        # ok path: make a stub MD file + a minimal QualityReport
        md_path = paper_dir / f"{stem}.md"
        md_path.write_text(f"# {stem}\n")
        rpt = p2m.QualityReport()
        return md_path, rpt
    return _fake


def _read_manifest(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in
            (out_dir / "manifest.jsonl").read_text().splitlines()
            if line.strip()]


def test_si_failure_keeps_main_paper_status_ok(tmp_path):
    """Main paper succeeds; SI raises MineruRunError. The entry must
    have status=ok (main is good), grade present, plus si_status=error
    + si_error_kind classifying the SI failure."""
    main_pdf = tmp_path / "paper.pdf"
    si_pdf = tmp_path / "paper_SI.pdf"
    main_pdf.write_bytes(b"%PDF-1.4 fake main")
    si_pdf.write_bytes(b"%PDF-1.4 fake si")

    si_err = lm.MineruRunError(
        returncode=1, pdf_path=si_pdf, kind="cuda_oom",
        stderr_tail="CUDA out of memory. Tried to allocate 2.50 GiB")
    do_convert = _fake_do_convert_factory({
        "paper": ("ok", None),
        "paper_SI": ("raise", si_err),
    })

    out_dir = tmp_path / "out"
    args = _Args(out_dir)
    p2m.run_batch([(main_pdf, si_pdf)], args, do_convert)

    entries = _read_manifest(out_dir)
    assert len(entries) == 1
    e = entries[0]
    assert e["status"] == "ok"  # main succeeded
    assert e["grade"] == "F"    # empty report -> overall 0 -> F (fine)
    assert e["si_status"] == "error"
    assert "MineruRunError" in e["si_error"]
    assert e["si_error_kind"] == "cuda_oom"
    assert e["si_error_pdf"] == "paper_SI.pdf"
    assert "CUDA out of memory" in e["si_error_stderr_tail"]
    # The main-paper error fields must NOT be present.
    assert "error" not in e
    assert "error_kind" not in e


def test_si_success_records_si_status_ok(tmp_path):
    """Happy path: both main and SI succeed. Entry has status=ok and
    si_status=ok; no si_error fields."""
    main_pdf = tmp_path / "paper.pdf"
    si_pdf = tmp_path / "paper_SI.pdf"
    main_pdf.write_bytes(b"%PDF")
    si_pdf.write_bytes(b"%PDF")
    do_convert = _fake_do_convert_factory({
        "paper": ("ok", None),
        "paper_SI": ("ok", None),
    })
    out_dir = tmp_path / "out"
    p2m.run_batch([(main_pdf, si_pdf)], _Args(out_dir), do_convert)

    e = _read_manifest(out_dir)[0]
    assert e["status"] == "ok"
    assert e["si_status"] == "ok"
    assert "si_error" not in e
    assert "si_error_kind" not in e


def test_main_failure_skips_si_attempt(tmp_path):
    """Main fails -> entry status=error, SI is NOT attempted (would
    waste GPU time on an unsalvageable paper). No si_* fields appear."""
    main_pdf = tmp_path / "paper.pdf"
    si_pdf = tmp_path / "paper_SI.pdf"
    main_pdf.write_bytes(b"%PDF")
    si_pdf.write_bytes(b"%PDF")
    main_err = lm.MineruRunError(
        returncode=1, pdf_path=main_pdf, kind="cuda_oom",
        stderr_tail="CUDA out of memory")
    si_calls = {"n": 0}

    def _do_convert(pdf_path, paper_dir, asset_prefix=""):
        if pdf_path.stem == "paper":
            raise main_err
        si_calls["n"] += 1
        md = paper_dir / f"{pdf_path.stem}.md"
        md.write_text("x")
        return md, p2m.QualityReport()

    out_dir = tmp_path / "out"
    p2m.run_batch([(main_pdf, si_pdf)], _Args(out_dir), _do_convert)

    e = _read_manifest(out_dir)[0]
    assert e["status"] == "error"
    assert e["error_kind"] == "cuda_oom"
    assert si_calls["n"] == 0  # SI never invoked
    assert "si_status" not in e


def test_manifest_entry_includes_audit_fields(tmp_path):
    """Each successful main entry has the figure/table audit counts +
    mismatch flags so `jq 'select(.figure_mismatch)'` filters problem
    papers. Build a do_convert that emits a markdown with a clear
    mismatch to confirm the wiring."""
    main_pdf = tmp_path / "paper.pdf"
    main_pdf.write_bytes(b"%PDF")

    def _do_convert(pdf_path, paper_dir, asset_prefix=""):
        md_path = paper_dir / f"{pdf_path.stem}.md"
        md_path.write_text(
            "Body cites Fig. 1, Fig. 2, Fig. 3, Table 1.\n\n"
            "![](assets/figure_1.jpg)\n"
            "**Table 1.** Caption.\n"
            "| a |\n|---|\n| 1 |\n"
        )
        return md_path, p2m.QualityReport()

    out_dir = tmp_path / "out"
    p2m.run_batch([(main_pdf, None)], _Args(out_dir), _do_convert)

    e = _read_manifest(out_dir)[0]
    assert e["status"] == "ok"
    assert e["figures_referenced"] == 3
    assert e["figures_inserted"] == 1
    assert e["figure_mismatch"] is True
    assert e["tables_referenced"] == 1
    assert e["tables_inserted"] == 1
    assert e["table_mismatch"] is False


def test_manifest_entry_includes_recoverable_tables_count(tmp_path):
    """When `table_mismatch=True` AND a sibling mineru/ output has the
    missing table, the manifest entry records the recoverable count
    so batch triage (`jq 'select(.recoverable_tables>0)'`) finds
    papers fixable via --recover-from-mineru."""
    import json as _json

    main_pdf = tmp_path / "paper.pdf"
    main_pdf.write_bytes(b"%PDF")

    def _do_convert(pdf_path, paper_dir, asset_prefix=""):
        md_path = paper_dir / f"{pdf_path.stem}.md"
        # Body refs Table 1 + Table 2, inserts only Table 1.
        md_path.write_text(
            "Refs: Table 1 and Table 2.\n\n"
            "**Table 1.** First.\n\n| a |\n|---|\n| 1 |\n"
        )
        # Stub mineru output containing Table 2 (the missing one)
        mineru_dir = paper_dir / "mineru"
        images = mineru_dir / "images"
        images.mkdir(parents=True)
        img = images / "t2.jpg"
        img.write_bytes(b"FAKE-JPG")
        middle = {
            "_backend": "pipeline", "_version_name": "3.1.7",
            "pdf_info": [{
                "page_idx": 1, "page_size": [612, 792],
                "para_blocks": [{
                    "type": "table", "bbox": [100, 100, 500, 300],
                    "index": 0, "score": 0.95,
                    "blocks": [
                        {"type": "table_caption",
                         "bbox": [100, 80, 500, 95],
                         "index": 0, "score": 0.95,
                         "lines": [{"bbox": [100, 80, 500, 95],
                                    "spans": [{"type": "text",
                                               "content": "Table 2. Recoverable.",
                                               "bbox": [100, 80, 500, 95]}]}]},
                        {"type": "table_body",
                         "bbox": [100, 100, 500, 300],
                         "index": 0, "score": 0.95,
                         "lines": [{"bbox": [100, 100, 500, 300],
                                    "spans": [{"type": "table",
                                               "html": "<table><tr><td>x</td></tr></table>",
                                               "image_path": "images/t2.jpg",
                                               "bbox": [100, 100, 500, 300]}]}]},
                    ],
                }],
                "discarded_blocks": [],
            }],
        }
        (mineru_dir / f"{pdf_path.stem}_middle.json").write_text(
            _json.dumps(middle))
        return md_path, p2m.QualityReport()

    out_dir = tmp_path / "out"
    p2m.run_batch([(main_pdf, None)], _Args(out_dir), _do_convert)

    e = _read_manifest(out_dir)[0]
    assert e["table_mismatch"] is True
    assert e["recoverable_tables"] == 1


def test_no_supplement_no_si_fields(tmp_path):
    """Paper with no supplement: entry has no si_* fields at all,
    preserving backward compatibility with consumers that don't
    expect them."""
    main_pdf = tmp_path / "paper.pdf"
    main_pdf.write_bytes(b"%PDF")
    do_convert = _fake_do_convert_factory({"paper": ("ok", None)})
    out_dir = tmp_path / "out"
    p2m.run_batch([(main_pdf, None)], _Args(out_dir), do_convert)

    e = _read_manifest(out_dir)[0]
    assert e["status"] == "ok"
    assert e["supplement"] is None
    assert "si_status" not in e
    assert "si_error" not in e


def test_si_non_mineru_error_still_isolates(tmp_path):
    """A non-MinerU SI exception (e.g. a ValueError from a downstream
    hook) also gets isolated -- it just doesn't have an error_kind."""
    main_pdf = tmp_path / "paper.pdf"
    si_pdf = tmp_path / "paper_SI.pdf"
    main_pdf.write_bytes(b"%PDF")
    si_pdf.write_bytes(b"%PDF")

    do_convert = _fake_do_convert_factory({
        "paper": ("ok", None),
        "paper_SI": ("raise", ValueError("malformed citation")),
    })
    out_dir = tmp_path / "out"
    p2m.run_batch([(main_pdf, si_pdf)], _Args(out_dir), do_convert)

    e = _read_manifest(out_dir)[0]
    assert e["status"] == "ok"
    assert e["si_status"] == "error"
    assert "ValueError" in e["si_error"]
    # No MineruRunError-specific fields since it wasn't that exception.
    assert "si_error_kind" not in e
    assert "si_error_stderr_tail" not in e


# === Pre-flight memory check ============================================


def test_available_memory_reads_proc_meminfo(monkeypatch):
    """The helper parses MemAvailable from /proc/meminfo and returns
    GiB. Mock the file to avoid depending on the runner's state."""
    fake = "MemTotal:       128000000 kB\nMemFree:        2000000 kB\nMemAvailable:   16777216 kB\n"

    import builtins
    real_open = builtins.open

    def _fake_open(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            import io
            return io.StringIO(fake)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _fake_open)
    gb = lm._available_memory_gb()
    assert gb is not None
    assert 15.5 < gb < 16.5  # 16 GiB-ish


def test_available_memory_returns_none_on_missing_file(monkeypatch):
    """macOS / sandboxed environment: no /proc/meminfo. Helper returns
    None and the check is silently skipped (no crash)."""
    import builtins
    real_open = builtins.open

    def _fake_open(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _fake_open)
    assert lm._available_memory_gb() is None


def test_pre_flight_warn_when_memory_tight(monkeypatch, caplog):
    """When MemAvailable falls below the 8 GiB safe threshold, the
    pre-flight emits a WARNING that names the recommended remediation
    (drop vLLM --gpu-memory-utilization)."""
    monkeypatch.setattr(lm, "_available_memory_gb", lambda: 6.5)
    with caplog.at_level("WARNING", logger="layout_mineru"):
        lm._check_memory_before_mineru(Path("/tmp/test.pdf"))
    msgs = [r.message for r in caplog.records]
    assert any("6.5 GiB available" in m for m in msgs)
    assert any("gpu-memory-utilization" in m for m in msgs)


def test_pre_flight_no_warn_when_memory_healthy(monkeypatch, caplog):
    """At 20 GiB available, no WARN -- just an INFO breadcrumb."""
    monkeypatch.setattr(lm, "_available_memory_gb", lambda: 20.0)
    with caplog.at_level("INFO", logger="layout_mineru"):
        lm._check_memory_before_mineru(Path("/tmp/test.pdf"))
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warns == []


def test_pre_flight_skips_silently_when_meminfo_unreadable(
        monkeypatch, caplog):
    """No /proc/meminfo (macOS, sandbox) -> the check returns silently;
    nothing logged."""
    monkeypatch.setattr(lm, "_available_memory_gb", lambda: None)
    with caplog.at_level("INFO", logger="layout_mineru"):
        lm._check_memory_before_mineru(Path("/tmp/test.pdf"))
    assert caplog.records == []
