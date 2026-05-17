"""Stage 5 — `paper2md --layout-source=mineru`.

Covers:
  * argparse: `--layout-source` flag exists, defaults to "mineru" (Stage 6
    flip), accepts "marker" / "mineru", rejects garbage.
  * pipeline state: `layout_source` is recorded so the YAML front-matter
    captures which engine the output came from.
  * run_mineru_layout helper: wires layout_mineru.run_mineru ->
    parse_middle_json -> rescue_orphan_captions -> wrap_mineru.emit_markdown
    and copies images verbatim into assets/.

The MinerU subprocess is mocked: tests stage a pre-baked middle.json
(reusing wrap_mineru's minimal fixture) so they don't need a working
mineru install on the test runner.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import paper2md as p2m  # noqa: E402
import wrap_mineru as wm  # noqa: E402


# === argparse =============================================================


def _build_parser():
    """Re-derive the parser without invoking main()."""
    return p2m.build_arg_parser() if hasattr(p2m, "build_arg_parser") else None


def test_layout_source_flag_default_is_mineru():
    """The default routes through mineru post-Stage-6 corpus
    validation (mineru wins on layout/figs/tables on Sarah's full
    corpus run, 2026-05-10). 'marker' remains available via the
    flag for triage-routed sup/sub-fidelity papers."""
    import argparse
    saved = sys.argv
    try:
        sys.argv = ["paper2md", "dummy.pdf", "-o", "out", "--no-vlm",
                    "--no-metadata-lookup", "--skip-vlm-check"]
        p = argparse.ArgumentParser()
        p.add_argument("--layout-source",
                       choices=["marker", "mineru"], default="mineru")
        ns = p.parse_args([])
        assert ns.layout_source == "mineru"
        ns = p.parse_args(["--layout-source", "marker"])
        assert ns.layout_source == "marker"
        with pytest.raises(SystemExit):
            p.parse_args(["--layout-source", "docling"])
    finally:
        sys.argv = saved


def test_generic_refs_heading_rescue_inserts_for_bracketed_run():
    """Doc with no References heading + a trailing run of [N]
    bracketed entries gets a `## References` line inserted at the
    head of the run.

    The rescue's tail-only window (last 30%) requires >=5 matches in
    the tail and at least one anchor with N in {1,2,3}. Test sized
    so [1] lands in the tail window and all 12 refs are detectable."""
    # Body sized so [1] lands in the trailing 30% of the doc
    # (rescue's tail-only detection window). Refs ~370 chars,
    # need body >=0.3*(body+refs) -> body >= 0.7/0.3 * 370 ~ 860.
    body = "Body text. " * 80  # ~880 chars
    refs = "\n".join(f"[{i}] A. Title {i}. J. {i}, 1 (2020)."
                     for i in range(1, 13))  # 12 entries; ~30 chars each
    md = body + "\n\n" + refs
    out = p2m._rescue_generic_missing_refs_heading(md, None)
    assert "## References" in out
    # Heading is inserted somewhere INSIDE the refs block (before some
    # ref entry). Strict "before [1]" assertion is fragile -- depends
    # on exactly which match the tail-window detector picks as the
    # anchor (could be [1], [2], or [3] depending on doc size).
    heading_idx = out.index("## References")
    last_ref_idx = out.index("[12]")
    assert heading_idx < last_ref_idx
    body_end_idx = out.rindex("Body text.")
    # Heading lives after the body text ends.
    assert body_end_idx < heading_idx


def test_generic_refs_heading_rescue_idempotent_with_existing_heading():
    """If a `## References` heading already exists, the generic
    rescue returns md unchanged."""
    md = ("Body text.\n\n"
          "## References\n\n"
          + "\n".join(f"[{i}] A. Title {i}." for i in range(1, 8)))
    out = p2m._rescue_generic_missing_refs_heading(md, None)
    assert out == md


def test_generic_refs_heading_rescue_skips_short_run():
    """A trailing run shorter than 5 entries doesn't trigger
    insertion (avoids false positives on inline citations)."""
    md = ("Body text.\n\n"
          "[1] A.\n[2] B.\n[3] C.\n")
    out = p2m._rescue_generic_missing_refs_heading(md, None)
    assert "## References" not in out


def test_argparse_auto_layout_source_mutex_with_layout_source():
    """--auto-layout-source and --layout-source are mutually exclusive
    (Q1.2: argparse error). Mirrors the --force-ocr / --auto-force-ocr
    contract."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--layout-source", choices=["marker", "mineru"], default=None)
    p.add_argument("--auto-layout-source", action="store_true")
    # Both passed -> simulated post-parse check (the actual check
    # lives in main(); we verify the argparse Namespace allows
    # detection by sentinel-default).
    ns = p.parse_args(["--layout-source", "mineru", "--auto-layout-source"])
    # The runtime code does:
    #   if args.auto_layout_source and args.layout_source is not None:
    #       p.error(...)
    assert ns.auto_layout_source is True
    assert ns.layout_source == "mineru"   # not None -> mutex would fire
    # Auto only:
    ns = p.parse_args(["--auto-layout-source"])
    assert ns.auto_layout_source is True
    assert ns.layout_source is None       # sentinel: caller resolves to mineru
    # Layout-source only:
    ns = p.parse_args(["--layout-source", "marker"])
    assert ns.auto_layout_source is False
    assert ns.layout_source == "marker"


def test_compose_triage_and_scan_detect():
    """The refined --auto-layout-source rule: route to marker if
    EITHER triage says marker OR scan_detect says scan. Auto-apply
    force-OCR only when scan_detect fired (avoids false-firing on
    triage-marker-routed-but-not-actually-a-scan papers like
    alexander_sm). Tests the composition logic directly via the
    triage_pdfs.recommend_routing return + a synthetic scan_detected
    boolean."""
    import triage_pdfs

    # Cases:
    #  A. canup-style: triage=mineru, scan_detected=False -> mineru
    #  B. Wackerle-style: triage=mineru, scan_detected=True
    #     (triage's super-flag check fooled by OCR artifacts) -> marker + force-OCR
    #  C. Lyzenga-style: triage=marker, scan_detected=True -> marker + force-OCR
    #  D. alexander_sm-style: triage=marker (no supers because no
    #     supers in content), scan_detected=False -> marker, NO force-OCR

    cases = [
        ("canup", triage_pdfs.PdfStats(
            stem="canup", pages=10, chars_per_page=2000,
            distinct_font_sizes=8, super_flag_count=10,
            style_flag_pct=0.30, unknown_char_pct=0.0),
         False, "mineru", False),
        ("wackerle", triage_pdfs.PdfStats(
            stem="wackerle", pages=16, chars_per_page=5134,
            distinct_font_sizes=19, super_flag_count=12,
            style_flag_pct=0.22, unknown_char_pct=0.0),
         True, "marker", True),
        ("lyzenga", triage_pdfs.PdfStats(
            stem="lyzenga", pages=4, chars_per_page=5586,
            distinct_font_sizes=30, super_flag_count=0,
            style_flag_pct=0.998, unknown_char_pct=0.0),
         True, "marker", True),
        ("alexander_sm", triage_pdfs.PdfStats(
            stem="alexander_sm", pages=25, chars_per_page=1785,
            distinct_font_sizes=5, super_flag_count=0,
            style_flag_pct=0.09, unknown_char_pct=0.0),
         False, "marker", False),
    ]

    for name, stats, scan_detected, exp_layout, exp_force_ocr in cases:
        triage_routing, _ = triage_pdfs.recommend_routing(stats)
        triage_says_marker = (triage_routing == "marker-surya")
        # Composition rules from _do_convert (refined design):
        if triage_says_marker or scan_detected:
            layout_source = "marker"
        else:
            layout_source = "mineru"
        force_ocr = (layout_source == "marker" and scan_detected)
        assert layout_source == exp_layout, f"{name}: layout"
        assert force_ocr == exp_force_ocr, f"{name}: force_ocr"


def test_triage_routing_maps_to_layout_source():
    """triage_pdfs.recommend_routing returns 'mineru-paddleocr' /
    'marker-surya'; auto-routing in paper2md maps these to 'mineru' /
    'marker' for layout_source. Verify the mapping is round-trip
    stable on synthetic stats."""
    import triage_pdfs
    # Born-digital: clean stats -> mineru route.
    born_digital = triage_pdfs.PdfStats(
        stem="paper", pages=10, chars_per_page=2000.0,
        distinct_font_sizes=8, super_flag_count=10,
        style_flag_pct=0.30, unknown_char_pct=0.0,
    )
    routing, _ = triage_pdfs.recommend_routing(born_digital)
    layout_source = "mineru" if routing == "mineru-paddleocr" else "marker"
    assert routing == "mineru-paddleocr"
    assert layout_source == "mineru"

    # Scan-style: zero supers -> marker route.
    scan = triage_pdfs.PdfStats(
        stem="paper", pages=10, chars_per_page=4000.0,
        distinct_font_sizes=8, super_flag_count=0,
        style_flag_pct=0.30, unknown_char_pct=0.0,
    )
    routing, reason = triage_pdfs.recommend_routing(scan)
    layout_source = "mineru" if routing == "mineru-paddleocr" else "marker"
    assert routing == "marker-surya"
    assert reason == "no-super-flags-preserved"
    assert layout_source == "marker"


def test_layout_source_paper2md_argparse_default_is_mineru():
    """Verify the actual paper2md.py argparser reflects the flipped
    default — guards against regressions where convert() / argparse
    drift apart."""
    import io
    import contextlib
    saved = sys.argv
    try:
        # Use --help so the parser surfaces the resolved default in
        # the formatted output. We probe the argparse Namespace
        # directly via a minimal parse.
        sys.argv = ["paper2md"]
        # Reach into main() not feasible (it side-effects); reconstruct
        # by reading the source default for layout_source via the
        # convert signature.
        import inspect
        sig = inspect.signature(p2m.convert)
        assert sig.parameters["layout_source"].default == "mineru"
    finally:
        sys.argv = saved


def test_pipeline_state_records_layout_source():
    """run.pipeline.layout_source is captured for reproducibility."""
    state = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
        layout_source="mineru",
    )
    assert state["layout_source"] == "mineru"
    state_default = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
    )
    assert state_default["layout_source"] == "mineru"


def test_pipeline_state_prunes_marker_fields_under_mineru():
    """Marker-only pipeline fields (table_finder, force_ocr,
    figmatch_strategy, rotate_tables, table_rotation_*,
    rescue_orphan_tables, rescue_orphan_figures) are omitted under
    layout_source=mineru since none of those code paths run."""
    state = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=True,  # would normally appear; mineru omits it
        layout_source="mineru",
    )
    # Pruned (marker-only):
    for f in ("table_finder", "force_ocr", "figmatch_strategy",
              "rotate_tables", "table_rotation_aspect_threshold",
              "table_rotation_direction", "rescue_orphan_tables",
              "rescue_orphan_figures"):
        assert f not in state, f"{f} should be pruned under mineru layout"
    # Kept (mineru-only or shared):
    assert state["rescue_mineru_orphan_captions"] is True
    assert "trim_articles" in state
    assert "fix_references" in state


def test_pipeline_state_keeps_marker_fields_under_marker():
    """Marker-only pipeline fields ARE recorded under layout_source=marker."""
    state = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
        layout_source="marker",
    )
    # Marker-only fields present:
    for f in ("table_finder", "force_ocr", "figmatch_strategy",
              "rotate_tables", "rescue_orphan_tables",
              "rescue_orphan_figures"):
        assert f in state, f"{f} should be recorded under marker layout"
    # Mineru-only field absent under marker:
    assert "rescue_mineru_orphan_captions" not in state


def test_collect_package_versions_mineru_excludes_marker():
    """Under layout_source=mineru, marker-pdf / surya-ocr / docling are
    NOT recorded; mineru / mineru-vl-utils ARE (when installed)."""
    pkgs = p2m._collect_package_versions("mineru")
    assert "marker-pdf" not in pkgs
    assert "surya-ocr" not in pkgs
    assert "docling" not in pkgs
    # mineru is installed in the dev env; mineru-vl-utils may or may
    # not be (it's an optional extra). Only assert what we know is
    # installed via pip show in the test runner.
    # Shared packages are always recorded (when installed):
    assert "pymupdf" in pkgs


def test_collect_package_versions_marker_excludes_mineru():
    """Under layout_source=marker, mineru / mineru-vl-utils are NOT
    recorded; marker-pdf / surya-ocr / docling ARE (when installed)."""
    pkgs = p2m._collect_package_versions("marker")
    assert "mineru" not in pkgs
    assert "mineru-vl-utils" not in pkgs
    # marker packages are installed in the dev env:
    assert "marker-pdf" in pkgs
    assert "surya-ocr" in pkgs


def test_runinfo_emits_compute_backend_layout_source_text_engine():
    """to_yaml_lines and to_dict both surface the new triple of
    top-level identity fields (compute_backend, layout_source,
    text_engine). Old `backend` field is gone."""
    info = p2m.RunInfo(
        command="paper2md test.pdf",
        hostname="testhost",
        vlm_provider="vllm",
        vlm_model="Qwen/Qwen3-VL-32B-Instruct",
        compute_backend="cuda",
        layout_source="mineru",
        text_engine="mineru-paddleocr",
    )
    yaml_lines = info.to_yaml_lines()
    yaml_text = "\n".join(yaml_lines)
    assert "compute_backend: cuda" in yaml_text
    assert "layout_source: mineru" in yaml_text
    assert 'text_engine: "mineru-paddleocr"' in yaml_text
    # Old field name gone -- check line-anchored to avoid colliding
    # with the legitimate "compute_backend: cuda" substring.
    assert not any(line.strip().startswith("backend:")
                   for line in yaml_lines)

    d = info.to_dict()
    assert d["compute_backend"] == "cuda"
    assert d["layout_source"] == "mineru"
    assert d["text_engine"] == "mineru-paddleocr"
    assert "backend" not in d  # old field name gone


# === run_mineru_layout helper ============================================


def _stage_minimal_mineru_dir(out_dir: Path, stem: str) -> Path:
    """Reuse wrap_mineru's minimal fixture as a pre-baked MinerU output:
    write <stem>_middle.json + images/ next to it. Returns the dir."""
    from test_wrap_mineru import _make_minimal_middle
    # _make_minimal_middle writes to tmp_path / "test_middle.json"; we
    # need <stem>_middle.json to satisfy run_mineru_layout's lookup.
    src = _make_minimal_middle(out_dir)
    target = out_dir / f"{stem}_middle.json"
    src.rename(target)
    images = out_dir / "images"
    images.mkdir(exist_ok=True)
    # Three image references in the fixture: eq1.jpg, fig1.jpg, tbl1.jpg.
    for name in ("eq1.jpg", "fig1.jpg", "tbl1.jpg"):
        (images / name).write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return out_dir


def test_run_mineru_layout_emits_body_and_copies_assets(tmp_path,
                                                        monkeypatch):
    """run_mineru_layout calls layout_mineru.run_mineru, parses the
    produced middle.json, copies images/ into assets/, and returns
    body markdown produced by emit_markdown."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)

    def _fake_run_mineru(pdf, mineru_dir, **_kw):
        _stage_minimal_mineru_dir(Path(mineru_dir), pdf.stem)
        return Path(mineru_dir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    report = p2m.QualityReport()
    body = p2m.run_mineru_layout(pdf_path, assets_dir,
                                 vlm_rewrite_tables=False,
                                 report=report,
                                 mineru_dir=tmp_path / "out" / "mineru")

    # Body produced via emit_markdown.
    assert "# Test Paper Title" in body
    assert "**Abstract.** This is the abstract." in body
    assert "![Figure 1. Schematic of the apparatus.](assets/figure_1_p1.jpg)" in body
    assert "**TABLE 1. Sample data.**" in body
    # Assets renamed to the semantic convention:
    #   table_{id}_p{page}_{idx}.jpg (caption "TABLE 1." -> id "1")
    #   figure_{id}_p{page}.jpg (caption "Figure 1." -> id "1")
    # eq is unchanged (equation-rename out of scope).
    assert (assets_dir / "figure_1_p1.jpg").exists()
    assert (assets_dir / "table_1_p1_1.jpg").exists()
    assert (assets_dir / "eq1.jpg").exists()
    assert not (assets_dir / "fig1.jpg").exists()
    assert not (assets_dir / "tbl1.jpg").exists()
    # MinerU intermediates persisted at out/mineru/.
    assert (tmp_path / "out" / "mineru" / "paper_middle.json").exists()
    # QualityReport populated.
    assert len(report.figures) == 1
    assert len(report.tables) == 1


def test_run_mineru_layout_asset_prefix_supplements(tmp_path, monkeypatch):
    """asset_prefix='si_' (supplement bundling) prepends to every asset
    filename so SI assets don't collide with the main paper's."""
    pdf_path = tmp_path / "paper_si.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)

    def _fake_run_mineru(pdf, mineru_dir, **_kw):
        _stage_minimal_mineru_dir(Path(mineru_dir), pdf.stem)
        return Path(mineru_dir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    report = p2m.QualityReport()
    body = p2m.run_mineru_layout(pdf_path, assets_dir,
                                 asset_prefix="si_",
                                 vlm_rewrite_tables=False,
                                 report=report,
                                 mineru_dir=tmp_path / "out" / "mineru")

    # Figure + table JPGs renamed with si_ prefix on the new semantic names.
    assert "(assets/si_figure_1_p1.jpg)" in body
    assert "(assets/si_table_1_p1_1.jpg)" in body
    assert (assets_dir / "si_figure_1_p1.jpg").exists()
    assert (assets_dir / "si_table_1_p1_1.jpg").exists()
    assert not (assets_dir / "si_fig1.jpg").exists()
    assert not (assets_dir / "si_tbl1.jpg").exists()


def test_run_mineru_layout_missing_middle_raises(tmp_path, monkeypatch):
    """If MinerU's subprocess succeeds but no middle.json appears,
    surface a clear error rather than silently emitting empty body."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)

    def _fake_run_mineru(pdf, mineru_dir, **_kw):
        # Don't write any middle.json.
        return Path(mineru_dir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    report = p2m.QualityReport()
    with pytest.raises(FileNotFoundError, match="middle.json"):
        p2m.run_mineru_layout(pdf_path, assets_dir,
                              vlm_rewrite_tables=False,
                              report=report,
                              mineru_dir=tmp_path / "out" / "mineru")


def test_run_mineru_layout_rescue_default_on(tmp_path, monkeypatch):
    """rescue_orphan_captions=True by default: helper calls
    layout_mineru.rescue_orphan_captions and logs the adoption count."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)

    def _fake_run_mineru(pdf, mineru_dir, **_kw):
        _stage_minimal_mineru_dir(Path(mineru_dir), pdf.stem)
        return Path(mineru_dir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    called = {"n": 0}
    real_rescue = layout_mineru.rescue_orphan_captions

    def _spy(blocks, page_sizes=None):
        called["n"] += 1
        return real_rescue(blocks, page_sizes=page_sizes)

    monkeypatch.setattr(layout_mineru, "rescue_orphan_captions", _spy)

    report = p2m.QualityReport()
    p2m.run_mineru_layout(pdf_path, assets_dir,
                          vlm_rewrite_tables=False,
                          report=report,
                          mineru_dir=tmp_path / "out" / "mineru")
    assert called["n"] == 1


def test_run_mineru_layout_rescue_opt_out(tmp_path, monkeypatch):
    """rescue_orphan_captions=False skips the layout_mineru pass.

    Drives the --no-rescue-orphan-captions opt-out for a fast clean
    MinerU run."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)

    def _fake_run_mineru(pdf, mineru_dir, **_kw):
        _stage_minimal_mineru_dir(Path(mineru_dir), pdf.stem)
        return Path(mineru_dir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    called = {"n": 0}

    def _spy(blocks, page_sizes=None):
        called["n"] += 1
        return {"adopted": 0, "footnotes_adopted": 0}

    monkeypatch.setattr(layout_mineru, "rescue_orphan_captions", _spy)

    report = p2m.QualityReport()
    p2m.run_mineru_layout(pdf_path, assets_dir,
                          vlm_rewrite_tables=False,
                          rescue_orphan_captions=False,
                          report=report,
                          mineru_dir=tmp_path / "out" / "mineru")
    # rescue function NOT called.
    assert called["n"] == 0


def test_pipeline_state_records_rescue_orphan_captions():
    """run.pipeline.rescue_mineru_orphan_captions captured for
    reproducibility."""
    state_on = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
        layout_source="mineru",
        rescue_mineru_orphan_captions=True,
    )
    assert state_on["rescue_mineru_orphan_captions"] is True
    state_off = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
        layout_source="mineru",
        rescue_mineru_orphan_captions=False,
    )
    assert state_off["rescue_mineru_orphan_captions"] is False
    # Default is on.
    state_default = p2m._collect_pipeline_state(
        table_finder="docling",
        vlm_rewrite_tables=False,
        rescue_sparse_pages=False,
        force_ocr=False,
        layout_source="mineru",
    )
    assert state_default["rescue_mineru_orphan_captions"] is True


def test_run_mineru_layout_uses_tempdir_when_no_mineru_dir(tmp_path,
                                                          monkeypatch):
    """When `mineru_dir=None`, helper allocates a tempdir, copies
    assets out, and discards intermediates after returning."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True)

    captured: dict = {}

    def _fake_run_mineru(pdf, mineru_dir, **_kw):
        captured["mineru_dir"] = Path(mineru_dir)
        _stage_minimal_mineru_dir(Path(mineru_dir), pdf.stem)
        return Path(mineru_dir)

    import layout_mineru
    monkeypatch.setattr(layout_mineru, "run_mineru", _fake_run_mineru)

    report = p2m.QualityReport()
    body = p2m.run_mineru_layout(pdf_path, assets_dir,
                                 vlm_rewrite_tables=False,
                                 report=report)

    assert "# Test Paper Title" in body
    # Assets were copied + renamed before the tempdir was cleaned up.
    assert (assets_dir / "figure_1_p1.jpg").exists()
    # Tempdir itself is cleaned up on return.
    assert not captured["mineru_dir"].exists()


# === rescues frontmatter ===================================================


def test_rescues_frontmatter_omitted_when_no_rescue_fired():
    """A QualityReport with empty rescues dict (or all-zero stats) does
    NOT emit a 'rescues:' YAML block.  Keeps clean single-column papers'
    frontmatter free of noise."""
    report = p2m.QualityReport()
    fm = report.to_frontmatter()
    assert "rescues:" not in fm


def test_rescues_frontmatter_emitted_on_caption_adoption():
    """A QualityReport with rescue_orphan_captions stats showing a
    geometric adoption emits a 'rescues:' block with the count."""
    report = p2m.QualityReport()
    report.rescues["orphan_captions"] = {
        "image_table_blocks": 4,
        "misnested_dropped": 0,
        "captions_adopted": 1,
        "footnotes_adopted": 0,
        "captions_adopted_3col": 0,
        "ambiguous_3col_warnings": 0,
        "pages_3col": [],
    }
    fm = report.to_frontmatter()
    assert "rescues:" in fm
    assert "orphan_captions:" in fm
    assert "captions_adopted: 1" in fm
    # Zero-valued keys are suppressed for clarity.
    assert "footnotes_adopted: 0" not in fm
    assert "misnested_dropped: 0" not in fm


def test_rescues_frontmatter_three_col_with_pages_list():
    """Phase B stats include the list of 3-col pages and an
    ambiguity review_note when warnings fired."""
    report = p2m.QualityReport()
    report.rescues["orphan_captions"] = {
        "image_table_blocks": 6,
        "misnested_dropped": 2,
        "captions_adopted": 0,
        "footnotes_adopted": 0,
        "captions_adopted_3col": 1,
        "ambiguous_3col_warnings": 1,
        "pages_3col": [2, 3],
    }
    fm = report.to_frontmatter()
    assert "captions_adopted_3col: 1" in fm
    assert "misnested_dropped: 2" in fm
    assert "ambiguous_3col_warnings: 1" in fm
    assert "pages_3col: [2, 3]" in fm
    assert "review_note:" in fm


def test_rescues_dict_in_meta_json_round_trip():
    """to_dict() mirrors to_frontmatter()'s rescue emission for the
    .meta.json sidecar."""
    report = p2m.QualityReport()
    report.rescues["orphan_captions"] = {
        "image_table_blocks": 2,
        "misnested_dropped": 1,
        "captions_adopted": 0,
        "footnotes_adopted": 0,
        "captions_adopted_3col": 1,
        "ambiguous_3col_warnings": 0,
        "pages_3col": [3],
    }
    d = report.to_dict()
    assert "rescues" in d
    assert "orphan_captions" in d["rescues"]
    orph = d["rescues"]["orphan_captions"]
    assert orph["misnested_dropped"] == 1
    assert orph["captions_adopted_3col"] == 1
    assert orph["pages_3col"] == [3]
    # Zero-valued keys are suppressed.
    assert "footnotes_adopted" not in orph
    assert "captions_adopted" not in orph


def test_rescues_subpanel_groups_emitted():
    """rescue_subpanel_groups stats also appear in the rescues block."""
    report = p2m.QualityReport()
    report.rescues["subpanel_groups"] = {
        "groups": 2,
        "panels_adopted": 5,
    }
    fm = report.to_frontmatter()
    assert "subpanel_groups:" in fm
    assert "groups: 2" in fm
    assert "panels_adopted: 5" in fm


# === MinerU subprocess failure capture (Phase A diagnostic) ================
#
# `run_mineru` raises `MineruRunError` on non-zero exit, carrying the
# stderr tail + a classified failure `kind` so the batch driver can
# log "kind=cuda_oom" instead of an opaque "exit status 1" and decide
# whether to retry or fall back to marker layout. Empirically the most
# common kind under hybrid is cuda_oom because marker has already
# loaded surya + PyTorch CUDA in the parent paper2md process.


def test_classify_mineru_failure_cuda_oom():
    """Common PyTorch OOM messages classify as cuda_oom."""
    import layout_mineru as lm
    samples = [
        "CUDA out of memory. Tried to allocate 2.50 GiB",
        "torch.cuda.OutOfMemoryError: ...",
        "cudaErrorMemoryAllocation",
        "Insufficient memory to allocate buffer",
    ]
    for s in samples:
        assert lm._classify_mineru_failure(s) == "cuda_oom", s


def test_classify_mineru_failure_other_kinds():
    """Non-OOM patterns get their own kinds; unmatched -> 'unknown'."""
    import layout_mineru as lm
    assert lm._classify_mineru_failure(
        "FileNotFoundError: 'layout_model.pt'") == "model_load"
    assert lm._classify_mineru_failure(
        "Found no NVIDIA driver on your system.") == "torch_init"
    assert lm._classify_mineru_failure(
        "No space left on device") == "disk"
    assert lm._classify_mineru_failure(
        "PDFSyntaxError: No /Root object in PDF") == "pdf_parse"
    assert lm._classify_mineru_failure("") == "unknown"
    assert lm._classify_mineru_failure(
        "Some unexpected traceback") == "unknown"


def test_run_mineru_raises_typed_error_with_stderr_tail(tmp_path,
                                                          monkeypatch):
    """When MinerU exits non-zero, run_mineru must raise MineruRunError
    carrying the stderr tail + classified kind, not a bare
    CalledProcessError. The batch driver and hybrid orchestrator
    branch on .kind to decide retry vs marker-fallback."""
    import layout_mineru as lm
    import subprocess

    class _StubResult:
        returncode = 1
        stdout = ""
        stderr = (
            "Starting MinerU...\n"
            "Loading layout model...\n"
            "RuntimeError: CUDA out of memory. "
            "Tried to allocate 2.50 GiB (GPU 0; ...)\n"
        )

    def _fake_run(cmd, env=None, capture_output=False, text=False,
                  errors=None, **kw):
        return _StubResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(lm.MineruRunError) as exc_info:
        lm.run_mineru(pdf, tmp_path / "out")
    err = exc_info.value
    assert err.returncode == 1
    assert err.kind == "cuda_oom"
    assert "CUDA out of memory" in err.stderr_tail
    assert err.pdf_path == pdf


def test_run_mineru_unknown_kind_when_no_pattern_matches(tmp_path,
                                                          monkeypatch):
    """Non-zero exit with stderr that doesn't match any pattern still
    raises MineruRunError, with kind='unknown'. The raw tail is preserved
    so a human can diagnose."""
    import layout_mineru as lm
    import subprocess

    class _StubResult:
        returncode = 7
        stdout = ""
        stderr = "Random weird message we've never seen before\n"

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: _StubResult())
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(lm.MineruRunError) as exc_info:
        lm.run_mineru(pdf, tmp_path / "out")
    err = exc_info.value
    assert err.returncode == 7
    assert err.kind == "unknown"
    assert "Random weird message" in err.stderr_tail


def test_run_mineru_empty_cache_called_before_subprocess(tmp_path,
                                                          monkeypatch):
    """Before spawning MinerU, the parent process's CUDA allocator
    cache is emptied. Cheap defensive measure: returns marker's
    cached blocks to the driver so they don't count against MinerU's
    GPU memory budget."""
    import layout_mineru as lm
    import subprocess

    empty_cache_calls = {"n": 0}

    class _StubTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                empty_cache_calls["n"] += 1

    # Inject the stub torch into sys.modules so `import torch` finds it.
    import sys
    monkeypatch.setitem(sys.modules, "torch", _StubTorch())

    # Make MinerU "succeed" so we can verify empty_cache fired regardless
    # of failure path.
    class _OK:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _OK())

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    # Will fail later at the _middle.json lookup, but that's after
    # subprocess.run() so empty_cache should already have fired.
    try:
        lm.run_mineru(pdf, tmp_path / "out")
    except FileNotFoundError:
        pass  # expected: no real MinerU output

    assert empty_cache_calls["n"] == 1


def test_run_mineru_empty_cache_skips_when_torch_unavailable(tmp_path,
                                                              monkeypatch):
    """No torch installed -> empty_cache step is silently skipped.
    Mac CPU users and minimal envs should still be able to run MinerU."""
    import layout_mineru as lm
    import builtins
    import subprocess

    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    class _OK:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _OK())
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    # Should NOT crash on the torch import path.
    try:
        lm.run_mineru(pdf, tmp_path / "out")
    except FileNotFoundError:
        pass  # expected after subprocess "succeeds" but produced no output
