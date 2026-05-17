"""layout_mineru.py — shared MinerU-layout reader for paper2md.

Stage 4 of the convergence plan. Houses the MinerU `middle.json` reader,
the `_MBlock` block dataclass, the subprocess wrapper for the `mineru`
CLI, and the orphan-caption-rescue post-pass. wrap_mineru.py is the
historical home for these; pulling them into a shared module so
paper2md.py can import the same code (Stage 5: paper2md uses MinerU
layout for figure/table cropping while keeping marker/surya for body
text).

Pure layout / parsing utility. Does NOT depend on paper2md, metadata_-
frontend, or any GPU/VLM machinery. Logging on the `layout_mineru`
logger.

What lives here:
  * _MBlock, ParsedDoc dataclasses
  * Helpers: _join_lines_text, _first_span, _bbox_reading_key
  * parse_middle_json(middle_path) -> ParsedDoc
  * run_mineru(pdf_path, mineru_dir, ...) subprocess wrapper
  * _locate_mineru_auto_dir, stage_existing_mineru_dir
  * Orphan-caption rescue group (Stage 3):
    - _CAPTION_WATERMARK_RES, _FIG_TAB_CAPTION_RE, _FOOTNOTE_OPENER_RE
    - _caption_is_watermark, _is_caption_text, _is_footnote_text
    - _bbox_y_adjacent, _bbox_x_compatible
    - rescue_orphan_captions(blocks)

What does NOT live here (stays in wrap_mineru.py):
  * HTML-table -> pipe-markdown converter
  * emit_markdown + per-block emitters
  * References-heading / footnote-refs / Knudson rescues
  * Article-trim (_vlm_compare_pages, trim_articles_in_blocks)
  * Run-info builder, CLI, _process_one orchestration
  * Data-repo population
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Patterns we recognise in MinerU's stderr so callers can branch on `kind`.
# `cuda_oom` is the most common failure mode under hybrid layout: marker
# has already loaded surya + PyTorch into the parent paper2md process,
# leaving too little GPU memory for MinerU's PaddleOCR + layout models.
# Mitigation lives in the caller (lower vLLM --gpu-memory-utilization;
# torch.cuda.empty_cache() in parent before spawn) and the run_mineru
# stderr capture so the failure is visible instead of "exit status 1".
_MINERU_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cuda_oom", r"CUDA out of memory"),
    ("cuda_oom", r"OutOfMemoryError"),
    ("cuda_oom", r"cudaErrorMemoryAllocation"),
    ("cuda_oom", r"cuMemAlloc.*failed"),
    ("cuda_oom", r"insufficient.*memory"),
    ("model_load", r"FileNotFoundError.*\.(?:pt|pth|onnx|paddle)"),
    ("model_load", r"Failed to download"),
    ("model_load", r"HuggingFace.*not.*found"),
    ("torch_init", r"Found no NVIDIA driver"),
    ("torch_init", r"CUDA driver version is insufficient"),
    ("disk", r"No space left on device"),
    ("pdf_parse", r"PDFSyntaxError"),
    ("pdf_parse", r"is not a valid PDF"),
)


# MinerU's middle.json schema is the canonical input for the hybrid
# splice + several layout-rescue hooks. Minor-version bumps have
# historically changed the schema in ways that silently break caption
# detection, table HTML extraction, and figure-asset paths. We pin to
# 3.1.7 in the env files + USAGE.md and warn (not fail) at runtime if
# the installed version drifts, so accidental `pip install --upgrade
# mineru` doesn't go undetected until output starts looking wrong.
EXPECTED_MINERU_VERSION = "3.1.7"

# Module-level flag so the warning fires AT MOST once per process,
# even if run_mineru is invoked multiple times in a batch.
_mineru_version_checked = False


class MineruNotInstalledError(RuntimeError):
    """Raised when paper2md is asked to run MinerU but the `mineru`
    package or its CLI entry-point is not present in the active env.

    MinerU is a separate `pip install "mineru[core]==3.1.7"` per
    environment-mac.yml / environment-gpu.yml -- it's not bundled with
    the conda env because it's only needed for `--layout-source
    mineru/hybrid/auto`. This error carries the install hint so the
    user doesn't have to grep docs."""


def _assert_mineru_installed() -> None:
    """Verify both the `mineru` Python package AND its CLI entry-point
    are available before paper2md spawns the subprocess. Raises
    `MineruNotInstalledError` with the install hint if either is missing.

    Checking BOTH catches: (a) the common case where the user hasn't
    installed mineru at all (no module + no CLI); and (b) the rarer
    case where module + CLI got out of sync (e.g., env activation
    issue, broken setuptools entry-point) -- in that case the user
    sees a more useful "module present but CLI not on PATH" hint than
    a bare FileNotFoundError from subprocess.
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            version("mineru")
            module_present = True
        except PackageNotFoundError:
            module_present = False
    except ImportError:  # pragma: no cover -- Python 3.8+ always has it
        module_present = True  # can't check; assume yes, fall through to CLI check

    cli_path = shutil.which("mineru")
    if not module_present and cli_path is None:
        raise MineruNotInstalledError(
            "MinerU is not installed in this environment. "
            "`--layout-source mineru/hybrid/auto` requires the mineru "
            "package + CLI. Install with:\n"
            f"    pip install \"mineru[core]=={EXPECTED_MINERU_VERSION}\"\n"
            "Then re-run paper2md. (The pin is load-bearing -- paper2md's "
            "splice + rescue hooks are tuned against MinerU "
            f"{EXPECTED_MINERU_VERSION}'s middle.json schema.) "
            "Alternative: pass `--layout-source marker` to skip MinerU."
        )
    if module_present and cli_path is None:
        raise MineruNotInstalledError(
            "The mineru Python package is installed but its `mineru` "
            "CLI is not on PATH. This usually means the env activation "
            "is partial or the package's console_scripts entry-point "
            "is broken. Try:\n"
            "    which mineru   # should resolve inside the active env\n"
            f"    pip install --force-reinstall \"mineru[core]=={EXPECTED_MINERU_VERSION}\""
        )


def _check_mineru_version() -> None:
    """One-shot startup check: log a WARNING if installed MinerU
    differs from `EXPECTED_MINERU_VERSION`. Soft-fails silently when
    mineru isn't installed (callers of run_mineru hit the
    MineruNotInstalledError preflight first) or when
    `importlib.metadata` can't read package metadata for any reason."""
    global _mineru_version_checked
    if _mineru_version_checked:
        return
    _mineru_version_checked = True
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover -- Python 3.8+ always has it
        return
    try:
        installed = version("mineru")
    except PackageNotFoundError:
        # mineru not installed; the subprocess call further down will
        # fail with a more useful error than a missing package warning.
        return
    except Exception:
        return
    if installed != EXPECTED_MINERU_VERSION:
        log.warning(
            "MinerU version %s installed; paper2md's hybrid splice + "
            "layout-rescue hooks were tuned against %s. Schema drift "
            "between MinerU minor versions has historically caused "
            "silent figure / table parsing errors. Pin to %s (see "
            "USAGE.md §2 and environment-gpu.yml) or audit the "
            "fudges in docs/dev/HYBRID_IMPLEMENTATION.md before "
            "trusting output.",
            installed, EXPECTED_MINERU_VERSION,
            EXPECTED_MINERU_VERSION,
        )


def _classify_mineru_failure(stderr: str) -> str:
    """Pattern-match MinerU's stderr to one of the recognised failure
    kinds. Returns 'unknown' when nothing matches; that's a signal to
    inspect the raw stderr tail in the log."""
    if not stderr:
        return "unknown"
    for kind, pat in _MINERU_FAILURE_PATTERNS:
        if re.search(pat, stderr, re.IGNORECASE):
            return kind
    return "unknown"


# MinerU's pipeline backend loads PaddleOCR + a layout model + PyTorch
# CUDA runtime; in practice it claims 5-6 GB of GPU memory at minimum
# on the Spark's unified-memory GB10. Under hybrid layout, marker has
# already loaded surya in the same process tree, so MinerU competes
# with the parent for whatever's left after vLLM's reservation.
_MINERU_MEMORY_WARN_GB = 8.0    # WARN if available drops below this
_MINERU_MEMORY_HARD_GB = 4.0    # at this point MinerU is very likely to OOM


def _available_memory_gb() -> Optional[float]:
    """Return the system's currently-available memory in GiB, or None if
    we can't read /proc/meminfo. Works on Linux only (Spark = Linux);
    on macOS this returns None and the check is silently skipped.

    Uses MemAvailable (best estimate of how much can be allocated
    without swapping) rather than MemFree (which excludes reclaimable
    page cache and is way too pessimistic on Spark)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _check_memory_before_mineru(pdf_path: Path) -> None:
    """Log a memory snapshot before spawning MinerU. WARNs if available
    memory is below `_MINERU_MEMORY_WARN_GB` so the user has a clear
    signal to drop vLLM's --gpu-memory-utilization (recommended 0.65
    for hybrid on the Spark GB10). This is preventative diagnostics
    only -- we still let MinerU try; small PDFs may fit even at low
    available memory, and the post-hoc MineruRunError classification
    will catch genuine OOMs."""
    avail = _available_memory_gb()
    if avail is None:
        return  # non-Linux or unreadable /proc; skip silently
    if avail < _MINERU_MEMORY_HARD_GB:
        log.warning(
            "MinerU pre-flight: only %.1f GiB available before "
            "spawning on %s -- HIGH RISK of CUDA OOM. Consider "
            "lowering vLLM --gpu-memory-utilization (0.65 recommended "
            "for hybrid on Spark) and re-trying.",
            avail, pdf_path.name)
    elif avail < _MINERU_MEMORY_WARN_GB:
        log.warning(
            "MinerU pre-flight: %.1f GiB available before spawning "
            "on %s (below %.0f GiB safe threshold). MinerU typically "
            "claims 5-6 GiB; tight. Recommend vLLM "
            "--gpu-memory-utilization 0.65 for hybrid workloads.",
            avail, pdf_path.name, _MINERU_MEMORY_WARN_GB)
    else:
        log.info("MinerU pre-flight: %.1f GiB available on %s",
                 avail, pdf_path.name)


class MineruRunError(RuntimeError):
    """Raised by `run_mineru` when the MinerU subprocess exits non-zero.

    Carries the exit code, the PDF path being processed, a classified
    failure `kind` (see _MINERU_FAILURE_PATTERNS) and the tail of the
    captured stderr so callers can:
      - log the actual reason instead of a generic CalledProcessError;
      - decide whether to retry (e.g. clear GPU cache + retry on
        cuda_oom) or fall back to a marker-only layout.

    Subclassing RuntimeError keeps it catchable by existing `except
    Exception` handlers in batch / supplement-pair code paths.
    """

    def __init__(self, returncode: int, pdf_path: Path, kind: str,
                 stderr_tail: str):
        self.returncode = returncode
        self.pdf_path = pdf_path
        self.kind = kind
        self.stderr_tail = stderr_tail
        super().__init__(
            f"MinerU exited {returncode} on {pdf_path.name} (kind={kind})")


log = logging.getLogger("layout_mineru")


# === parsed-doc dataclass + middle.json walker =============================


@dataclass
class _MBlock:
    """Single para_block from MinerU's middle.json, normalised."""
    type: str                                  # text/title/abstract/ref_text/image/chart/table/interline_equation
    page_idx: int
    index: int                                 # MinerU's intra-page ordering index
    bbox: tuple                                # (x0, y0, x1, y1)
    text: Optional[str] = None                 # body text for text-like blocks; caption text for image/chart/table
    image_path: Optional[str] = None           # for image/chart/table/interline_equation bodies
    html: Optional[str] = None                 # for table_body
    latex: Optional[str] = None                # for interline_equation
    text_level: Optional[int] = None           # heading level (1..6) for title blocks
    footnote: Optional[str] = None             # for table footnote
    # Sub-panel consolidation (rescue_subpanel_groups). Populated on
    # the *primary* image block of a multi-panel figure group.
    subpanel_paths: list = field(default_factory=list)   # additional image_paths
    subpanel_bboxes: list = field(default_factory=list)  # parallel to subpanel_paths; used by hybrid splice to sort panels into reading order
    panel_letter: Optional[str] = None                   # letter extracted from primary's caption prefix
    figure_number: Optional[int] = None                  # caption's "Figure N" / "FIG. N" number for alt-text


@dataclass
class ParsedDoc:
    pages: int
    blocks: list[_MBlock] = field(default_factory=list)
    backend: str = ""
    version: str = ""
    # MinerU's classifier separates page-bottom journal/DOI furniture
    # (`footer`) from numbered citation footnotes (`page_footnote`).
    # We collect the latter into a sidecar list — they're skipped from
    # the main body emit, but the rescue_footnote_refs pass mines them
    # for old-style numbered-footnote references when the body lacks
    # a usable References section.
    footnote_blocks: list[_MBlock] = field(default_factory=list)
    # Page geometry from middle.json's pdf_info[*].page_size = [w, h].
    # Indexed by 0-based page_idx -> (width, height).  Consumed by the
    # rescue passes that need a column-layout signal (3-col Science).
    page_sizes: dict = field(default_factory=dict)


def _join_lines_text(lines: list) -> str:
    """Join span content from a block's `lines` field into flowing prose.

    Matches MinerU's own emitter (pipeline_middle_json_mkcontent._join_-
    rendered_span): spans within a line are space-joined; lines within a
    paragraph block are space-joined; a trailing hyphen on one line that
    precedes a lowercase-starting next line is treated as a soft hyphen
    and stripped (de-hyphenation: "Earth spin-" + "ning" -> "Earth
    spinning"). A trailing hyphen before an uppercase-starting line is
    preserved (proper-noun / compound: "NASA-" + "JPL" -> "NASA- JPL"
    via space-join).

    Inline equations are wrapped in $...$ and interline equations in
    $$...$$ so they remain renderable when caption / footnote text is
    later emitted.

    The function returns a single flowing string; visual PDF line breaks
    are NOT preserved as semantic breaks.
    """
    parts: list[str] = []
    for ln in lines or []:
        line_pieces: list[str] = []
        for span in ln.get("spans", []) or []:
            t = span.get("type")
            content = span.get("content")
            if t == "interline_equation":
                if content:
                    line_pieces.append(f"$${content}$$")
            elif t == "inline_equation":
                if content:
                    line_pieces.append(f"${content}$")
            elif content:
                line_pieces.append(content)
        if line_pieces:
            parts.append(" ".join(line_pieces))
    if not parts:
        return ""
    out = parts[0]
    for nxt in parts[1:]:
        if out.endswith("-") and nxt and nxt[0].islower():
            out = out[:-1] + nxt
        else:
            out = out + " " + nxt
    return out.strip()


def _first_span(child: dict) -> dict:
    lines = child.get("lines") or []
    if not lines:
        return {}
    spans = lines[0].get("spans") or []
    return spans[0] if spans else {}


def _detect_three_column_page(text_blocks: list, page_width: float) -> bool:
    """True when the page's text blocks fall into three distinct x-center
    clusters -- the diagnostic for an old Science / AAAS 3-column body
    layout.

    Heuristic: bin every text-shaped block's x-center into bins of width
    page_width/6.  Page is 3-column when AT LEAST three distinct bins
    are populated AND the total block count is >=4 AND no single bin
    holds ALL blocks.  The block-count floor excludes cover / sidebar
    pages with too few text blocks to draw conclusions from; the "not
    all in one bin" check excludes 1-column pages that happen to have
    a few outlier blocks (running header at top-left, page number at
    bottom-right, etc.).

    Tuned against the young / cuk / canup2012 corpus: young page 3 has
    only 6 text blocks (most space is charts) but they clearly cluster
    around x=118, 296, 474 -- the relaxed >=1-per-bin rule catches it.
    Modern 1-column papers (canup Nature 2014) have all text-block
    centers near page_width/2 and fail the >=3-bins check cleanly.

    Pure function on the block list -- no MinerU API call.  Gated check
    for the page-wide caption-by-label-match rescue (canup, modern
    Nature 1-column papers must NOT trigger that pass).
    """
    if not text_blocks or page_width <= 0:
        return False
    bin_w = page_width / 6.0
    counts: dict[int, int] = {}
    total = 0
    for b in text_blocks:
        bb = b.bbox if hasattr(b, "bbox") else b.get("bbox")
        if not bb or len(bb) < 4:
            continue
        x_center = (float(bb[0]) + float(bb[2])) / 2.0
        key = int(x_center // bin_w)
        counts[key] = counts.get(key, 0) + 1
        total += 1
    if total < 4:
        return False
    if len(counts) < 3:
        return False
    # Reject single-column with outliers: one bin holds >=80% of blocks.
    if max(counts.values()) >= 0.8 * total:
        return False
    return True


def _bbox_reading_key(bbox: tuple) -> tuple[float, float]:
    """Sort key for ordering caption/footnote fragments in reading
    order. Top-to-bottom by y0, then left-to-right by x0 within the
    same row. Returns (0, 0) when bbox is missing/malformed."""
    if not bbox or len(bbox) < 2:
        return (0.0, 0.0)
    y0 = float(bbox[1])
    x0 = float(bbox[0])
    return (y0, x0)


def parse_middle_json(middle_path: Path) -> ParsedDoc:
    """Walk MinerU's middle.json and return a flat ordered list of _MBlock.

    Order is (page_idx, index). Each block's payload is normalised so the
    emit functions don't have to know about MinerU's nested structure.
    """
    raw = json.loads(middle_path.read_text())
    pdf_info = raw.get("pdf_info", []) or []
    blocks: list[_MBlock] = []
    for page in pdf_info:
        page_idx = page.get("page_idx", 0)
        for b in page.get("para_blocks", []) or []:
            t = b.get("type")
            bbox = tuple(b.get("bbox") or ())
            idx = b.get("index", 0)
            if t in ("text", "title", "abstract", "ref_text"):
                text = _join_lines_text(b.get("lines", []))
                level = b.get("level") or b.get("text_level")
                blocks.append(_MBlock(
                    type=t, page_idx=page_idx, index=idx, bbox=bbox,
                    text=text, text_level=level,
                ))
            elif t == "interline_equation":
                span = _first_span(b)
                blocks.append(_MBlock(
                    type=t, page_idx=page_idx, index=idx, bbox=bbox,
                    latex=span.get("content"),
                    image_path=span.get("image_path"),
                ))
            elif t in ("image", "chart"):
                # Wide figures with captions split across multiple
                # narrow columns appear in middle.json as multiple
                # `image_caption` children of one parent block. Collect
                # all fragments and join in reading order (top-to-bottom,
                # left-to-right by child bbox); the parser used to
                # overwrite, keeping only the last fragment.
                cap_frags: list[tuple[tuple[float, float], str]] = []
                body_image = None
                for c in b.get("blocks", []) or []:
                    ct = c.get("type")
                    cbb = tuple(c.get("bbox") or ())
                    if ct in ("image_caption", "chart_caption"):
                        ctxt = _join_lines_text(c.get("lines", []))
                        if ctxt:
                            cap_frags.append((_bbox_reading_key(cbb), ctxt))
                    elif ct in ("image_body", "chart_body"):
                        span = _first_span(c)
                        body_image = span.get("image_path")
                cap_frags.sort()
                caption_text = " ".join(txt for _, txt in cap_frags)
                # Apply the OCR-label-reorder repair to pre-classified
                # captions too (PaddleOCR can misorder Fig. N. labels
                # inside a correctly-classified image_caption block --
                # cuk Fig 2/3/4 in the corpus).
                caption_text = (
                    _repair_misordered_caption_label(caption_text)
                    if caption_text else caption_text
                )
                blocks.append(_MBlock(
                    type=t, page_idx=page_idx, index=idx, bbox=bbox,
                    text=caption_text or None,
                    image_path=body_image,
                ))
            elif t == "table":
                # Same fragment-merge for tables: 3-column papers
                # routinely emit 2-3 `table_caption` and/or
                # `table_footnote` fragments inside one table block.
                cap_frags: list[tuple[tuple[float, float], str]] = []
                foot_frags: list[tuple[tuple[float, float], str]] = []
                body_html = None
                body_image = None
                for c in b.get("blocks", []) or []:
                    ct = c.get("type")
                    cbb = tuple(c.get("bbox") or ())
                    if ct == "table_caption":
                        ctxt = _join_lines_text(c.get("lines", []))
                        if ctxt:
                            cap_frags.append((_bbox_reading_key(cbb), ctxt))
                    elif ct == "table_body":
                        span = _first_span(c)
                        body_html = span.get("html")
                        body_image = span.get("image_path")
                    elif ct == "table_footnote":
                        ftxt = _join_lines_text(c.get("lines", []))
                        if ftxt:
                            foot_frags.append(
                                (_bbox_reading_key(cbb), ftxt))
                cap_frags.sort()
                foot_frags.sort()
                caption_text = " ".join(txt for _, txt in cap_frags)
                footnote_text = " ".join(txt for _, txt in foot_frags)
                # Same OCR-label-reorder repair as the image / chart
                # branch above (canup2012 Table 1 in the corpus).
                caption_text = (
                    _repair_misordered_caption_label(caption_text)
                    if caption_text else caption_text
                )
                blocks.append(_MBlock(
                    type=t, page_idx=page_idx, index=idx, bbox=bbox,
                    text=caption_text or None,
                    html=body_html,
                    image_path=body_image,
                    footnote=footnote_text or None,
                ))
            else:
                # Unknown type: keep as text if it has lines, else skip.
                lines = b.get("lines")
                if lines:
                    blocks.append(_MBlock(
                        type=t or "unknown", page_idx=page_idx, index=idx, bbox=bbox,
                        text=_join_lines_text(lines),
                    ))
    blocks.sort(key=lambda b: (b.page_idx, b.index))

    # Second pass: capture page_footnote blocks from discarded_blocks/.
    # MinerU's classifier already separates these from page-bottom
    # publisher furniture (which lands under type=footer / header /
    # page_number — we ignore those). Sorted by (page, bbox-y) so the
    # rescue can assume document order.
    footnote_blocks: list[_MBlock] = []
    for page in pdf_info:
        page_idx = page.get("page_idx", 0)
        for b in page.get("discarded_blocks", []) or []:
            if b.get("type") != "page_footnote":
                continue
            bbox = tuple(b.get("bbox") or ())
            text = _join_lines_text(b.get("lines", []))
            if not text:
                continue
            footnote_blocks.append(_MBlock(
                type="page_footnote",
                page_idx=page_idx,
                index=b.get("index", 0),
                bbox=bbox,
                text=text,
            ))
    footnote_blocks.sort(
        key=lambda b: (b.page_idx, b.bbox[1] if b.bbox else 0)
    )

    page_sizes: dict = {}
    for page in pdf_info:
        ps = page.get("page_size") or []
        if len(ps) >= 2:
            page_sizes[page.get("page_idx", 0)] = (float(ps[0]), float(ps[1]))

    return ParsedDoc(
        pages=len(pdf_info),
        blocks=blocks,
        backend=raw.get("_backend", ""),
        version=raw.get("_version_name", ""),
        footnote_blocks=footnote_blocks,
        page_sizes=page_sizes,
    )


# === MinerU subprocess invocation ==========================================


def run_mineru(pdf_path: Path, mineru_dir: Path, *,
               backend: str = "pipeline",
               extra_args: list[str] = ()) -> Path:
    """Invoke MinerU as a subprocess and stage its output flat inside
    `mineru_dir`.

    MinerU writes to <work>/<stem>/auto/{stem}_middle.json + co-located
    images/. We move the auto/ contents flat into `mineru_dir` (no
    per-stem nesting) so the layout matches the wrapper's directory
    convention:

        <paper_dir>/mineru/<stem>_middle.json
        <paper_dir>/mineru/<stem>_content_list.json
        <paper_dir>/mineru/<stem>_model.json
        <paper_dir>/mineru/<stem>_layout.pdf
        <paper_dir>/mineru/<stem>_origin.pdf
        <paper_dir>/mineru/<stem>_span.pdf
        <paper_dir>/mineru/<stem>.md            (MinerU's flat MD)
        <paper_dir>/mineru/images/              (full set, verbatim)

    Sets MINERU_VIRTUAL_VRAM_SIZE=1 to force batch_size=1 on Apple
    Silicon (MinerU 3.1.6's get_vram() has no MPS path; default fall-
    through value triggers a hang at "Predict: 0/N").

    Raises `MineruRunError` (subclass of RuntimeError) on non-zero
    exit; the exception carries the classified failure `kind`,
    stderr tail, and the PDF path so callers can branch (retry on
    cuda_oom, fall back to marker-only on persistent failure, log
    the actual reason instead of an opaque "exit status 1").
    Returns mineru_dir on success.
    """
    mineru_dir = mineru_dir.resolve()
    mineru_dir.mkdir(parents=True, exist_ok=True)
    # MinerU insists on creating <out>/<stem>/<backend-subdir>/. Use a
    # tmp child of mineru_dir so we can move the auto/ contents flat
    # afterwards without leaving the per-stem nesting on disk.
    tmp_root = mineru_dir / "_mineru_staging"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    # Free the parent process's CUDA allocator caches before spawning
    # MinerU. Under hybrid layout, marker+surya have already loaded
    # PyTorch CUDA models in the parent; without this, those cached
    # blocks count against the GPU memory budget for MinerU's
    # subprocess. Empirically the bigger lever is vLLM's
    # --gpu-memory-utilization (recommend 0.65 for hybrid on Spark);
    # this empty_cache() is a cheap complement. Best-effort: skip
    # silently if torch isn't importable or CUDA isn't available.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Installed-or-not check FIRST: if mineru is missing entirely, fail
    # with an actionable install hint rather than a bare FileNotFoundError
    # from subprocess.run() further down.
    _assert_mineru_installed()

    # Pin check (once per process): warn if installed MinerU version
    # diverges from what the splice + rescue hooks were tuned against.
    _check_mineru_version()

    # Pre-flight memory check (Linux only). Log a clear WARN when
    # available memory drops near MinerU's footprint so the user
    # knows to drop vLLM's --gpu-memory-utilization before debugging.
    _check_memory_before_mineru(pdf_path)

    env = os.environ.copy()
    env.setdefault("MINERU_VIRTUAL_VRAM_SIZE", "1")
    cmd = [
        "mineru", "-p", str(pdf_path), "-o", str(tmp_root),
        "-b", backend, "-l", "en",
        *extra_args,
    ]
    log.info("Running MinerU: %s", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        # Tail to the last ~50 lines so the log doesn't get drowned in
        # MinerU's startup banner / model-download spam.
        stderr = result.stderr or ""
        tail = "\n".join(stderr.splitlines()[-50:])
        kind = _classify_mineru_failure(stderr)
        log.warning(
            "MinerU subprocess exited %d on %s (kind=%s); "
            "stderr tail:\n%s",
            result.returncode, pdf_path.name, kind,
            tail or "(empty stderr)",
        )
        raise MineruRunError(
            returncode=result.returncode, pdf_path=pdf_path,
            kind=kind, stderr_tail=tail)

    # Locate the produced auto/ (or vlm/) subdir.
    stem = pdf_path.stem
    produced = _locate_mineru_auto_dir(tmp_root / stem, stem)
    if produced is None:
        raise FileNotFoundError(
            f"MinerU did not produce a *_middle.json under {tmp_root / stem}"
        )

    # Move flat into mineru_dir.
    for entry in produced.iterdir():
        dst = mineru_dir / entry.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(entry), str(dst))

    shutil.rmtree(tmp_root, ignore_errors=True)
    return mineru_dir


def _locate_mineru_auto_dir(stem_dir: Path, stem: str) -> Optional[Path]:
    """Probe a MinerU per-stem dir for auto/ or vlm/ etc. Returns the
    subdir containing <stem>_middle.json, or None."""
    if not stem_dir.is_dir():
        return None
    direct = stem_dir / "auto"
    if direct.is_dir() and (direct / f"{stem}_middle.json").exists():
        return direct
    for child in stem_dir.iterdir():
        if child.is_dir() and (child / f"{stem}_middle.json").exists():
            return child
    return None


def stage_existing_mineru_dir(src_dir: Path, mineru_dir: Path) -> Path:
    """For --skip-mineru-run: copy or accept an existing MinerU output.

    `src_dir` may point at:
      - a flat layout dir already containing *_middle.json + images/
        (e.g. a previous wrap_mineru run's mineru/), or
      - MinerU's nested <stem>/auto/ subdir.

    Returns the path to use as the MinerU dir for parsing. If `src_dir`
    is already `mineru_dir`, returns it unchanged (no-op). Otherwise
    copies the relevant contents into `mineru_dir` flat and returns
    `mineru_dir`.
    """
    src_dir = src_dir.resolve()
    mineru_dir = mineru_dir.resolve()
    if src_dir == mineru_dir:
        return mineru_dir

    middle_jsons = sorted(src_dir.glob("*_middle.json"))
    if middle_jsons:
        actual_src = src_dir
    else:
        # Maybe nested: src_dir/<stem>/auto/.
        stem_dir_candidates = [d for d in src_dir.iterdir() if d.is_dir()]
        actual_src = None
        for sd in stem_dir_candidates:
            probe = _locate_mineru_auto_dir(sd, sd.name)
            if probe is not None:
                actual_src = probe
                break
        if actual_src is None:
            raise FileNotFoundError(
                f"No *_middle.json found under {src_dir} or its child "
                f"<stem>/auto/ subdirs"
            )

    mineru_dir.mkdir(parents=True, exist_ok=True)
    for entry in actual_src.iterdir():
        dst = mineru_dir / entry.name
        if entry.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)
    log.info("Staged %d MinerU file(s) from %s to %s",
             sum(1 for _ in actual_src.iterdir()), actual_src, mineru_dir)
    return mineru_dir


# === orphan caption / footnote rescue ======================================


_CAPTION_WATERMARK_RES = (
    re.compile(r"\bDownloaded\s+from\b", re.IGNORECASE),
    re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE),
    re.compile(r"\bwww\.[a-z0-9-]+\.(?:com|org)\b", re.IGNORECASE),
    re.compile(r"\bonlinelibrary\.wiley\.com", re.IGNORECASE),
    re.compile(r"\bsciencemag\.org\b|\bscience\.org\b", re.IGNORECASE),
    re.compile(r"\bnature\.com\b", re.IGNORECASE),
    re.compile(r"©\s*\d{4}", re.IGNORECASE),
)


_FIG_TAB_CAPTION_RE = re.compile(
    # No word-boundary anchor: OCR sometimes renders the caption label
    # adjacent to body text with no space (cuk p2: "Formation of the"
    # "Fig. 1.lunar disk"). \b would require a word boundary before
    # Fig which isn't present. The combined `Label\.?\s*[A-Z]?\d+`
    # pattern is selective enough on its own — body text rarely has
    # "fig"/"figure"/"table" as a substring followed immediately (or
    # after a period) by a digit.
    r"(?:Fig|Figure|Table|Plate|Scheme|Box)\.?\s*[A-Z]?\d+",
    re.IGNORECASE,
)


_FOOTNOTE_OPENER_RE = re.compile(
    r"^\s*(?:[\*†‡§¶#]|[a-z]\)\s|\(\d+\)\s|\d+\)\s|Notes?\s*[:.])",
)


# === caption-label reorder =================================================
#
# PaddleOCR on Science (AAAS)-style captions, where the bold "Fig. N."
# label has a different font weight than the title text, can mis-attribute
# the label to the WRONG visual line of the caption and merge it with the
# adjacent word with NO space.  Example signatures seen in the corpus:
#   cuk    Fig 1: "Formation of the Fig. 1.lunar disk from Earth's..."
#   canup2 Fig 1: "An SPH simulation Fig. 1.of a moderately oblique,..."
#   canup2 Fig 2: "Compositional differ- Fig. 2.ence between the disk..."
#   canup2 Tbl 1: "Properties of candidate impacts. ... Shown are Table 1.the..."
# All show: figure/table label glued to a LOWERCASE next word with no
# separator, label sitting mid-caption rather than at the start.  The
# no-space-after-period + lowercase-next-char combination is the OCR-
# merge fingerprint -- body cross-refs like "shown in Fig. 1. The..."
# always have a space after the label period (PaddleOCR emits the space
# whenever the source PDF has rendered one) so they do NOT match.

_MISORDERED_LABEL_RE = re.compile(
    # (?i:...) is a local case-insensitive group: applies to the label
    # alternation only, NOT to the lookahead.  The lookahead requires
    # the next two characters to form a word-start: any letter followed
    # by a lowercase letter (i.e., a 2+ char word starting either lower-
    # case 'lunar' or title-case 'Moon').  This excludes legitimate
    # panel-letter cross-refs like 'Fig. 3.A panel' (uppercase letter
    # followed by space) and 'Fig. 3.b' (single lowercase letter at end
    # of sentence) while still firing on the corpus's OCR-merge signatures
    # (cuk Fig 4 'Fig. 4.Moon', canup Fig 1 'Fig. 1.of', etc.).
    r"(?P<label>(?i:Fig(?:ure)?|Table|Plate|Scheme|Box)\.?\s*\d+[A-Za-z]?\.)"
    r"(?=[A-Za-z][a-z])",
)

# Bail-out guard: caption already starts with a proper "Label N. " token
# (label, period, whitespace).  In that case the in-place label is the
# correct one and any mid-string match found by _MISORDERED_LABEL_RE is
# a cross-ref artifact that this rescue should NOT touch.
_PROPER_LABEL_AT_START_RE = re.compile(
    r"^\s*(?i:Fig(?:ure)?|Table|Plate|Scheme|Box)\.?\s*\d+[A-Za-z]?\.\s",
)


# Same as _PROPER_LABEL_AT_START_RE but captures the kind ('Fig', 'Table',
# 'Plate', etc.) and the number, for the Phase B label-match scan that
# pairs caption candidates with parent figures.  Phase B treats "Fig. N."
# and "Table N." captions independently -- a "Fig. 2." candidate is not
# eligible to caption a table parent and vice versa.
_PROPER_LABEL_CAPTURE_RE = re.compile(
    r"^\s*(?P<kind>(?i:Fig(?:ure)?|Table|Plate|Scheme|Box))\.?\s*"
    r"(?P<num>\d+)[A-Za-z]?\.\s",
)


def _caption_kind_and_number(text: Optional[str]) -> Optional[tuple]:
    """Extract (kind, number) from a caption that starts with a proper
    label, e.g. 'Fig. 2. Plot of...' -> ('fig', 2); 'Table 1. Sample...'
    -> ('table', 1).  Returns None if the text doesn't start with a
    proper caption label.

    The kind is collapsed to a canonical lowercase token: 'fig' for
    Fig/Figure/FIG/etc., 'table', 'plate', 'scheme', 'box'.
    """
    if not text:
        return None
    m = _PROPER_LABEL_CAPTURE_RE.match(text)
    if not m:
        return None
    raw_kind = m.group("kind").lower()
    if raw_kind.startswith("fig"):
        kind = "fig"
    else:
        kind = raw_kind
    try:
        num = int(m.group("num"))
    except (TypeError, ValueError):
        return None
    return (kind, num)


def _parent_caption_kind(parent: "_MBlock") -> str:
    """Canonical kind ('fig' or 'table') for matching Phase B candidates
    to parent image/chart/table blocks."""
    if parent.type == "table":
        return "table"
    return "fig"  # image and chart both render as figures


def _repair_misordered_caption_label(text: Optional[str]) -> Optional[str]:
    """Splice an OCR-mis-attributed "Fig. N." / "Table N." label back to
    the front of an adopted caption.

    No-op when:
      * text is empty / None
      * caption already starts with a properly-formatted label
        (label + period + whitespace at the start of the string)
      * no OCR-merge signature is present in the body

    When the merge signature IS present (label + no-space + lowercase
    body word), the caption is restructured as:
        {label} {prefix-up-to-label.rstrip()} {tail-after-label}

    De-hyphenation: if the prefix's last char is '-' and the tail starts
    with a lowercase letter, the trailing hyphen is stripped and the
    two segments are concatenated without a separating space.  Handles
    the canup2012 Fig 2 case where the line-end soft-hyphen of
    "Compositional differ-" was kept by _join_lines_text because the
    OCR-mis-attributed label sat between it and "ence".

    Idempotent: a correctly-formatted caption passes through unchanged.
    """
    if not text:
        return text
    if _PROPER_LABEL_AT_START_RE.match(text):
        return text
    m = _MISORDERED_LABEL_RE.search(text)
    if not m:
        return text
    label = m.group("label")
    prefix = text[:m.start()].rstrip()
    tail = text[m.end("label"):]
    if not prefix:
        # Label is at the start of the caption but missing the space
        # that a proper label has after its period.  Just insert the
        # space; do not reorder.
        return f"{label} {tail}"
    if prefix.endswith("-") and tail and tail[0].islower():
        return f"{label} {prefix[:-1]}{tail}"
    return f"{label} {prefix} {tail}"


def _caption_is_watermark(text: Optional[str]) -> bool:
    """True when the caption text is publisher-watermark furniture
    (download stamp, copyright line, bare URL) rather than a real
    figure/table caption. Treated as 'empty' for adoption purposes."""
    if not text or not text.strip():
        return False
    s = text.strip()
    # Real captions typically lead with Fig./Table; never wholly a URL.
    if _FIG_TAB_CAPTION_RE.match(s):
        return False
    return any(p.search(s) for p in _CAPTION_WATERMARK_RES)


def _is_caption_text(text: Optional[str]) -> bool:
    """Caption signature: contains a `Fig\\.|Figure\\.|Table\\.` token
    within the first 100 chars. Permissive enough to catch cuk's
    'Formation of theFig. 1.lunar disk' (label mid-string)."""
    if not text:
        return False
    return bool(_FIG_TAB_CAPTION_RE.search(text[:100]))


def _is_footnote_text(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(_FOOTNOTE_OPENER_RE.match(text))


def _bbox_y_adjacent(parent_bbox: tuple, candidate_bbox: tuple,
                     tolerance: float = 60.0) -> bool:
    """True if candidate is within `tolerance` pt above/below parent
    OR vertically overlaps it (caption to the side of figure)."""
    if (not parent_bbox or len(parent_bbox) < 4
            or not candidate_bbox or len(candidate_bbox) < 4):
        return False
    p_y0, p_y1 = float(parent_bbox[1]), float(parent_bbox[3])
    c_y0, c_y1 = float(candidate_bbox[1]), float(candidate_bbox[3])
    if c_y1 <= p_y0 + 0.5 and (p_y0 - c_y1) <= tolerance:
        return True
    if c_y0 >= p_y1 - 0.5 and (c_y0 - p_y1) <= tolerance:
        return True
    overlap = min(p_y1, c_y1) - max(p_y0, c_y0)
    return overlap > 0


def _bbox_x_compatible(parent_bbox: tuple, candidate_bbox: tuple,
                       page_width: float = 612.0,
                       overlap_min: float = 0.4,
                       side_gap_max: float = 50.0) -> bool:
    """True if the candidate's x-range fits one of three patterns:
    (1) horizontally overlaps the parent (caption above/below figure),
    (2) is full-page-width (caption above wide table), or
    (3) is side-by-side with a small horizontal gap (cuk p2 pattern:
        caption block in left column, figure spans right two columns)."""
    if (not parent_bbox or len(parent_bbox) < 4
            or not candidate_bbox or len(candidate_bbox) < 4):
        return False
    p_x0, p_x1 = float(parent_bbox[0]), float(parent_bbox[2])
    c_x0, c_x1 = float(candidate_bbox[0]), float(candidate_bbox[2])
    overlap = max(0.0, min(p_x1, c_x1) - max(p_x0, c_x0))
    p_w = max(1.0, p_x1 - p_x0)
    c_w = max(1.0, c_x1 - c_x0)
    # (1) Overlap.
    if overlap / p_w >= overlap_min or overlap / c_w >= overlap_min:
        return True
    # (2) Full-page-width.
    if c_w / max(1.0, page_width) > 0.6:
        return True
    # (3) Side-by-side: candidate adjacent to parent with a small gap.
    h_gap = max(c_x0 - p_x1, p_x0 - c_x1)
    if 0 <= h_gap < side_gap_max:
        return True
    return False


def rescue_orphan_captions(blocks: list[_MBlock],
                           page_sizes: Optional[dict] = None) -> dict:
    """Detect mis-classified text blocks adjacent to figure/table
    blocks and adopt them as caption / footnote.

    Three layers, applied per page:

    1. PHASE A (validate-and-drop): walk image/chart/table parents whose
       caption is populated.  If the caption text doesn't start with a
       proper caption label (`Fig. N.` / `Table N.` etc.) and isn't a
       watermark, drop it -- MinerU's layout classifier wrongly nested
       body text under the figure (young p3 Fig 2).  Increments
       `misnested_dropped` and sets `parent.text = None` so the rescue
       below can run.

    2. GEOMETRIC adoption: original behavior.  When an image / chart /
       table has an empty caption (or a watermark-only caption like
       'Downloaded from science.org...') AND a nearby `text` block has
       a caption signature (`Fig.\\s*N`, `Table\\s*N`, etc.), adopt the
       closest geometric candidate.

    3. PHASE B (3-col page-wide label-match): only on pages classified
       as 3-column by `_detect_three_column_page` AND only when no
       geometric candidate was found.  Scans the entire page for text
       blocks starting with `Fig. N. ` / `Table N. ` and adopts on the
       CLEAN SINGLE MATCH rule: if exactly one orphaned parent has
       `need_caption=True` AND exactly one candidate starts with that
       same figure-number label, adopt.  Otherwise log a warning and
       leave for manual review.

    Footnote adoption: empty `table_footnote` slot AND a nearby text
    block opens with `*†‡§` or `Notes:`. Layer 2 only.

    Mutates blocks in place. Adopted text blocks have their type set
    to `_adopted_caption` / `_adopted_footnote` so emit_markdown skips
    them; their content has already been merged into the parent's
    text/footnote field.

    Closes Stage-3 issues identified in cuk p2 (Fig 1 caption-as-text)
    and alexander p2 (Table 1 caption + notes both classified as text).
    Phase A/B added 2026-05-11 for young p3 Fig 2 (3-col Science).

    `page_sizes` is `{page_idx: (width, height)}` from
    `ParsedDoc.page_sizes`.  Required for Phase B's 3-col gating;
    Phase B is skipped (no-op) if not provided.

    Returns stats dict.
    """
    info = {"image_table_blocks": 0,
            "misnested_dropped": 0,
            "captions_adopted": 0,
            "footnotes_adopted": 0,
            "captions_adopted_3col": 0,
            "ambiguous_3col_warnings": 0,
            "pages_3col": []}
    if not blocks:
        return info

    by_page: dict[int, list[_MBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_idx, []).append(b)

    def _gap_to_parent(parent_bbox: tuple, cand_bbox: tuple) -> float:
        """Smallest vertical gap from candidate to parent (0 if any
        y-overlap). Used to pick the CLOSEST caption when multiple
        candidates are adjacency-eligible — avoids adopting body-text
        paragraphs further from the figure when a real caption block
        sits closer."""
        p_y0, p_y1 = float(parent_bbox[1]), float(parent_bbox[3])
        c_y0, c_y1 = float(cand_bbox[1]), float(cand_bbox[3])
        if c_y1 <= p_y0:  # candidate fully above
            return p_y0 - c_y1
        if c_y0 >= p_y1:  # candidate fully below
            return c_y0 - p_y1
        return 0.0  # vertical overlap

    for page_idx, page_blocks in by_page.items():
        # 3-column detection -- gates the Phase A/B combined pass below.
        # Single-/two-column papers (canup Nature 2014, modern Nature
        # 2-col, etc.) skip the new rescue layer entirely; their existing
        # captions (Nature 'Figure N |' headers, panel prefixes, sub-
        # panel stubs) are preserved unchanged.
        page_w = 612.0
        if page_sizes and page_idx in page_sizes:
            page_w = page_sizes[page_idx][0]
        text_blocks_for_page = [b for b in page_blocks
                                if b.type in ("text", "ref_text")]
        is_three_col = _detect_three_column_page(
            text_blocks_for_page, page_w)
        if is_three_col:
            info["pages_3col"].append(page_idx + 1)

        for parent in page_blocks:
            if parent.type not in ("image", "chart", "table"):
                continue
            info["image_table_blocks"] += 1
            need_caption = (not parent.text
                            or _caption_is_watermark(parent.text))
            need_footnote = (parent.type == "table"
                             and not parent.footnote)
            if not need_caption and not need_footnote:
                continue

            # Gather all geometrically eligible candidates. Pick by
            # minimum vertical gap so the CLOSEST caption-shaped block
            # wins (avoids body-paragraph false positives further away).
            cap_candidates: list[tuple[float, _MBlock]] = []
            foot_candidates: list[tuple[float, _MBlock]] = []
            for cand in page_blocks:
                if cand is parent or cand.type != "text" or not cand.text:
                    continue
                if cand.bbox and len(cand.bbox) >= 4:
                    h = float(cand.bbox[3]) - float(cand.bbox[1])
                    if h > 500:
                        continue
                if not _bbox_y_adjacent(parent.bbox, cand.bbox):
                    continue
                if not _bbox_x_compatible(parent.bbox, cand.bbox):
                    continue
                gap = _gap_to_parent(parent.bbox, cand.bbox)
                if need_caption and _is_caption_text(cand.text):
                    cap_candidates.append((gap, cand))
                if need_footnote and _is_footnote_text(cand.text):
                    foot_candidates.append((gap, cand))

            if need_caption and cap_candidates:
                cap_candidates.sort(key=lambda t: t[0])
                _, cand = cap_candidates[0]
                if (parent.text
                        and not _caption_is_watermark(parent.text)):
                    parent.text = (parent.text.strip() + " "
                                   + cand.text.strip())
                else:
                    parent.text = cand.text.strip()
                parent.text = _repair_misordered_caption_label(parent.text)
                cand.type = "_adopted_caption"
                info["captions_adopted"] += 1
                log.info("rescue_orphan_captions: adopted caption "
                         "for page %d %s (idx=%d): %r",
                         parent.page_idx + 1, parent.type,
                         parent.index, cand.text[:80])

            if need_footnote and foot_candidates:
                foot_candidates.sort(key=lambda t: t[0])
                _, cand = foot_candidates[0]
                if parent.footnote:
                    parent.footnote = (parent.footnote.strip() + " "
                                       + cand.text.strip())
                else:
                    parent.footnote = cand.text.strip()
                cand.type = "_adopted_footnote"
                info["footnotes_adopted"] += 1
                log.info("rescue_orphan_captions: adopted footnote "
                         "for page %d table (idx=%d): %r",
                         parent.page_idx + 1, parent.index,
                         cand.text[:80])

        # === PHASE A + B: 3-col page-wide caption rescue ==============
        # Only on pages classified as 3-column by the x-center detector.
        # Combines two operations atomically:
        #
        #   PHASE A (drop): mark a parent's caption as "needs replace-
        #       ment" when it is empty, a publisher watermark, OR
        #       doesn't carry a Fig/Table label (misnested body text).
        #
        #   PHASE B (adopt): for each parent in "needs replacement",
        #       look for a unique caption-shaped text block on the page
        #       starting with a matching `Fig. N.` / `Table N.` label.
        #
        # Critical: Phase A only DROPS when Phase B has a replacement.
        # If a parent's caption looks misnested but no Phase B candidate
        # exists, the existing caption is preserved (some label-less
        # captions in the corpus -- cuk Table 1, canup2012 Fig 3 -- are
        # the real caption stripped of its leading label by OCR; losing
        # them is worse than keeping the imperfect label-less version).
        #
        # Ambiguous cases (>=2 candidates with the same Fig. N.) log a
        # warning and are left for manual review per the user's "clean
        # single match" rule.
        if not is_three_col:
            continue

        # Catalog parents needing replacement.  Each entry is
        # (parent, reason) where reason in {'empty', 'watermark',
        # 'misnested'}; only 'misnested' increments misnested_dropped
        # when actually swapped.
        needs_replacement: list = []
        for p in page_blocks:
            if p.type not in ("image", "chart", "table"):
                continue
            if not p.text:
                needs_replacement.append((p, "empty"))
            elif _caption_is_watermark(p.text):
                needs_replacement.append((p, "watermark"))
            elif (not _is_caption_text(p.text)
                    and not _is_figure_caption_stub(p.text)):
                needs_replacement.append((p, "misnested"))
        if not needs_replacement:
            continue

        # Caption-shaped candidates: text blocks (not yet adopted) whose
        # text starts with a proper "Fig. N. " / "Table N. " label.
        candidates: list[_MBlock] = []
        for cand in page_blocks:
            if cand.type != "text" or not cand.text:
                continue
            kn = _caption_kind_and_number(cand.text)
            if kn is not None:
                candidates.append(cand)

        # Group candidates by (kind, number).
        by_label: dict = {}
        for cand in candidates:
            kn = _caption_kind_and_number(cand.text)
            if kn is None:
                continue
            by_label.setdefault(kn, []).append(cand)

        # Warn on ambiguous labels (2+ candidates with the same kind+num).
        for kn, lst in by_label.items():
            if len(lst) > 1:
                log.warning(
                    "rescue_orphan_captions[3col]: ambiguous caption for "
                    "%s %d on page %d -- %d candidates start with the "
                    "same label, leaving unchanged for manual review",
                    kn[0], kn[1], page_idx + 1, len(lst))
                info["ambiguous_3col_warnings"] += 1

        # Filter to unique single-match candidates, partition by kind.
        unique_by_kind: dict = {"fig": [], "table": [],
                                "plate": [], "scheme": [], "box": []}
        for kn, lst in by_label.items():
            if len(lst) == 1:
                unique_by_kind.setdefault(kn[0], []).append(
                    (kn[1], lst[0]))

        # Adopt per "needs replacement" parent.  Skip if multiple parents
        # of the same kind share the same single candidate set (we cannot
        # disambiguate which fig number goes to which parent).
        parents_by_kind: dict = {}
        for p, reason in needs_replacement:
            parents_by_kind.setdefault(
                _parent_caption_kind(p), []).append((p, reason))

        for kind, parent_list in parents_by_kind.items():
            candidates_for_kind = unique_by_kind.get(kind, [])
            if not candidates_for_kind:
                # No replacement available.  Phase A's misnested-detect
                # found nothing better -- preserve the existing caption
                # for parents marked 'misnested' (don't strip without
                # replacement); leave 'empty' / 'watermark' parents
                # alone as before.
                continue
            if len(parent_list) != 1 or len(candidates_for_kind) != 1:
                log.warning(
                    "rescue_orphan_captions[3col]: page %d has %d "
                    "needs-replacement %s parent(s) and %d unique "
                    "candidate(s); cannot disambiguate -- leaving "
                    "unchanged",
                    page_idx + 1, len(parent_list), kind,
                    len(candidates_for_kind))
                info["ambiguous_3col_warnings"] += 1
                continue
            parent, reason = parent_list[0]
            fig_num, cand = candidates_for_kind[0]
            if reason == "misnested":
                log.info(
                    "rescue_orphan_captions[3col]: dropping misnested "
                    "caption on page %d %s (idx=%d) and replacing: %r",
                    page_idx + 1, parent.type, parent.index,
                    parent.text[:80])
                info["misnested_dropped"] += 1
            parent.text = _repair_misordered_caption_label(
                cand.text.strip())
            cand.type = "_adopted_caption"
            info["captions_adopted_3col"] += 1
            log.info(
                "rescue_orphan_captions[3col]: page %d %s (idx=%d) <- "
                "%s %d candidate: %r",
                page_idx + 1, parent.type, parent.index,
                kind, fig_num, cand.text[:80])

    log.info("rescue_orphan_captions: %d image/table block(s); "
             "misnested dropped=%d, captions adopted=%d "
             "(geometric=%d, 3col=%d), footnotes adopted=%d, "
             "ambiguous warnings=%d, 3-col pages=%s",
             info["image_table_blocks"], info["misnested_dropped"],
             info["captions_adopted"] + info["captions_adopted_3col"],
             info["captions_adopted"], info["captions_adopted_3col"],
             info["footnotes_adopted"],
             info["ambiguous_3col_warnings"],
             info["pages_3col"])
    return info


# ============================================================================
# Sub-panel consolidation
#
# Two distinct code sections per the design separation rule:
#   Section 1: caption predicates (pure string functions; no _MBlock
#              knowledge; reusable by other rescue passes).
#   Section 2: group reconstruction (operates on _MBlock list; calls
#              into Section 1 for caption classification).
#
# In scope: figures only.  Pattern A (canup-style real-caption-with-
# letter-prefix + stub-letter siblings) and Pattern C (luo-style
# stub-Figure-N siblings).  Pattern B (cross-figure caption absorption)
# is guarded against (skip when primary's caption contains 2+ figure
# anchors) but NOT fixed here.  Pattern D (multi-page table same-
# caption duplicates) and ALL table cases are explicitly out of scope
# -- separate evaluation.
# ============================================================================


# === Section 1: caption predicates =========================================
# Operate on a single caption STRING.  No knowledge of _MBlock.

# All caption styles seen in the corpus: "Figure 1", "Fig. 1", "Fig 1",
# "FIG. 1".  Optional period after Fig.
_FIG_LABEL_RE_FRAG = r"(?:Figure|Fig\.?|FIG\.?)"

# Leading panel-letter prefix, in any of three observed forms:
#   "b Figure 1 | text"        bare letter (Nature, canup)
#   "(c) Figure 1. body..."    parenthesized (feng's J. Geophys. Res.)
#   "b) Figure 1. body..."     closing-paren-only (ocampo's AGU style;
#                              MinerU reads the panel-label glyph
#                              `a)` printed near the figure)
# All three forms must be applied consistently across the caption
# predicates below so that primary/stub detection, letter extraction,
# and figure-number extraction agree.
_PANEL_PREFIX_FRAG = r"(?:(?:\(\s*[a-h]\s*\)|[a-h]\s*\)|[a-h])\s+)?"

# A "primary" figure caption: optional leading panel-letter prefix,
# then the figure label, then a separator ([|.,:] or whitespace before
# a capital letter), then substantial body text (>=10 chars).
_PRIMARY_FIG_CAPTION_RE = re.compile(
    rf"^\s*{_PANEL_PREFIX_FRAG}{_FIG_LABEL_RE_FRAG}\s+\d+"
    r"(?:\s*[|.,:]\s*|\s+(?=[A-Z(]))"
    r"\S.{9,}",
    re.IGNORECASE | re.DOTALL,
)

# A "stub" caption is one of:
#   - just a panel letter (or letters): "a", "b a", "  c  ", "(a)", "a)", "a b)"
#   - just a figure label: "Figure 1", "Fig. 7"
#   - empty / whitespace-only (None handled by the predicate, not regex)
# Closing-paren form (`a)`, `b)`, `a b)`) is the ocampo AGU style;
# parenthesized form (`(a)`) is feng's J. Geophys. Res. style; bare
# form (`a`, `a b`) is canup's Nature style.
_STUB_FIG_CAPTION_RE = re.compile(
    rf"^\s*(?:"
    rf"\(\s*[a-h]\s*\)\s*|"               # parenthesized: (a)
    rf"[a-h]\s*\)?(?:\s+[a-h]\s*\)?)*\s*|"  # bare or closing-paren: a / a) / a b / a b)
    rf"{_FIG_LABEL_RE_FRAG}\s+\d+\s*"
    rf")$",
    re.IGNORECASE,
)

# Strip a leading panel-letter from a caption like "b Figure 1 | text"
# -> ("Figure 1 | text", "b") or "(c) Figure 1." -> ("Figure 1.", "c")
# or "b) Figure 1." -> ("Figure 1.", "b"). Only fires when followed by
# a figure label (so we don't accidentally strip "a" from real text).
# Three alternation groups for the three observed forms.
_LEADING_PANEL_LETTER_RE = re.compile(
    rf"^\s*(?:"
    rf"\(\s*([a-h])\s*\)|"       # group 1: (a) parens
    rf"([a-h])\s*\)|"            # group 2: a) closing paren only
    rf"([a-h])"                  # group 3: a bare
    rf")\s+(?={_FIG_LABEL_RE_FRAG}\s+\d+)",
    re.IGNORECASE,
)

# Counts figure-header anchors -- a "header" is a figure label at the
# start of the caption OR preceded by sentence-ending punctuation
# (`. `, `! `, `? `).  This distinguishes "Fig. 6. <body>. Fig. 7.
# <body>." (two headers -- Pattern B) from "Figure 3 | ... from the
# Fig. 1 simulation. ..." (one header; Fig. 1 is a cross-reference
# inside body text, not a second header).  Drives the Pattern B guard.
_FIG_HEADER_ANCHOR_RE = re.compile(
    rf"(?:^|(?<=[.!?]\s)){_FIG_LABEL_RE_FRAG}\s+\d+",
    re.IGNORECASE,
)

# Extract the figure number from a primary or stub caption.  Used by
# the grouping pass to refuse adopting "FIG. 5" stubs into "FIG. 6"
# primaries (Merrill 1922 pattern: each stub IS its own figure, not
# a panel of the next figure).
_FIG_NUMBER_RE = re.compile(
    rf"^\s*{_PANEL_PREFIX_FRAG}{_FIG_LABEL_RE_FRAG}\s+(\d+)",
    re.IGNORECASE,
)


def _is_figure_caption_primary(caption: Optional[str]) -> bool:
    """True if `caption` looks like a real figure caption with
    substantial body text after the label.  Used to identify the
    primary block in a multi-panel group."""
    if not caption:
        return False
    return bool(_PRIMARY_FIG_CAPTION_RE.match(caption))


def _is_figure_caption_stub(caption: Optional[str]) -> bool:
    """True if `caption` is a stub: empty, single panel letter, or
    just a 'Figure N' label with no body text."""
    if not caption or not caption.strip():
        return True
    return bool(_STUB_FIG_CAPTION_RE.match(caption))


def _strip_leading_panel_letter(
        caption: Optional[str]) -> tuple[str, Optional[str]]:
    """Return (cleaned_caption, extracted_letter | None).

    'b Figure 1 | text'  -> ('Figure 1 | text', 'b')
    '(c) Figure 1. body' -> ('Figure 1. body',  'c')
    'Figure 1 | text'    -> ('Figure 1 | text', None)
    """
    if not caption:
        return caption or "", None
    m = _LEADING_PANEL_LETTER_RE.match(caption)
    if not m:
        return caption, None
    # Three alternation groups for the three observed forms:
    # `(c)` -> group 1; `c)` -> group 2; bare `c` -> group 3.
    letter = (m.group(1) or m.group(2) or m.group(3)).lower()
    cleaned = caption[m.end():]
    return cleaned, letter


def _count_figure_anchors(caption: Optional[str]) -> int:
    """How many figure-HEADER anchors appear in the caption.  A header
    is a figure label at the start of the caption OR preceded by
    sentence-ending punctuation (`. `, `! `, `? `).  Cross-references
    inside body text (e.g. "from the Fig. 1 simulation") DO NOT count.
    Drives the Pattern B guard (skip consolidation when >=2)."""
    if not caption:
        return 0
    return len(_FIG_HEADER_ANCHOR_RE.findall(caption))


# Counts panel references in a caption: "a, ", "b, ", "(c) " etc.
# Used as a sanity cap on letter-stub adoption: a primary whose
# caption has zero panel references probably isn't a multi-panel
# figure even when stubs are spatially adjacent (Merrill 1922 plate
# pattern: page of chondrule photos, two of which got captions like
# "Fig. 2.—One of the same chondrules broken..." with no panel
# letters in the body -- adopting the surrounding empty-caption
# stubs as panels is wrong).
_PANEL_REFERENCE_RE = re.compile(
    r"\b([a-h])[,.]\s+\S",
    re.IGNORECASE,
)


def _count_panel_references(caption: Optional[str]) -> int:
    """Count distinct panel letters referenced in a caption.

    Matches 'a, body', 'b. body', etc.  Returns the number of distinct
    panel letters (so 'a, ... a, ...' counts as 1).  Caps the number
    of letter-only stubs that may be adopted as panels of this
    primary -- a 0-reference caption can't legitimately be a
    multi-panel figure.
    """
    if not caption:
        return 0
    letters = {m.group(1).lower()
               for m in _PANEL_REFERENCE_RE.finditer(caption)}
    return len(letters)


def _extract_figure_number(caption: Optional[str]) -> Optional[int]:
    """Extract the figure number from a caption like 'FIG. 6 ...' or
    'Figure 14' or 'b Figure 1 | ...'.  Returns None when the caption
    is letter-only ('a'), empty, or otherwise lacks a Fig N anchor at
    the start.

    Used by the grouping pass to enforce: a stub with explicit figure
    number N may only adopt to a primary whose figure number is also
    N.  Without this check, a 'FIG. 5' stub on the same page as a
    'FIG. 6 <body>' primary would be falsely adopted as panel-of-FIG-6
    (Merrill 1922 pattern).  Letter-only / empty stubs return None
    here and fall through to the reading-order flow.
    """
    if not caption:
        return None
    m = _FIG_NUMBER_RE.match(caption)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


# === Section 2: group reconstruction =======================================
# Operates on a _MBlock list.  Calls into Section 1 for predicates.


def _subpanel_letter_assignments(
        primary_letter: Optional[str], n_panels: int) -> list[str]:
    """Generate panel letters for a consolidated multi-panel figure.

    `n_panels` is the total panel count (primary + subpanels).
    `primary_letter` is the letter extracted from the primary's caption
    prefix, or None.

    When primary_letter is set, place it in the primary's slot (always
    index 0 — primary is rendered first) and assign a-z to the other
    panels in reading order, skipping the primary's letter.
    Otherwise label all panels by reading order (a, b, c, ...).
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if primary_letter and primary_letter in alphabet:
        labels = [primary_letter]
        used = {primary_letter}
        i = 0
        while len(labels) < n_panels and i < len(alphabet):
            ch = alphabet[i]
            if ch not in used:
                labels.append(ch)
                used.add(ch)
            i += 1
        return labels
    return list(alphabet[:n_panels])


def _group_subpanels(image_blocks: list) -> list:
    """Group image/chart blocks on a page into (primary, stubs) tuples.

    Two-pass design, one pass per stub kind:

      Pass 1 (letter-only / empty stubs): flow by reading order to
      the NEXT primary.  This matches the canonical caption-below-
      panels layout where panel images appear above the primary's
      caption.  Trailing stubs (after the last primary) attach to
      the last primary as a caption-above-panels safety fallback.

      Pass 2 (numbered stubs like 'FIG. 5', 'Figure 6'): adopt ONLY
      to a primary with the same figure number.  This protects
      against the Merrill 1922 pattern where each stub IS its own
      figure (separate from the primary on the same page) -- a
      'FIG. 5' stub MUST NOT become a panel of a 'FIG. 6 <body>'
      primary, even when they share a page.

    Returns [(primary, stubs), ...] with primaries in their original
    order.  No mutation -- callers decide what to do with the groups.
    """
    primaries_with_nums = [
        (b, _extract_figure_number(b.text))
        for b in image_blocks
        if _is_figure_caption_primary(b.text)
    ]
    if not primaries_with_nums:
        return []
    nums_to_primary: dict = {}
    for p, n in primaries_with_nums:
        if n is not None and n not in nums_to_primary:
            nums_to_primary[n] = p
    primary_ids = {id(p) for p, _ in primaries_with_nums}
    groups: dict = {id(p): [] for p, _ in primaries_with_nums}

    # Pass 1: letter-only / empty stubs flow to the NEXT primary in
    # reading order.  Trailing stubs attach to the LAST primary
    # (caption-above-panels fallback).
    pending_letter_stubs: list = []
    for b in image_blocks:
        if id(b) in primary_ids:
            if pending_letter_stubs:
                groups[id(b)].extend(pending_letter_stubs)
                pending_letter_stubs = []
            continue
        if not _is_figure_caption_stub(b.text):
            continue
        if _extract_figure_number(b.text) is None:
            pending_letter_stubs.append(b)
    if pending_letter_stubs:
        last_primary = primaries_with_nums[-1][0]
        groups[id(last_primary)].extend(pending_letter_stubs)

    # Pass 1 cap: drop letter-stub adoptions when the count exceeds
    # max(panel_refs, MAX_DEFAULT).  Two cases:
    #   - Caption explicitly references panels ('a, body. b, body.'):
    #     use that count as the cap.  canup Fig 3 has 3 refs and 2
    #     stubs -> keeps; a hypothetical 5-stub adoption to a 3-ref
    #     primary would drop.
    #   - Caption has no panel references (jacquet Fig 1 captions use
    #     'panels A and B' prose that the regex doesn't catch):
    #     fall back to a fixed cap.  Multi-panel figures rarely exceed
    #     6 panels; 4 is conservative.  The Merrill 1922 plate page
    #     (10 empty-caption stubs surrounding two primaries with no
    #     panel references) trips the cap and stays unmerged.
    # Numbered-stub adoptions in Pass 2 are NOT capped here -- the
    # figure-number match is itself a strong signal.
    MAX_LETTER_STUBS_DEFAULT = 4
    for primary, _ in primaries_with_nums:
        n_refs = _count_panel_references(primary.text)
        cap = max(n_refs, MAX_LETTER_STUBS_DEFAULT)
        n_letter_stubs = len(groups[id(primary)])
        if n_letter_stubs > cap:
            log.info("rescue_subpanel_groups: dropping %d letter-stub "
                     "adoption(s) from primary fig=%s -- exceeds cap "
                     "(refs=%d, cap=%d; plate-page guard)",
                     n_letter_stubs,
                     _extract_figure_number(primary.text) or "?",
                     n_refs, cap)
            groups[id(primary)] = []

    # Pass 2: numbered stubs adopt iff their number matches a primary.
    for b in image_blocks:
        if id(b) in primary_ids or not _is_figure_caption_stub(b.text):
            continue
        stub_num = _extract_figure_number(b.text)
        if stub_num is None:
            continue  # already handled in pass 1
        target = nums_to_primary.get(stub_num)
        if target is not None:
            groups[id(target)].append(b)
        # else: orphan -- stays as standalone image in emit_markdown.

    return [(p, groups[id(p)]) for p, _ in primaries_with_nums]


def rescue_subpanel_groups(blocks: list) -> dict:
    """Per-page consolidation of multi-panel figure groups.  Operates
    on `image` and `chart` blocks (MinerU classifies multi-panel
    figures as either, depending on visual content -- canup's Figs 1-3
    come out as 'chart').  Tables are out of scope.

    For each (primary, stubs) group from _group_subpanels (which
    enforces figure-number matching for numbered stubs and reading-
    order flow for letter-only stubs):
      - extract leading panel-letter from primary's caption (if any)
      - collect stubs' image_paths into primary.subpanel_paths
      - mark stubs' type as '_adopted_subpanel' so emit_markdown skips them

    Pattern B guard: skip consolidation when primary's caption
    contains 2+ figure HEADER anchors (MinerU has absorbed two
    captions; sentence-boundary check prevents false-firing on
    cross-references like "from the Fig. 1 simulation").

    Returns {'groups': N, 'panels_adopted': M} for logging.
    """
    from collections import defaultdict

    page_to_image_blocks: dict = defaultdict(list)
    for b in blocks:
        if b.type in ("image", "chart"):
            page_to_image_blocks[b.page_idx].append(b)

    groups_n = 0
    panels_adopted = 0
    for page_idx, image_blocks in page_to_image_blocks.items():
        if len(image_blocks) < 2:
            continue
        for primary, stubs in _group_subpanels(image_blocks):
            if not stubs:
                continue   # primary with no panels; nothing to consolidate
            # Pattern B guard.
            if _count_figure_anchors(primary.text or "") >= 2:
                log.info("rescue_subpanel_groups: skipping group on "
                         "page %d -- primary caption contains 2+ "
                         "figure anchors (pattern B; needs separate "
                         "fix)", page_idx + 1)
                continue
            cleaned, letter = _strip_leading_panel_letter(primary.text)
            primary.text = cleaned
            primary.panel_letter = letter
            # Cache caption's figure number for emit_markdown alt-text
            # (so renderings show the actual "Figure 6" rather than
            # the emission-order counter, which can drift when MinerU
            # skips figures).
            primary.figure_number = _extract_figure_number(cleaned)
            adopted = [s for s in stubs if s.image_path]
            primary.subpanel_paths = [s.image_path for s in adopted]
            primary.subpanel_bboxes = [s.bbox for s in adopted]
            for s in stubs:
                s.type = "_adopted_subpanel"
            groups_n += 1
            panels_adopted += len(stubs)
            log.info("rescue_subpanel_groups: page %d, fig=%s, "
                     "primary letter=%s, %d panels adopted",
                     page_idx + 1,
                     primary.figure_number or "?",
                     letter or "?",
                     len(stubs))

    info = {"groups": groups_n, "panels_adopted": panels_adopted}
    if groups_n:
        log.info("rescue_subpanel_groups: %d figure group(s) "
                 "consolidated; %d panel(s) adopted",
                 groups_n, panels_adopted)
    return info
