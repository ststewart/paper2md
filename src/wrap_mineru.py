"""wrap_mineru.py — paper2md value-adds layered on MinerU body extraction.

License: MIT (paper2md). Calls MinerU as a separate subprocess; users who
distribute MinerU in their environment must comply with its AGPL-3.0 terms
separately. The wrapper itself (this file + paper2md helpers it imports) is
MIT.

Provenance: MinerU is developed by Shanghai AI Lab (opendatalab). Disclosed
for institutional review; users for whom Chinese-origin tooling is a
sensitivity should use paper2md's marker-based pipeline instead.

This wrapper consumes MinerU's `<stem>_middle.json` structured output (which
already pairs figures/tables/charts with captions) and re-emits markdown with
paper2md's frontmatter, references rescue, copyright/OA metadata, data-repo
detection, and HDF5 bundling layered on top.

Usage:
    python src/wrap_mineru.py paper.pdf -o out/                   (default: pipeline backend)
    python src/wrap_mineru.py paper.pdf -o out/ --vlm-tables      (paper2md VLM cleanup on suspicious tables)
    python src/wrap_mineru.py paper.pdf -o out/ --mineru-backend hybrid-auto-engine
    python src/wrap_mineru.py --skip-mineru-run --mineru-dir <auto/> paper.pdf -o out/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import metadata_frontend
import paper2md as p2m
# Layout-related machinery (block dataclasses, middle.json reader,
# subprocess wrapper, orphan-caption rescue) was moved to layout_mineru
# in Stage 4 of the convergence plan so paper2md.py can consume the
# same code in Stage 5. Re-imported into this module's namespace so
# existing call sites (and tests) continue to work via `wm._MBlock`,
# `wm.parse_middle_json`, etc.
from layout_mineru import (
    _MBlock,
    ParsedDoc,
    _bbox_reading_key,
    _bbox_x_compatible,
    _bbox_y_adjacent,
    _CAPTION_WATERMARK_RES,
    _caption_is_watermark,
    _FIG_TAB_CAPTION_RE,
    _first_span,
    _FOOTNOTE_OPENER_RE,
    _is_caption_text,
    _is_footnote_text,
    _join_lines_text,
    _locate_mineru_auto_dir,
    parse_middle_json,
    rescue_orphan_captions,
    run_mineru,
    stage_existing_mineru_dir,
)

log = logging.getLogger("wrap_mineru")


# === HTML table → pipe markdown converter ==================================
#
# Pure-stdlib html.parser. Returns (markdown_or_None, reason_or_empty).
# Returns None if the table has rowspan/colspan/nested tables/empty result.
# Caller is expected to fall back to the raw HTML or to a VLM rewrite.


_EQ_RE = re.compile(r"\$([^$]+)\$")  # post-processing safety, unused for emit

# MinerU emits <eq>LATEX</eq> inside HTML table cells. The pipe-MD
# converter handles this inline (see _TableHTMLParser.handle_endtag),
# but for raw-HTML fallback tables (rowspan/colspan rejected) we
# translate the tag to `$...$` so markdown viewers render the math.
# DOTALL because some MinerU equations span newlines.
_EQ_TAG_RE = re.compile(r"<eq>(.*?)</eq>", re.DOTALL)


class _TableHTMLParser(HTMLParser):
    """Parses MinerU's HTML table cells into structured rows.

    Each row is a list[(text, colspan)] tuple. rowspan>1 still rejects
    the whole table (pipe-MD has no rowspan). colspan>1 is permitted
    and resolved at conversion time as a section-subheader candidate.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int]]] = []
        self.cur_row: Optional[list[tuple[str, int]]] = None
        self.cur_cell: Optional[list[str]] = None
        self.cur_cell_colspan: int = 1
        self.in_eq = False
        self.eq_buf: list[str] = []
        self.bad: Optional[str] = None
        self._depth_table = 0
        self._headers_marked: bool = False  # first row was wrapped in <thead>

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth_table += 1
            if self._depth_table > 1:
                self.bad = "nested-table"
        elif tag == "thead":
            self._headers_marked = True
        elif tag == "tr":
            self.cur_row = []
        elif tag in ("td", "th"):
            self.cur_cell = []
            self.cur_cell_colspan = 1
            try:
                cs = a.get("colspan")
                if cs is not None:
                    self.cur_cell_colspan = max(1, int(cs))
                rs = a.get("rowspan")
                if rs is not None and int(rs) > 1:
                    # rowspan still rejects: pipe-MD can't represent
                    # cells that span multiple rows.
                    self.bad = f"rowspan={rs}"
            except ValueError:
                pass
        elif tag == "eq":
            self.in_eq = True
            self.eq_buf = []
        elif tag == "br":
            if self.cur_cell is not None:
                self.cur_cell.append("<br>")

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth_table -= 1
        elif tag in ("td", "th"):
            if self.cur_row is not None and self.cur_cell is not None:
                cell = "".join(self.cur_cell).strip()
                # Collapse whitespace runs but preserve <br>.
                cell = re.sub(r"\s+", " ", cell).strip()
                # Pipe characters inside cells must be escaped.
                cell = cell.replace("|", "\\|")
                self.cur_row.append((cell, self.cur_cell_colspan))
            self.cur_cell = None
            self.cur_cell_colspan = 1
        elif tag == "tr":
            if self.cur_row is not None:
                self.rows.append(self.cur_row)
            self.cur_row = None
        elif tag == "eq":
            expr = "".join(self.eq_buf).strip()
            # Wrap as inline LaTeX. Cell text content can contain $...$.
            if self.cur_cell is not None and expr:
                self.cur_cell.append(f"${expr}$")
            self.in_eq = False
            self.eq_buf = []

    def handle_data(self, data):
        if self.in_eq:
            self.eq_buf.append(data)
        elif self.cur_cell is not None:
            self.cur_cell.append(data)


def html_to_pipe_md(html: str) -> tuple[Optional[str], str]:
    """Convert a `<table>...</table>` HTML string to a GFM pipe table.

    Returns (markdown, "") on success. Returns (None, reason) when not
    convertible: rowspan>1, nested tables, zero rows, inconsistent
    column counts across data rows, or colspan-bearing rows that
    can't be classified as section subheaders.

    Section-subheader rows (full-width row with colspan AND <=2 non-
    empty cells) are emitted as a markdown row with bolded content
    in column 1 and empty cells filling the rest, e.g.:
        | **0.99 M_E target with 2.3 hours spin and 3.0 L_EM** |  |  |
    This salvages tables where publishers use a single full-width
    cell as a group divider between data blocks (cuk Table 1 pattern).
    """
    if not html or not html.strip():
        return None, "empty-html"
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as e:
        return None, f"parse-error: {e}"
    if parser.bad:
        return None, parser.bad
    rows = parser.rows
    if not rows:
        return None, "no-rows"

    # Classify each row.
    #   data:    every cell has colspan==1 (cleanly representable as pipe)
    #   section: at least one cell has colspan>1 AND <=2 non-empty cells
    #            (publisher subheader pattern; cuk Table 1 style)
    #   skip:    every cell is empty (continuation noise)
    #   reject:  has colspan>1 cells but >2 non-empty cells — ambiguous,
    #            we don't try to align such rows to the data-row grid
    classified: list[tuple[str, object]] = []
    for r in rows:
        non_empty = [(t, cs) for t, cs in r if t.strip()]
        if not non_empty:
            classified.append(("skip", None))
            continue
        has_wide_cell = any(cs > 1 for _, cs in r)
        if not has_wide_cell:
            classified.append(("data", r))
            continue
        if len(non_empty) <= 2:
            content = " ".join(t for t, _ in non_empty)
            classified.append(("section", content))
        else:
            return None, (
                f"colspan-with-multi-content "
                f"(non_empty={len(non_empty)})"
            )

    # Determine canonical width from data rows.
    data_rows = [r for kind, r in classified if kind == "data"]
    if not data_rows:
        return None, "no-data-rows"
    col_counts = {len(r) for r in data_rows}
    if len(col_counts) != 1:
        return None, f"inconsistent-cols ({sorted(col_counts)})"
    width = next(iter(col_counts))
    if width == 0:
        return None, "zero-width"

    # Build pipe markdown. First data row is the header.
    out: list[str] = []
    header_emitted = False
    n_sections = 0
    for kind, payload in classified:
        if kind == "skip":
            continue
        if kind == "data":
            cells = [t for t, _ in payload]
            if not header_emitted:
                out.append("| " + " | ".join(cells) + " |")
                out.append("|" + "|".join([" --- "] * width) + "|")
                header_emitted = True
            else:
                out.append("| " + " | ".join(cells) + " |")
        elif kind == "section":
            if not header_emitted:
                # Section row before any data row: drop silently.
                # Pipe markdown can't represent it cleanly without a
                # header above it; the table image still preserves it.
                continue
            empty_tail = " | ".join([""] * (width - 1))
            text = payload
            out.append(f"| **{text}** | {empty_tail} |")
            n_sections += 1

    if not header_emitted:
        # Synthesize a body row so the pipe is well-formed (matches old
        # behavior for tables that have no body data rows).
        return None, "no-data-after-classification"

    # If we have a body of zero data rows after the header, pad with one
    # blank row (pipe-MD requires at least one body row to render).
    body_count = sum(1 for k, _ in classified if k == "data")
    if body_count == 1:
        out.append("| " + " | ".join([""] * width) + " |")

    return "\n".join(out), ""


# === per-block markdown emitters ===========================================


def _format_caption(text: Optional[str], default: str) -> str:
    """Strip leading FIG./Fig./TABLE markers IF the caption is empty
    otherwise return the caption text. Used for figure alt-text and
    italicised display-line; does not modify the displayed prose."""
    if not text:
        return default
    return text.strip()


def _alt_text_safe(text: str) -> str:
    """Markdown alt-text cannot contain unescaped ] characters or
    newlines. Collapse and escape."""
    s = re.sub(r"\s+", " ", text).strip()
    return s.replace("[", "(").replace("]", ")")


def equation_block_to_md(block: _MBlock, assets_rel: str,
                         asset_prefix: str = "") -> str:
    """Emit display-mode LaTeX + image link (when an image_path is
    present, the rendered glyph image acts as a fidelity check)."""
    body = ""
    if block.latex:
        latex = block.latex.strip()
        # Some MinerU LaTeX uses \tag{N} which KaTeX accepts but pandoc
        # math-mode rejects. Pass through unchanged; downstream renderer
        # decides.
        body = f"$$\n{latex}\n$$"
    if block.image_path:
        rel = f"{assets_rel}/{asset_prefix}{block.image_path}"
        if body:
            return f"{body}\n\n![equation]({rel})"
        return f"![equation]({rel})"
    return body


def image_block_to_md(block: _MBlock, kind: str, fig_idx: int,
                      assets_rel: str, asset_prefix: str = "") -> str:
    """Emit a paired figure/chart: alt-text contains the caption so RAG
    over figures finds the image. The caption also appears as italic
    body text below for normal-reader rendering.

    For multi-panel figure groups (block.subpanel_paths populated by
    layout_mineru.rescue_subpanel_groups), emit one image-link per
    panel (primary's image_path first, then subpanel_paths in
    reading order) and ONE shared italic caption line below.  Panel
    letters in alt-text come from layout_mineru._subpanel_letter_assignments,
    which preserves block.panel_letter (the letter extracted from the
    primary's caption) when present.
    """
    caption = block.text or f"{kind.title()} {fig_idx}"

    # Multi-panel group rendering.
    subpanel_paths = list(getattr(block, "subpanel_paths", None) or [])
    if subpanel_paths and block.image_path:
        from layout_mineru import _subpanel_letter_assignments
        n_panels = 1 + len(subpanel_paths)
        letters = _subpanel_letter_assignments(
            getattr(block, "panel_letter", None), n_panels)
        # Prefer the caption's actual figure number ('Figure 6') over
        # the emission-order fig_idx (which can drift when MinerU skips
        # figures or classifies some as 'chart'). figure_number is
        # populated by rescue_subpanel_groups on the primary block.
        caption_fig_num = getattr(block, "figure_number", None)
        display_num = caption_fig_num if caption_fig_num is not None else fig_idx
        # primary's image_path is rendered first; the primary's panel
        # label is letters[0] (which will be block.panel_letter if set).
        all_paths = [block.image_path] + subpanel_paths
        out_lines: list[str] = []
        for letter, path in zip(letters, all_paths):
            rel = f"{assets_rel}/{asset_prefix}{path}"
            alt = f"{kind.title()} {display_num} panel {letter}"
            out_lines.append(f"![{alt}]({rel})")
        return "\n\n".join(out_lines) + f"\n\n*{caption}*"

    # Single-panel (existing behavior).
    alt = _alt_text_safe(caption)
    if block.image_path:
        rel = f"{assets_rel}/{asset_prefix}{block.image_path}"
        return f"![{alt}]({rel})\n\n*{caption}*"
    # Image-less chart/image — still emit caption.
    return f"*{caption}*"


def _vlm_table_rewrite(image_path: Path, caption: Optional[str],
                       pil_image: "Optional[object]" = None
                       ) -> Optional[str]:
    """Crop-image VLM rewrite using paper2md's existing TABLE_PROMPT.
    Returns markdown table on success, None on failure or empty result.

    By default reads the table image from disk (MinerU has already
    cropped it as JPG). When `pil_image` is provided, uses that PIL
    Image directly -- bypasses the JPG-on-disk round trip. This is
    the test variant for the hybrid-splice path that renders fresh
    from the PDF via paper2md's render_crop(): the hypothesis is
    that MinerU's intermediate JPG carries compression artifacts
    that the vLLM vision encoder chokes on, while a PyMuPDF-direct
    render produces cleaner bytes the VLM handles correctly. See
    docs/dev/HYBRID_IMPLEMENTATION.md fudge #9 for evidence.
    """
    if pil_image is not None:
        img = pil_image
    else:
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            log.warning("vlm_table_rewrite: failed to load image %s: %s",
                        image_path, e)
            return None
    prompt = p2m.TABLE_PROMPT
    if caption:
        prompt = (
            "The table caption is: '" + caption.strip() + "'. " + prompt
        )
    result = p2m.vlm(prompt, img, max_tokens=6000, max_retries=2)
    if not result:
        return None
    md = result.strip()
    # Strip any preamble before the first pipe-table line.
    pipe_start = md.find("\n|")
    if pipe_start > 0:
        # If the preceding chunk doesn't contain a pipe row already
        # (e.g. the model emitted "Here is the table:\n|..."), strip it.
        if not md[:pipe_start].lstrip().startswith("|"):
            md = md[pipe_start + 1:]
    return md.strip() or None


def _figure_filename_for_panel(figure_id: str, letter: Optional[str],
                               page_idx: Optional[int],
                               ext: str) -> str:
    """Build the semantic filename for one figure panel.

    Patterns (parallels the table convention `table_{id}_p{page}_{idx}`):
      - Single panel: `figure_{id}_p{page}{ext}` (e.g. `figure_1_p3.jpg`)
      - Multi panel: `figure_{id}{letter}_p{page}{ext}` (e.g.
        `figure_1a_p3.jpg`, `figure_1b_p3.jpg`)
      - Unknown page: `figure_{id}{letter?}{ext}`

    The `id` is sanitized (`.` -> `_`) the same way as table ids.
    Extension is preserved (caller passes `.jpg` / `.jpeg` / `.png`).
    """
    safe_id = _sanitize_table_id_for_filename(figure_id) or figure_id
    suffix = letter if letter else ""
    if page_idx is not None:
        page_str = f"_p{page_idx + 1}"
    else:
        page_str = ""
    return f"figure_{safe_id}{suffix}{page_str}{ext}"


def _rename_figure_image_file(src_basename: str, dst_basename: str,
                              assets_abs: Path,
                              asset_prefix: str) -> Optional[str]:
    """Rename one figure image file on disk and return the new
    basename (without prefix). Idempotent: if the source is missing,
    the target already exists, or src == dst, the rename is a no-op
    and the appropriate basename is returned. Returns None when no
    usable name could be settled (file truly missing both ways).
    """
    if src_basename == dst_basename:
        return dst_basename
    src = assets_abs / f"{asset_prefix}{src_basename}"
    dst = assets_abs / f"{asset_prefix}{dst_basename}"
    if dst.exists() and not src.exists():
        return dst_basename  # already renamed in a prior run
    if not src.exists():
        return None  # nothing to rename; caller falls back to src
    if dst.exists():
        return dst_basename  # both exist (re-run without --clean);
                              # leave dst and caller drops the src
    try:
        src.rename(dst)
        return dst_basename
    except OSError as e:
        log.warning("Could not rename figure image %s -> %s: %s",
                    src, dst, e)
        return src_basename


def _rename_mineru_figure_image(block, assets_abs: Path,
                                asset_prefix: str) -> None:
    """Rename MinerU's content-hash figure JPG (and any multi-panel
    subpanel files) to the semantic
    `{prefix}figure_{id}{letter?}_p{page}{ext}` convention. Mutates
    `block.image_path` and `block.subpanel_paths` so downstream emit
    paths (image_block_to_md, paper2md._render_image_only) naturally
    pick up the new names.

    Only fires when the figure id can be derived (via
    paper2md._hybrid_fig_id_from_block). Unknown-id blocks keep
    their MinerU hash names; the emit path links them as-is.

    Multi-panel: letters are assigned via the same alphabetic-
    visual-ordering rule used at emit time. Primary panel + each
    subpanel get the matching letter.

    Idempotent: re-running on already-renamed files is a no-op.
    """
    if not block.image_path:
        return
    figure_id = p2m._hybrid_fig_id_from_block(block)
    if figure_id is None:
        return  # keep hash names; emit path links them as-is
    page_idx = getattr(block, "page_idx", None)
    primary_path = block.image_path
    primary_ext = Path(primary_path).suffix or ".jpg"
    subpanel_paths = list(getattr(block, "subpanel_paths", None) or [])

    if subpanel_paths:
        from layout_mineru import _subpanel_letter_assignments
        n_panels = 1 + len(subpanel_paths)
        letters = _subpanel_letter_assignments(
            getattr(block, "panel_letter", None), n_panels)
        all_paths = [primary_path] + subpanel_paths
        new_paths: list[str] = []
        for letter, path in zip(letters, all_paths):
            ext = Path(path).suffix or ".jpg"
            new_basename = _figure_filename_for_panel(
                figure_id, letter, page_idx, ext)
            resolved = _rename_figure_image_file(
                path, new_basename, assets_abs, asset_prefix)
            new_paths.append(resolved if resolved else path)
        block.image_path = new_paths[0]
        block.subpanel_paths = new_paths[1:]
    else:
        new_basename = _figure_filename_for_panel(
            figure_id, None, page_idx, primary_ext)
        resolved = _rename_figure_image_file(
            primary_path, new_basename, assets_abs, asset_prefix)
        if resolved:
            block.image_path = resolved


def _rename_mineru_table_image(block, table_idx: int,
                               assets_abs: Path, asset_prefix: str,
                               table_id: Optional[str]) -> None:
    """Rename MinerU's content-hash table JPG to the semantic
    `{prefix}table_{id}_p{page}_{idx}.{ext}` (or the id-less form
    `{prefix}table_p{page}_{idx}.{ext}` when the caption parse
    didn't yield an id). Mutates `block.image_path` so the inline
    image link and TableScore both reference the new name.

    Mirrors `paper2md._rename_table_asset_semantic` but lives here
    so the mineru-only emit_markdown path can call it without
    importing back into paper2md. Idempotent: re-running on an
    already-renamed file is a no-op."""
    if not block.image_path:
        return
    if getattr(block, "page_idx", None) is None:
        return
    suffix = Path(block.image_path).suffix or ".jpg"
    safe_id = _sanitize_table_id_for_filename(table_id)
    page_str = f"p{block.page_idx + 1}_{table_idx}"
    if safe_id is not None:
        new_basename = f"table_{safe_id}_{page_str}{suffix}"
    else:
        new_basename = f"table_{page_str}{suffix}"
    src = assets_abs / f"{asset_prefix}{block.image_path}"
    dst = assets_abs / f"{asset_prefix}{new_basename}"
    if src == dst:
        return
    if dst.exists() and not src.exists():
        block.image_path = new_basename
        return
    if not src.exists():
        return
    if dst.exists():
        block.image_path = new_basename
        return
    try:
        src.rename(dst)
        block.image_path = new_basename
    except OSError as e:
        log.warning("Could not rename table image %s -> %s: %s",
                    src, dst, e)


def _sanitize_table_id_for_filename(table_id: Optional[str]) -> Optional[str]:
    """Make a paper-relative table id safe to embed in a filename stem.
    Replaces `.` (would collide with the extension parser on ids like
    "A.4") with `_`. Other chars (alphanumeric, Roman, S-prefix) are
    already filesystem-safe. Returns None when input is None / empty
    so callers can fall through to the legacy positional-only name."""
    if table_id is None:
        return None
    s = table_id.strip()
    if not s:
        return None
    return s.replace(".", "_")


def _hybrid_table_sidecar_name(page_idx: Optional[int], table_idx: int,
                               asset_prefix: str,
                               table_id: Optional[str]) -> str:
    """Build the sidecar `.md` filename. When `table_id` is known, it's
    prepended to the page+idx to make the file's paper-relative
    identity obvious (the paired JPG follows the same convention via
    `_rename_table_asset_semantic`). Falls back to the positional-only
    naming for marker layout where the id isn't extracted."""
    safe_id = _sanitize_table_id_for_filename(table_id)
    if page_idx is not None:
        page_str = f"p{page_idx + 1}_{table_idx}"
    else:
        page_str = f"{table_idx}"
    if safe_id is not None:
        return f"{asset_prefix}table_{safe_id}_{page_str}.md"
    return f"{asset_prefix}table_{page_str}.md"


def _write_hybrid_table_sidecar(body: str, page_idx: Optional[int],
                                table_idx: int, assets_abs: Path,
                                assets_rel: str, asset_prefix: str,
                                table_id: Optional[str] = None,
                                ) -> tuple[str, str]:
    """Write the table body (pipe-md or VLM-rewritten) to a sidecar `.md`
    in `assets/`, mirroring the marker-layout convention. Returns
    `(sidecar_filename, link_suffix)` so the caller can record the
    filename on `TableScore.sidecar` and append the link below the
    inline table.

    `table_idx` is the positional counter (1, 2, 3...) used for the
    sidecar FILENAME (filesystem-stable across runs). `table_id` is the
    actual table number from the paper ("1", "A.4") used in the link
    TEXT so a reader sees the right number. When `table_id` is not
    provided (e.g., legacy callers in marker layout), the link text
    falls back to `table_idx` -- preserves the prior behavior.

    Naming:
      - `{prefix}table_{id}_p{page}_{idx}.md` when both id AND page
        are known (typical hybrid / mineru-only path; e.g.
        `table_A_4_p11_2.md` for Table A.4). The id has `.` replaced
        with `_` for filesystem safety (so the stem-vs-extension
        parser doesn't choke on `A.4.md`).
      - `{prefix}table_p{page}_{idx}.md` when id is unknown but page
        is (marker layout, or mineru blocks where caption parsing
        failed).
      - `{prefix}table_{idx}.md` when neither id nor page is known.
    """
    sidecar_name = _hybrid_table_sidecar_name(
        page_idx, table_idx, asset_prefix, table_id)
    assets_abs.mkdir(parents=True, exist_ok=True)
    (assets_abs / sidecar_name).write_text(body)
    log.info("Wrote table sidecar: %s", sidecar_name)
    link_label = table_id if table_id is not None else table_idx
    suffix = (f"\n[Table {link_label} — separate markdown]"
              f"({assets_rel}/{sidecar_name})\n")
    return sidecar_name, suffix


def table_block_to_md(block: _MBlock, table_idx: int, assets_rel: str,
                      vlm_tables: bool, assets_abs: Path,
                      report: p2m.QualityReport,
                      asset_prefix: str = "",
                      pil_image_override: "Optional[object]" = None,
                      table_id: "Optional[str]" = None,
                      inline_in_body: bool = True,
                      vlm_force: bool = False,
                      ) -> str:
    """Decide between pipe-MD, VLM-rewritten pipe-MD, or raw HTML based
    on the gating logic in the plan. Always emits the table image link
    afterwards so a downstream reader can verify cell-by-cell fidelity.
    On any path that produces a real markdown body (clean pipe-md or
    VLM rewrite), also write a per-table `.md` sidecar in `assets/`
    matching the marker-layout convention (`{prefix}table_p{page}_{idx}.md`).
    The HTML-fallback path skips the sidecar -- HTML in a `.md` file is
    not useful to downstream readers.

    `pil_image_override` (hybrid layout): when provided, used directly
    by the VLM rewrite path instead of loading MinerU's saved JPG from
    disk. Lets the hybrid splice pass a fresh PyMuPDF-rendered crop so
    the VLM sees clean PDF bytes rather than JPG-compressed pixels.

    `table_id` (hybrid layout): the actual table number from the paper
    (e.g. "1", "A.4"). When provided, used in the sidecar link text so
    a reader sees the correct id. Defaults to `table_idx` for backward
    compatibility (marker layout's process_tables passes only the
    positional counter).

    `inline_in_body` controls whether the inline pipe-md / VLM-rewrite
    / HTML fallback is emitted in the returned body string. Default
    True (marker layout's standalone path needs inline content because
    no other table representation exists). The hybrid splice passes
    False -- it preserves marker's own pipe-md in the body, and the
    only table representation it needs in the body is the image +
    sidecar link. The sidecar `.md` is still written either way; the
    flag only changes what goes into the body. When False AND the
    sidecar path fired (clean pipe-md or VLM rewrite), the returned
    string is just `<image_md><footnote_md><sidecar_suffix>` with no
    inline table body or `**caption**` header. When False AND the
    sidecar path did NOT fire (no-html or HTML fallback), we still
    skip the inline emission -- those paths produce no usable sidecar
    so the splice gets nothing in the body, but the audit still won't
    flag it as missing because marker's caption + inline table remain.

    `vlm_force` (only effective when `vlm_tables=True`): force the VLM
    rewrite even on tables whose `html_to_pipe_md` conversion was
    clean. The VLM output goes to the sidecar `.md` for highest
    fidelity; the BODY keeps the clean pipe-md (cheap conversion)
    rather than swapping in the VLM rewrite. This way `--vlm-tables-
    force` adds high-fidelity sidecars without changing what readers
    see inline. When the forced VLM call returns empty or errors,
    the sidecar falls back to the clean pipe-md so the user still
    gets a sidecar (post_redo_reason='vlm-empty-fallback-to-pipe-md').
    """
    caption = block.text or f"Table {table_idx}"
    image_rel = (f"{assets_rel}/{asset_prefix}{block.image_path}"
                 if block.image_path else None)
    image_md = (f"\n\n![{_alt_text_safe(caption)}]({image_rel})"
                if image_rel else "")
    # Paper-relative table id ("1", "S1", "I", "A.4") for the
    # TableScore.matched_table field. Prefer the explicit `table_id`
    # passed by the hybrid splice (already canonicalized at call site);
    # fall back to extracting from this block's caption text for the
    # mineru-only path that doesn't pass a table_id.
    matched_table = table_id
    if matched_table is None and block.text:
        matched_table = p2m._extract_table_number(block.text)
    # Use the resolved matched_table for the sidecar filename + link
    # text too, not just the TableScore field. Without this, the
    # mineru-only path (which doesn't pass `table_id`) writes
    # `table_p9_2.md` even though we know the id is, e.g., "3".
    # The body link text picks up the id, AND the filename does too.
    if table_id is None and matched_table is not None:
        table_id = matched_table

    # Footnote block (rendered AFTER the table+image). Prefixed with
    # "Notes:" so the under-table footnote text is visually labeled
    # in the rendered markdown rather than floating in italics with
    # no context. Strip a leading "Notes:" / "Note:" if MinerU's text
    # already includes one to avoid doubling.
    fn = (block.footnote or "").lstrip()
    if fn:
        m = re.match(r"^Notes?\s*:\s*", fn, flags=re.IGNORECASE)
        if m:
            fn = fn[m.end():]
        footnote_md = f"\n\n*Notes: {fn}*"
    else:
        footnote_md = ""

    if not block.html:
        report.tables.append(p2m.TableScore(
            index=table_idx, page=block.page_idx + 1, located=False,
            jpeg_saved=bool(block.image_path),
            pre_redo_reason="no-html-in-middle.json",
            vlm_redone=False, post_redo_reason=None, score=0.0,
            matched_table=matched_table,
        ))
        if not inline_in_body:
            # Hybrid: no usable sidecar to link (no HTML to convert)
            # and no inline body desired -- emit just the image.
            return f"{image_md}{footnote_md}\n"
        return f"\n**{caption}**{image_md}{footnote_md}\n"

    pipe_md, reason = html_to_pipe_md(block.html)

    if pipe_md is not None and p2m.table_is_suspicious(pipe_md) is None:
        # Clean conversion. Two sub-paths depending on --vlm-tables-force.

        if vlm_force and vlm_tables:
            # FORCED-VLM-ON-CLEAN-TABLE path: body keeps the cheap
            # pipe-md; the VLM rewrite goes to the sidecar for higher
            # fidelity (sub/super, Greek, units, math). When the VLM
            # call fails, sidecar falls back to the clean pipe-md so
            # the user still gets a sidecar (`post_redo_reason=
            # 'vlm-empty-fallback-to-pipe-md'`).
            rewritten = None
            img_path = (assets_abs / (asset_prefix + block.image_path)
                        if block.image_path else None)
            have_input = pil_image_override is not None or (
                img_path is not None and img_path.exists())
            if have_input:
                source = ("PIL override (PDF-render)"
                          if pil_image_override is not None
                          else str(block.image_path))
                log.info("Table %d: --vlm-tables-force on a clean "
                         "table; invoking VLM rewrite for sidecar "
                         "from %s (body keeps the clean pipe-md)",
                         table_idx, source)
                rewritten = _vlm_table_rewrite(
                    img_path, block.text, pil_image=pil_image_override)
                if rewritten and p2m.table_is_suspicious(rewritten) is not None:
                    # VLM produced something but it's also suspicious
                    # -- discard. Sidecar will fall back to pipe-md.
                    rewritten = None
            sidecar_body = rewritten if rewritten else pipe_md
            sidecar_name, sidecar_suffix = _write_hybrid_table_sidecar(
                sidecar_body, block.page_idx, table_idx,
                assets_abs, assets_rel, asset_prefix,
                table_id=table_id)
            post_redo_reason = (None if rewritten
                                else "vlm-empty-fallback-to-pipe-md")
            report.tables.append(p2m.TableScore(
                index=table_idx, page=block.page_idx + 1, located=True,
                jpeg_saved=bool(block.image_path),
                pre_redo_reason="force",
                vlm_redone=rewritten is not None,
                post_redo_reason=post_redo_reason,
                score=1.0,  # body is still clean pipe-md
                sidecar=sidecar_name,
                matched_table=matched_table,
            ))
        else:
            # Standard clean path: no VLM call, sidecar = pipe-md.
            sidecar_name, sidecar_suffix = _write_hybrid_table_sidecar(
                pipe_md, block.page_idx, table_idx,
                assets_abs, assets_rel, asset_prefix,
                table_id=table_id)
            report.tables.append(p2m.TableScore(
                index=table_idx, page=block.page_idx + 1, located=True,
                jpeg_saved=bool(block.image_path),
                pre_redo_reason=None, vlm_redone=False,
                post_redo_reason=None, score=1.0,
                sidecar=sidecar_name,
                matched_table=matched_table,
            ))

        if not inline_in_body:
            # Hybrid: marker's pipe-md stays in body verbatim; we only
            # add the image + sidecar link as additive, non-corrupting
            # additions next to marker's caption.
            return f"{image_md}{footnote_md}{sidecar_suffix}\n"
        return (f"\n**{caption}**\n\n{pipe_md}{image_md}{footnote_md}"
                f"{sidecar_suffix}\n")

    pre_reason = reason or p2m.table_is_suspicious(pipe_md) or "suspicious"

    # Suspicious or not convertible — try VLM rewrite when enabled.
    if vlm_tables and (block.image_path or pil_image_override is not None):
        img_path = (assets_abs / (asset_prefix + block.image_path)
                    if block.image_path else None)
        # Fire the VLM call if either:
        #   - a pil_image_override is provided (hybrid PDF-render path), OR
        #   - the JPG fallback exists on disk.
        have_input = pil_image_override is not None or (
            img_path is not None and img_path.exists())
        if have_input:
            source = ("PIL override (PDF-render)"
                      if pil_image_override is not None
                      else str(block.image_path))
            log.info("Table %d: pipe conversion failed (%s); invoking "
                     "paper2md VLM rewrite from %s", table_idx, pre_reason,
                     source)
            rewritten = _vlm_table_rewrite(
                img_path, block.text, pil_image=pil_image_override)
            if rewritten and p2m.table_is_suspicious(rewritten) is None:
                sidecar_name, sidecar_suffix = _write_hybrid_table_sidecar(
                    rewritten, block.page_idx, table_idx,
                    assets_abs, assets_rel, asset_prefix,
                    table_id=table_id)
                report.tables.append(p2m.TableScore(
                    index=table_idx, page=block.page_idx + 1, located=True,
                    jpeg_saved=True, pre_redo_reason=pre_reason,
                    vlm_redone=True, post_redo_reason=None, score=0.85,
                    sidecar=sidecar_name,
                    matched_table=matched_table,
                ))
                if not inline_in_body:
                    return (f"{image_md}{footnote_md}{sidecar_suffix}\n")
                return (f"\n**{caption}**\n\n{rewritten}{image_md}"
                        f"{footnote_md}{sidecar_suffix}\n")
            post_reason = (
                p2m.table_is_suspicious(rewritten)
                if rewritten else "vlm-empty-or-failed"
            )
        else:
            post_reason = "table-image-missing-on-disk"
            log.warning("Table %d: image path %s not found; skipping VLM",
                        table_idx, img_path)
    else:
        post_reason = ("vlm-disabled" if not vlm_tables
                       else "no-image-path-for-vlm")

    # Fallback: keep raw HTML + image. Translate <eq>...</eq> (MinerU's
    # tag for cell-level math) to inline LaTeX `$...$` so equations
    # render in markdown viewers even though the surrounding table
    # stays as HTML. This is purely a syntax bridge: <eq> isn't a
    # standard HTML tag and isn't rendered by markdown processors;
    # `$...$` is the conventional inline-math delimiter.
    fallback_html = _EQ_TAG_RE.sub(r"$\1$", block.html or "")
    report.tables.append(p2m.TableScore(
        index=table_idx, page=block.page_idx + 1, located=True,
        jpeg_saved=bool(block.image_path),
        pre_redo_reason=pre_reason, vlm_redone=bool(vlm_tables),
        post_redo_reason=post_reason, score=0.45,
        matched_table=matched_table,
    ))
    if not inline_in_body:
        # Hybrid HTML-fallback path: no good sidecar (raw HTML isn't
        # useful as a .md sidecar). Emit just the image so the visual
        # reference is preserved; marker's caption + pipe-md stay in
        # the body untouched. No sidecar link to include either.
        return f"{image_md}{footnote_md}\n"
    return f"\n**{caption}**\n\n{fallback_html}{image_md}{footnote_md}\n"


# === document assembly =====================================================


_HEADING_PREFIX = {1: "# ", 2: "## ", 3: "### ", 4: "#### ", 5: "##### ", 6: "###### "}


def emit_markdown(doc: ParsedDoc, assets_rel: str, assets_abs: Path,
                  vlm_tables: bool, report: p2m.QualityReport,
                  strip_chart_details: bool = False,
                  asset_prefix: str = "",
                  vlm_force: bool = False,
                  table_workers: int = 1) -> str:
    """Walk parsed blocks and produce body markdown.

    `assets_rel` is the path used in markdown links (e.g. `assets`).
    `assets_abs` is the on-disk path used for VLM image reads.
    `report` is mutated: per-table TableScore entries are appended.

    `strip_chart_details` removes MinerU hybrid-mode `<details><summary>line</summary>...</details>`
    blocks from text content. Off by default (preserve and let consumer decide).

    `asset_prefix` is prepended to every image filename in markdown
    links and to the on-disk lookup for the VLM table rewrite path
    (e.g. 'si_' for supplement bundling).

    `table_workers > 1`: dispatch the per-table `table_block_to_md`
    calls concurrently via a ThreadPoolExecutor. The VLM call inside
    each task is the only network-bound step; vLLM batches concurrent
    requests server-side, so workers > 1 is the recommended setting
    on vLLM. LM Studio users may want workers=1 (single-threaded).
    Mirrors the hybrid splice's parallelization.
    """
    out: list[str] = []
    # Tables get a placeholder slot in `out` and a deferred task list;
    # everything else is computed synchronously in this loop. After
    # the loop, table tasks run (concurrently when table_workers>1)
    # and fill their slots.
    table_tasks: list[tuple] = []  # (slot_idx, table_idx, block)
    fig_idx = 0
    chart_idx = 0
    table_idx = 0

    for b in doc.blocks:
        t = b.type
        # Adopted-into-parent blocks (captions/footnotes lifted by
        # rescue_orphan_captions) are skipped — their content already
        # lives in the parent figure/table block's text/footnote slot.
        if t.startswith("_adopted"):
            continue
        if t == "title":
            level = b.text_level if b.text_level and 1 <= b.text_level <= 6 else 2
            prefix = _HEADING_PREFIX[level]
            if b.text:
                out.append(f"{prefix}{b.text.strip()}")
        elif t == "abstract":
            if b.text:
                out.append("**Abstract.** " + b.text.strip())
        elif t in ("text", "ref_text"):
            if b.text:
                txt = b.text.strip()
                if strip_chart_details:
                    txt = _strip_chart_details_blocks(txt)
                out.append(txt)
        elif t == "interline_equation":
            md = equation_block_to_md(b, assets_rel, asset_prefix)
            if md:
                out.append(md)
        elif t == "image":
            fig_idx += 1
            # Rename MinerU's content-hash JPG (and any multi-panel
            # subpanel files) to the semantic
            # `figure_{id}{letter?}_p{page}{ext}` convention BEFORE
            # image_block_to_md reads block.image_path. Mutates block
            # in place; unknown-id blocks keep their hash names.
            _rename_mineru_figure_image(b, assets_abs, asset_prefix)
            out.append(image_block_to_md(b, "figure", fig_idx, assets_rel,
                                         asset_prefix))
            if b.image_path:
                # Prefer the caption-derived figure id when known so
                # the FigureScore.matched_figure reflects the paper-
                # relative id (e.g. "1", "S1", "12"), not the
                # emission-order positional idx.
                fid = p2m._hybrid_fig_id_from_block(b) or str(fig_idx)
                report.figures.append(p2m.FigureScore(
                    filename=asset_prefix + b.image_path,
                    caption_produced=bool(b.text),
                    caption_length=len(b.text or ""),
                    score=1.0 if b.text else 0.6,
                    matched_figure=fid,
                ))
        elif t == "chart":
            chart_idx += 1
            _rename_mineru_figure_image(b, assets_abs, asset_prefix)
            out.append(image_block_to_md(b, "figure", chart_idx, assets_rel,
                                         asset_prefix))
            if b.image_path:
                fid = p2m._hybrid_fig_id_from_block(b) or str(chart_idx)
                report.figures.append(p2m.FigureScore(
                    filename=asset_prefix + b.image_path,
                    caption_produced=bool(b.text),
                    caption_length=len(b.text or ""),
                    score=1.0 if b.text else 0.6,
                    matched_figure=fid,
                ))
        elif t == "table":
            table_idx += 1
            # Reserve a slot; the table-VLM call may take 5-180s, so
            # defer it to a parallel pass after the cheap blocks are
            # all emitted.
            out.append(None)
            table_tasks.append((len(out) - 1, table_idx, b))
        else:
            if b.text:
                out.append(b.text.strip())

    # Dispatch tables: parallel via ThreadPoolExecutor when
    # table_workers>1 and there's more than one table to run.
    def _run_one_table(args):
        slot_idx, t_idx, blk = args
        # Rename MinerU's content-hash JPG to the semantic
        # `table_{id}_p{page}_{idx}.jpg` (or the id-less form when
        # the caption parse fails). This is the mineru-only-layout
        # equivalent of `_rename_table_asset_semantic` in
        # paper2md.py's hybrid path; without it, the JPG keeps its
        # hash name while the sidecar .md uses the semantic name.
        derived_id = (p2m._extract_table_number(blk.text)
                      if blk.text else None)
        _rename_mineru_table_image(blk, t_idx, assets_abs,
                                   asset_prefix, derived_id)
        return slot_idx, table_block_to_md(
            blk, t_idx, assets_rel, vlm_tables, assets_abs, report,
            asset_prefix, vlm_force=vlm_force,
        )

    if table_workers > 1 and len(table_tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        log.info("emit_markdown: dispatching %d table VLM call(s) "
                 "concurrently (workers=%d)",
                 len(table_tasks), table_workers)
        with ThreadPoolExecutor(max_workers=table_workers) as pool:
            results = list(pool.map(_run_one_table, table_tasks))
    else:
        results = [_run_one_table(t) for t in table_tasks]

    for slot_idx, content in results:
        out[slot_idx] = content

    # report.tables may have been appended out-of-order by concurrent
    # workers. Sort by table index to restore deterministic order
    # (mirrors the hybrid splice's post-sort).
    if table_workers > 1 and len(table_tasks) > 1 and report.tables:
        report.tables.sort(key=lambda t: getattr(t, "index", 0))

    return "\n\n".join(p for p in out if p)


_CHART_DETAILS_RE = re.compile(
    r"<details>\s*<summary>line</summary>.*?</details>",
    re.DOTALL,
)


def _strip_chart_details_blocks(md: str) -> str:
    """Remove MinerU hybrid-mode <details><summary>line</summary>...</details>
    chart-data reconstruction blocks from the body. The numbers inside
    are VLM-estimated (not in the source PDF) and should not be quoted
    as facts."""
    return _CHART_DETAILS_RE.sub("", md)


# === Generalised references-heading rescue ================================
#
# Many journals emit a trailing reference list with no `## References`
# heading: section-chunked embedding pipelines need the heading to locate
# refs. We detect four common styles and insert a heading conservatively.
#
#   numbered  : `1. Author...`, `2) Author...`
#   bracketed : `[1] Author...`, `[2] Author...`
#   footnote  : `1A. Author...` (Knudson-style no-space digit-then-letter)
#   ay (a-y)  : `Smith (2020)`, `Smith, J. and Doe, K. (2020). ...`
#
# Trigger gates per style require:
#   * No existing `## References`/Bibliography/Notes heading anywhere
#     (the rescue is heading-injection only).
#   * A "tail run" of >=N contiguous matches in the last 30% of the
#     document.
#   * Numbered/bracketed/footnote: the run must include at least one
#     anchor with number 1, 2, or 3 (refs lists begin at 1/2/3).
#   * Author-year: stricter -- requires >=10 matches in a contiguous
#     block (separated by <300 chars), in the last 25% of the doc, to
#     suppress false positives on inline citations.
#
# The footnote style additionally REFORMATS each ref `1A. X` -> `- 1. A. X`
# so it renders as a bulleted list. The other styles keep their format.

_REFS_HEADING_RE = re.compile(
    r"(?im)^#{1,3}\s+(?:references|bibliography|notes|literature\s+cited|"
    r"references\s+and\s+notes)\b"
)

_NUMBERED_REF_PAT = re.compile(r"(?m)^[ \t]*(\d{1,3})[\.\)]\s+\S")
_BRACKETED_REF_PAT = re.compile(r"(?m)^[ \t]*\[(\d{1,3})\]\s+\S")
_FOOTNOTE_REF_PAT = re.compile(r"(?m)^(\d{1,3})([A-Z]\.\s)")
_AUTHOR_YEAR_REF_PAT = re.compile(
    r"(?m)^[A-Z][A-Za-z\-\.,& ]{1,120}"
    r"(?:\([12]\d{3}[a-z]?\)|,\s+[12]\d{3}[a-z]?[\.\,])"
)


def _detect_run(matches: list, max_gap_chars: int = 250) -> Optional[tuple[int, int]]:
    """Given a list of regex match objects, find the LAST contiguous run
    where adjacent matches start within `max_gap_chars` of each other.
    Returns (start_offset, count) or None if no run >= 5."""
    if not matches:
        return None
    runs: list[tuple[int, int]] = []
    cur_start = matches[0].start()
    cur_count = 1
    cur_last_end = matches[0].end()
    for m in matches[1:]:
        if m.start() - cur_last_end <= max_gap_chars:
            cur_count += 1
            cur_last_end = m.end()
        else:
            runs.append((cur_start, cur_count, cur_last_end))
            cur_start = m.start()
            cur_count = 1
            cur_last_end = m.end()
    runs.append((cur_start, cur_count, cur_last_end))
    if not runs:
        return None
    last = runs[-1]
    if last[1] < 5:
        return None
    return (last[0], last[1])


def _detect_numbered_or_bracketed(md: str, pat: re.Pattern,
                                  *, label: str) -> Optional[int]:
    """Return inject offset for `## References` if a trailing numbered/
    bracketed ref run is detected; else None."""
    cut = int(len(md) * 0.7)
    tail_matches = list(pat.finditer(md, cut))
    if len(tail_matches) < 5:
        return None
    # Anchor check: at least one match in the tail with N in {1,2,3}.
    anchored = [m for m in tail_matches if int(m.group(1)) in (1, 2, 3)]
    if not anchored:
        return None
    anchor = anchored[0]  # earliest low-N match in the tail
    log.info("rescue_missing_refs_heading[%s]: anchor at offset %d "
             "(N=%s, %d tail matches)", label, anchor.start(),
             anchor.group(1), len(tail_matches))
    return anchor.start()


def _detect_author_year(md: str) -> Optional[int]:
    """Author-year tail run detector. Strict: >=10 contiguous matches in
    last 25% of doc."""
    cut = int(len(md) * 0.75)
    tail_matches = list(_AUTHOR_YEAR_REF_PAT.finditer(md, cut))
    run = _detect_run(tail_matches, max_gap_chars=400)
    if run is None or run[1] < 10:
        return None
    log.info("rescue_missing_refs_heading[author-year]: %d-entry run "
             "starting at offset %d", run[1], run[0])
    return run[0]


def _detect_footnote(md: str) -> Optional[int]:
    """Knudson-style no-space footnote (1A.) tail-run detector. Returns
    inject offset; the caller is responsible for reformatting matches."""
    cut = int(len(md) * 0.7)
    tail_matches = list(_FOOTNOTE_REF_PAT.finditer(md, cut))
    if len(tail_matches) < 5:
        return None
    midpoint = len(md) // 2
    matches_after_mid = list(_FOOTNOTE_REF_PAT.finditer(md, midpoint))
    if not matches_after_mid:
        return None
    min_n = min(int(m.group(1)) for m in matches_after_mid)
    if min_n > 3:
        return None
    anchor = next(m for m in matches_after_mid
                  if int(m.group(1)) == min_n)
    log.info("rescue_missing_refs_heading[footnote]: anchor at offset %d "
             "(N=%d, %d tail matches)", anchor.start(), min_n,
             len(tail_matches))
    return anchor.start()


def rescue_missing_refs_heading(md: str) -> str:
    """If `md` lacks a References heading and has a detectable trailing
    run of refs in any of the four styles, inject `## References` at
    the head of the run.

    Detection order: footnote (Knudson-style) -> bracketed -> numbered
    -> author-year. The footnote pass also reformats the entries to
    a clean bulleted style; the other passes leave entries unchanged
    (their format already renders correctly).

    Idempotent: a doc with a heading is returned unchanged.
    """
    if not md:
        return md

    # When a heading already exists, only the footnote-style entries
    # benefit from reformatting (they don't render as a list otherwise).
    # The other styles render fine as-is and we leave them alone.
    if _REFS_HEADING_RE.search(md):
        if _FOOTNOTE_REF_PAT.search(md):
            return _FOOTNOTE_REF_PAT.sub(r"- \1. \2", md)
        return md

    # 1. Footnote (Knudson). Reformat + inject.
    pos = _detect_footnote(md)
    if pos is not None:
        head = md[:pos].rstrip()
        tail = _FOOTNOTE_REF_PAT.sub(r"- \1. \2", md[pos:])
        return head + "\n\n## References\n\n" + tail

    # 2. Bracketed.
    pos = _detect_numbered_or_bracketed(md, _BRACKETED_REF_PAT,
                                        label="bracketed")
    if pos is not None:
        return md[:pos].rstrip() + "\n\n## References\n\n" + md[pos:]

    # 3. Numbered.
    pos = _detect_numbered_or_bracketed(md, _NUMBERED_REF_PAT,
                                        label="numbered")
    if pos is not None:
        return md[:pos].rstrip() + "\n\n## References\n\n" + md[pos:]

    # 4. Author-year (strictest).
    pos = _detect_author_year(md)
    if pos is not None:
        return md[:pos].rstrip() + "\n\n## References\n\n" + md[pos:]

    return md


# Backward-compat alias for test code referring to the original name.
def format_knudson_refs(md: str) -> str:
    """Deprecated: use rescue_missing_refs_heading. Kept as a thin
    alias so external callers don't break."""
    return rescue_missing_refs_heading(md)


# === Page-footnote references rescue =======================================
#
# Old-style scientific papers (pre-2000 J. Appl. Phys., Phys. Rev., etc.)
# put numbered references in page-bottom footnotes rather than a
# consolidated bibliography. MinerU's classifier puts these under
# `discarded_blocks` with `type=page_footnote` (separate from page-
# header/footer publisher furniture). Walking those blocks recovers
# the references for an entire class of pre-2000 papers that would
# otherwise score references.score=0.0.
#
# Three discrimination layers (composed; all must fire for the rescue
# to activate):
#
#   Layer 1 (free): MinerU type label `page_footnote` already excludes
#                   journal-name / DOI / copyright furniture (those land
#                   under `footer`).
#   Layer 2:        per-block bibliographic-shape filter — must contain
#                   `(YYYY)` in parens, must not start with "*"/"**" /
#                   "doi.org" / "doi:". canup's author-affiliations
#                   page_footnote (no year) is rejected here.
#   Layer 3:        activation gate on existing references quality —
#                   only fire when references.score < 0.4 AND
#                   entry_count < 3 AND no API ref rescue has appended.

# Year anywhere in text (loose). Catches `(1959)`, `Theory of Elasticity
# (Addison-Wesley ... 1959)`, `Colloque International 1961 ...`, etc.
# A 4-digit token starting with 19xx/20xx is a strong bibliographic
# signal; very rare in non-citation page-footnote text.
_FOOTNOTE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_FOOTNOTE_LEADING_NUM_RE = re.compile(r"^(\d{1,3})\s+(.+)$", re.DOTALL)
# Splits a multi-ref block on the boundary between two consecutive
# numbered footnotes that got concatenated by MinerU (e.g.,
#   "9 M. H. Rice... (1960). 10 R. G. McQueen (private...)" ).
_FOOTNOTE_MULTI_SPLIT_RE = re.compile(
    r"(?<=\.)\s+(?=\d{1,3}\s+[A-Z]\.\s)"
)
# Use a literal ## References heading so paper2md.score_references
# counts the appended entries. Provenance is recorded in
# report.references.rescue_applied = "footnote-refs:N".
_FOOTNOTE_REFS_HEADING = "## References"


def rescue_footnote_refs(md: str,
                          footnote_blocks: list[_MBlock],
                          report: p2m.QualityReport) -> str:
    """Rescue: append a `## References (from page footnotes)` section
    when the body lacks a usable references section AND `footnote_blocks`
    contains bibliographic-shape entries.

    Triggers only when ALL of:
      - report.references is not None
      - report.references.score < 0.4
      - report.references.entry_count < 3
      - report.references.api_refs_appended == 0  (no API rescue ran)
      - >= 5 footnote blocks pass the bibliographic-shape filter
      - the rescue's own heading isn't already present (idempotent)

    Soft-fail: any failure path returns md unchanged. Records
    `report.references.rescue_applied = "footnote-refs:N"` on success.
    """
    if not footnote_blocks:
        return md
    rs = report.references
    if rs is None:
        return md
    # Layer 3: only when local refs are poor AND no API rescue fired.
    if rs.score >= 0.4 or rs.entry_count >= 3:
        return md
    if rs.api_refs_appended:
        return md
    # Idempotent: if any References / Bibliography / etc. heading
    # already exists in the body, don't append a duplicate. (Re-uses
    # the same regex rescue_missing_refs_heading uses.)
    if _REFS_HEADING_RE.search(md):
        return md

    # Layer 2: bibliographic-shape filter.
    candidates: list[str] = []
    for fb in footnote_blocks:    # already sorted by (page_idx, bbox_y)
        text = (fb.text or "").strip()
        if not text:
            continue
        if text.startswith(("*", "**")):
            continue
        lower = text.lower()
        if "doi.org" in lower or lower.startswith("doi:"):
            continue
        if not _FOOTNOTE_YEAR_RE.search(text):
            continue
        candidates.append(text)

    if len(candidates) < 5:
        return md

    # Split multi-ref blocks (MinerU sometimes concatenates two adjacent
    # footnotes into one block).
    flat_refs: list[str] = []
    for c in candidates:
        flat_refs.extend(_FOOTNOTE_MULTI_SPLIT_RE.split(c))

    # Reconstruct missing leading numbers from document order.
    out: list[str] = []
    last_n = 0
    for r in flat_refs:
        r = r.strip()
        if not r:
            continue
        m = _FOOTNOTE_LEADING_NUM_RE.match(r)
        if m:
            n = int(m.group(1))
            body_text = m.group(2)
            last_n = n
        else:
            last_n += 1
            n = last_n
            body_text = r
        out.append(f"- {n}. {body_text}")

    if len(out) < 5:
        return md

    log.info("rescue_footnote_refs: appended %d ref(s) from %d footnote "
             "block(s) (refs.score was %.2f, entries=%d)",
             len(out), len(candidates), rs.score, rs.entry_count)
    rs.rescue_applied = (
        f"footnote-refs:{len(out)}"
        if rs.rescue_applied is None
        else f"{rs.rescue_applied}+footnote-refs:{len(out)}"
    )
    rs.score_pre_rescue = rs.score
    appended = (
        "\n\n" + _FOOTNOTE_REFS_HEADING + "\n\n"
        + "\n".join(out) + "\n"
    )
    return md.rstrip() + appended


# === data-repo population (local, respects allow_network) ==================


def _populate_data_repos_local(md: str, report: p2m.QualityReport,
                               *, allow_network: bool,
                               fetch_summaries: bool) -> None:
    """Stand-in for paper2md._populate_data_repos that doesn't depend on
    the OFFLINE / FETCH_DATA_REPOS module globals. Same behavior:
    extract data-repo links via regex, optionally enrich with API
    summaries when fetch_summaries=True and allow_network=True.
    """
    try:
        from data_repos import extract_data_links, fetch_summary
    except ImportError as e:
        log.debug("data_repos module unavailable: %s", e)
        return
    try:
        links = extract_data_links(md)
    except Exception as e:
        log.warning("Data-repo extraction failed (%s); continuing", e)
        return
    if not links:
        return
    log.info("Detected %d data-repository link(s)", len(links))
    if fetch_summaries and allow_network:
        try:
            import requests
            with requests.Session() as session:
                for link in links:
                    try:
                        fetch_summary(link, session=session, timeout=8.0)
                    except Exception as e:
                        log.debug("fetch_summary failed for %s: %s",
                                  link.url, e)
        except Exception as e:
            log.warning("Data-repo enrichment failed (%s); links recorded "
                        "without summaries", e)
    elif fetch_summaries and not allow_network:
        log.info("--offline: data-repo enrichment skipped; links recorded "
                 "without summaries")
    report.data_repos = links


# === Article-boundary trim (VLM probe on first/last PDF pages) =============
#
# Some PDFs concatenate multiple articles. paper2md's marker pipeline has a
# trim_to_first_article pass; for the MinerU wrapper we operate on
# middle.json blocks (each carries page_idx) so the trim is a simple
# block filter once the boundary pages are known.
#
# Strategy:
#   1. Trailing-article: probe last page vs first page (1 VLM call). If
#      different, binary-search backward for the first DIFFERENT page.
#      Set tail_cutoff = that page index.
#   2. Leading-article: if main article is >=2 pages, probe first page
#      vs the last in-main-article page. If different, binary-search
#      forward for the first SAME-as-last page. Set head_cutoff.
#   3. Drop blocks with page_idx < head_cutoff or page_idx >= tail_cutoff.
#
# Each VLM call uses paper2md's TRIM_PROMPT (which expects "page 1 vs
# candidate page" framing). For the leading-article probe we re-orient
# the prompt by passing the last in-main-article page as the anchor.


def _vlm_compare_pages(doc, anchor_idx: int, candidate_idx: int,
                       dpi: int = 100) -> Optional[dict]:
    """Generalised page-vs-page same-article probe via paper2md's vlm().

    Returns:
        {"same": True,  "title": None}             same article
        {"same": False, "title": str | None}       different article
        None                                        VLM error
    """
    try:
        from PIL import Image
        pix1 = doc[anchor_idx].get_pixmap(dpi=dpi)
        pix2 = doc[candidate_idx].get_pixmap(dpi=dpi)
        img1 = Image.frombytes("RGB", (pix1.width, pix1.height), pix1.samples)
        img2 = Image.frombytes("RGB", (pix2.width, pix2.height), pix2.samples)
        max_w = max(img1.width, img2.width)
        if img1.width != max_w:
            img1 = img1.resize((max_w, int(img1.height * max_w / img1.width)))
        if img2.width != max_w:
            img2 = img2.resize((max_w, int(img2.height * max_w / img2.width)))
        sep_h = 8
        composite = Image.new(
            "RGB", (max_w, img1.height + sep_h + img2.height), (0, 0, 0))
        composite.paste(img1, (0, 0))
        composite.paste(img2, (0, img1.height + sep_h))

        ans = p2m.vlm(p2m.TRIM_PROMPT, composite, max_tokens=80)
        if not ans:
            return None
        lines = [ln.strip() for ln in ans.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        first = lines[0].upper().split()[0] if lines[0].split() else ""
        if "SAME" in first:
            return {"same": True, "title": None}
        if "DIFFERENT" in first:
            title = lines[1] if len(lines) > 1 else None
            if title and title.upper() == "NONE":
                title = None
            return {"same": False, "title": title}
        return None
    except Exception as e:
        log.warning("Article-trim VLM probe (%d vs %d) failed: %s",
                    anchor_idx + 1, candidate_idx + 1, e)
        return None


def _find_trailing_boundary(doc, n: int) -> tuple[int, Optional[str]]:
    """Return (cutoff, title) where cutoff is the 0-indexed page where the
    trailing different article starts and title is the new article's title
    as reported by the VLM (None if no boundary or no title returned).
    Returns (n, None) if no boundary detected (single article).

    The TRIM_PROMPT is anchor=page-0 + candidate=later-page, which is the
    orientation the prompt was written for. We do NOT search for a leading
    boundary -- the prompt's "title in BOTTOM => DIFFERENT" heuristic
    false-positives on title pages, which is what every real paper's
    page 0 looks like.
    """
    if n < 2:
        return n, None
    last = _vlm_compare_pages(doc, 0, n - 1)
    if last is None:
        return n, None
    if last["same"]:
        return n, None
    boundary_title = last["title"]
    lo, hi = 0, n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        check = _vlm_compare_pages(doc, 0, mid)
        if check is None:
            return n, None  # abort safely
        if check["same"]:
            lo = mid
        else:
            hi = mid
            if check["title"]:
                boundary_title = check["title"]
    return hi, boundary_title


def _find_title_in_blocks(title: str, blocks: list[_MBlock],
                          min_page: int) -> Optional[tuple[int, int]]:
    """Locate the boundary block: search blocks at page_idx >= min_page
    for a contiguous N-word slice of `title` (N decreasing from full
    title length down to 3). Case-insensitive, punctuation-insensitive,
    whitespace-collapsed.

    Returns (page_idx, index) of the LATEST matching block in reading
    order, or None if no match. "Latest" mirrors paper2md._find_title_-
    line_in_md: a forward reference to the new article's title can
    appear earlier (in the prior article's body); the new article's
    actual title is later in reading order.

    A None result is the abort signal for the trim: a real article
    boundary has a real title that MinerU would have emitted as a block;
    missing title => VLM hallucinated the boundary.
    """
    title_words = re.findall(r"\w+", title.lower())
    if len(title_words) < 3:
        return None
    candidates = sorted(
        (b for b in blocks if b.page_idx >= min_page),
        key=lambda b: (b.page_idx, b.index),
    )
    block_norms = [
        (b.page_idx, b.index,
         " ".join(re.findall(r"\w+", (b.text or "").lower())))
        for b in candidates
    ]
    for window in range(len(title_words), 2, -1):
        for start in range(len(title_words) - window + 1):
            phrase = " ".join(title_words[start:start + window])
            if not phrase:
                continue
            best: Optional[tuple[int, int]] = None
            for (pi, idx, btext) in block_norms:
                if phrase in btext:
                    best = (pi, idx)
            if best is not None:
                return best
    return None


def trim_articles_in_blocks(blocks: list[_MBlock], doc, n_pages: int
                            ) -> tuple[list[_MBlock], dict]:
    """Drop blocks belonging to a trailing different article.

    `doc` is a fitz.Document opened on the source PDF. `n_pages` is the
    page count.

    Trailing-only by design (matches paper2md.trim_to_first_article).
    Validates the boundary by string-matching the VLM-reported title
    against block text on the trimmed pages: if no match, the trim is
    aborted on the assumption the VLM hallucinated a boundary on a
    single-article PDF.

    Cut is BLOCK-granular, not page-granular: when the boundary block
    sits mid-page (the boundary page contains the kept article's tail
    AND the new article's head), only blocks at-or-after the boundary
    block are dropped. Block ordering is `(page_idx, index)` reading
    order. This matches paper2md, which cuts at a markdown line.

    Returns (filtered_blocks, info) where info has keys:
        head_trimmed: int (always 0; leading trim removed)
        tail_trimmed: int (count of trailing pages partially or fully
                            dropped, == n_pages - boundary_page)
        main_pages: tuple[int, int] (0, boundary_page)
        boundary_block: tuple[int, int] | None
                        (page_idx, index) of the boundary block; the
                        block itself and everything after it is dropped.
                        None when no trim occurred.
        vlm_calls: int
    """
    info = {"head_trimmed": 0, "tail_trimmed": 0,
            "main_pages": (0, n_pages), "boundary_block": None,
            "vlm_calls": 0}
    if n_pages < 2:
        return blocks, info

    tail_cutoff, boundary_title = _find_trailing_boundary(doc, n_pages)
    if tail_cutoff == n_pages:
        return blocks, info

    boundary = (_find_title_in_blocks(boundary_title, blocks, tail_cutoff)
                if boundary_title else None)
    if boundary is None:
        log.warning("Article-trim: VLM reported a trailing boundary at "
                    "page %d with title %r, but the title was not found "
                    "in any block on the trimmed pages. Likely a VLM "
                    "hallucination on a single-article PDF; NOT trimming. "
                    "See USAGE.md §5 (article-boundary trim).",
                    tail_cutoff + 1, boundary_title)
        return blocks, info

    boundary_page, boundary_index = boundary
    filtered = [b for b in blocks
                if (b.page_idx, b.index) < (boundary_page, boundary_index)]
    info.update({
        "tail_trimmed": n_pages - boundary_page,
        "main_pages": (0, boundary_page),
        "boundary_block": (boundary_page, boundary_index),
    })
    log.info("Article-trim: cut at page %d block #%d (title=%r); "
             "tail spans %d page(s)",
             boundary_page + 1, boundary_index, boundary_title,
             n_pages - boundary_page)
    return filtered, info


# === asset copy (still wrap_mineru-specific; layout module handles the
# subprocess + middle.json reading, but copy_assets is part of the
# wrap_mineru output convention) ============================================


def copy_assets(src_images: Path, dst_assets: Path,
                asset_prefix: str = "") -> int:
    """Copy MinerU's images/ directory to dst_assets. Returns count
    copied. Existing files overwritten.

    `asset_prefix` is prepended to each filename on copy. Used for SI
    bundling so SI assets don't collide with main assets in a shared
    assets/ dir (e.g. asset_prefix='si_' produces si_abc123.jpg)."""
    if not src_images.is_dir():
        return 0
    dst_assets.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_images.iterdir():
        if f.is_file():
            shutil.copy2(f, dst_assets / (asset_prefix + f.name))
            n += 1
    return n


# === RunInfo helper ========================================================


# Packages whose versions land in the run-info `packages:` block of
# every wrap_mineru output. Recorded for reproducibility:
#  - mineru / mineru-vl-utils  : the body extractor + hybrid-VLM extras.
#  - torch / transformers      : underlying ML stack used by mineru and
#                                paper2md helpers.
#  - pymupdf                   : PDF reading for trim_articles + paper2md
#                                metadata frontend.
#  - pillow                    : image decoding for the table-VLM rewrite.
#  - openai / anthropic        : VLM client SDKs (when --provider is set).
#  - requests                  : Crossref / OpenAlex / Unpaywall calls.
#  - h5py                      : --hdf5 bundling.
#  - python-dotenv             : .env loading.
#  - numpy                     : transitive dep widely relied on.
# Skips marker-pdf / surya-ocr / docling -- those are paper2md (marker
# spine) specific and not used by the wrap_mineru path.
_WRAP_MINERU_RECORDED_PACKAGES = (
    "mineru",
    "mineru-vl-utils",
    "torch",
    "transformers",
    "pymupdf",
    "pillow",
    "openai",
    "anthropic",
    "requests",
    "h5py",
    "python-dotenv",
    "numpy",
)


def _collect_package_versions() -> dict:
    """Best-effort: read installed versions of the wrap_mineru-relevant
    packages. Skips any not present (e.g. anthropic when only lmstudio
    is configured)."""
    from importlib.metadata import PackageNotFoundError, version
    out: dict = {}
    for name in _WRAP_MINERU_RECORDED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _resolve_vlm_endpoint(vlm_provider: str) -> str:
    """Return the VLM endpoint URL when configure_client() has run on
    behalf of this wrapper run, else empty string.

    Note: paper2md's module-level `client` is initialized to the
    lmstudio default at import time (whether or not we'll use it),
    so we cannot just read its base_url unconditionally. We only
    record an endpoint when vlm_provider is set to a real provider
    (i.e., when the wrapper called configure_client itself, indicated
    by vlm_provider != 'none')."""
    if vlm_provider in ("", "none"):
        return ""
    if vlm_provider == "anthropic":
        return "https://api.anthropic.com"
    client = getattr(p2m, "client", None)
    if client is None:
        return ""
    base_url = getattr(client, "base_url", None)
    return str(base_url) if base_url else ""


def _build_run_info(args, vlm_provider: str, vlm_model: str) -> p2m.RunInfo:
    cmd_str = " ".join(shlex.quote(a) for a in sys.argv)
    # Snapshot every CLI knob that affects the output. Trim outcome
    # (trim_head / trim_tail) is added later in _process_one.
    pipeline = {
        "wrap_mineru": True,
        "mineru_backend": args.mineru_backend,
        "vlm_tables": bool(args.vlm_tables),
        "trim_articles": bool(args.trim_articles),
        "strip_chart_details": bool(args.strip_chart_details),
        "no_vlm": bool(args.no_vlm),
        "skip_vlm_check": bool(args.skip_vlm_check),
        "no_metadata_lookup": bool(args.no_metadata_lookup),
        "use_journal_rescue": bool(args.use_journal_rescue),
        "rescue_orphan_captions": bool(args.rescue_orphan_captions),
        "fetch_data_repos": bool(args.fetch_data_repos),
        "offline": bool(args.offline),
    }
    return p2m.RunInfo(
        command=cmd_str,
        hostname=socket.gethostname(),
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        python_version=".".join(str(v) for v in sys.version_info[:3]),
        # compute_backend = hardware (cuda/mps/cpu) -- left empty here
        # because wrap_mineru doesn't pin it the way paper2md does;
        # MinerU's subprocess detects its own GPU. layout_source +
        # text_engine carry the engine identity.
        compute_backend="",
        layout_source="mineru",
        text_engine=f"mineru-{args.mineru_backend}",
        vlm_endpoint=_resolve_vlm_endpoint(vlm_provider),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        pipeline=pipeline,
        packages=_collect_package_versions(),
    )


# === End-of-run summary ====================================================


def _emit_run_summary(report: p2m.QualityReport, out_md: Path,
                      args, trim_info: Optional[dict] = None) -> None:
    """Banner emitted after each paper. Mirrors paper2md._emit_run_summary
    but tuned for wrap_mineru's pipeline state."""
    elapsed = (report.run_info.elapsed_sec
               if report.run_info is not None else 0.0)
    log.info("=" * 60)
    log.info("wrap_mineru complete: %s", out_md.name)
    log.info("  Total time: %.1f s", elapsed)
    log.info("  Grade:      %s (overall: %.2f)",
             report.grade(), report.overall())
    log.info("  Backend:    mineru-%s", args.mineru_backend)

    # References summary.
    if report.references is not None:
        r = report.references
        bits = [f"score={r.score:.2f}", f"style={r.style}",
                f"entries={r.entry_count}",
                f"section_count={r.section_count}"]
        if r.journal_slug:
            bits.append(f"journal={r.journal_slug}")
        if r.api_source:
            bits.append(f"api={r.api_source} (+{r.api_refs_appended})")
        log.info("  References: %s", ", ".join(bits))

    # Tables: clean vs fallback split.
    if report.tables:
        n = len(report.tables)
        clean = sum(1 for t in report.tables if t.score >= 0.9)
        vlm_redone = sum(1 for t in report.tables if t.vlm_redone)
        fallback = n - clean - vlm_redone
        log.info("  Tables:     %d (%d clean, %d VLM-rewritten, "
                 "%d HTML fallback)", n, clean, vlm_redone, fallback)

    # Figures.
    if report.figures:
        nf = len(report.figures)
        captioned = sum(1 for f in report.figures if f.caption_produced)
        log.info("  Figures:    %d (%d with captions)", nf, captioned)

    # Article-boundary trim outcome.
    if trim_info and (trim_info.get("head_trimmed", 0)
                      or trim_info.get("tail_trimmed", 0)):
        log.info("  Article-trim: head=%d tail=%d page(s) dropped "
                 "(main pages %d..%d)",
                 trim_info["head_trimmed"], trim_info["tail_trimmed"],
                 trim_info["main_pages"][0] + 1,
                 trim_info["main_pages"][1])

    # Copyright shape.
    if report.metadata is not None:
        m = report.metadata
        bits = []
        if m.doi:
            bits.append(f"doi={m.doi}")
        if m.journal_slug:
            bits.append(f"journal={m.journal_slug}")
        if m.license:
            bits.append(f"license={m.license}")
        if m.safe_to_distribute:
            bits.append(f"safe={m.safe_to_distribute}")
        if bits:
            log.info("  Copyright:  %s", ", ".join(bits))

    log.info("=" * 60)


# === User annotations ======================================================


def _build_user_annotations(args) -> Optional[p2m.UserAnnotations]:
    """Read --user / --collection / --note flags + env-var fallbacks.
    Returns None when all three are empty (so the YAML block is omitted)."""
    ua = p2m.UserAnnotations(
        user=(args.user or os.environ.get("PAPER2MD_USER") or None),
        collection=(args.collection
                    or os.environ.get("PAPER2MD_COLLECTION") or None),
        note=(args.note or None),
    )
    if ua.is_empty():
        return None
    return ua


# === Batch mode ============================================================


def _run_batch(pairs: list[tuple[Path, Optional[Path]]], args,
               vlm_provider: str, vlm_model: str) -> int:
    """Sequential or thread-pooled batch runner. Returns failure count."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "manifest.jsonl"
    manifest_lock = __import__("threading").Lock()

    def _per_paper(main_pdf: Path, si_pdf: Optional[Path]) -> dict:
        t_start = time.time()
        paper_dir = args.outdir / main_pdf.stem
        out_md_existing = paper_dir / f"{main_pdf.stem}.md"
        if (out_md_existing.exists()
                and not args.force
                and not args.clean_output):
            return {"pdf": str(main_pdf),
                    "supplement": str(si_pdf) if si_pdf else None,
                    "status": "skipped",
                    "elapsed_sec": 0.0}
        try:
            run_info = _build_run_info(args, vlm_provider, vlm_model)
            out_md, report = _process_one(main_pdf, args.outdir, args,
                                          run_info)
            main_paper_dir = out_md.parent
            si_md = None
            si_report = None
            if si_pdf is not None:
                si_run = _build_run_info(args, vlm_provider, vlm_model)
                si_md, si_report = _process_one(
                    si_pdf, args.outdir, args, si_run,
                    is_si=True,
                    paper_dir_override=main_paper_dir,
                    asset_prefix="si_",
                )
            if args.hdf5:
                bundle = paper_dir / f"{main_pdf.stem}.h5"
                p2m.write_hdf5_bundle(
                    bundle, main_md=out_md, main_report=report,
                    main_pdf=main_pdf,
                    si_md=si_md, si_report=si_report, si_pdf=si_pdf,
                )
            overall = report.overall()
            status = "ok"
            if (args.quality_threshold is not None
                    and overall < args.quality_threshold):
                status = "low_quality"
            entry = {
                "pdf": str(main_pdf),
                "supplement": str(si_pdf) if si_pdf else None,
                "status": status,
                "grade": report.grade(),
                "overall": round(overall, 3),
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            if report.metadata is not None:
                entry["copyright"] = report.metadata.manifest_dict()
            return entry
        except Exception as e:
            log.exception("Failed: %s", main_pdf.name)
            return {"pdf": str(main_pdf),
                    "supplement": str(si_pdf) if si_pdf else None,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_sec": round(time.time() - t_start, 1)}

    deadline_s = float(args.paper_timeout) if args.paper_timeout else 0.0

    def _bounded(main_pdf: Path, si_pdf: Optional[Path]) -> dict:
        if deadline_s <= 0:
            return _per_paper(main_pdf, si_pdf)
        import threading as _threading
        box: dict = {}

        def _go() -> None:
            try:
                box["result"] = _per_paper(main_pdf, si_pdf)
            except BaseException as e:
                box["error"] = e

        t = _threading.Thread(target=_go, daemon=True,
                              name=f"wrap_mineru-batch-{main_pdf.stem}")
        t_start = time.time()
        t.start()
        t.join(deadline_s)
        if t.is_alive():
            log.error("[batch] paper-timeout (%.0fs): %s -- abandoning "
                      "thread (may hold MinerU subprocess) and continuing",
                      deadline_s, main_pdf.name)
            return {
                "pdf": str(main_pdf),
                "supplement": str(si_pdf) if si_pdf else None,
                "status": "timeout",
                "error": f"timed out after {deadline_s:.0f} s",
                "elapsed_sec": round(time.time() - t_start, 1),
            }
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _record(entry: dict) -> None:
        line = json.dumps(entry)
        with manifest_lock:
            with manifest_path.open("a") as f:
                f.write(line + "\n")
        log.info("[batch] %s", line)

    failures = 0
    fail_statuses = {"error", "timeout"}
    workers = max(1, args.workers)
    if workers == 1:
        for main_pdf, si_pdf in pairs:
            entry = _bounded(main_pdf, si_pdf)
            _record(entry)
            if entry["status"] in fail_statuses:
                failures += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_bounded, m, s) for m, s in pairs]
            for fut in as_completed(futs):
                entry = fut.result()
                _record(entry)
                if entry["status"] in fail_statuses:
                    failures += 1
    return failures


# === CLI driver ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wrap_mineru.py",
        description="paper2md value-adds layered on MinerU body extraction.",
        epilog=(
            "Calls MinerU (AGPL-3.0) as a subprocess; users distributing "
            "MinerU must comply with AGPL terms separately. The wrapper "
            "itself is MIT."
        ),
    )
    p.add_argument("pdf", type=Path, nargs="?", default=None,
                   help="Input PDF. Use --batch instead for a folder of PDFs.")
    p.add_argument("-o", "--outdir", type=Path, required=True,
                   help="Output directory (one subdir per paper will be created).")
    p.add_argument("--supplement", type=Path,
                   help="Optional supplement PDF; runs the wrapper a second "
                        "time and merges output into the bundle.")
    p.add_argument("--hdf5", action="store_true",
                   help="Emit <stem>.h5 alongside <stem>.md.")
    p.add_argument("--clean-mineru", "--clean", dest="clean_mineru",
                   action="store_true",
                   help="Delete intermediate MinerU work_dir "
                        "(mineru/, mineru_si/) on success. Per-paper "
                        ".md/.h5/.meta.json/assets are kept. "
                        "(--clean is a deprecated alias.)")
    p.add_argument("--clean-output", action="store_true",
                   help="Wipe the entire per-paper output directory "
                        "BEFORE running -- assets, .md, .h5, "
                        ".meta.json, mineru/, everything. Use for "
                        "re-runs when stale files from a prior run "
                        "could mask the new output. Skipped for "
                        "supplement processing (would clobber the "
                        "main paper's just-written files).")

    p.add_argument("--mineru-backend",
                   choices=["pipeline", "hybrid-auto-engine"],
                   default="pipeline",
                   help="MinerU backend. pipeline (default) is fastest "
                        "(~30 s/paper, no MinerU VLM). hybrid-auto-engine "
                        "uses MinerU's bundled 2.3 GB VLM "
                        "(~3-17 min/paper) for cell-level tables and "
                        "chart-data reconstruction; emits <details> blocks "
                        "with VLM-estimated curve readings.")
    p.add_argument("--strip-chart-details", action="store_true",
                   help="Strip MinerU hybrid-mode <details><summary>line</summary> "
                        "chart-data blocks from the body. Off by default; "
                        "use this when downstream RAG might quote the "
                        "VLM-estimated values as facts.")

    p.add_argument("--vlm-tables", action="store_true",
                   help="Enable image-based paper2md VLM rewrite for tables "
                        "that fail clean HTML->pipe conversion. Fires only "
                        "on suspicious tables (rowspan/colspan/etc.). Off by "
                        "default. Independent of --mineru-backend.")
    p.add_argument("--provider", default=None,
                   help="paper2md VLM provider (lmstudio | vllm | openai | "
                        "anthropic). Default: lmstudio (or $VLM_PROVIDER).")
    p.add_argument("--model", default=None,
                   help="paper2md VLM model. Default: $VLM_MODEL or the "
                        "provider's built-in default.")
    p.add_argument("--no-vlm", action="store_true",
                   help="Disable paper2md's external VLM entirely (raw HTML "
                        "on suspicious tables). Does NOT disable MinerU's "
                        "internal VLM (--mineru-backend hybrid-auto-engine).")
    p.add_argument("--trim-articles", action="store_true",
                   help="Detect and trim adjacent papers from the first "
                        "and/or last PDF pages using a VLM page-vs-page "
                        "comparison. Drops MinerU blocks whose page_idx "
                        "is outside the main article boundary. Off by "
                        "default. Uses paper2md's vlm() (--provider).")

    p.add_argument("--no-metadata-lookup", action="store_true",
                   help="Skip metadata_frontend.resolve() (no Crossref/"
                        "OpenAlex/Unpaywall calls).")
    p.add_argument("--use-journal-rescue", action="store_true",
                   help="Enable journal-aware reference rescue passes. "
                        "(Off by default; APS-missing-heading rescue and "
                        "Knudson-style ref reformat always run.)")
    p.add_argument("--rescue-orphan-captions", action="store_true",
                   help="Adopt mis-classified text blocks adjacent to "
                        "figure/table blocks as captions / footnotes "
                        "(closes cuk p2, alexander p2 patterns). Also "
                        "treats publisher-watermark text in caption "
                        "slots as 'empty' for adoption purposes. "
                        "Default off; per Stage 3 plan.")
    p.add_argument("--fetch-data-repos", action="store_true",
                   help="Enrich detected data-repository links with API "
                        "summaries (Zenodo/Dryad/figshare/etc.). Implies "
                        "network access.")
    p.add_argument("--offline", action="store_true",
                   help="No network: skip metadata API calls, references "
                        "API rescue, and data-repo enrichment.")

    p.add_argument("--mineru-arg", action="append", default=[],
                   help="Repeatable; passed through to mineru CLI as "
                        "additional argument. E.g. --mineru-arg "
                        "--device-mode --mineru-arg cpu.")
    p.add_argument("--skip-mineru-run", action="store_true",
                   help="Skip the MinerU subprocess; consume an existing "
                        "auto/ output directory specified via --mineru-dir. "
                        "Intended for offline replay / testing.")
    p.add_argument("--mineru-dir", type=Path,
                   help="Existing MinerU auto/ directory containing "
                        "<stem>_middle.json and images/. Required when "
                        "--skip-mineru-run is passed.")

    # Preflight / VLM safety
    p.add_argument("--skip-vlm-check", action="store_true",
                   help="Skip the VLM endpoint preflight ping. By default, "
                        "if any VLM feature is enabled (--vlm-tables / "
                        "--trim-articles), the wrapper pings the configured "
                        "endpoint with a cheap models.list() request before "
                        "running MinerU and exits in seconds if the VLM is "
                        "unreachable.")

    # User annotations
    g_user = p.add_argument_group("user annotations")
    g_user.add_argument("--user", default=None, metavar="NAME",
                        help="Curator name. Recorded in `user:` YAML block. "
                             "Falls back to $PAPER2MD_USER. Useful for "
                             "multi-curator corpora.")
    g_user.add_argument("--collection", default=None, metavar="NAME",
                        help="Collection / project tag. Recorded in `user:` "
                             "YAML block. Falls back to $PAPER2MD_COLLECTION. "
                             "In --batch mode, applied to every paper.")
    g_user.add_argument("--note", default=None, metavar="TEXT",
                        help="Free-form note about this run. Recorded in "
                             "`user:` YAML block as a YAML block-scalar so "
                             "newlines survive. In --batch mode the note is "
                             "applied to every paper.")

    # Batch mode
    g_batch = p.add_argument_group("batch mode")
    g_batch.add_argument("--batch", default=None, metavar="PATH_OR_GLOB",
                         help="Run on every PDF in a directory or glob "
                              "instead of a single PDF positional. Mutually "
                              "exclusive with the positional pdf arg.")
    g_batch.add_argument("--workers", type=int, default=1,
                         help="Parallel papers in --batch mode (default 1). "
                              "Each MinerU subprocess is itself parallel; "
                              "raise this only when MinerU runs CPU-bound.")
    g_batch.add_argument("--force", action="store_true",
                         help="In --batch mode, re-run papers even if their "
                              "output markdown already exists.")
    g_batch.add_argument("--no-pair", action="store_true",
                         help="Disable supplement auto-pairing in --batch "
                              "mode. Each PDF is treated as standalone.")
    g_batch.add_argument("--pair-regex", default=None,
                         help="Custom SI-suffix regex for supplement "
                              "pairing. Must define a 'stem' named capture "
                              "group. Default: matches _SI / _S1 / _sm / "
                              "_supplement / etc.")
    g_batch.add_argument("--paper-timeout", type=float, default=1800.0,
                         help="Per-paper deadline in seconds (default 1800 "
                              "= 30 min). Pass 0 to disable. Applies in "
                              "--batch mode only.")
    g_batch.add_argument("--quality-threshold", type=float, default=None,
                         help="Mark papers with overall < this value as "
                              "'low_quality' in manifest.jsonl. Does not "
                              "block the run.")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging (DEBUG level).")

    # Allow positional pdf to be optional when --batch is used.
    return p


def _configure_paper2md_vlm(provider: Optional[str], model: Optional[str]):
    """Configure paper2md's VLM client. Returns (provider_name, model_name)."""
    p2m.configure_client(provider=provider, model=model)
    return p2m._PROVIDER, p2m.VLM_MODEL


def _process_one(pdf_path: Path, outdir: Path, args,
                 run_info: p2m.RunInfo, *,
                 is_si: bool = False,
                 paper_dir_override: Optional[Path] = None,
                 asset_prefix: str = "",
                 ) -> tuple[Path, p2m.QualityReport]:
    """Run the wrapper on a single PDF (main or supplement). Returns
    (markdown_path, QualityReport).

    `run_info` is attached to the report early; elapsed_sec is finalised
    here before .md and .meta.json are written, so downstream consumers
    (and the HDF5 bundle) see consistent run metadata.

    Supplement bundling (when `is_si=True`):
      * Set `paper_dir_override` to the main paper's dir so SI artifacts
        land alongside the main markdown instead of in a sibling dir.
      * Set `asset_prefix='si_'` so SI assets don't collide with main
        assets in the shared assets/ directory.
      * SI's MinerU output goes to `<paper_dir>/mineru_si/` (a sibling
        of the main `mineru/` dir).
    """
    t0 = time.time()
    stem = pdf_path.stem
    paper_dir = (paper_dir_override
                 if paper_dir_override is not None
                 else outdir / stem)
    if (getattr(args, "clean_output", False)
            and not is_si
            and paper_dir.is_dir()):
        log.info("--clean-output: wiping %s", paper_dir)
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = paper_dir / "assets"
    mineru_subdir_name = "mineru_si" if is_si else "mineru"
    mineru_dir = paper_dir / mineru_subdir_name

    if args.skip_mineru_run:
        if not args.mineru_dir or not args.mineru_dir.is_dir():
            raise SystemExit("--skip-mineru-run requires --mineru-dir <path>")
        mineru_dir = stage_existing_mineru_dir(args.mineru_dir, mineru_dir)
    else:
        extra = list(args.mineru_arg or [])
        mineru_dir = run_mineru(pdf_path, mineru_dir,
                                backend=args.mineru_backend,
                                extra_args=extra)

    middle_candidates = sorted(mineru_dir.glob("*_middle.json"))
    if not middle_candidates:
        raise FileNotFoundError(
            f"No *_middle.json found under {mineru_dir}"
        )
    middle_path = middle_candidates[0]
    if middle_path.stem.removesuffix("_middle") != stem:
        log.info("Using middle.json: %s (stem mismatch with input %s)",
                 middle_path.name, stem)
    doc = parse_middle_json(middle_path)
    log.info("Parsed %d blocks across %d page(s) from %s",
             len(doc.blocks), doc.pages, middle_path.name)

    n_copied = copy_assets(mineru_dir / "images", assets_dir,
                           asset_prefix=asset_prefix)
    log.info("Copied %d image asset(s) to %s%s",
             n_copied, assets_dir,
             f" (prefix={asset_prefix!r})" if asset_prefix else "")

    # Build report and emit markdown body.
    report = p2m.QualityReport()
    report.edits = p2m.EditsAnnotation()
    # Attach run_info early; elapsed_sec finalised right before write.
    report.run_info = run_info
    # User annotations: same instance reused across batch papers.
    user_anno = _build_user_annotations(args)
    if user_anno is not None:
        report.user_annotations = user_anno

    # Article-boundary trim (optional, VLM-driven).
    trim_info = {"head_trimmed": 0, "tail_trimmed": 0,
                 "main_pages": (0, doc.pages)}
    if args.trim_articles and not args.no_vlm:
        try:
            import fitz
            with fitz.open(pdf_path) as fdoc:
                doc.blocks, trim_info = trim_articles_in_blocks(
                    doc.blocks, fdoc, doc.pages,
                )
        except Exception as e:
            log.warning("Article-trim aborted (%s); continuing with full doc", e)
    elif args.trim_articles and args.no_vlm:
        log.warning("--trim-articles requires VLM; --no-vlm overrides. "
                    "Skipping trim.")
    # Record trim outcome on the run info so it surfaces in frontmatter
    # and meta.json regardless of whether the trim fired.
    if report.run_info is not None:
        report.run_info.pipeline["trim_head"] = int(trim_info["head_trimmed"])
        report.run_info.pipeline["trim_tail"] = int(trim_info["tail_trimmed"])

    # Orphan caption/footnote rescue (flag-gated). Adopts mis-classified
    # text blocks into adjacent figure/table parents BEFORE sup/sub
    # recovery so adopted captions also get the markup pass.
    if args.rescue_orphan_captions:
        try:
            orph_info = rescue_orphan_captions(
                doc.blocks, page_sizes=doc.page_sizes)
            if report.run_info is not None:
                report.run_info.pipeline["orphan_captions_adopted"] = int(
                    orph_info["captions_adopted"]
                    + orph_info.get("captions_adopted_3col", 0)
                )
                report.run_info.pipeline["orphan_footnotes_adopted"] = int(
                    orph_info["footnotes_adopted"]
                )
        except Exception as e:
            log.warning("rescue_orphan_captions failed (%s); continuing", e)

    md_body = emit_markdown(
        doc, assets_rel="assets", assets_abs=assets_dir,
        vlm_tables=args.vlm_tables and not args.no_vlm,
        report=report,
        strip_chart_details=args.strip_chart_details,
        asset_prefix=asset_prefix,
    )

    # paper2md value-adds (drop-in).
    md_body = rescue_missing_refs_heading(md_body)

    # Metadata resolution.
    if not args.no_metadata_lookup:
        try:
            report.metadata = metadata_frontend.resolve(
                pdf_path, allow_network=not args.offline,
            )
            log.info("Metadata: doi=%s, journal=%s, license=%s",
                     report.metadata.doi, report.metadata.journal_slug,
                     report.metadata.license)
        except Exception as e:
            log.warning("metadata_frontend.resolve raised %s; continuing", e)

    # APS missing-heading rescue (always-on).
    md_body = p2m._rescue_aps_missing_refs_heading(md_body, report)

    # References score.
    p2m._populate_references_score(md_body, report, doc=None)

    # API ref rescue when score is poor and DOI known.
    rs = report.references
    if (rs is not None and report.metadata is not None
            and getattr(report.metadata, "doi", None)
            and not args.offline):
        # Trigger condition mirrors paper2md._rescue_append_api_refs's gate.
        score = rs.score if rs.score is not None else 0.0
        gate = (
            score < 0.5
            or rs.section_count == 0
            or (rs.style == "numbered"
                and not rs.numbered_continuous
                and rs.entry_count >= 3)
        )
        if gate:
            md_body = p2m._rescue_append_api_refs(
                md_body, report, allow_network=True,
            )

    # Footnote-style refs rescue: when local AND API rescues failed but
    # MinerU classified some discarded_blocks as page_footnote (old
    # J. Appl. Phys. / Phys. Rev. footnote-citation style), mine them.
    # Self-gates on poor refs.score + zero api_refs_appended; cheap
    # no-op otherwise.
    if doc.footnote_blocks:
        md_body_after = rescue_footnote_refs(md_body, doc.footnote_blocks,
                                              report)
        if md_body_after is not md_body:
            md_body = md_body_after
            # Re-score so the manifest reflects the appended refs.
            p2m._populate_references_score(md_body, report, doc=None)

    # Data-repo links (always-on extraction; enrichment opt-in).
    _populate_data_repos_local(
        md_body, report,
        allow_network=not args.offline,
        fetch_summaries=bool(args.fetch_data_repos),
    )

    # Finalise run timing before writing artifacts.
    if report.run_info is not None:
        report.run_info.elapsed_sec = time.time() - t0

    # Frontmatter + emit.
    out_md = paper_dir / f"{stem}.md"
    out_md.write_text(report.to_frontmatter() + md_body + "\n")

    out_meta = paper_dir / f"{stem}.meta.json"
    out_meta.write_text(json.dumps(
        report.to_dict(), indent=2, default=str, ensure_ascii=False,
    ))

    if args.clean_mineru and mineru_dir.is_dir():
        shutil.rmtree(mineru_dir, ignore_errors=True)
        log.info("Cleaned MinerU output dir: %s", mineru_dir)

    _emit_run_summary(report, out_md, args, trim_info)
    return out_md, report


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input mode: positional pdf XOR --batch.
    if args.pdf and args.batch:
        parser.error("pass either a positional pdf or --batch, not both")
    if not args.pdf and not args.batch:
        parser.error("missing pdf path (or use --batch <dir|glob>)")
    if args.supplement and args.batch:
        parser.error("--supplement is for single-paper mode; in --batch, "
                     "supplements are auto-paired by stem suffix")
    if args.skip_mineru_run and args.batch:
        parser.error("--skip-mineru-run is single-paper only")
    if args.skip_mineru_run and not args.mineru_dir:
        parser.error("--skip-mineru-run requires --mineru-dir")

    # Configure paper2md's VLM when any VLM-using feature is enabled.
    vlm_provider, vlm_model = "none", "none"
    needs_vlm = (args.vlm_tables or args.trim_articles) and not args.no_vlm
    if needs_vlm:
        vlm_provider, vlm_model = _configure_paper2md_vlm(
            args.provider, args.model,
        )
        # Preflight: ping the configured endpoint with a cheap models.list().
        # Exits in seconds if misconfigured instead of failing mid-run.
        if not args.skip_vlm_check:
            err = p2m.check_vlm_reachable()
            if err is not None:
                log.error("VLM preflight failed: %s", err)
                log.error("To bypass this check, pass --skip-vlm-check (or "
                          "--no-vlm to skip the VLM hooks entirely).")
                return 2
            log.info("VLM preflight OK (%s @ %s)", vlm_provider, vlm_model)

    # ----- Batch mode -----
    if args.batch:
        try:
            pdfs = p2m.discover_pdfs(args.batch)
        except SystemExit as e:
            log.error("--batch: %s", e)
            return 2
        log.info("[batch] %d PDF(s) discovered", len(pdfs))

        if args.no_pair:
            pairs: list[tuple[Path, Optional[Path]]] = [(p, None) for p in pdfs]
            standalone_si: list[Path] = []
        else:
            if args.pair_regex:
                try:
                    pair_re = re.compile(args.pair_regex)
                    if "stem" not in pair_re.groupindex:
                        parser.error("--pair-regex must define a 'stem' "
                                     "named group")
                except re.error as e:
                    parser.error(f"--pair-regex: {e}")
            else:
                pair_re = p2m.SI_SUFFIX_RE
            pairs, standalone_si = p2m.pair_supplements(pdfs, pair_re)
            n_paired = sum(1 for _, si in pairs if si is not None)
            log.info("[batch] %d pair(s), %d standalone, %d orphan SI",
                     n_paired, len(pairs) - n_paired, len(standalone_si))
            if standalone_si:
                log.warning("[batch] orphan SI files (will run as main): %s",
                            ", ".join(p.name for p in standalone_si))
                pairs += [(s, None) for s in standalone_si]

        n_failed = _run_batch(pairs, args, vlm_provider, vlm_model)
        log.info("[batch] %d / %d paper(s) failed",
                 n_failed, len(pairs))
        return 0 if n_failed == 0 else 1

    # ----- Single-paper mode -----
    main_run_info = _build_run_info(args, vlm_provider, vlm_model)
    out_md, main_report = _process_one(
        args.pdf, args.outdir, args, main_run_info,
    )
    main_paper_dir = out_md.parent

    si_md_path: Optional[Path] = None
    si_report: Optional[p2m.QualityReport] = None
    if args.supplement:
        si_run_info = _build_run_info(args, vlm_provider, vlm_model)
        # SI artifacts land in main's paper_dir with si_ asset prefix.
        si_md_path, si_report = _process_one(
            args.supplement, args.outdir, args, si_run_info,
            is_si=True,
            paper_dir_override=main_paper_dir,
            asset_prefix="si_",
        )

    if args.hdf5:
        bundle_path = out_md.with_suffix(".h5")
        p2m.write_hdf5_bundle(
            bundle_path=bundle_path,
            main_md=out_md, main_report=main_report, main_pdf=args.pdf,
            si_md=si_md_path, si_report=si_report, si_pdf=args.supplement,
        )
        log.info("Wrote HDF5 bundle: %s", bundle_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
