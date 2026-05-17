"""Recover audit-flagged missing tables from MinerU's pre-existing
output.

After a hybrid run, paper2md leaves MinerU's middle.json + cropped
images under `<paper-dir>/mineru/`. The audit may show that the
hybrid splice dropped a table that MinerU detected fine -- in that
case the table is RECOVERABLE without rerunning anything.

This module orchestrates the recovery:

  1. `find_recoverable_tables(paper_md)`  -- pure read; returns the
     subset of audit-flagged missing tables that MinerU has crops for.

  2. `stage_one_table(paper_md, item, ...)` -- copies MinerU's image
     into assets/, calls VLM (or html_to_pipe_md), writes a sidecar
     .md, appends to the `pending_recoveries:` frontmatter block.

  3. `do_recover_from_mineru(paper_md, vlm_override)` -- top-level
     entry; runs (1) then (2) for every found item.

  4. `confirm_recovery(paper_md, kind, id)` -- promotes a staged entry
     from `pending_recoveries:` -> `edits:` (signals "user pasted").

The body of the .md is NEVER auto-edited. The user pastes the sidecar
contents over the garbled / missing table. This module's job is to
remove every step EXCEPT the actual paste.

Pinned MinerU schema: 3.1.7 (see environment-gpu.yml + HYBRID_-
IMPLEMENTATION.md). Schema drift on a version bump will surface as a
`find_recoverable_tables` regression; re-audit before adopting.
"""
from __future__ import annotations

import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

# Sibling-module imports. paper2md / layout_mineru / wrap_mineru
# define the heavy lifting (middle.json parser, html_to_pipe_md, VLM
# caller). Imported lazily inside functions where practical to keep
# this module's surface light for tests.

import manual_edit
import audit_md as _audit

log = logging.getLogger("mineru_recovery")


# ---------------------------------------------------------------------
# Frontmatter helpers (delegate to manual_edit's parameterized helpers)
# ---------------------------------------------------------------------

PENDING_BLOCK = "pending_recoveries"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------
# Step 1: discovery
# ---------------------------------------------------------------------


# Some MinerU table blocks have NO `table_caption` child -- the "Table
# N" label is embedded as the first content row of the table HTML
# instead. Caption-anchoring then fails entirely (this is precisely
# why hybrid drops these tables). For RECOVERY we look in the first
# ~250 chars of the HTML for a "Table N" pattern as a secondary id
# source. The pattern matches the same id forms as
# paper2md._HYBRID_TBL_NUM_RE so canonical-id dedup still works.
_HTML_TABLE_ID_RE = re.compile(
    r"(?:Supplementary\s+|Supplement(?:ary)?\s+|Supp\.\s+)?"
    r"Table\s+(S?\d+|S?[IVXLC]+|[A-Z]\d+|[A-Z]\.\d+)\b",
    re.IGNORECASE,
)


def _resolve_image_path(
    image_path: Optional[str],
    mineru_dir: Path,
    paper_dir: Path,
) -> Optional[Path]:
    """MinerU's middle.json `image_path` is sometimes a bare filename
    (`aa635fb...jpg`), sometimes `images/aa635fb...jpg`. The actual
    image may live in `mineru/images/`, `mineru/`, or -- once the
    hybrid pipeline has moved it -- in the paper's `assets/`. Return
    the first candidate that exists on disk, or None."""
    if not image_path:
        return None
    candidates = [
        mineru_dir / image_path,
        mineru_dir / "images" / image_path,
        paper_dir / "assets" / image_path,
        # Bare-basename in assets (image_path might be already-pathed)
        paper_dir / "assets" / Path(image_path).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _extract_id_from_html(html: Optional[str]) -> Optional[str]:
    """Pull a Table id from the first ~250 chars of HTML when MinerU
    didn't classify a caption block."""
    if not html:
        return None
    m = _HTML_TABLE_ID_RE.search(html[:250])
    return m.group(1) if m else None


def _middle_json_for(paper_md: Path) -> Optional[Path]:
    """Locate `<paper-dir>/mineru/<stem>_middle.json` for the given .md.

    Stem is `paper_md.stem`. SI files (`<stem>_sm.md`) auto-resolve to
    `<stem>_sm_middle.json`, so a main paper's recovery never
    accidentally pulls SI tables and vice versa."""
    mineru_dir = paper_md.parent / "mineru"
    if not mineru_dir.is_dir():
        return None
    candidate = mineru_dir / f"{paper_md.stem}_middle.json"
    return candidate if candidate.exists() else None


def find_recoverable_tables(paper_md: Path) -> list[dict]:
    """For every Table id REFERENCED in the body but NOT inserted in
    the `.md`, look in MinerU's `middle.json` for a matching table
    block. Returns one dict per recoverable table:

        {
            "id": "3",            # raw id from the body ref
            "canonical_id": "3",  # Roman -> arabic, S-prefix preserved
            "page_idx": 4,        # 0-based page index from MinerU
            "image_path": Path(   # absolute path to MinerU's cropped JPG
                "<paper-dir>/mineru/images/<hash>.jpg"),
            "html": "<table>...",
            "caption_text": "Table 3. Summary of ...",
        }

    Items whose `id` is referenced but NOT found in middle.json are NOT
    returned -- those are unrecoverable (genuine extraction failure)
    and need a manual `--replace-table` instead.
    """
    middle_path = _middle_json_for(paper_md)
    if middle_path is None:
        log.info("No MinerU output for %s (looked at %s/mineru/); "
                 "nothing recoverable.", paper_md.name,
                 paper_md.parent.name)
        return []

    # Lazy import: layout_mineru pulls in numpy + json walking on first
    # import. Tests that don't need this branch shouldn't pay the cost.
    sys.path.insert(0, str(Path(__file__).parent))
    import layout_mineru

    parsed = layout_mineru.parse_middle_json(middle_path)

    # Build a canonical-id -> block index. _extract_table_number is in
    # paper2md.py; reuse it via the cached _HYBRID_TBL_NUM_RE pattern.
    import paper2md as p2m

    mineru_tables: dict[str, list] = {}
    for blk in parsed.blocks:
        if blk.type != "table":
            continue
        raw_id = p2m._extract_table_number(blk.text)
        if raw_id is None:
            # Fallback: many tables lack a classified caption block
            # but embed "Table N" as the first HTML row. Pull from
            # there.
            raw_id = _extract_id_from_html(blk.html)
        if raw_id is None:
            continue
        canon = _audit._canonical_tbl_id(raw_id)
        mineru_tables.setdefault(canon, []).append(blk)

    # Audit the paper for referenced-but-missing ids. Use the auto-
    # detection of SI sibling so S-prefixed refs are handled correctly.
    audit = _audit.audit_md(paper_md)
    refs = set(audit["tables_referenced"])
    inserted = {
        _audit._canonical_tbl_id(x) for x in audit["tables_inserted"]
    }

    recoverable: list[dict] = []
    seen_canon: set[str] = set()
    for raw_ref in refs:
        canon = _audit._canonical_tbl_id(raw_ref)
        if canon in inserted:
            continue  # already in the .md
        if canon in seen_canon:
            continue  # already added under a different ref alias
        blocks = mineru_tables.get(canon, [])
        if not blocks:
            continue  # not in MinerU's output -> unrecoverable
        seen_canon.add(canon)
        blk = blocks[0]
        # Resolve image_path against several plausible locations. MinerU
        # writes table images to `mineru/images/<hash>.jpg` originally,
        # but the hybrid pipeline may have moved them to `assets/` even
        # when the splice failed -- so the file is wherever it ended up.
        img_abs = _resolve_image_path(
            blk.image_path, middle_path.parent, paper_md.parent)
        recoverable.append({
            "id": raw_ref,
            "canonical_id": canon,
            "page_idx": blk.page_idx,
            "image_path": img_abs,
            "html": blk.html,
            "caption_text": blk.text,
        })
    return recoverable


# ---------------------------------------------------------------------
# Step 2: staging one table
# ---------------------------------------------------------------------


def detect_original_vlm_mode(paper_md: Path) -> bool:
    """Read `run.pipeline.vlm_rewrite_tables` from the paper's
    frontmatter. Returns False when unset/unreadable (the safe
    default that doesn't burn VLM tokens unexpectedly)."""
    try:
        fm, _ = manual_edit._split_frontmatter(paper_md.read_text())
    except OSError:
        return False
    if fm is None:
        return False
    try:
        parsed = yaml.safe_load(fm.strip("-\n").strip()) or {}
    except yaml.YAMLError:
        return False
    if not isinstance(parsed, dict):
        return False
    run = parsed.get("run") or {}
    pipeline = run.get("pipeline") or {}
    return bool(pipeline.get("vlm_rewrite_tables"))


def _html_to_pipe_md_safe(html: Optional[str]) -> tuple[str, str]:
    """Return (sidecar_content, source_tag). Source is one of
    'mineru_html' (clean conversion), 'mineru_html_raw' (rowspan/-
    colspan defeated the converter; HTML preserved verbatim for the
    user to clean), 'mineru_html_empty' (no HTML in the block)."""
    if not html:
        return ("<!-- MinerU did not produce a table HTML body. -->\n",
                "mineru_html_empty")
    try:
        import wrap_mineru
    except Exception as e:  # pragma: no cover -- defensive
        log.warning("wrap_mineru import failed (%s); writing raw HTML.", e)
        return (html, "mineru_html_raw")
    pipe_md, reason = wrap_mineru.html_to_pipe_md(html)
    if pipe_md:
        return (pipe_md.rstrip() + "\n", "mineru_html")
    # Conversion gave up -- preserve raw HTML in a comment header.
    return (
        f"<!-- MinerU table has rowspan/colspan that pipe-md can't "
        f"represent ({reason}). Raw HTML preserved -- clean manually. "
        f"-->\n{html}\n",
        "mineru_html_raw",
    )


def _check_not_confirmed(fm: str, kind: str, id_: str) -> None:
    """Refuse to re-stage a recovery the user has already confirmed
    (the entry got promoted to `edits:` and the user presumably pasted).
    Re-running would clobber their work silently."""
    existing_edits = manual_edit._extract_block_entries(fm, "edits")
    for e in existing_edits:
        if (e.get("kind") == kind
                and str(e.get("id")) == str(id_)
                and e.get("recovered_from") == "mineru"):
            raise RuntimeError(
                f"Recovery for {kind} {id_} already confirmed in the "
                f"`edits:` frontmatter block. Run "
                f"`paper2md --revert-edit {kind.split('_')[0]} {id_} "
                f"{Path('<paper.md>')}` first if you want to redo.")


def stage_one_table(
    paper_md: Path,
    item: dict,
    *,
    use_vlm: bool,
    vlm_callback: Optional[Callable[[Path, Optional[str]],
                                    Optional[str]]] = None,
    vlm_model_name: Optional[Callable[[], str]] = None,
) -> dict:
    # Late-bind the defaults so tests can monkeypatch
    # manual_edit._real_vlm_table_call after import.
    if vlm_callback is None:
        vlm_callback = manual_edit._real_vlm_table_call
    if vlm_model_name is None:
        vlm_model_name = manual_edit._real_vlm_model_name
    """Stage one recoverable table into `assets/` + `pending_recoveries:`
    in the paper's frontmatter. Body of paper_md is NOT modified.

    `item` is one element from `find_recoverable_tables`'s output.
    `use_vlm` controls the sidecar source: True -> VLM transcribes
    the MinerU image; False -> direct HTML conversion. If VLM is
    requested but fails, we fall back to HTML (logged as
    `mineru_html_fallback` source so the user knows what they got).
    """
    table_id = str(item["id"])
    paper_dir = paper_md.parent
    assets_dir = paper_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    fm, body = manual_edit._split_frontmatter(paper_md.read_text())
    if fm is None:
        fm = "---\n---\n"
        body = paper_md.read_text()

    _check_not_confirmed(fm, "table_replacement", table_id)

    # 1. Copy the MinerU image into assets/
    src_image = item.get("image_path")
    if src_image is None or not Path(src_image).exists():
        raise FileNotFoundError(
            f"MinerU image for Table {table_id} missing on disk "
            f"(expected at {src_image}); cannot recover")
    src_image = Path(src_image)
    ext = src_image.suffix.lower().lstrip(".") or "jpg"
    # Sanitize id for filesystem (A.4 -> A_4). Same rule as
    # wrap_mineru's auto sidecars; only the filename gets sanitized,
    # the frontmatter records the original id with its `.`.
    import wrap_mineru as _wm
    safe_id = _wm._sanitize_table_id_for_filename(table_id) or table_id
    asset_name = f"user_edit_table_{safe_id}.{ext}"
    asset_path = assets_dir / asset_name
    shutil.copyfile(src_image, asset_path)
    img_sha = manual_edit._sha256_of(asset_path)

    # 2. Produce sidecar content (VLM transcription or HTML conversion)
    sidecar_name = f"user_edit_table_{safe_id}.md"
    sidecar_path = assets_dir / sidecar_name
    source = "unknown"
    vlm_model = None
    if use_vlm:
        log.info("Recovery: VLM-transcribing Table %s from %s ...",
                 table_id, asset_name)
        try:
            md_table = vlm_callback(asset_path, item.get("caption_text"))
        except Exception as e:
            log.warning("Recovery VLM call raised %s: %s; falling back "
                        "to MinerU HTML.", type(e).__name__, e)
            md_table = None
        if md_table:
            sidecar_path.write_text(md_table.rstrip() + "\n")
            source = "vlm_rewrite"
            vlm_model = vlm_model_name()
        else:
            log.warning("Recovery VLM returned empty for Table %s; "
                        "falling back to MinerU HTML.", table_id)
            content, src_tag = _html_to_pipe_md_safe(item.get("html"))
            sidecar_path.write_text(content)
            # Tag explicitly so the user knows this happened.
            source = "mineru_html_fallback" if src_tag == "mineru_html" \
                else f"{src_tag}_fallback"
    else:
        content, src_tag = _html_to_pipe_md_safe(item.get("html"))
        sidecar_path.write_text(content)
        source = src_tag

    # 3. Append to pending_recoveries: frontmatter block
    caption_line = manual_edit._find_caption_line(body, "table", table_id)
    entry = {
        "kind": "table_replacement",
        "id": table_id,
        "image": f"assets/{asset_name}",
        "image_sha256": img_sha,
        "sidecar": f"assets/{sidecar_name}",
        "source": source,
        "vlm_model": vlm_model,
        "mineru_page_idx": item.get("page_idx"),
        "staged_at": _utc_now_iso(),
        "caption_line": caption_line,
    }
    new_fm = manual_edit._merge_block_entry(fm, entry, PENDING_BLOCK)
    paper_md.write_text(new_fm + body)

    return {
        "id": table_id,
        "asset": asset_path,
        "sidecar": sidecar_path,
        "source": source,
        "caption_line": caption_line,
        "paste_hint": manual_edit._format_table_paste_hint(
            paper_md, table_id, sidecar_path, caption_line),
    }


# ---------------------------------------------------------------------
# Step 3: confirm (promote pending -> edits)
# ---------------------------------------------------------------------


def confirm_recovery(paper_md: Path, kind: str, id_: str) -> dict:
    """Promote a `pending_recoveries:` entry to `edits:` with two added
    fields: `recovered_from: mineru` (audit trail) and `confirmed_at`
    timestamp. The pending entry is removed. Idempotent on the
    edits-side: if an edits entry with the same kind+id already exists,
    that entry is replaced (so a re-confirm refreshes confirmed_at)."""
    if kind not in ("table", "figure"):
        raise ValueError(f"kind must be 'table' or 'figure', got {kind!r}")
    full_kind = f"{kind}_replacement"

    text = paper_md.read_text()
    fm, body = manual_edit._split_frontmatter(text)
    if fm is None:
        raise RuntimeError("no frontmatter present in paper.md")

    new_fm, dropped = manual_edit._drop_block_entry(
        fm, full_kind, str(id_), PENDING_BLOCK)
    if dropped is None:
        raise LookupError(
            f"no pending recovery for {kind} {id_} in {paper_md.name}")

    promoted = dict(dropped)
    promoted["recovered_from"] = "mineru"
    promoted["confirmed_at"] = _utc_now_iso()
    new_fm = manual_edit._merge_block_entry(new_fm, promoted, "edits")
    paper_md.write_text(new_fm + body)
    return {"kind": kind, "id": str(id_), "entry": promoted}


# ---------------------------------------------------------------------
# Step 4: top-level orchestrator
# ---------------------------------------------------------------------


def do_recover_from_mineru(
    paper_md: Path,
    *,
    vlm_override: Optional[bool] = None,
) -> dict:
    """Walk audit's missing-tables list, recover each one for which
    MinerU has a crop. Returns a summary dict suitable for printing.

    `vlm_override`: True -> force VLM, False -> force HTML, None ->
    follow the paper's original `run.pipeline.vlm_rewrite_tables`
    setting (the default; preserves the original quality bar)."""
    if not paper_md.exists():
        raise FileNotFoundError(f"paper not found: {paper_md}")

    if vlm_override is None:
        use_vlm = detect_original_vlm_mode(paper_md)
        vlm_origin = "original"
    else:
        use_vlm = vlm_override
        vlm_origin = "override"

    items = find_recoverable_tables(paper_md)
    if not items:
        return {
            "paper": paper_md,
            "use_vlm": use_vlm,
            "vlm_origin": vlm_origin,
            "recovered": [],
            "summary": "No recoverable tables -- audit clean OR "
                       "missing tables are not in MinerU's output.",
        }

    recovered: list[dict] = []
    for item in items:
        try:
            r = stage_one_table(paper_md, item, use_vlm=use_vlm)
            recovered.append(r)
        except Exception as e:
            log.error("Recovery failed for Table %s: %s", item["id"], e)
            recovered.append({
                "id": item["id"],
                "error": str(e),
            })
    return {
        "paper": paper_md,
        "use_vlm": use_vlm,
        "vlm_origin": vlm_origin,
        "recovered": recovered,
        "summary": f"{len([r for r in recovered if 'error' not in r])} "
                   f"of {len(items)} table(s) staged "
                   f"(VLM={'on' if use_vlm else 'off'}, "
                   f"source={vlm_origin}).",
    }
