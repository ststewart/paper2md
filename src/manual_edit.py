"""User-driven manual replacement of a table or figure in an existing
paper2md `.md` output.

Use case: the splice produced a garbled table or selected the wrong
figure asset. The user crops a clean image of the correct
table/figure and points this tool at the paper. For tables, the VLM
transcribes the crop to pipe-md and writes a sidecar `.md` for the
user to paste in. For figures, the asset is swapped in-place
(original archived as `.orig`). In both cases the action is logged
under a top-level `edits:` block in the paper's frontmatter so the
provenance is auditable.

CLI entry points live in `paper2md.py`:
    paper2md --replace-table <ID> <image> <paper.md> [--note "..."]
    paper2md --replace-fig   <ID> <image> <paper.md> [--note "..."]
    paper2md --revert-edit   <kind> <ID> <paper.md>

The body is NEVER modified for tables -- only the assets/ folder and
the frontmatter. The user pastes the sidecar contents manually; this
deliberately stops short of a fragile auto-replace.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

log = logging.getLogger("manual_edit")


# ---------------------------------------------------------------------
# Frontmatter manipulation
# ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Compact UTC timestamp matching the rest of paper2md's frontmatter."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_frontmatter(text: str) -> tuple[Optional[str], str]:
    """Return (frontmatter_inclusive, body) or (None, text) if no FM."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    fm = text[:end + 5]
    body = text[end + 5:]
    return fm, body


# A top-level YAML block (e.g. `edits:` or `pending_recoveries:`).
# PyYAML safe_dump emits list children with the `- ` marker at column 0
# (not indented), so the alternation accepts: (a) a list-item start
# `- <content>`, (b) an indented continuation line, or (c) a blank
# line. Terminates at the next top-level key, the closing `---`, or
# EOF. The closing `---` delimiter doesn't match `-[ \t]` because
# after the first dash it has another dash, not whitespace.
def _block_re(name: str) -> "re.Pattern[str]":
    return re.compile(
        r"(?m)^" + re.escape(name) + r":[ \t]*\n"
        r"(?:(?:-[ \t].*\n)|(?:[ \t]+\S.*\n)|(?:[ \t]*\n))*",
    )


_EDITS_BLOCK_RE = _block_re("edits")


def _extract_block_entries(fm: str, block_name: str = "edits") -> list[dict]:
    """Find a top-level block by name, parse, return its list (or [])."""
    pat = _block_re(block_name)
    m = pat.search(fm)
    if not m:
        return []
    block = m.group(0)
    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError as e:
        log.warning("Could not parse existing `%s:` block (%s); "
                    "previous entries will be lost.", block_name, e)
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get(block_name), list):
        return parsed[block_name]
    return []


# Backward-compatible alias used by older call-sites in this module.
def _extract_existing_edits(fm: str) -> list[dict]:
    return _extract_block_entries(fm, "edits")


def _emit_block_yaml(block_name: str, entries: list[dict]) -> str:
    """Render a top-level list-of-dicts block as YAML. sort_keys=False
    keeps human ordering."""
    return yaml.safe_dump({block_name: entries},
                          sort_keys=False,
                          default_flow_style=False,
                          allow_unicode=True)


def _merge_block_entry(fm: str, new_entry: dict,
                       block_name: str = "edits") -> str:
    """Idempotent merge in any top-level block: an existing entry with
    the same kind+id is replaced; else the new entry is appended. The
    block is always re-emitted at the end (just before closing `---`)."""
    existing = _extract_block_entries(fm, block_name)
    kept = [
        e for e in existing
        if not (e.get("kind") == new_entry["kind"]
                and str(e.get("id")) == str(new_entry["id"]))
    ]
    kept.append(new_entry)
    fm_no_block = _block_re(block_name).sub("", fm)
    return _inject_block(fm_no_block, _emit_block_yaml(block_name, kept))


# Backward-compatible wrapper -- preserves the old name for callers in
# this module.
def _merge_edits_block(fm: str, new_entry: dict) -> str:
    return _merge_block_entry(fm, new_entry, "edits")


def _drop_block_entry(fm: str, kind: str, id_: str,
                      block_name: str = "edits"
                      ) -> tuple[str, Optional[dict]]:
    """Remove the entry matching kind+id from any top-level block.
    Returns (new_fm, dropped_entry); dropped is None on no match."""
    existing = _extract_block_entries(fm, block_name)
    dropped = None
    kept = []
    for e in existing:
        if (e.get("kind") == kind and str(e.get("id")) == str(id_)
                and dropped is None):
            dropped = e
            continue
        kept.append(e)
    fm_no_block = _block_re(block_name).sub("", fm)
    if kept:
        new_fm = _inject_block(
            fm_no_block, _emit_block_yaml(block_name, kept))
    else:
        new_fm = fm_no_block
    return new_fm, dropped


def _drop_edits_entry(fm: str, kind: str, id_: str
                      ) -> tuple[str, Optional[dict]]:
    return _drop_block_entry(fm, kind, id_, "edits")


def _inject_block(fm_no_block: str, block_yaml: str) -> str:
    """Place a YAML block immediately before the closing `---`."""
    if fm_no_block.endswith("---\n"):
        head = fm_no_block[:-4]
        if head and not head.endswith("\n"):
            head += "\n"
        return head + block_yaml + "---\n"
    return fm_no_block + block_yaml + "---\n"


# Backward-compat alias.
def _inject_edits(fm_no_edits: str, edits_yaml: str) -> str:
    return _inject_block(fm_no_edits, edits_yaml)


# ---------------------------------------------------------------------
# Body inspection helpers
# ---------------------------------------------------------------------


def _find_caption_line(body: str, kind: str, id_: str) -> Optional[int]:
    """1-based line number of the canonical caption matching kind+id."""
    if kind == "table":
        pat = re.compile(
            r"(?i)^\s*\**\s*Table\s+" + re.escape(id_)
            + r"\b\s*[:.\*\|]",
        )
    else:
        pat = re.compile(
            r"(?i)^\s*\**\s*(?:Extended\s+Data\s+)?Fig(?:ure|\.)\s+"
            + re.escape(id_) + r"\b",
        )
    for i, line in enumerate(body.splitlines(), 1):
        if pat.search(line):
            return i
    return None


_IMG_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$")


def _find_figure_asset(body: str, fig_id: str,
                       assets_dir: Path) -> Optional[Path]:
    """Resolve the on-disk asset bound to a figure id. Two strategies:
    (1) caption-proximity -- find the `Fig. <id>` caption, scan a small
    window around it for the nearest `![](href)` image line; (2)
    convention fallback -- `assets/figure_<id>.{jpg,jpeg,png,webp}`."""
    lines = body.splitlines()
    cap_pat = re.compile(
        r"(?i)^\s*\**\s*(?:Extended\s+Data\s+)?Fig(?:ure|\.)\s+"
        + re.escape(fig_id) + r"\b",
    )
    for i, line in enumerate(lines):
        if not cap_pat.search(line):
            continue
        window = (list(range(max(0, i - 8), i))
                  + list(range(i + 1, min(len(lines), i + 8))))
        for j in window:
            m = _IMG_LINE_RE.match(lines[j])
            if not m:
                continue
            href = m.group(1)
            paper_dir = assets_dir.parent
            cand = (paper_dir / href).resolve()
            if cand.exists():
                return cand
            base = href.rsplit("/", 1)[-1]
            cand = assets_dir / base
            if cand.exists():
                return cand
        break
    for ext in ("jpg", "jpeg", "png", "webp"):
        cand = assets_dir / f"figure_{fig_id}.{ext}"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------
# VLM callback (injectable for tests)
# ---------------------------------------------------------------------


def _real_vlm_table_call(image_path: Path,
                         caption: Optional[str]) -> Optional[str]:
    """Production VLM caller: imports paper2md, reuses TABLE_PROMPT."""
    from PIL import Image
    import paper2md as p2m
    img = Image.open(image_path).convert("RGB")
    prompt = p2m.TABLE_PROMPT
    if caption:
        prompt = f"The table caption is: '{caption.strip()}'. " + prompt
    result = p2m.vlm(prompt, img, max_tokens=6000, max_retries=2)
    if not result:
        return None
    md = result.strip()
    pipe_start = md.find("\n|")
    if pipe_start > 0 and not md[:pipe_start].lstrip().startswith("|"):
        md = md[pipe_start + 1:]
    return md.strip() or None


def _real_vlm_model_name() -> str:
    try:
        import paper2md as p2m
        return getattr(p2m, "VLM_MODEL", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def do_replace_table(
    paper_md: Path,
    table_id: str,
    image: Path,
    note: Optional[str] = None,
    *,
    vlm_callback: Callable[[Path, Optional[str]],
                           Optional[str]] = _real_vlm_table_call,
    vlm_model_name: Callable[[], str] = _real_vlm_model_name,
) -> dict:
    """Replace a table by VLM-transcribing the user's image crop.

    Side effects:
      - Image is copied into `assets/user_edit_table_<id>.<ext>`.
      - A sidecar `assets/user_edit_table_<id>.md` is written with
        the VLM transcription (pipe-md).
      - `paper.md` frontmatter gets/updates an `edits:` entry.
      - The paper body is NOT modified -- the user pastes manually.
    """
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    if not paper_md.exists():
        raise FileNotFoundError(f"paper not found: {paper_md}")

    paper_dir = paper_md.parent
    assets_dir = paper_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    ext = image.suffix.lower().lstrip(".") or "png"
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "png"
    # Sanitize the id for filesystem safety (A.4 -> A_4), matching
    # the convention used by wrap_mineru's auto-generated sidecars.
    # The human-readable id (with `.`) is still recorded in the
    # frontmatter `edits:` entry; only the filename gets sanitized.
    import wrap_mineru as _wm
    safe_id = _wm._sanitize_table_id_for_filename(table_id) or table_id
    asset_image_name = f"user_edit_table_{safe_id}.{ext}"
    asset_image = assets_dir / asset_image_name
    shutil.copyfile(image, asset_image)
    img_sha = _sha256_of(asset_image)

    log.info("Calling VLM to transcribe table %s from %s ...",
             table_id, asset_image_name)
    md_table = vlm_callback(asset_image, None)
    if not md_table:
        raise RuntimeError(
            f"VLM returned empty result for table {table_id}; "
            "verify the VLM server is reachable and the crop "
            "contains a legible table.")

    sidecar_name = f"user_edit_table_{safe_id}.md"
    sidecar = assets_dir / sidecar_name
    sidecar.write_text(md_table.rstrip() + "\n")

    text = paper_md.read_text()
    fm, body = _split_frontmatter(text)
    if fm is None:
        fm = "---\n---\n"

    entry = {
        "kind": "table_replacement",
        "id": str(table_id),
        "image": f"assets/{asset_image_name}",
        "image_sha256": img_sha,
        "sidecar": f"assets/{sidecar_name}",
        "vlm_model": vlm_model_name(),
        "timestamp": _utc_now_iso(),
    }
    if note:
        entry["note"] = note

    new_fm = _merge_edits_block(fm, entry)
    paper_md.write_text(new_fm + body)

    cap_line = _find_caption_line(body, "table", str(table_id))
    return {
        "kind": "table",
        "id": str(table_id),
        "image": asset_image,
        "sidecar": sidecar,
        "caption_line": cap_line,
        "paste_hint": _format_table_paste_hint(
            paper_md, table_id, sidecar, cap_line),
    }


def do_replace_fig(
    paper_md: Path,
    fig_id: str,
    image: Path,
    note: Optional[str] = None,
) -> dict:
    """Swap a figure's asset in-place. Archives the original as `.orig`
    (only on first run) and overwrites the bound path. No VLM call.
    """
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    if not paper_md.exists():
        raise FileNotFoundError(f"paper not found: {paper_md}")

    paper_dir = paper_md.parent
    assets_dir = paper_dir / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"no assets/ folder at {assets_dir}; not a paper2md output?")

    text = paper_md.read_text()
    fm, body = _split_frontmatter(text)
    if fm is None:
        fm = "---\n---\n"

    target = _find_figure_asset(body, str(fig_id), assets_dir)
    if target is None:
        raise FileNotFoundError(
            f"could not locate the asset bound to Fig. {fig_id} in "
            f"{paper_md}. The caption may be missing, or no `![](...)` "
            f"image line is near it. --replace-fig requires the "
            f"figure to already be inserted; if it's missing entirely, "
            f"a manual paste of the sidecar caption + image line is "
            f"needed instead.")

    orig = target.with_suffix(target.suffix + ".orig")
    archived = False
    if not orig.exists():
        shutil.copyfile(target, orig)
        archived = True

    shutil.copyfile(image, target)
    img_sha = _sha256_of(target)

    entry = {
        "kind": "figure_replacement",
        "id": str(fig_id),
        "image": f"assets/{target.name}",
        "image_sha256": img_sha,
        "original_archived_as": (f"assets/{orig.name}"
                                 if orig.exists() else None),
        "timestamp": _utc_now_iso(),
    }
    if note:
        entry["note"] = note

    new_fm = _merge_edits_block(fm, entry)
    paper_md.write_text(new_fm + body)

    return {
        "kind": "figure",
        "id": str(fig_id),
        "image": target,
        "archived_original": orig if archived else None,
    }


def do_revert(paper_md: Path, kind: str, id_: str) -> dict:
    """Revert a recorded edit.

    For `figure`: restore `.orig` over the bound asset, delete the
    archived original, drop the frontmatter entry.
    For `table`: drop the frontmatter entry. The sidecar `.md` and the
    asset crop stay on disk (the user may have pasted from them and we
    can't safely unpaste). Effectively a "discard intent" operation.
    """
    if not paper_md.exists():
        raise FileNotFoundError(f"paper not found: {paper_md}")
    if kind not in ("table", "figure"):
        raise ValueError(f"kind must be 'table' or 'figure', got {kind!r}")

    text = paper_md.read_text()
    fm, body = _split_frontmatter(text)
    if fm is None:
        raise RuntimeError("no frontmatter present; nothing to revert")

    full_kind = f"{kind}_replacement"
    new_fm, dropped = _drop_edits_entry(fm, full_kind, str(id_))
    if dropped is None:
        raise LookupError(
            f"no recorded edit for kind={kind} id={id_}")

    restored: Optional[Path] = None
    if kind == "figure":
        orig_rel = dropped.get("original_archived_as")
        target_rel = dropped.get("image")
        if orig_rel and target_rel:
            paper_dir = paper_md.parent
            orig = paper_dir / orig_rel
            target = paper_dir / target_rel
            if orig.exists():
                shutil.copyfile(orig, target)
                orig.unlink()
                restored = target

    paper_md.write_text(new_fm + body)
    return {
        "kind": kind,
        "id": str(id_),
        "restored": restored,
        "dropped_entry": dropped,
    }


# ---------------------------------------------------------------------
# Paste-hint formatting
# ---------------------------------------------------------------------


def _format_table_paste_hint(paper_md: Path, table_id: str,
                             sidecar: Path,
                             caption_line: Optional[int]) -> str:
    try:
        rel_sidecar = sidecar.relative_to(paper_md.parent)
    except ValueError:
        rel_sidecar = sidecar
    if caption_line is None:
        return (
            f"Sidecar written: {rel_sidecar}\n"
            f"  • Open {paper_md.name}, search for the Table {table_id} "
            f"caption.\n"
            f"  • Replace the garbled table block below the caption "
            f"with the contents of {rel_sidecar}."
        )
    return (
        f"Sidecar written: {rel_sidecar}\n"
        f"  • {paper_md.name}: Table {table_id} caption is at line "
        f"{caption_line}.\n"
        f"  • Replace the garbled table block below it with the "
        f"contents of {rel_sidecar}."
    )
