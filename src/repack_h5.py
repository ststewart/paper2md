#!/usr/bin/env python3
"""repack_h5: rebuild an HDF5 bundle from edited loose files.

After running paper2md.py with --hdf5 you have a bundle plus the loose
markdown + assets/ on disk. If you edit the .md file (fix an extracted
table, correct an OCR slip, polish the citation) the bundle goes stale.
This script reads the current on-disk content and writes a new bundle in
place, preserving the original per-document metadata (source_pdf,
markdown_path, overall, grade) read from the existing bundle.

Usage:
    python repack_h5.py paper.h5
        Reads <stem>.md (and optional <si-stem>.md) and assets/ from the
        directory containing paper.h5; rewrites paper.h5 atomically.

    python repack_h5.py paper.h5 --from /path/to/edited/dir
        Reads loose files from a different directory.

    python repack_h5.py paper.h5 -o new_paper.h5
        Write to a different output path (leaves the input bundle alone).

What changes in the new bundle:
    - markdown content (and assets/*.md) — read fresh from disk
    - assets binary content (jpegs, pngs) — read fresh from disk
    - root attrs `created_at` -> now; `tool` -> "repack_h5.py"
    - root attr `original_created_at` -> previous bundle's created_at
      (added so consumers can spot a repack)

What is preserved unchanged from the old bundle:
    - root attrs: schema_version, vlm_model
    - per-doc attrs: source_pdf, markdown_path, overall, grade

Depends only on h5py + numpy + stdlib; no paper2md.py import.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SUPPORTED_SCHEMA_VERSIONS = {1}


def _attr(obj, name: str, default=None):
    if name not in obj.attrs:
        return default
    v = obj.attrs[name]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def _read_str(ds) -> str:
    accessor = ds.asstr()
    if ds.ndim == 0:
        return accessor[()]
    return accessor[0]


def _copy_doc(src_h5, group: str, dst_h5, src_dir: Path,
              str_dtype) -> tuple[str, str]:
    """Copy /<group> from src_h5 to dst_h5, but read the markdown text
    from <src_dir>/<markdown_path>. Also reads the .meta.json sidecar
    next to the .md (file <stem>.meta.json) when present. Returns
    (md_filename, summary)."""
    if group not in src_h5:
        raise SystemExit(f"bundle missing /{group} group; cannot repack")
    g_old = src_h5[group]
    md_name = _attr(g_old, "markdown_path")
    if not md_name:
        raise SystemExit(f"/{group} has no markdown_path attr; "
                         "old bundle predates the schema this script supports")
    md_path = src_dir / md_name
    if not md_path.is_file():
        raise SystemExit(f"expected markdown not found: {md_path}")

    text = md_path.read_text(encoding="utf-8")
    g_new = dst_h5.create_group(group)
    g_new.create_dataset("markdown", data=[text], dtype=str_dtype,
                         compression="gzip", compression_opts=9)

    # Carry the .meta.json sidecar through if it exists on disk OR was
    # in the source bundle. Disk wins (a user may have hand-edited the
    # JSON the same way they hand-edit the .md).
    meta_name = (_attr(g_old, "meta_json_path")
                 or (md_path.stem + ".meta.json"))
    meta_path = src_dir / meta_name
    meta_text = None
    if meta_path.is_file():
        meta_text = meta_path.read_text(encoding="utf-8")
    elif "meta_json" in g_old:
        meta_text = _read_str(g_old["meta_json"])
    if meta_text is not None:
        g_new.create_dataset("meta_json", data=[meta_text], dtype=str_dtype,
                             compression="gzip", compression_opts=9)
        g_new.attrs["meta_json_path"] = meta_name

    for k in ("source_pdf", "markdown_path", "overall", "grade"):
        v = _attr(g_old, k)
        if v is not None:
            g_new.attrs[k] = v

    grade = _attr(g_old, "grade", "?")
    overall = float(_attr(g_old, "overall", 0.0))
    meta_tag = " + meta.json" if meta_text is not None else ""
    summary = (f"{group:11s} <- {md_name}{meta_tag}  "
               f"({len(text):,} chars, grade {grade}, score {overall:.2f})")
    return md_name, summary


def _pack_assets(dst_h5, assets_dir: Path, str_dtype) -> tuple[int, int]:
    """Walk assets_dir and mirror every .jpg/.jpeg/.png/.md into
    /assets/. Returns (file_count, byte_count)."""
    g = dst_h5.create_group("assets")
    if not assets_dir.is_dir():
        g.attrs["count"] = 0
        return 0, 0

    kept = 0
    total = 0
    skipped: list[str] = []
    for asset in sorted(assets_dir.iterdir()):
        if not asset.is_file():
            continue
        suffix = asset.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            raw = asset.read_bytes()
            data = np.frombuffer(raw, dtype=np.uint8)
            g.create_dataset(asset.name, data=data)
            kept += 1
            total += len(raw)
        elif suffix == ".md":
            text = asset.read_text(encoding="utf-8")
            g.create_dataset(asset.name, data=[text], dtype=str_dtype,
                             compression="gzip", compression_opts=9)
            kept += 1
            total += len(text.encode("utf-8"))
        else:
            skipped.append(asset.name)
    g.attrs["count"] = kept
    if skipped:
        print(f"  skipped {len(skipped)} non-standard asset(s): "
              f"{', '.join(skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}")
    return kept, total


def repack(bundle: Path, src_dir: Path, out_path: Path) -> None:
    if not bundle.is_file():
        raise SystemExit(f"bundle not found: {bundle}")
    if not src_dir.is_dir():
        raise SystemExit(f"source dir not found: {src_dir}")

    # Write to a sibling .tmp file then rename, so a crash mid-write
    # doesn't leave the user with a half-written bundle in place of the
    # working one. h5py opens in "w" mode (truncating), but we only
    # touch the real target on rename.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with h5py.File(bundle, "r") as src:
        schema = int(_attr(src, "schema_version", 0) or 0)
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise SystemExit(
                f"schema_version={schema} not in {sorted(SUPPORTED_SCHEMA_VERSIONS)}; "
                "this repacker only handles bundles produced by paper2md.py "
                f"schema versions {sorted(SUPPORTED_SCHEMA_VERSIONS)}.")
        original_created = _attr(src, "created_at", "?")
        vlm_model = _attr(src, "vlm_model", "?")

        print(f"Repacking {bundle}")
        print(f"  source dir: {src_dir}")
        print(f"  schema {schema}, original vlm: {vlm_model}, "
              f"original created: {original_created}")

        with h5py.File(tmp_path, "w") as dst:
            dst.attrs["schema_version"] = schema
            dst.attrs["tool"] = "repack_h5.py"
            dst.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
            dst.attrs["original_created_at"] = original_created
            dst.attrs["vlm_model"] = vlm_model

            str_dtype = h5py.string_dtype("utf-8")

            print("Documents:")
            _, summary = _copy_doc(src, "main", dst, src_dir, str_dtype)
            print(f"  {summary}")
            if "supplement" in src:
                _, summary = _copy_doc(src, "supplement", dst, src_dir, str_dtype)
                print(f"  {summary}")

            print(f"Assets <- {src_dir / 'assets'}/:")
            kept, total = _pack_assets(dst, src_dir / "assets", str_dtype)
            print(f"  packed {kept} file(s), {total:,} bytes")

    os.replace(tmp_path, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("bundle", type=Path,
                   help="path to the existing .h5 bundle (read for "
                        "metadata; rewritten in place unless -o is given)")
    p.add_argument("--from", dest="src_dir", type=Path, default=None,
                   metavar="DIR",
                   help="directory holding the edited markdown and "
                        "assets/ (default: parent of BUNDLE)")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write the new bundle to this path instead of "
                        "overwriting BUNDLE")
    args = p.parse_args()

    src_dir = args.src_dir if args.src_dir is not None else args.bundle.parent
    out_path = args.out if args.out is not None else args.bundle
    repack(args.bundle, src_dir, out_path)


if __name__ == "__main__":
    main()
