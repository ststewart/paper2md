#!/usr/bin/env python3
"""unpack_h5: inverse of paper2md.py's --hdf5 flag.

Takes a bundle produced by `paper2md.py --hdf5` and restores the
human-readable outputs: the main markdown, supplement markdown
(if present), and every file under assets/.

Usage:
    python unpack_h5.py paper.h5                  # -> ./paper/
    python unpack_h5.py paper.h5 -o outdir
    python unpack_h5.py paper.h5 --stdout main    # print main markdown

Produces at OUT_DIR (default: <bundle-stem>/):
    <main-stem>.md              main document markdown
    <main-stem>.meta.json       structured frontmatter mirror (when bundled)
    <si-stem>.md                supplement markdown, if the bundle had one
    <si-stem>.meta.json         supplement frontmatter mirror, if present
    assets/
        *.jpg, *.jpeg, *.png    binary (written as-is)
        *.md                    text table sidecars

Depends only on h5py and stdlib; no paper2md.py import.
"""

import argparse
import sys
from pathlib import Path

import h5py


SUPPORTED_SCHEMA_VERSIONS = {1}


def _read_str(ds) -> str:
    """Return a str from an HDF5 string dataset, regardless of whether
    it was stored as a scalar or a 1-element 1-D array."""
    accessor = ds.asstr()
    if ds.ndim == 0:
        return accessor[()]
    return accessor[0]


def _attr(obj, name: str, default=None):
    """Read an HDF5 attribute and coerce bytes -> str for old files."""
    if name not in obj.attrs:
        return default
    v = obj.attrs[name]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def _write_asset(ds, path: Path) -> int:
    """Write an /assets/* dataset to disk. Text (.md) is decoded; other
    suffixes are treated as raw bytes. Returns bytes written."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        text = _read_str(ds)
        path.write_text(text, encoding="utf-8")
        return len(text.encode("utf-8"))
    # Binary: jpeg/png/etc., stored as uint8 arrays
    data = ds[()]
    if hasattr(data, "tobytes"):
        raw = data.tobytes()
    else:
        raw = bytes(data)
    path.write_bytes(raw)
    return len(raw)


def _extract_doc(h5, group: str, out_dir: Path) -> Path:
    """Write /<group>/markdown to <out_dir>/<markdown_path>, plus the
    /<group>/meta_json sidecar (when present) to
    <out_dir>/<meta_json_path>. Returns the markdown output path."""
    g = h5[group]
    md_name = _attr(g, "markdown_path", default=f"{group}.md")
    text = _read_str(g["markdown"])
    out_path = out_dir / str(md_name)
    out_path.write_text(text, encoding="utf-8")

    # Structured frontmatter mirror. Symmetric with paper2md.py's
    # write_hdf5_bundle, which always packs this when --hdf5 is on.
    # Defaults the on-disk filename to <stem>.meta.json when the
    # attr is absent (older bundles).
    meta_json_written = False
    if "meta_json" in g:
        meta_name = _attr(g, "meta_json_path",
                          default=f"{Path(str(md_name)).stem}.meta.json")
        meta_text = _read_str(g["meta_json"])
        (out_dir / str(meta_name)).write_text(meta_text, encoding="utf-8")
        meta_json_written = True

    grade = _attr(g, "grade", "?")
    overall = float(_attr(g, "overall", 0.0))
    src = _attr(g, "source_pdf", "?")
    extras = " + meta.json" if meta_json_written else ""
    print(f"  {group:10s} -> {out_path.name}{extras}  "
          f"({len(text):,} chars, grade {grade}, score {overall:.2f}, from {src})")
    return out_path


def unpack(bundle: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    with h5py.File(bundle, "r") as h5:
        schema = int(_attr(h5, "schema_version", 0) or 0)
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            print(f"WARNING: schema_version={schema} not in "
                  f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}; unpacker may "
                  f"not handle it correctly.", file=sys.stderr)
        tool = _attr(h5, "tool", "?")
        created = _attr(h5, "created_at", "?")
        vlm = _attr(h5, "vlm_model", "?")
        print(f"Bundle: {bundle}")
        print(f"  tool: {tool}   created: {created}   vlm: {vlm}")
        print(f"  schema_version: {schema}")

        print("Documents:")
        _extract_doc(h5, "main", out_dir)
        if "supplement" in h5:
            _extract_doc(h5, "supplement", out_dir)

        if "assets" in h5:
            print(f"Assets -> {assets_dir}/:")
            count = 0
            total_bytes = 0
            for name in sorted(h5["assets"]):
                ds = h5[f"assets/{name}"]
                wrote = _write_asset(ds, assets_dir / name)
                total_bytes += wrote
                count += 1
                print(f"  {name}  ({wrote:,} bytes)")
            print(f"  ({count} files, {total_bytes:,} bytes)")


def print_doc_to_stdout(bundle: Path, which: str) -> None:
    """Print just the main or supplement markdown to stdout. Useful for
    piping: `unpack_h5.py paper.h5 --stdout main | grep '^#'`."""
    with h5py.File(bundle, "r") as h5:
        if which not in h5:
            print(f"error: /{which} not present in bundle", file=sys.stderr)
            sys.exit(2)
        sys.stdout.write(_read_str(h5[f"{which}/markdown"]))


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("bundle", type=Path, help="path to the .h5 bundle")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: <bundle-stem>/)")
    p.add_argument("--stdout", choices=["main", "supplement"], default=None,
                   help="instead of unpacking to disk, write the named "
                        "document's markdown to stdout and exit")
    args = p.parse_args()

    if not args.bundle.exists():
        print(f"error: {args.bundle} not found", file=sys.stderr)
        sys.exit(1)

    if args.stdout is not None:
        print_doc_to_stdout(args.bundle, args.stdout)
        return

    out_dir = args.out if args.out is not None else Path(args.bundle.stem)
    unpack(args.bundle, out_dir)


if __name__ == "__main__":
    main()
