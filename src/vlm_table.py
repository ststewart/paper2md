#!/usr/bin/env python3
"""Standalone VLM table transcription.

One-off extraction of a single table image to markdown pipe-md OR
RFC 4180 CSV. Reuses paper2md's VLM client configuration -- pick
provider via the same env vars (`VLM_PROVIDER`, `VLM_MODEL`,
`VLLM_BASE_URL`, `LM_STUDIO_URL`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`).

Usage:
    python -m vlm_table table.png                  # markdown to stdout
    python -m vlm_table table.png -f csv           # CSV to stdout
    python -m vlm_table table.png -o result.md     # markdown to file
    python -m vlm_table table.png -f csv -o data.csv
    python -m vlm_table table.png -c "Caption text"   # caption hint

After `pip install -e .`, also available as `vlm-table`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import paper2md as p2m

log = logging.getLogger("vlm_table")

# Reuses paper2md's TABLE_PROMPT for markdown (same prompt the
# in-pipeline table-rewrite uses, so output style stays consistent
# across the codebase). CSV uses its own prompt below.

TABLE_PROMPT_CSV = (
    "This image is one table cropped from a scientific document. "
    "Convert it to RFC 4180 CSV. Rules: "
    "(1) Output the HEADER row first; one row per data row. "
    "(2) Quote any cell that contains a comma, a double-quote, or a "
    "newline; escape an embedded double-quote by doubling it. "
    "(3) Each row on its own line, terminated by a single newline. "
    "(4) Preserve every row, column, header, and cell exactly. "
    "(5) For numeric values with units or uncertainty notation, keep "
    "them as written (e.g., '12.5 +/- 0.3', '950 GPa', '6.30 +- 0.22'). "
    "(6) Footnote markers (a, b, c, *, dagger, double-dagger) stay "
    "attached to their cell as written. "
    "(7) For columns without an explicit header label, output an empty "
    "header cell (two consecutive commas) rather than inventing a label. "
    "Output ONLY the CSV body -- no preamble, no commentary, no code "
    "fences."
)


def transcribe_table_image(
    image_path: Path,
    fmt: str = "md",
    caption: Optional[str] = None,
    max_tokens: int = 6000,
) -> str:
    """Call the configured VLM with `image_path` and return the table
    transcription as a string. `fmt` is 'md' or 'csv'. `caption`, if
    provided, is prepended to the prompt as context the VLM uses to
    disambiguate (helpful for tables with non-obvious column meaning)."""
    if fmt not in ("md", "csv"):
        raise ValueError(f"fmt must be 'md' or 'csv', got {fmt!r}")
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    prompt = TABLE_PROMPT_CSV if fmt == "csv" else p2m.TABLE_PROMPT
    if caption:
        prompt = (
            f"The table caption is: '{caption.strip()}'. " + prompt
        )
    result = p2m.vlm(prompt, img, max_tokens=max_tokens, max_retries=2)
    if not result:
        raise RuntimeError(
            "VLM returned an empty response. Verify the VLM server is "
            "reachable, the image contains a legible table, and "
            f"`{fmt}` output isn't being silently rejected."
        )
    text = result.strip()
    # Defensive: strip a leading code-fence if the model emitted one
    # despite the "no fences" instruction.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0].rstrip()
    return text + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="vlm-table",
        description=__doc__.split("\n\n", 1)[0],
    )
    p.add_argument("image", type=Path,
                   help="image file containing a single table "
                        "(.png, .jpg, .jpeg, .webp)")
    p.add_argument("-f", "--format", dest="fmt",
                   choices=["md", "csv"], default="md",
                   help="output format: markdown pipe-table (default) "
                        "or RFC 4180 CSV")
    p.add_argument("-o", "--output", type=Path, default=None,
                   metavar="FILE",
                   help="write to FILE instead of stdout")
    p.add_argument("-c", "--caption", default=None, metavar="TEXT",
                   help="optional caption text to prepend to the "
                        "prompt as VLM context (helps with ambiguous "
                        "column meanings)")
    p.add_argument("--max-tokens", type=int, default=6000,
                   help="VLM max output tokens (default 6000; bump for "
                        "very wide tables that get truncated)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="DEBUG-level logging")
    args = p.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not args.image.is_file():
        p.error(f"image not found: {args.image}")

    # configure_client() reads $VLM_PROVIDER + per-provider env vars and
    # initializes the right SDK client. Same setup as the main pipeline.
    p2m.configure_client()

    try:
        text = transcribe_table_image(
            args.image, fmt=args.fmt, caption=args.caption,
            max_tokens=args.max_tokens,
        )
    except RuntimeError as e:
        log.error("%s", e)
        return 2

    if args.output:
        args.output.write_text(text)
        log.info("Wrote %d bytes to %s", len(text), args.output)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
