#!/usr/bin/env python3
"""paper2md v0.4.0 -- scientific PDF -> Markdown for chunked-by-section
embedding and reproducible re-extraction. MIT License (see LICENSE).
Authors: Sarah T. Stewart; Claude (Anthropic, Opus 4.7).

Marker is the OCR / layout spine; PyMuPDF + a vision LLM handle a
sequence of targeted post-passes for tables, figures, page rescues,
citations, and output bundling. Runs on Apple Silicon (MPS, via LM
Studio), NVIDIA (CUDA, via vLLM), or CPU; --provider also accepts
openai and anthropic for runs against frontier APIs. Backend
(--backend) and provider (--provider) are auto-detected from the
hardware.

Pipeline (every step is optional; the toggle states are recorded in
run.pipeline of the YAML front-matter so a reader can reproduce the
run from the artifact alone):

  Preflight    ping the configured VLM endpoint; fail fast on
               misconfig before the multi-minute marker run.

  Metadata pre-pass (CPU-only, network):
    - DOI / arXiv lookup against OpenAlex, Unpaywall, Europe PMC,
      OSTI / DOE PAGES; resolves license, OA status, and optionally
      swaps to a downloaded OA copy of the PDF.

  Deterministic body pre-passes (no VLM):
    - strip_line_numbers          lineno-package artifacts;
    - strip_running_footers       "AUTHOR ET AL. N of M" lines;
    - strip_journal_page_headers  "LETTERS" / "NATURE GEOSCIENCE
                                  DOI: ..." interleaved page headers;
    - strip_math_labels           KaTeX-incompatible \\label{...} and
                                  \\eqref{...} cross-references;
    - normalise_span_anchored_refs  Elsevier / Icarus refhub cleanup;
    - fix_reference_numbering     back-fill missing ref numbers;
    - consolidate_footnote_references  lift `<sup>N</sup>`-prefixed
                                  footnote-references into a single
                                  ## References section at end;
    - merge_reference_sections    consolidate Nature-style "main +
                                  Methods References" splits into
                                  one ## References at end of doc.
    - trim_to_first_article       VLM-binary-search for concatenated-
                                  article boundary (only pre-pass
                                  that uses the VLM; gated on
                                  use_vlm).

  VLM hooks:
    Hook 0   citation synthesis from page 1 + DOI.
    Hook 1   per-table: locate via table-finder bbox or caption-page
             bypass, render JPEG crop, VLM-rewrite to clean markdown
             + per-table .md sidecar.
    Hook 1.5 orphan-caption rescue: line-leading "Table N." captions
             with no markdown table or sidecar following get re-
             extracted from a page-image VLM call.
    Hook 2   match each extracted image to an author figure caption
             (default `dup-detect` strategy resolves same-caption
             duplicates by reclassifying losers); drop unmatched.
    Hook 3   re-extract pages whose text layer is unusually sparse.

  Post-passes:
    - data-repository link extraction: scan for DOIs / URLs at
      Zenodo, Dryad, Dataverse, figshare, OSF, PANGAEA, ESS-DIVE,
      Mendeley Data, ICPSR, CaltechDATA. --fetch-data-repos opts
      into one HTTP GET per deposit for title / license / file list.
    Hook 4   heuristic quality scoring -> YAML front-matter.

Output:
  <stem>.md          markdown body + multi-block YAML front-matter
                     (copyright, user annotations, run / reproducibility
                     metadata, data-repo links, quality);
  <stem>.meta.json   structured mirror of the front-matter for jq;
  assets/            per-table / per-figure JPEGs + .md sidecars;
  <stem>.h5          optional self-contained HDF5 bundle (--hdf5).

Install: environment-mac.yml (Apple Silicon, pinned marker/surya)
         or environment-gpu.yml (NVIDIA, CUDA 13).

Run
  python paper2md.py paper.pdf -o out/
  python paper2md.py paper.pdf --supplement paper_SI.pdf -o out/
  python paper2md.py --batch corpus/ -o out/ --workers 4
  python paper2md.py paper.pdf --hdf5
  python paper2md.py paper.pdf --no-vlm                  # marker only
  python paper2md.py paper.pdf --text-only               # VLM only
  python paper2md.py paper.pdf --provider openai         # gpt-4.1
  python paper2md.py paper.pdf --provider anthropic      # sonnet 4.6
  python paper2md.py paper.pdf --user "Sarah Stewart" \\
      --collection moon-formation \\
      --note "second pass with looser figure prompt"
  python paper2md.py paper.pdf --fetch-data-repos        # pull deposit summaries
"""

import argparse
import base64
import io
import json
import logging
import os
import platform
import re
import sys

# macOS + conda + pip-installed mineru routinely produces two copies of
# libomp.dylib in the same process (one from PyTorch / surya / Pillow /
# pymupdf via numpy-OpenBLAS, one from PaddlePaddle + opencv-python that
# `pip install mineru[core]` brings in). LLVM OpenMP detects the duplicate
# and aborts with SIGABRT before any work happens (OMP Error #15). Setting
# this BEFORE any C-extension imports lets both copies coexist; the
# theoretical risk of mixed-runtime parallel reductions doesn't apply to
# paper2md's process model (PyTorch and Paddle don't share threadpool-
# resident data; Paddle runs in a separate MinerU subprocess). Use
# setdefault so an explicit user override (set / unset by the shell) wins.
# Gated on Darwin so non-Mac users see no change.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# When run as a script (`python paper2md.py`), Python loads this file as
# `__main__`. Helper modules that do `import paper2md` would then trigger
# a SECOND load under the name `paper2md`, creating two copies of every
# module-level global. `configure_client` mutates the `__main__` copy; the
# helper-side `paper2md.client` keeps its default (lmstudio:1234). Result:
# wrap_mineru's `p2m.vlm()` connects to a port that's not listening. Alias
# `paper2md` -> this module in sys.modules so all imports share one copy.
if __name__ == "__main__":
    sys.modules.setdefault("paper2md", sys.modules["__main__"])

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI, OpenAIError

# Load .env (CWD first, then this script's directory) so API keys / polite-pool
# emails for the metadata front-end are available without shell sourcing.
# Soft-fail if python-dotenv isn't installed; env vars must come from the shell
# in that case.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
    _load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

__version__ = "0.4.0"
__license__ = "MIT"
__authors__ = "Sarah T. Stewart; Claude (Anthropic, Opus 4.7)"
# Per-version Zenodo DOI for citation. Update once per release: reserve
# the DOI on Zenodo (Upload -> New version -> Reserve DOI), paste the
# bare 10.5281/zenodo.NNNNN identifier here, commit, tag, push -- the
# GitHub-Zenodo integration then publishes the release WITH the
# reserved DOI so this string stays canonical. The placeholder
# "10.5281/zenodo.RESERVED" means the citation has not yet been wired
# to a Zenodo deposit; the runtime banner falls back to a no-DOI
# citation when it sees the placeholder.
__doi__ = "10.5281/zenodo.20262917"
__citation__ = (
    "Stewart, S. T., & Claude (Anthropic, Opus 4.7). (2026). "
    "paper2md (v{version}) [Software]. {license} License."
)
__citation_with_doi__ = (
    "Stewart, S. T., & Claude (Anthropic, Opus 4.7). (2026). "
    "paper2md (v{version}) [Software]. {license} License. "
    "https://doi.org/{doi}"
)

log = logging.getLogger("paper2md")

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

# Provider selection: lmstudio (default, local), openai, or anthropic.
# CLI `--provider` can override VLM_PROVIDER at runtime via configure_client().
_PROVIDER = os.environ.get("VLM_PROVIDER", "lmstudio").lower()

# Per-provider default model (overridable via VLM_MODEL env var or CLI).
_PROVIDER_DEFAULT_MODEL = {
    "lmstudio": "qwen3-vl-32b-instruct-mlx",
    # BF16, not FP8: vllm 0.19.1's CUTLASS FP8 GEMM kernel crashes on the
    # Spark's GB10 (sm_121, Blackwell desktop). BF16 uses standard PyTorch
    # matmul which works on sm_121 via PTX. Override with $VLM_MODEL when a
    # working sm_121 FP8 path lands.
    "vllm": "Qwen/Qwen3-VL-32B-Instruct",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}
VLM_MODEL = os.environ.get("VLM_MODEL", _PROVIDER_DEFAULT_MODEL.get(_PROVIDER, "qwen3-vl-32b-instruct-mlx"))

# Deterministic sampling defaults sent on EVERY VLM request. Was
# previously NOT set; servers fell back to OpenAI-compat defaults
# (temperature=1.0, seed=None) which produced run-to-run drift on
# the same paper. Pinning here makes paper2md output reproducible
# end-to-end (modulo floating-point ordering inside the model).
# Overridable via env vars or CLI flags (--vlm-temperature /
# --vlm-seed). Recorded in every output's `run:` frontmatter for
# publication-grade reproducibility (USAGE.md §18).
#
# Anthropic accepts temperature but NOT seed; the Anthropic branch of
# vlm() silently drops the seed and logs a one-shot warning at
# configure_client time.
try:
    VLM_TEMPERATURE = float(os.environ.get("VLM_TEMPERATURE", "0.0"))
except ValueError:
    log.warning("$VLM_TEMPERATURE is not a valid float; using 0.0")
    VLM_TEMPERATURE = 0.0
try:
    VLM_SEED: Optional[int] = int(os.environ.get("VLM_SEED", "42"))
except ValueError:
    log.warning("$VLM_SEED is not a valid int; using 42")
    VLM_SEED = 42

# Markdown table: header row, separator row, then >=1 body rows.
TABLE_RE = re.compile(
    r"(?:^\|[^\n]*\|[ \t]*\n)"
    r"(?:^\|[-: |]+\|[ \t]*\n)"
    r"(?:^\|[^\n]*\|[ \t]*\n?)+",
    re.MULTILINE,
)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Author figure-caption patterns. Two journal conventions in the wild:
#   Nature-style:   "Figure 1 | Title..." or "**Figure 1** | **Title...**"
#                   -- the pipe separator is the identifying feature.
#   Elsevier-style: "**Fig. 1.** Title..." -- the number-then-period is
#                   entirely inside the bold span, title follows.
# Both patterns exclude inline references like "Fig. 2 shows..." which
# lack the pipe / bold-close-after-period.
# The figure ID capture is `S?\d+` so both main-paper ("Figure 1") and
# supplement ("Figure S1", "Fig. S3") numbering are handled by the same
# regex; figure IDs flow through the pipeline as strings.
# Figure-id alternation accepts either a plain integer ("12"),
# a supplementary "S"-prefixed integer ("S1", "S12"), an
# appendix-style "Letter.Number" ("B.13", "D.14") used by some
# Elsevier journals (e.g. jacquet's Icarus paper), or the
# AGU/Wiley appendix style "LetterNumber" without a separator
# ("A1", "B1") used in journals like JGR/Geophysical Research
# Letters. Captured group is the raw id string; the captions
# dict keys it directly so downstream code distinguishes
# "1" / "S1" / "B.13" / "A1" by inspection.
FIG_ID_PAT = r"(?:S?\d+|[A-Z]\d+|[A-Z]\.\d+)"

# All three "main figure" patterns include a (?<!Extended Data ) and
# (?<!Ext\. Data ) negative lookbehind so they don't capture Nature's
# "Extended Data Fig. N" inline references or captions as if they were
# main-paper figures. Those are caught by the dedicated ED patterns
# below and stored under "ED1", "ED2", ... fig_ids so they don't
# collide with main-paper "1", "2". The 500-char title cap accommodates
# long math-heavy captions (Nature ED figures, Elsevier appendices).
# The (?<!\!\[) lookbehind keeps PIPE (and ED_PIPE) from matching
# inside markdown image alt-text "![Figure 4 | Title](assets/...)" --
# without it the lazy title match runs past the closing "]" into the
# URL until it hits the ".jpeg" period, swallowing the asset path.
FIG_CAPTION_RE_PIPE = re.compile(
    r"(?<!Extended Data )(?<!Ext\. Data )(?<!\!\[)(?:\*\*)?(?:Figure|Fig\.?)\s+(" + FIG_ID_PAT + r")(?:\*\*)?\s*\|\s*(?:\*\*)?([^.\n]{3,500}?)(?=[.*\n]|$)",
    re.IGNORECASE,
)
FIG_CAPTION_RE_ELSEVIER = re.compile(
    r"(?<!Extended Data )(?<!Ext\. Data )\*\*\s*(?:Figure|Fig\.?)\s+(" + FIG_ID_PAT + r")\s*\.\s*\*\*\s*([^.\n]{3,500}?)(?=[.\n]|$)",
    re.IGNORECASE,
)
# AGU / some Nature derivatives bold the entire "Figure N. Title" block
# instead of just "Figure N.". The closing ** sits AFTER the title (with
# or without a trailing period inside the bold span). Captures everything
# between the period and the closing ** as the title; extract_figure_captions
# strips a trailing period if present.
#   "**Figure 1. Experimental configuration.** (a) ..."   → id=1, title=Experimental configuration
#   "**Figure 2. VISAR data** Raw (Top) ..."              → id=2, title=VISAR data
FIG_CAPTION_RE_BOLD_BLOCK = re.compile(
    r"(?<!Extended Data )(?<!Ext\. Data )\*\*\s*(?:Figure|Fig\.?)\s+(" + FIG_ID_PAT + r")\s*\.\s*([^*\n]{3,500}?)\s*\*\*",
    re.IGNORECASE,
)
# Plain-text figure caption (no bold, no pipe). Anchored at line
# start with optional marker span anchor; the line-start anchor
# keeps inline body references ("We see in Fig. 5 that...") from
# triggering. Body references typically lack the period right
# after the figure number, but to be doubly safe we require the
# leading <span> *or* line-start position.
#   <span id="page-4-0"></span>Fig. 5. Mesostasis/glass Na/Al vs. SiO2...   (Elsevier with anchor)
#   Fig. B.13. Mesostasis K/Na ratio vs. silica.                          (Elsevier appendix, no anchor)
# The optional "(...)" between the figure id and the period catches
# APS-style qualifiers ("FIG. 1 (color).", "Fig. 2 (Color online).")
# common in PRL/PRB and other Physical Review journals; without it
# the regex only matches the rare unqualified caption.
# Separator is "period+space" OR "space-before-Capital-or-paren": the
# second alternative catches captions where the post-id period was
# dropped by older OCR ("Fig. 2 Measured shock..." in Boslough 1988)
# while still rejecting body refs whose next word is lowercase
# ("Fig. 5 shows...", "Fig. 5 of Smith [1990]"). The (?-i:...)
# inline group disables IGNORECASE just for that lookahead so
# "Fig. 2 measured" (lowercase 'm') won't slip through.
FIG_CAPTION_RE_PLAIN = re.compile(
    r'^(?:<span\s+id="page-\d+-\d+"></span>\s*)?'
    r'(?:Figure|Fig\.?)\s+(' + FIG_ID_PAT + r')'
    r'(?:\s+\([^)\n]{1,40}\))?'
    r'(?:\s*\.\s+|\s+(?=(?-i:[A-Z(])))'
    r'([^.\n]{3,500}?)(?=[.\n]|$)',
    re.IGNORECASE | re.MULTILINE,
)
# Nature "Extended Data" figures. Same paper can reference them inline
# as "(Extended Data Fig. 1)" — those are body references, NOT captions
# (they get filtered by being shorter than 3 chars after the number,
# or they don't have a pipe / bold structure). Two real-caption
# variants observed in young-full.pdf:
#   "**Extended Data Fig. 1 | Plot of surface temperatures...**"   (whole-bold, pipe inside)
#   "**Extended Data Fig. 1** | Plot of surface temperatures..."   (pipe outside, header-only bold)
# Captured fig_id is just the digit; extract_figure_captions prepends
# "ED" so "1" → "ED1" in the captions list. Sort key in
# extract_figure_captions orders these after main figures and after
# supplementary "S" figures.
FIG_CAPTION_RE_ED_PIPE = re.compile(
    r"(?<!\!\[)(?:\*\*)?Extended\s+Data\s+(?:Figure|Fig\.?)\s+(\d+)(?:\*\*)?\s*\|\s*(?:\*\*)?([^.\n]{3,500}?)(?=[.*\n]|$)",
    re.IGNORECASE,
)
FIG_CAPTION_RE_ED_BOLD_BLOCK = re.compile(
    r"\*\*\s*Extended\s+Data\s+(?:Figure|Fig\.?)\s+(\d+)\s*\|\s*([^*\n]{3,500}?)\s*\*\*",
    re.IGNORECASE,
)

# Used by the page-image fallback to find a markdown table's caption
# ("Table I", "Table II.", "Table 3:", "Supplementary Table S1", etc.)
# in the surrounding context so we can search for the same caption in
# the PDF text and identify which page hosts the table. Roman numerals
# up to L (50) and arabic numerals are both common in scientific
# articles. Supplement tables prefixed with "S" ("Table S1", "Table SII")
# and the "Supplementary " literal prefix that some publishers use are
# both supported, so SI runs find their own tables. The "[A-Z]\d+"
# alternative covers AGU/Wiley appendix-style ids ("Table A1", "Table
# B1"); the "*" added to the trailing lookahead lets bold-wrapped
# captions like "**Table A1**Title..." match (the closing "**" is the
# next char after the id, not whitespace/punctuation).
TABLE_CAPTION_RE = re.compile(
    r"\b(?:Supplementary\s+|Supplement(?:ary)?\s+)?Table\s+(S?\d+|S?[IVXLC]+|[A-Z]\d+)\b(?=[\.\:\s\*])",
    re.IGNORECASE,
)

# Stricter line-leading caption pattern used by the orphan-caption
# rescue. Requires the line to BEGIN with a Table-N caption (optionally
# wrapped in markdown bold) followed by a period, colon, or bold-close
# and then the descriptor text. Inline body references like "...as
# shown in Table 2" don't match. Captures the full caption header
# ("Table I", "Supplementary Table S6", "Table A1") plus the rest of
# the descriptor sentence on the same line so the rescue prompt can
# quote it verbatim. The optional <span> prefix matches marker's
# anchor tags ("<span id=\"page-12-1\"></span>**Table B1**..."); the
# bold-close "(?:\*\*\s*|\s*\*{0,2}\s*[\.:]\s*)" alternation handles
# AGU "**Table A1**Title" (no separator) alongside the standard
# "Table 1." and "Table 1:" forms.
TABLE_CAPTION_LINE_RE = re.compile(
    r"(?im)^\s*"
    r'(?:<span\s+id="page-\d+-\d+"></span>\s*)*'
    r"\*{0,2}\s*"
    r"(?P<full>(?:Supplementary\s+|Supplement(?:ary)?\s+|Supp\.\s+)?"
    r"Table\s+(?P<id>S?\d+|S?[IVXLC]+|[A-Z]\d+|[A-Z]\.\d+))"
    r"\b(?:\*\*\s*|\s*\*{0,2}\s*[\.:]\s*)"
    r"(?P<rest>[^\n]*)$"
)

# === Hybrid layout (--layout-source=hybrid) anchors ====================
# Line-anchored caption regexes used by _align_marker_to_mineru_layout.
# Purpose: locate a caption LINE in marker_md so a MinerU figure / table
# block can be spliced (image link above the caption; table HTML in
# place of marker's table region). They only need the figure number;
# the caption text comes from marker (canonical body OCR).
#
# False positives (a body line that happens to start with "Fig. N") would
# splice a spurious image; we filter aggressively. The trailing `[.|]`
# lookahead requires either a period or pipe after the figure id, which
# rejects body sentences ("Fig. 5 shows...", "Fig. 5 of Smith [1990]")
# while accepting captions ("Fig. 1.", "Fig. 1 |", "Fig. 1 (color).").
HYBRID_FIG_LINE_RE = re.compile(
    r'^(?:<span\s+id="page-\d+-\d+"></span>\s*)?'
    r'(?:\*\*)?'
    r'(?:Extended\s+Data\s+)?'
    r'(?:Figure|Fig\.?)\s+'
    r'(' + FIG_ID_PAT + r')'
    r'(?:\s+\([^)\n]{1,40}\))?'
    # Allow optional closing `**` (Nature-style `**Figure 1** | **Caption`)
    # between the fig id and the punctuation. Up to 2 asterisks so we
    # match either a single-asterisk or full bold close, but not enough
    # to swallow other markdown structures.
    r'(?=\s*\*{0,2}\s*[.|])',
    re.IGNORECASE | re.MULTILINE,
)
# For tables, reuse TABLE_CAPTION_LINE_RE (already line-anchored, tolerates
# bold wrappers and span anchors, captures the id in named group "id").
HYBRID_TABLE_LINE_RE = TABLE_CAPTION_LINE_RE


def _normalise_fig_id(raw: str) -> str:
    """Map a captured figure id to its base id for cross-side matching.

    Marker's caption may emit the panel letter on the figure id ("Fig.
    1A.") while MinerU's `figure_number` carries only the integer (1).
    Strip a trailing single-letter panel suffix when the rest of the id
    is purely numeric or starts with a known prefix (S, ED, A1...).

    >>> _normalise_fig_id("1")    # "1"
    >>> _normalise_fig_id("1A")   # "1"
    >>> _normalise_fig_id("S2")   # "S2"
    >>> _normalise_fig_id("S2B")  # "S2"
    >>> _normalise_fig_id("ED1")  # "ED1"
    >>> _normalise_fig_id("A1")   # "A1"  (Elsevier appendix; letter is part of id)
    >>> _normalise_fig_id("A.1")  # "A.1"
    """
    if not raw:
        return raw
    m = re.match(r"^(S?\d+|ED\d+)([A-Z])$", raw)
    if m:
        return m.group(1)
    return raw


_HYBRID_TBL_NUM_RE = re.compile(
    r"^\s*\**\s*(?:Supplementary\s+|Supplement(?:ary)?\s+|Supp\.\s+)?"
    r"Table\s+(S?\d+|S?[IVXLC]+|[A-Z]\d+|[A-Z]\.\d+)\b",
    re.IGNORECASE,
)

_HYBRID_FIG_NUM_FROM_TEXT_RE = re.compile(
    r"^\s*\**\s*(?:Extended\s+Data\s+)?(?:Figure|Fig\.?)\s+("
    + FIG_ID_PAT + r")",
    re.IGNORECASE,
)

# Defensive: when MinerU's PaddleOCR concatenates a publisher-typeset
# panel marker (`(a)`, `(b)`, `[c]`, `a.`, `b.`) onto the front of the
# caption text -- a real pattern on multi-panel scientific figures
# where the panel labels are vertically beside the subpanel images
# (e.g. Wackerle1962 Fig. 9: caption text begins "(b) FIG. 9(a) and
# (b)...") -- the leading-anchored fig-id regex misses the figure
# number. This fragment strips one such marker so we can retry the
# match. Bounded forms only: a single letter a-h with a paren or
# period delimiter. Won't match arbitrary parenthetical text.
_HYBRID_FIG_PANEL_PREFIX_RE = re.compile(
    r"^\s*[\(\[]?\s*[a-hA-H]\s*[\)\].\:]\s+"
)


def _hybrid_fig_id_from_block(blk) -> Optional[str]:
    """Figure id (as a string) for a MinerU image/chart block.

    Prefers `blk.figure_number` (populated by rescue_subpanel_groups
    on multi-panel primaries); falls back to parsing `blk.text` for
    a leading caption label. Returns None when neither yields an id
    -- the block is then routed to the EOD figure fallback.

    When the leading-anchor match fails AND the caption text begins
    with a panel marker like `(b) `, strip the marker and retry. This
    rescues OCR'd captions where the panel label fragment was glued
    onto the front of the main caption.
    """
    if getattr(blk, "figure_number", None) is not None:
        return str(blk.figure_number)
    text = getattr(blk, "text", None)
    if not text:
        return None
    m = _HYBRID_FIG_NUM_FROM_TEXT_RE.match(text)
    if m:
        return m.group(1)
    # Fallback: strip a leading panel marker and retry. Without this,
    # captions like "(b) FIG. 9(a) and (b). Pressure vs..." silently
    # drop their figure id and the splice never inserts the image.
    panel_m = _HYBRID_FIG_PANEL_PREFIX_RE.match(text)
    if panel_m:
        m = _HYBRID_FIG_NUM_FROM_TEXT_RE.match(text[panel_m.end():])
        if m:
            return m.group(1)
    return None


def _extract_table_number(text: Optional[str]) -> Optional[str]:
    """Extract the table id from a MinerU table block's caption text.

    Returns None when the text is empty or doesn't begin with a Table
    label. Used to group MinerU table blocks by id for hybrid splicing.
    """
    if not text:
        return None
    m = _HYBRID_TBL_NUM_RE.match(text)
    return m.group(1) if m else None

# Below this many characters per square point, a page's text layer is
# probably broken or missing and worth a VLM rescue. Used as the
# trigger input for Hook 3 (rescue_sparse_pages, opt-in via
# --rescue-sparse-pages). NOT used for quality scoring as of
# 2026-05; see QualityReport.overall docstring.
SPARSE_CHARS_PER_PT2 = 0.005

CROP_DPI = 220
PAGE_DPI = 170

FULL_PAGE_TABLE_FRAC = 0.85  # bbox area / page area above this = "full page"
TABLE_JPEG_QUALITY = 90

# Sideways-table detection. A wide landscape table rotated 90 degrees
# onto a portrait page emerges from the table-finder as a tall+narrow
# crop (the cell rows become columns from the renderer's POV). When
# the cropped image's height/width ratio exceeds this threshold, hook
# 1 rotates the crop before sending it to the VLM (and saves the
# rotated JPG as the human-visible asset). The fallback path also
# pre-renders the whole page so a follow-up VLM call can re-do the
# transcription on the full page if the rotated-crop transcription
# looks broken.
#
# Defaults are tuned to the silica-shock corpus where a real
# landscape table at ~5:1 aspect rotates to a ~0.2 ratio. Threshold
# 2.5 catches that case with margin while being less likely to fire
# on a tall non-rotated table (long single-column ratings list, page
# of references rendered as a "table"). Override either via env var.
ROTATE_TABLES = True
try:
    TABLE_ROTATION_ASPECT_THRESHOLD = float(
        os.environ.get("PAPER2MD_TABLE_ROTATION_THRESHOLD", "2.5"))
except ValueError:
    TABLE_ROTATION_ASPECT_THRESHOLD = 2.5
TABLE_ROTATION_DIRECTION = (
    os.environ.get("PAPER2MD_TABLE_ROTATION_DIRECTION", "ccw").strip().lower()
    or "ccw"
)
if TABLE_ROTATION_DIRECTION not in ("ccw", "cw"):
    TABLE_ROTATION_DIRECTION = "ccw"

# Phase-2 journal-aware reference rescue (off by default while we tune).
# When enabled (via --use-journal-rescue or programmatic toggle), the
# dispatcher routes papers whose references_score falls below
# PAPER2MD_REF_RESCUE_THRESHOLD into a journal-specific rescue keyed by
# the DOI-derived journal_slug. Working papers (which validate above the
# threshold) never see the new code. See docs/USAGE.md and
# tests/test_journal_rescue.py for details. All three knobs are env-
# tunable so calibration can iterate without code changes.
RESCUE_JOURNAL_REFS = False
try:
    REF_RESCUE_THRESHOLD = float(
        os.environ.get("PAPER2MD_REF_RESCUE_THRESHOLD", "0.65"))
except ValueError:
    REF_RESCUE_THRESHOLD = 0.65
try:
    REF_RESCUE_LOR = float(
        os.environ.get("PAPER2MD_REF_RESCUE_LOR", "8.0"))
except ValueError:
    REF_RESCUE_LOR = 8.0
REF_RESCUE_DRY_RUN = (
    os.environ.get("PAPER2MD_REF_RESCUE_DRY_RUN", "").strip().lower()
    in ("1", "true", "yes", "on")
)

# Skip VLM captioning for image crops smaller than this; they're almost
# always journal banners, icons, header rules, or extraction fragments
# rather than real figures, and the VLM will hallucinate captions on them.
CAPTION_MIN_BYTES = 10_000

# Module-level toggle for the deterministic reference-renumber pass.
# Flipped by --no-fix-refs in main(). Lives at module scope so both
# convert() and convert_text_only() see the same setting.
FIX_REFERENCES = True

# Module-level toggle for the article-boundary trim pre-pass. Flipped by
# --no-trim-articles in main().
TRIM_ARTICLES = True

# Module-level toggle for the footnote-reference consolidation pre-pass.
# Flipped by --no-consolidate-footnotes in main().
CONSOLIDATE_FOOTNOTES = True

# Module-level toggle for the Elsevier/Icarus span-anchored reference
# normalisation pre-pass (re-bullet plain-span lines, rejoin split DOI
# links, strip refhub link wrappers). Flipped by --no-normalise-refs.
NORMALISE_REFS = True

# Module-level toggle for the running-footer strip pre-pass that removes
# "<AUTHOR> ET AL.  N of M" lines (AGU / Wiley / society-journal style).
# Flipped by --no-strip-footers in main().
STRIP_RUNNING_FOOTERS = True

# Module-level toggle for the math-label strip pre-pass that removes
# KaTeX-incompatible \label{...} (and rewrites \eqref{...}) inside
# display-math blocks. Flipped off by --no-strip-math-labels.
STRIP_MATH_LABELS = True

# Module-level toggle for the journal page-header strip pre-pass that
# removes 'LETTERS' / 'NATURE GEOSCIENCE DOI: [...]' page-header
# artifacts that marker preserves between body paragraphs. Pure regex,
# no VLM. Flipped off by --no-strip-page-headers.
STRIP_PAGE_HEADERS = True

# Module-level toggle for the per-page download-watermark stripper.
# Subscription PDFs from publishers like Wiley have a multi-line stamp
# rendered behind / around every page giving the DOI URL, the
# downloading institution, the download date, and a link to the
# Terms-and-Conditions / Creative-Commons license. The stamp is
# per-page and frequently splits across lines (column-narrow URL
# wrap), so it accumulates many times in the body and interrupts
# real paragraphs / equations. No VLM, pure regex. Flipped off by
# --no-strip-publisher-stamps.
STRIP_PUBLISHER_STAMPS = True

# Module-level toggle for the reference-section consolidation pre-pass
# that merges multiple '## References' / '# **References**' sections
# (e.g. Nature's main-text references + Methods references) into one
# consolidated section at the end of the document. No VLM. Flipped off
# by --no-merge-references.
MERGE_REFERENCES = True

# Module-level toggle for the orphan-reference-cluster injector that
# detects bulleted numbered-reference entries living OUTSIDE any
# `## References` heading and prepends a synthetic heading above
# them, so merge_reference_sections can consolidate them with the
# existing section. Common cause: marker splits a multi-page
# references list across a column boundary so some entries end up
# above the heading. No VLM. Flipped off by --no-inject-orphan-refs.
INJECT_ORPHAN_REFS = True

# Module-level toggle for normalise_references_section, the pass that
# tidies the contents of a `## References` section: merges column-
# break continuation lines into their parent entry, pulls the author
# address / "(Received ...)" lines out of the bullet list, and
# applies uniform "- " bulleting. No VLM. Flipped off by
# --no-tidy-refs.
TIDY_REFS = True

# Module-level toggle for the orphan-caption table rescue post-pass.
# After hook 1 (process_tables), scans the body for line-leading
# "Table N." captions that aren't followed by either a markdown table
# or an emitted ![Table N (page P)] image link, and renders the caption
# page to ask the VLM to extract the table directly. Catches the
# "marker rendered the table as a paragraph" failure mode (Mode A);
# see USAGE §5.1. Needs the VLM (skipped under --no-vlm). Flipped off
# by --no-rescue-orphan-tables.
RESCUE_ORPHAN_TABLES = True

# Module-level toggle for the orphan-figure-caption rescue post-pass.
# After hook 2 (caption_figures), scans the markdown for "Fig. N."
# captions whose figure id has no preceding ![Figure N | ...] image
# markdown -- i.e. marker layout missed the figure region entirely
# (e.g. jacquet 2026 Fig. 5: caption block misclassified as paragraph,
# no figure crop emitted). Locates the caption page in the PDF and
# crops the page-region above the caption as the figure asset; falls
# back to a whole-page render when the above-caption area is too
# small. No VLM. Flipped off by --no-rescue-orphan-figures.
RESCUE_ORPHAN_FIGURES = True

# Module-level toggle for the sparse-page rescue (hook 3). Off by
# default since v0.2: the density-only trigger fires on legitimate-but-
# short pages (end-of-references tails, figure-dominant pages, blank
# divider pages) and the rescue output is appended as a tail section
# rather than spliced into the body, so even on a true-positive the
# value-add over marker's existing output is limited. Opt in via
# --rescue-sparse-pages on scanned PDFs / Adobe-Paper-Capture-vintage
# OCR corpora where the text layer is genuinely broken. Always skipped
# under --no-vlm.
RESCUE_SPARSE_PAGES = False

# Module-level toggle for the data-repository link extraction post-pass.
# Always-on by default; pure regex, no network. Flipped off by
# --no-data-repos in main(). When enabled, scans the body for
# DOIs/URLs pointing at Zenodo, Dryad, Dataverse, figshare, OSF,
# PANGAEA, ESS-DIVE, Mendeley Data, ICPSR, CaltechDATA. See
# src/data_repos.py.
DATA_REPOS = True

# Opt-in: when True, paper2md hits each detected data-repository's
# public API (one HTTP GET per unique deposit) to enrich the YAML/JSON
# with the deposit's title, description, license, and file list. Off
# by default to keep runs hermetic. Flipped on by --fetch-data-repos.
FETCH_DATA_REPOS = False
FETCH_DATA_REPOS_TIMEOUT = 8.0

# When True, every network-touching scholarly-API path is skipped:
# copyright/preprint metadata resolution, references API fallback, OA
# PDF download, data-repository enrichment. Local VLMs (lmstudio,
# vllm) still work; OpenAI/Anthropic providers will fail naturally.
# Flipped on by --offline. Skipped steps are recorded in
# `copyright.provenance` (network-disabled), `run.pipeline.offline`,
# and `quality.references.api_skipped_offline` so downstream
# consumers can see the gaps.
OFFLINE = False

# Hook-2 figure-caption matching strategy. Set by --figmatch-strategy.
# Token-based; multiple tokens may be combined with "+", e.g.
# "page-prior+dup-detect". See docs/FIGURE_MATCH_TEST_PLAN.md.
#   "single"      — one VLM call per image; no cross-image post-pass.
#   "page-prior"  — append "extracted from page N" to the prompt before
#                   classifying so the VLM can use page co-location as
#                   a signal.
#   "dup-detect"  — DEFAULT. After classification, for any fig_id
#                   assigned to ≥2 images, run a YES/NO confirmation
#                   per conflicting image; demoted images go through a
#                   second classification pass with the contested
#                   option excluded so a real figure that just looks
#                   like the contested one can still find its actual
#                   caption. Iterates until the set of choices is
#                   stable (max 3 iterations). Adds 1-2 extra VLM
#                   calls per duplicate group; on most papers that's
#                   0-2 extra calls total. Validated on a 6-paper
#                   corpus: 84.4% → 96.9% aggregate accuracy vs
#                   single-pass baseline, no regressions on easy
#                   cases. See docs/FIGURE_MATCH_TEST_PLAN.md.
#   "vote"        — call the classifier 3× and take the majority.
FIGMATCH_STRATEGY = "dup-detect"


# OpenAI-compatible client (serves lmstudio + openai). Anthropic uses a
# separate SDK loaded lazily. Both are (re)initialized by configure_client().
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
_anthropic_client = None  # populated on demand when provider=anthropic


def configure_client(provider: Optional[str] = None,
                     model: Optional[str] = None) -> None:
    """Reconfigure the global VLM client/model from CLI/env.

    Call once from main() before any vlm() invocation. `provider`
    beats $VLM_PROVIDER; `model` beats $VLM_MODEL. Defaults: provider=lmstudio,
    model=per-provider default from _PROVIDER_DEFAULT_MODEL.
    """
    global client, _anthropic_client, _PROVIDER, VLM_MODEL
    p = (provider or os.environ.get("VLM_PROVIDER", "lmstudio")).lower()
    if p == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set; required for --provider openai")
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif p == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set; required for --provider anthropic")
        try:
            import anthropic
        except ImportError:
            raise SystemExit("anthropic package not installed; run: pip install anthropic")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    elif p == "vllm":
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        client = OpenAI(base_url=base_url, api_key="EMPTY")
    else:
        base_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        client = OpenAI(base_url=base_url, api_key="lm-studio")
        p = "lmstudio"

    _PROVIDER = p
    if model:
        VLM_MODEL = model
    elif os.environ.get("VLM_MODEL"):
        VLM_MODEL = os.environ["VLM_MODEL"]
    else:
        VLM_MODEL = _PROVIDER_DEFAULT_MODEL.get(p, VLM_MODEL)
    log.info("VLM provider: %s, model: %s", _PROVIDER, VLM_MODEL)
    log.info("VLM sampling: temperature=%s, seed=%s (deterministic; "
             "override via --vlm-temperature / --vlm-seed or "
             "$VLM_TEMPERATURE / $VLM_SEED)",
             VLM_TEMPERATURE, VLM_SEED if VLM_SEED is not None else "None")
    if _PROVIDER == "anthropic" and VLM_SEED is not None:
        log.warning("VLM seed (%s) is ignored under --provider anthropic; "
                    "Anthropic's Messages API does not accept a seed "
                    "parameter. Only temperature applies.", VLM_SEED)


SYSTEM = "You are a precise document digitizer. Output only what is asked, no commentary."

TABLE_PROMPT = (
    "This image is one table cropped from a scientific journal article. "
    "Convert it to a GitHub-flavored markdown table, preserving every row, "
    "column, header, and cell exactly. Some tables have one or more "
    "columns without a header label -- typically the rightmost column "
    "holding a phase/state label, a footnote marker, or a category. "
    "Preserve such columns; leave the header cell blank (output `| |` for "
    "that header position) rather than inventing a label or merging the "
    "column into a neighbour. The separator row must still have the same "
    "number of `---` segments as there are columns. If there are "
    "footnotes or notes labeled with letter markers (a, b, c, ...) or "
    "symbol markers (*, dagger, double-dagger) immediately below the "
    "table body, include them as a `**Notes:**` block after the table -- "
    "one note per line, each prefixed by its marker (e.g. `a` "
    "Description...). Otherwise output only the markdown table -- no "
    "preamble, no commentary."
)

# Used by the page-image fallback when the table-finder can't locate a
# table but the caption text is present somewhere in the PDF. We render
# the caption's page and ask the VLM to extract the named table from it
# while ignoring everything else on the page.
TABLE_PAGE_PROMPT = (
    "This image is a full page from a scientific article. The page may "
    "contain running text, figures, and one or more tables. The target "
    "table may be rotated 90 degrees relative to the rest of the page "
    "(landscape table on a portrait page); if so, mentally orient it so "
    "its column headers run horizontally across the top of your output. "
    "Extract the table whose caption begins with: '{caption}'. Convert "
    "it to a GitHub-flavored markdown table, preserving every row, "
    "column, header, and cell exactly. Keep math in $...$ form where "
    "appropriate. Some tables have one or more columns without a "
    "header label -- typically the rightmost column holding a "
    "phase/state label, a footnote marker, or a category. Preserve "
    "such columns; leave the header cell blank (output `| |` for that "
    "header position) rather than inventing a label or merging the "
    "column into a neighbour. The separator row must still have the "
    "same number of `---` segments as there are columns. If that table "
    "has footnotes or notes labeled with letter markers (a, b, c, ...) "
    "or symbol markers (*, dagger, double-dagger) immediately below "
    "it, include them as a `**Notes:**` block after the table -- one "
    "note per line, each prefixed by its marker. Ignore body text, "
    "figures, and any other tables on the page. Output only the "
    "markdown table (and the notes block when applicable) -- no "
    "preamble, no commentary. If a table matching that caption is not "
    "visible on this page, output the single word: SKIP"
)

CAPTION_PROMPT = (
    "This image is extracted from a scientific article. If it is a "
    "journal banner, logo, icon, header/footer rule, decorative element, "
    "or any non-figure fragment, respond with exactly one word: SKIP. "
    "Otherwise write one concise sentence (under 25 words) describing "
    "what the figure depicts, for use as alt-text. Do not restate the "
    "figure number. Output only the sentence or SKIP."
)

PAIR_PROMPT_HEADER = (
    "This image was extracted from a scientific article. Below is a "
    "numbered list of that article's figure captions. Which caption "
    "describes this image? Respond with just the number. Reply 0 ONLY "
    "if the image is clearly a non-figure artifact (a journal banner, "
    "logo, page rule, icon, or extraction fragment from a header or "
    "footer). If the image is a figure (plot, diagram, photograph, "
    "schematic, or multi-panel composite) but you are uncertain which "
    "caption matches, pick your best guess from the list -- do not "
    "reply 0.\n\n"
)

CITATION_PROMPT = (
    "This is the first page of a scientific article. Produce a single "
    "bibliographic citation in the style this journal uses in its own "
    "reference list. Include authors (in the journal's author order and "
    "initial/full-name convention), title, journal name, volume, issue "
    "or article number if present, page range or article ID, and year. "
    "Format the title in **bold** and the journal name in *italics*. "
    "If a DOI or publisher URL is visible on the page, append it at the "
    "end as a markdown link: [https://doi.org/10.xxxx/yyyy](https://doi.org/10.xxxx/yyyy). "
    "A bare DOI like '10.1038/s41586-024-xxxxx' should be promoted to "
    "'https://doi.org/10.1038/s41586-024-xxxxx'.\n\n"
    "Output the citation on a single line. No preamble, no quotes, no "
    "bullet point -- just the citation text. Ignore any marginal line "
    "numbers (standalone digits in margins from LaTeX's lineno package). "
    "If you cannot find enough information on this page to build a "
    "usable citation, output the single word: SKIP"
)

PAGE_PROMPT = (
    "This is one page from a two-column scientific journal article. Convert "
    "the content to clean Markdown. Preserve reading order, headings, and "
    "math (as LaTeX in $...$). Skip running headers, footers, page numbers, "
    "and marginal line numbers (standalone digits that appear in the "
    "left/right margins every few lines -- common in manuscripts using "
    "LaTeX's lineno package). Output only the markdown."
)

TRIM_PROMPT = (
    "This image is a vertical concatenation of two pages from a PDF: TOP "
    "is page 1 (the article I want to extract); BOTTOM is a candidate "
    "page from later in the same PDF. Decide whether the bottom page "
    "belongs to the SAME article as page 1 (continuation of the body, "
    "references, methods, supplementary section) or a DIFFERENT article "
    "that happens to be in the same PDF (the next article in the journal "
    "issue, an editorial, an unrelated News & Views piece).\n"
    "\n"
    "Visual signals for DIFFERENT: bottom page has its own title, its own "
    "author byline, an opening paragraph introducing a new topic.\n"
    "Visual signals for SAME: bottom page continues mid-sentence, mid-"
    "paragraph, or has a section heading like References/Methods/"
    "Acknowledgements that clearly belongs to page 1's article.\n"
    "\n"
    "Reply in this EXACT format, two lines and nothing else:\n"
    "  Line 1: SAME or DIFFERENT\n"
    "  Line 2: only if DIFFERENT, the new article's title (the title of "
    "the article that the bottom page belongs to), 3-12 words, exactly "
    "as visible. If you cannot read the title clearly, write NONE."
)


def check_vlm_reachable(timeout: float = 5.0) -> Optional[str]:
    """Ping the configured VLM endpoint with a cheap request. Returns
    None on success, or an error string on failure.

    Anthropic gets a soft pass: there's no free reachability check that
    doesn't burn a billable token, so we only verify the SDK loaded.
    """
    if _PROVIDER == "anthropic":
        if _anthropic_client is None:
            return "anthropic SDK not initialized; configure_client() not called?"
        return None

    try:
        # GET /v1/models on the OpenAI-compat client. Cheap, no auth-dependent
        # state, returns the list of served models. Works for lmstudio, vllm,
        # openai. Tight timeout: this should be < 1s on a healthy local server.
        client.with_options(timeout=timeout).models.list()
        return None
    except Exception as e:
        endpoint = getattr(client, "base_url", "?")
        return (f"VLM provider '{_PROVIDER}' not reachable at {endpoint}: "
                f"{type(e).__name__}: {e}")


def vlm(prompt: str, image: Image.Image, max_tokens: int = 1500,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        raise_on_error: bool = False) -> Optional[str]:
    """Vision call: prompt + PIL image -> text. Dispatches on the active
    provider (lmstudio/openai use the OpenAI chat.completions shape;
    anthropic uses the messages API with image content blocks).

    `timeout` (seconds) caps the per-call wait. None uses the SDK
    default (600s for the OpenAI client). Set explicitly for
    long-running rescues so the pipeline fails fast instead of
    hanging through the SDK's default retry-after-10-min behavior.

    `max_retries` controls SDK-level retries on timeout / 5xx /
    connection errors. None uses the SDK default (2). Set to 0 for
    fail-fast on hard-capped calls so a single timeout doesn't fan
    out into 3x180s = 9min of wasted wall time.

    `raise_on_error` controls error reporting: when False (default),
    the function logs and returns None on `OpenAIError` for callers
    that just want "did it work?". When True, exceptions propagate
    to the caller so application-level retry logic can branch on the
    specific error subclass (e.g. retry on APIConnectionError but
    bail on BadRequestError)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if _PROVIDER == "anthropic":
        # Anthropic accepts `temperature` but NOT a `seed` parameter.
        # Drop the seed; the one-shot warning in configure_client
        # surfaced this at startup.
        try:
            msg = _anthropic_client.messages.create(
                model=VLM_MODEL,
                max_tokens=max_tokens,
                temperature=VLM_TEMPERATURE,
                system=SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
        except Exception as e:
            if raise_on_error:
                raise
            log.warning("Anthropic VLM call failed: %s", e)
            return None
        text = msg.content[0].text if msg.content else ""
        log.debug("VLM reply (max_tokens=%d): %r", max_tokens, text or "")
        return text.strip() or None
    # OpenAI's newer model families (gpt-5, o-series reasoning models)
    # rejected `max_tokens` and require `max_completion_tokens` instead.
    # gpt-4o / gpt-4.1 still accept `max_tokens`. vLLM and LM Studio
    # implement the OpenAI-compatible API and accept `max_tokens`. So
    # only swap the parameter name when we're talking to api.openai.com
    # AND the model is a newer family.
    token_kwarg = "max_tokens"
    if _PROVIDER == "openai" and any(
        VLM_MODEL.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
    ):
        token_kwarg = "max_completion_tokens"
    # Pass reasoning_effort=none through extra_body so reasoning-mode
    # models (Gemma 4, GLM-4 thinking variants, etc.) emit the answer
    # directly within paper2md's small max_tokens budgets instead of
    # spending all 5 tokens on internal reasoning. Non-reasoning models
    # silently ignore the field (verified on Qwen3-VL via vLLM and on
    # gpt-4.1 via OpenAI). For OpenAI's actual reasoning models (gpt-5,
    # o-series) the field is recognised and honoured per spec, but we
    # still need the larger token budget — see USAGE.md §13.6.
    opts: dict = {}
    if timeout is not None:
        opts["timeout"] = timeout
    if max_retries is not None:
        opts["max_retries"] = max_retries
    cli = client.with_options(**opts) if opts else client
    # Deterministic sampling: temperature + seed go in every request.
    # vLLM and LM Studio honor both; OpenAI honors temperature on all
    # models and seed on most chat models (silently ignored otherwise).
    sampling_kwargs: dict = {"temperature": VLM_TEMPERATURE}
    if VLM_SEED is not None:
        sampling_kwargs["seed"] = VLM_SEED
    try:
        resp = cli.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
            extra_body={"reasoning_effort": "none"},
            **sampling_kwargs,
            **{token_kwarg: max_tokens},
        )
    except OpenAIError as e:
        if raise_on_error:
            raise
        log.warning("VLM call failed (%s): %s", type(e).__name__, e)
        return None
    text = resp.choices[0].message.content or ""
    log.debug("VLM reply (max_tokens=%d): %r", max_tokens, text)
    return text.strip() or None


def render_crop(doc: fitz.Document, page_idx: int, bbox: Iterable[float],
                dpi: int = CROP_DPI) -> Image.Image:
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(bbox))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def render_page(doc: fitz.Document, page_idx: int, dpi: int = PAGE_DPI) -> Image.Image:
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _crop_looks_sideways(crop: "Image.Image") -> bool:
    """Return True iff the cropped table image looks like a wide
    landscape table rotated 90 deg onto a portrait page (tall+narrow
    in the renderer's coordinate frame).

    Tests pass MagicMock objects whose .size doesn't unpack cleanly
    -- treat those as not-rotated and pass through unchanged. Tiny
    crops (< 100 px on either edge) also fall through; they're
    indistinguishable from artifacts at this scale."""
    if not ROTATE_TABLES:
        return False
    try:
        w, h = crop.size
        w = int(w)
        h = int(h)
    except (TypeError, ValueError, AttributeError):
        return False
    if w < 100 or h < 100:
        return False
    return h / max(1, w) > TABLE_ROTATION_ASPECT_THRESHOLD


def _rotate_upright(img: "Image.Image") -> "Image.Image":
    """Rotate a sideways-table image 90 degrees so its column headers
    run horizontally across the top.

    `TABLE_ROTATION_DIRECTION` describes the ORIGINAL rotation that
    the publisher applied when laying the wide table onto a portrait
    page; this function applies the inverse rotation to undo it.
    Default 'ccw' (original page rotated CCW with table-top at LEFT
    page edge) -> applies CW 90 deg = PIL's ROTATE_270. 'cw' (table-
    top at RIGHT page edge) -> applies CCW 90 deg = PIL's ROTATE_90."""
    if TABLE_ROTATION_DIRECTION == "cw":
        return img.transpose(Image.ROTATE_90)
    return img.transpose(Image.ROTATE_270)


def extract_figure_captions(md: str) -> list[tuple[str, str]]:
    """Return [(figure_id, title)] sorted by figure id. One entry per
    author caption block found in the markdown; duplicates on the same
    figure id keep the first match. Runs both pipe (Nature) and
    Elsevier patterns.

    figure_id is a string: "1", "12", "S1", "S12" -- uppercase S prefix
    when present, normalized on read. Supplements use S-prefixed IDs;
    main papers use plain integer strings."""
    seen: dict[str, str] = {}
    # Run ED patterns first so "ED1" is registered before the negative
    # lookbehind on the main patterns has a chance to fail; this is
    # belt-and-suspenders since the lookbehind alone should be enough.
    for pattern in (FIG_CAPTION_RE_ED_PIPE, FIG_CAPTION_RE_ED_BOLD_BLOCK):
        for m in pattern.finditer(md):
            fig_id = "ED" + m.group(1)
            title = m.group(2).strip().rstrip(",;:.").strip()
            if title and fig_id not in seen:
                seen[fig_id] = title
    for pattern in (FIG_CAPTION_RE_PIPE, FIG_CAPTION_RE_ELSEVIER,
                    FIG_CAPTION_RE_BOLD_BLOCK, FIG_CAPTION_RE_PLAIN):
        for m in pattern.finditer(md):
            fig_id = m.group(1).upper()  # e.g. "1", "S12", "B.13"
            title = m.group(2).strip().rstrip(",;:.").strip()
            if title and fig_id not in seen:
                seen[fig_id] = title
    # Sort: main figures first (ascending), then supplement S-IDs,
    # then appendix letter.number ids, then Nature Extended Data ED-IDs.
    def sort_key(item):
        fid = item[0]
        if fid.startswith("ED"):
            return (3, int(fid[2:]) if fid[2:].isdigit() else 0, fid)
        if "." in fid and fid[0].isalpha():
            # Appendix style "B.13" -> sort by letter, then number
            letter, _, num = fid.partition(".")
            return (2, letter, int(num) if num.isdigit() else 0)
        if fid.startswith("S"):
            return (1, int(fid[1:]) if fid[1:].isdigit() else 0, fid)
        try:
            return (0, int(fid), fid)
        except ValueError:
            return (0, 0, fid)
    return sorted(seen.items(), key=sort_key)


def table_is_suspicious(md: str) -> Optional[str]:
    rows = [r for r in md.splitlines() if r.strip().startswith("|")]
    if len(rows) < 3:
        return "fewer than 3 rows"
    counts = [r.count("|") for r in rows]
    if max(counts) - min(counts) > 1:
        return f"inconsistent column count ({min(counts)}..{max(counts)})"
    body = rows[2:]
    if not body:
        return "no body rows"
    blank = sum(1 for r in body if re.match(r"^\|[\s|]+$", r))
    if blank / len(body) > 0.4:
        return f"{blank}/{len(body)} body rows blank"
    return None


def _find_title_line_in_md(title: str, md: str) -> Optional[int]:
    """Return the index of the LAST markdown line containing `title` (or a
    contiguous N-word slice of it, N decreasing from full title length down
    to 3). Case-insensitive, punctuation-insensitive, whitespace-collapsed.
    Returns None if nothing matches.

    "Last" because the title might also appear in the prior article's body
    (e.g., a forward reference); the new article's actual title heading is
    later in the markdown.
    """
    title_words = re.findall(r"\w+", title.lower())
    if len(title_words) < 3:
        return None
    md_lines = md.splitlines()
    md_norm = [" ".join(re.findall(r"\w+", ln.lower())) for ln in md_lines]
    for window in range(len(title_words), 2, -1):
        for start in range(len(title_words) - window + 1):
            phrase = " ".join(title_words[start:start + window])
            best = None
            for i, ln_norm in enumerate(md_norm):
                if phrase and phrase in ln_norm:
                    best = i
            if best is not None:
                return best
    return None


def _delete_orphan_page_assets(md_after: str, assets_dir: Path) -> int:
    """Delete page-tagged image files (`_page_N_*` / `si__page_N_*`) that
    are no longer referenced in the trimmed markdown. Leaves table JPEGs
    and table sidecars alone (they don't exist yet at trim time anyway,
    but be defensive). Returns count deleted."""
    if not assets_dir.is_dir():
        return 0
    referenced = set(re.findall(r"!\[[^\]]*\]\(assets/([^)]+)\)", md_after))
    pattern = re.compile(r"^(?:si_)?_page_\d+_")
    deleted = 0
    for asset in assets_dir.iterdir():
        if not asset.is_file():
            continue
        if not pattern.match(asset.name):
            continue
        if asset.name in referenced:
            continue
        asset.unlink()
        deleted += 1
    return deleted


def _check_page_vs_first(doc: "fitz.Document", page_idx: int,
                         dpi: int = 100) -> Optional[dict]:
    """Render page 0 + page `page_idx`, vstack, ask the VLM whether they
    belong to the same article. Returns:
        {"same": True,  "title": None}                — same article
        {"same": False, "title": "<title>" or None}   — different, with title
        None                                          — VLM error
    """
    try:
        pix1 = doc[0].get_pixmap(dpi=dpi)
        pix2 = doc[page_idx].get_pixmap(dpi=dpi)
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

        ans = vlm(TRIM_PROMPT, composite, max_tokens=80)
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
        log.warning("Article-trim VLM check on page %d failed: %s",
                    page_idx + 1, e)
        return None


def trim_to_first_article(md: str, doc: "fitz.Document",
                          assets_dir: Path) -> tuple[str, int]:
    """If the PDF concatenates multiple articles, trim markdown to the first.

    Common in Nature News & Views (2-page bundles), journal-issue downloads
    (whole issue PDF), OCR'd archives (multiple papers glued together).

    Algorithm: check the LAST page first. If it's the same article as page 1,
    the whole PDF is one article and we return unchanged (1 VLM call).
    Otherwise binary-search backwards to find the first page that's part of
    a different article (log2(N) additional VLM calls). Cut the markdown at
    that boundary using the new article's title (returned by the VLM) for a
    string match, falling back to the first image reference whose page index
    is >= the boundary.

    Orphaned `_page_N_*` image files are deleted after a successful trim
    so they don't end up in the HDF5 bundle.

    Returns (trimmed_md, n_pages_trimmed). Soft-fails: any VLM error
    returns (md, 0) with the markdown and assets untouched. v1 still
    handles only ONE boundary per PDF — multi-article (3+) PDFs need a
    re-run on the trimmed output.
    """
    n = len(doc)
    if n < 2:
        return md, 0

    # 1 VLM call: is the LAST page the same article as page 1?
    last_check = _check_page_vs_first(doc, n - 1)
    if last_check is None:
        return md, 0
    if last_check["same"]:
        log.info("Article-trim: page %d is the same article as page 1; "
                 "no trim needed", n)
        return md, 0

    # Boundary exists. Binary search backward.
    # Invariant: page `lo` is SAME (lo starts at 0 = page 1, definitely same);
    #            page `hi` is DIFFERENT (hi starts at n-1, just confirmed).
    # Loop tightens until lo + 1 == hi, at which point hi is the first
    # DIFFERENT page (1-indexed: PDF page hi+1 is the first page of the
    # different article).
    lo, hi = 0, n - 1
    boundary_title = last_check["title"]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        check = _check_page_vs_first(doc, mid)
        if check is None:
            log.warning("Article-trim: VLM check failed at page %d; aborting",
                        mid + 1)
            return md, 0
        if check["same"]:
            lo = mid
        else:
            hi = mid
            if check["title"]:
                boundary_title = check["title"]

    cutoff_page = hi  # 0-indexed first DIFFERENT page
    n_trimmed = n - cutoff_page

    cutoff_idx: Optional[int] = None
    if boundary_title:
        cutoff_idx = _find_title_line_in_md(boundary_title, md)

    if cutoff_idx is not None:
        trimmed = "\n".join(md.splitlines()[:cutoff_idx]).rstrip() + "\n"
        log.info("Article-trim: cut at line %d (title %r); removed %d page(s) "
                 "starting at PDF page %d", cutoff_idx, boundary_title,
                 n_trimmed, cutoff_page + 1)
    else:
        # Title-not-matched is a strong signal that the VLM's
        # boundary claim was wrong: a real article boundary would
        # have a real title, which marker would have rendered to
        # markdown. The earlier image-reference fallback ("cut at
        # first page>=cutoff image") was too aggressive -- a single
        # smaller / less-reliable VLM hallucinating a boundary on
        # a single-article PDF could nuke 5 of 6 pages. Safer to
        # warn and leave the document untrimmed; if the PDF really
        # is multi-article, the user can edit by hand.
        log.warning("Article-trim: VLM reported boundary at page %d "
                    "with title %r, but the title was not found in the "
                    "marker markdown. Likely a VLM hallucination on a "
                    "single-article PDF; NOT trimming. Re-run with a "
                    "more capable VLM if your PDF really is multi-"
                    "article and the trim is needed. See USAGE.md "
                    "§5 (article-boundary trim).",
                    cutoff_page + 1, boundary_title)
        return md, 0

    deleted = _delete_orphan_page_assets(trimmed, assets_dir)
    if deleted:
        log.info("Article-trim: removed %d orphaned page-tagged asset(s)",
                 deleted)
    return trimmed, n_trimmed


# Some journals (e.g. older J. Appl. Phys.) cite via numbered footnotes at
# the bottom of each page rather than a consolidated bibliography. After
# marker conversion those footnote-reference lines end up scattered through
# the body. The consolidate pre-pass lifts them into a single References
# section at the end so downstream consumers (vector-DB chunking by section
# heading, especially) get a clean references chunk.
#
# Tags actually emitted by marker for these:
#   <sup>1</sup> F. W. Neilson and W. B. Benedick, Bull. Am. Phys. Soc...
# Plus a known garble where marker mis-OCRs nested <sup>:
#   <sup>&lt;sup>17</sup></sup> F. B. Bundy, Trans. Am. Soc. Mech. Eng...
# Plus the unrecoverable case where the digit itself is mangled and
# nothing useful remains, which we leave in place:
#   <sup>::</sup> P. W. Bridgman and I. Simon...
# Marker's actual nested-sup garble looks like `<sup>&</sup>lt;sup>N</sup>`
# (the `<` of `&lt;` appears to be lost during OCR rather than rendering as
# the entity), so the literal pattern in the markdown is:
#   <sup>&</sup>lt;sup>17</sup>
# Normalize that to plain `<sup>17</sup>` so the line-start match works.
FOOTNOTE_NESTED_RE = re.compile(r"<sup>&</sup>lt;sup>(\d+)</sup>")
_SUP_HEAD_RE = re.compile(r"<sup>(\d{1,3})</sup>")


def _looks_like_ref_body_start(s: str) -> bool:
    """True if `s` starts with characters typical of an author name (after
    optional whitespace and markdown bold/italic tokens). Used to discriminate
    a NEW footnote-reference boundary from an embedded `<sup>NNN</sup>`
    (year, page number) inside the body of an existing reference."""
    i = 0
    while i < len(s) and s[i] in " \t*_":
        i += 1
    return i < len(s) and s[i].isalpha()
_REFS_HEADING_RE = re.compile(
    r"^#+\s+(References|Bibliography|Notes)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def consolidate_footnote_references(md: str) -> tuple[str, int]:
    """Lift `<sup>N</sup>`-prefixed footnote-reference lines out of the body
    and append them as a single `## References` numbered list.

    Behavior:
      - Pre-normalizes marker's nested-`<sup>` garble so those refs are
        captured cleanly.
      - Operates only on lines whose first non-whitespace token is
        `<sup>N</sup>` (digit). In-body `<sup>3</sup>` citation markers
        appear mid-line and are not touched.
      - Lines whose <sup> is unreadable (e.g. `<sup>::</sup>`) don't match
        the digit regex and stay in place.
      - Multiple `<sup>N</sup>` segments on a single line (e.g.
        `<sup>30</sup>... <sup>31</sup>...`) are split into separate refs.
      - Idempotent: skipped if a `## References` / `## Bibliography` /
        `## Notes` heading already exists.
      - Refs sorted by number; duplicate numbers logged but kept (so the
        user can compare and pick the correct version).
      - Inserted before any `## VLM page rescues` block at the document
        end; otherwise appended.

    Returns (new_md, count_consolidated).
    """
    if _REFS_HEADING_RE.search(md):
        return md, 0

    md = FOOTNOTE_NESTED_RE.sub(r"<sup>\1</sup>", md)

    lifted: list[tuple[int, str]] = []
    keep_lines: list[str] = []
    seen: set[int] = set()
    duplicates: list[int] = []

    for line in md.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("<sup>") or not _SUP_HEAD_RE.match(stripped):
            keep_lines.append(line)
            continue

        all_positions = list(_SUP_HEAD_RE.finditer(stripped))
        # The leading <sup> at column 0 is always a ref boundary. Any later
        # <sup>NNN</sup> on the same line is treated as a NEW ref only if
        # what follows looks like the start of an author name (e.g.
        # "S. M. Stishov" or "**F.** R. Boyd") rather than an embedded
        # year/page superscript like " <sup>749</sup> (1960)" inside a body.
        ref_positions = all_positions[:1]
        for p in all_positions[1:]:
            if _looks_like_ref_body_start(stripped[p.end():]):
                ref_positions.append(p)
        line_refs: list[tuple[int, str]] = []
        for i, m in enumerate(ref_positions):
            num = int(m.group(1))
            body_start = m.end()
            body_end = (ref_positions[i + 1].start()
                        if i + 1 < len(ref_positions) else len(stripped))
            body = stripped[body_start:body_end].strip()
            if body:
                line_refs.append((num, body))

        if not line_refs:
            keep_lines.append(line)
            continue

        for num, body in line_refs:
            if num in seen:
                duplicates.append(num)
            seen.add(num)
            lifted.append((num, body))

    if not lifted:
        return md, 0

    if duplicates:
        log.warning("consolidate_footnote_references: duplicate ref number(s) "
                    "kept (compare to pick the correct version): %s",
                    sorted(set(duplicates)))

    lifted.sort(key=lambda x: x[0])
    refs_block = "## References\n\n" + "\n".join(
        f"{num}. {body}" for num, body in lifted)

    body = "\n".join(keep_lines).rstrip()
    rescue = re.search(r"^##\s+VLM page rescues\s*$", body, re.MULTILINE)
    if rescue:
        before = body[:rescue.start()].rstrip()
        after = body[rescue.start():]
        new_md = f"{before}\n\n{refs_block}\n\n{after}\n"
    else:
        new_md = f"{body}\n\n{refs_block}\n"
    return new_md, len(lifted)


# Marker emits each numbered reference in the bibliography as a bullet line
# anchored by a span id. The leading reference number is sometimes dropped
# (marker confuses it with a footnote-link superscript during column reflow);
# the surviving id-N is always sequential per PDF page. We use the surviving
# numbers as anchors to back-fill the missing ones.
REF_LINE_RE = re.compile(
    r'^(- <span id="page-(\d+)-(\d+)"></span>)\s*(\d+)?\.?\s*(.*)$',
    re.MULTILINE,
)

# Plain-span ref line (no leading bullet). Marker emits these for the
# first cluster of refs on the references page in some Elsevier/Icarus
# extracts; the bulleted form (REF_LINE_RE) appears for later refs.
# Re-bulleting these makes them visible to fix_reference_numbering.
_PLAIN_SPAN_REF_RE = re.compile(
    r'^<span id="page-(\d+)-(\d+)"></span>\s*\S',
)

# Refhub Elsevier cross-reference link wrappers: `[Title](http://refhub...)`.
# Strip to just the title text -- the URL is publisher-internal noise.
# The URL itself contains balanced inner parens (e.g.
# `S0019-1035(26)00036-9/sbXX`), so the link target uses CommonMark's
# balanced-parens rule. A naive `[^)]+` regex would stop at the first
# inner `)` and leave a trailing URL fragment behind. The alternation
# `[^()]+|\([^)]*\)` consumes either a non-paren run or one balanced
# paren pair, repeated until the final terminating `)`.
_REFHUB_LINK_RE = re.compile(
    r'\[([^\]]+)\]\(https?://refhub\.elsevier\.com/'
    r'(?:[^()]+|\([^)]*\))*\)'
)

# Split DOI links produced by Marker when column reflow breaks a DOI URL
# across two `[text](url)` segments. Form 1: `[http://dx.doi.org/](u1) [10.x/y](u2)`.
# Form 2: any escaped-paren DOI like `10.1016/0012-821X\(80\)90038-2`. We
# rejoin to a single canonical `<https://doi.org/...>` autolink. The
# captured DOI must look like a real DOI suffix (10.<registrant>/<rest>).
_SPLIT_DOI_LINK_RE = re.compile(
    r'\[(?:https?://)?(?:dx\.)?doi\.org/?\]\([^)]*\)\s*'
    r'\[(10\.[^\]]+)\]\([^)]*\)'
)

# Idempotent escape-undo for DOIs that came through with backslash-escaped
# parens (e.g. `10.1016/0012-821X\(80\)90038-2`). Applied only inside the
# rejoined `<...>` autolink, so we don't touch real markdown link escapes.
_ESCAPED_PAREN_RE = re.compile(r'\\([()])')


def normalise_span_anchored_refs(md: str) -> tuple[str, dict]:
    """Pre-pass for Elsevier/Icarus-style references that come out as
    span-anchored lines mixed with markdown-link clutter.

    Three transforms, all idempotent:
      1. Re-bullet plain `<span id="page-X-N">...` ref lines (no leading
         `- `) so the existing fix_reference_numbering pass can see them.
         We only bullet *contiguous runs* of such lines (>=2 in a row),
         to avoid mass-bulleting unrelated span anchors elsewhere.
      2. Rejoin split DOI links (`[http://dx.doi.org/](u) [10.x/y](u)`)
         into a single `<https://doi.org/10.x/y>` autolink. Backslash-
         escaped parens inside the DOI are unescaped.
      3. Strip refhub Elsevier cross-reference link wrappers
         (`[Title](http://refhub.elsevier.com/...)` -> `Title`).

    Returns (cleaned_md, counts_dict). counts_dict has keys
    `bulletted`, `dois_rejoined`, `refhub_stripped`."""
    counts = {"bulletted": 0, "dois_rejoined": 0, "refhub_stripped": 0}

    # Transform 1: re-bullet near-contiguous clusters of plain-span ref
    # lines. "Near-contiguous" allows a single blank line between matches
    # (Marker emits Icarus refs with a blank between every entry). A
    # cluster needs >= 2 plain-span lines to qualify, defending against
    # singleton incidental anchor spans elsewhere in the body.
    lines = md.split("\n")
    match_idx = [i for i, ln in enumerate(lines) if _PLAIN_SPAN_REF_RE.match(ln)]
    to_bullet: set[int] = set()
    i = 0
    while i < len(match_idx):
        j = i
        # Extend the cluster while the next match is within 2 lines (i.e.
        # consecutive or separated by exactly one blank line).
        while (j + 1 < len(match_idx)
               and match_idx[j + 1] - match_idx[j] <= 2):
            j += 1
        if j - i + 1 >= 2:
            for k in range(i, j + 1):
                to_bullet.add(match_idx[k])
        i = j + 1
    if to_bullet:
        for idx in to_bullet:
            lines[idx] = "- " + lines[idx]
        counts["bulletted"] = len(to_bullet)
    md = "\n".join(lines)

    # Transform 2: rejoin split DOI links.
    def _rejoin(m: re.Match) -> str:
        doi = _ESCAPED_PAREN_RE.sub(r"\1", m.group(1)).rstrip(".")
        counts["dois_rejoined"] += 1
        return f"<https://doi.org/{doi}>"
    md = _SPLIT_DOI_LINK_RE.sub(_rejoin, md)

    # Transform 3: strip refhub link wrappers.
    def _strip_refhub(m: re.Match) -> str:
        counts["refhub_stripped"] += 1
        return m.group(1)
    md = _REFHUB_LINK_RE.sub(_strip_refhub, md)

    return md, counts


def fix_reference_numbering(md: str) -> tuple[str, int]:
    """Repair missing numeric prefixes on bibliography lines.

    Detects bullets of the form `- <span id="page-X-N"></span>...` and, for
    each PDF page that has at least one reference with a surviving leading
    number, computes the per-page offset (`number = N + offset`) and fills
    in the missing numbers on every other reference on that page.

    Pages with no surviving anchor are left alone (would require guessing).
    Returns (cleaned_md, count_repaired). Idempotent: lines that already
    have a number are unchanged.
    """
    matches = list(REF_LINE_RE.finditer(md))
    if not matches:
        return md, 0

    # Per-page anchor: pick the first surviving (id_n -> num) on each page.
    page_offsets: dict[int, int] = {}
    page_items: dict[int, list[tuple[int, Optional[int]]]] = {}
    for m in matches:
        page = int(m.group(2))
        id_n = int(m.group(3))
        captured = int(m.group(4)) if m.group(4) else None
        page_items.setdefault(page, []).append((id_n, captured))
        if captured is not None and page not in page_offsets:
            page_offsets[page] = captured - id_n

    # Cross-page propagation: if page P has no anchor but P+1 does, infer
    # P's offset from P+1 assuming references are contiguous.
    pages_sorted = sorted(page_items)
    for i, page in enumerate(pages_sorted):
        if page in page_offsets:
            continue
        # forward: previous-page-end + 1 = this-page-id-0
        if i > 0:
            prev = pages_sorted[i - 1]
            if prev in page_offsets:
                prev_count = len(page_items[prev])
                page_offsets[page] = page_offsets[prev] + prev_count
                continue
        # backward: next-page-id-0 - this-page-count = this-page-id-0
        if i < len(pages_sorted) - 1:
            nxt = pages_sorted[i + 1]
            if nxt in page_offsets:
                this_count = len(page_items[page])
                page_offsets[page] = page_offsets[nxt] - this_count

    # Second backward sweep so a chain of unanchored pages can resolve once
    # a downstream anchor is found.
    for i in range(len(pages_sorted) - 2, -1, -1):
        page = pages_sorted[i]
        if page in page_offsets:
            continue
        nxt = pages_sorted[i + 1]
        if nxt in page_offsets:
            this_count = len(page_items[page])
            page_offsets[page] = page_offsets[nxt] - this_count

    repaired = 0

    def _sub(m: re.Match) -> str:
        nonlocal repaired
        page = int(m.group(2))
        id_n = int(m.group(3))
        captured = m.group(4)
        body = m.group(5)
        if captured is not None:
            return m.group(0)  # already numbered
        if page not in page_offsets:
            return m.group(0)  # no inference possible
        num = id_n + page_offsets[page]
        if num < 1:
            return m.group(0)  # garbage offset, don't write nonsense
        repaired += 1
        return f"{m.group(1)} {num}. {body}".rstrip()

    return REF_LINE_RE.sub(_sub, md), repaired


# Running-footer pattern: AGU / Wiley / many society journals interleave
# "<AUTHOR> ET AL.  N of M" between body paragraphs as the running footer.
# Marker leaves these in the markdown body. Restrict the prefix to a
# bounded length so we don't accidentally clobber prose that mentions
# "et al" followed somewhere later by an "N of M" fragment; require the
# "N of M" to be immediately after "et al" with only a single token gap.
RUNNING_FOOTER_RE = re.compile(
    r"^[^|\n]{1,60}?\bet\s+al\.?,?\s+\d+\s+of\s+\d+\s*$",
    re.IGNORECASE,
)

# Bare publication-type label that appears as a standalone page-header
# in many journals (Nature, Nature Geoscience, etc.). Matched line-
# leading after stripping any markdown bold wrapper.
_PUB_TYPE_LABEL_RE = re.compile(
    r"^(?:LETTERS?|ARTICLES?|REVIEWS?|PERSPECTIVES?|"
    r"COMMENTAR(?:Y|IES)|EDITORIAL|FRONT\s+MATTER|"
    r"BRIEF\s+COMMUNICATIONS?|RESEARCH\s+ARTICLES?|"
    r"NEWS\s*(?:&|AND)\s*VIEWS|PROGRESS\s+ARTICLE|"
    r"ANALYSIS|INSIGHT|FOCUS|HIGHLIGHTS?)\s*$",
    re.IGNORECASE,
)

# Journal-name keywords that, in combination with "DOI:" + a publication-
# type word, identify a Nature-style page header line (often interleaved
# between body paragraphs by marker).
_PAGE_HEADER_JOURNAL_RE = re.compile(
    r"\bNATURE\s+\w+(?:\s+\w+)?\b", re.IGNORECASE
)
_PAGE_HEADER_DOI_RE = re.compile(r"\bDOI:\s*\S", re.IGNORECASE)
_PAGE_HEADER_PUB_TYPE_RE = re.compile(
    r"\b(?:LETTERS?|ARTICLES?|REVIEWS?)\b", re.IGNORECASE
)


def _looks_like_page_header_line(line: str) -> bool:
    """Heuristic: does this line look like a journal page-header
    artifact (e.g. 'LETTERS', '**NATURE GEOSCIENCE DOI: [...]**
    LETTERS', 'LETTERS NATURE GEOSCIENCE DOI: 10.1038/...')?

    Two recognized forms:
      Form A: bare publication-type label, optionally bold-wrapped
              ('LETTERS', '**ARTICLES**', 'Brief Communication').
      Form B: a single line that contains a Nature-style journal
              name AND 'DOI:' AND a publication-type word, total
              length under 250 chars (reference entries with the
              same words are typically much longer).
    """
    s = line.strip()
    if not s or len(s) > 250:
        return False
    # Strip markdown bold/italic markers and inline links so the
    # underlying content matches whether or not marker emphasized it.
    plain = re.sub(r"\*{1,3}", "", s)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
    plain = plain.strip()
    if _PUB_TYPE_LABEL_RE.match(plain):
        return True
    if (_PAGE_HEADER_JOURNAL_RE.search(plain)
            and _PAGE_HEADER_DOI_RE.search(plain)
            and _PAGE_HEADER_PUB_TYPE_RE.search(plain)):
        return True
    return False


def strip_journal_page_headers(md: str) -> tuple[str, int]:
    """Remove journal page-header artifacts that marker preserved
    from the PDF text layer (e.g. Nature's 'LETTERS' label and the
    'NATURE GEOSCIENCE DOI: [...]' header line interleaved between
    body paragraphs). Returns (cleaned_md, count_stripped).

    Skips lines inside fenced code blocks, markdown tables, and
    list items (so reference entries that happen to mention 'LETTER'
    or contain a journal-name + DOI aren't accidentally dropped).
    """
    out_lines: list[str] = []
    in_code = False
    stripped = 0
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code or s.startswith("|"):
            out_lines.append(line)
            continue
        # Bullet / numbered-list lines are reference entries in our
        # corpus and routinely contain journal names + DOIs.
        if s.startswith("- ") or s.startswith("* ") or re.match(r"^\d+\.\s", s):
            out_lines.append(line)
            continue
        if _looks_like_page_header_line(line):
            stripped += 1
            continue
        out_lines.append(line)
    return "\n".join(out_lines), stripped


# Per-publisher download-watermark patterns. Each subscription PDF
# server stamps every page of the downloaded copy with a multi-line
# block giving the DOI URL, downloading institution + date, and
# Terms-and-Conditions / Creative-Commons attribution. Marker
# preserves these as body paragraphs, where they accumulate on
# every page (8x in an 8-page paper) and frequently split a real
# paragraph or equation. Patterns below are line-level: each pattern
# matches one or more lines of a stamp block. They're checked
# anywhere in the line (re.search), case-insensitive.
_PUBLISHER_STAMP_LINE_RES = (
    # Wiley Online Library download stamp. Three-line block (URL +
    # institution + date + Terms&Conditions wrap) seen on
    # subscription downloads from onlinelibrary.wiley.com:
    #   onlinelibrary.wiley.com/doi/10.1029/JB093iB06p06477 by
    #   Arizona State University, Wiley Online Library on
    #   [01/05/2026]. See the Terms and Conditions
    #   (https://onlinelibrary.wiley.com/terms-and-conditions) on
    #   Wiley Online Library for rules of use; OA articles are
    #   governed by the applicable Creative Commons License
    #
    # Pattern 0 is anchored at line start (with optional scheme +
    # subdomain prefix) so it matches the stamp's bare-URL line
    # WITHOUT also matching legitimate Wiley URL citations in body
    # text or reference lists, which always appear mid-line after
    # "Retrieved from " or inside parentheses. Patterns 1-5 carry
    # boilerplate phrases that don't appear in normal scientific
    # writing.
    re.compile(r"^\s*(?:https?://)?(?:[a-z0-9-]+\.)?onlinelibrary\.wiley\.com",
               re.IGNORECASE),
    re.compile(r"Wiley Online Library on \[\d", re.IGNORECASE),
    re.compile(r"Wiley Online Library for rules of use", re.IGNORECASE),
    re.compile(r"See the Terms and Conditions \(https?://onlinelibrary",
               re.IGNORECASE),
    re.compile(r"^\s*(?:and|terms)-conditions\)\s*on Wiley", re.IGNORECASE),
    re.compile(r"OA articles are governed by the applicable "
               r"Creative Commons", re.IGNORECASE),
)


def strip_publisher_stamps(md: str) -> tuple[str, int]:
    """Remove per-page download watermarks that subscription publishers
    add to PDFs at delivery time. Currently handles Wiley Online
    Library's 3-4 line per-page stamp (DOI URL + downloading
    institution + date + Terms-and-Conditions / Creative Commons
    boilerplate). The stamp is per-page so a single paper accumulates
    many copies; marker emits each fragment as a separate paragraph
    that interrupts the body.

    Idempotent: clean documents return unchanged with count=0.
    Skips lines inside fenced code blocks and markdown tables so
    DOI URLs in legitimate citation lists aren't accidentally
    dropped (the stamp regexes are URL-fragment-or-boilerplate
    matches, not exact-DOI matches, so a paper that references its
    own DOI in body prose would be at risk without this guard).

    Returns (cleaned_md, n_lines_stripped). After stripping, runs
    of 3+ blank lines collapse back to 2 so the document doesn't
    accumulate visual gaps where the stamps used to live."""
    lines = md.split("\n")
    out: list[str] = []
    in_code = False
    n_stripped = 0
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or s.startswith("|"):
            out.append(line)
            continue
        if any(p.search(line) for p in _PUBLISHER_STAMP_LINE_RES):
            n_stripped += 1
            continue
        out.append(line)
    if n_stripped == 0:
        return md, 0
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, n_stripped


# Reference-section heading: matches at any markdown depth (#, ##, ###,
# ####) and tolerates bold-wrapped titles ("# **References**"), trailing
# colon/punctuation, and a wide set of section-title variants:
#   References / Reference / Bibliography / Cited References
#   Notes and References / References and Notes / References & Notes
#   Method[s] References (Nature Methods convention)
# Case-insensitive so all-caps section titles ("REFERENCES",
# "**REFERENCES AND NOTES**") match too. Leading "Notes and " /
# "Method[s] " and trailing " and Notes" cover journal-specific
# variants seen in Science (Refs+Notes inline) and Nature
# (Methods has its own References subsection).
_REFS_HEADING_RE = re.compile(
    r"(?m)^(?P<hashes>#{1,4})\s*(?:\*{0,2}\s*)?"
    r"(?P<title>"
    r"(?:Notes\s+and\s+|Methods?\s+)?References?(?:\s+(?:and|&)\s+Notes?)?"
    r"|Bibliography"
    r"|Cited\s+References?"
    r")"
    r"\s*(?:\*{0,2}\s*)?[\.:]?\s*$",
    re.IGNORECASE,
)


def merge_reference_sections(md: str) -> tuple[str, int]:
    """Consolidate multiple `## References` (or peer aliases) into a
    single section at the very end of the document.

    Many journals append a separate references list after the Methods
    section (Nature's Methods + Methods References pattern). For
    chunked-by-section embedding the cleanest output is one
    consolidated reference list at the end of the document, with the
    Methods (and any other sections) preserved in their original
    positions before it.

    Algorithm:
      1. Find every heading whose title matches references/bibliography
         (any depth, optional bold).
      2. For each heading, capture the section span (from the heading
         to the next heading at the SAME or higher depth, or end of
         file).
      3. If 0 or 1 sections found, return md unchanged.
      4. Otherwise: concatenate the bodies (heading line dropped) in
         document order, remove the original sections from md, and
         append `## References\n\n<concatenated body>` at the end.

    Returns (cleaned_md, n_sections_merged). When the function returns
    n_sections_merged > 0 the body has been mutated even when the
    final section count is 1 (we still emit the canonical `##
    References` header).
    """
    matches = list(_REFS_HEADING_RE.finditer(md))
    if len(matches) < 2:
        return md, 0

    # Find the section span for each match. End is the start of the
    # next heading at SAME or higher depth (depth = number of #).
    heading_re = re.compile(r"(?m)^(#{1,4})\s+\S")
    spans: list[tuple[int, int, str]] = []  # (start, end, body_text)
    for i, m in enumerate(matches):
        depth = len(m.group("hashes"))
        section_start = m.start()
        body_start = m.end()
        # Find next heading at depth <= this one.
        end = len(md)
        for nm in heading_re.finditer(md, body_start):
            nd = len(nm.group(1))
            if nd <= depth:
                end = nm.start()
                break
        body = md[body_start:end].strip("\n")
        spans.append((section_start, end, body))

    # Build the output: remove every reference section, then append
    # a single consolidated one. Splice in reverse order so positions
    # remain valid.
    out = md
    for start, end, _ in sorted(spans, key=lambda t: t[0], reverse=True):
        out = out[:start] + out[end:]
    # Strip any trailing whitespace introduced by the splices.
    out = out.rstrip() + "\n\n"
    consolidated_body = "\n\n".join(b for _, _, b in spans if b.strip())
    out += f"## References\n\n{consolidated_body}\n"
    return out, len(spans)


# Bullet-prefixed reference entry: bullet, optional <span> page
# anchor, then a digit (the ref number, with or without a separating
# period or whitespace). Catches marker's signature for numbered
# reference entries even when the leading author name was joined to
# the number ("- 47A. F. Danilyuk, ...") or split across columns
# ("- <span id='page-17-0'></span>46From manufacturers ...").
_BULLETED_NUMBERED_REF_RE = re.compile(
    r"^\s*[-*]\s+"
    r"(?:<span\s+id=\"[^\"]*\"></span>\s*)?"
    r"\d+",
    re.MULTILINE,
)


# Citation-year sniff: a 4-digit year inside parens or as a standalone
# token. Used to discriminate real reference clusters (most lines
# carry a year) from enumerated body lists ("1. Install Python", "2.
# Open the terminal", AASTEX feature lists) that share the bulleted-
# number shape but never carry a publication year.
_REF_YEAR_HINT_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")


def _refs_section_line_spans(md: str) -> list[tuple[int, int]]:
    """Return (start_line_idx, end_line_idx) line spans of every
    existing `## References` (or alias) section. End is exclusive.
    Used by inject_orphan_ref_clusters to filter out runs that are
    already inside a section."""
    matches = list(_REFS_HEADING_RE.finditer(md))
    if not matches:
        return []
    next_h_re = re.compile(r"(?m)^(#{1,6})\s+\S")
    spans: list[tuple[int, int]] = []
    for m in matches:
        depth = len(m.group("hashes"))
        section_start_pos = m.start()
        body_start_pos = m.end()
        end_pos = len(md)
        for nm in next_h_re.finditer(md, body_start_pos):
            if len(nm.group(1)) <= depth:
                end_pos = nm.start()
                break
        start_line = md[:section_start_pos].count("\n")
        end_line = md[:end_pos].count("\n")
        spans.append((start_line, end_line))
    return spans


def inject_orphan_ref_clusters(md: str) -> tuple[str, int]:
    """Detect bulleted reference clusters that aren't inside any
    existing ## References section and prepend a synthetic
    ## References heading to each, so merge_reference_sections can
    consolidate them with the existing section in the next pass.

    Marker can split a multi-page References section across a
    column or page boundary in such a way that some entries land
    above (or far away from) the heading line. paper2md's
    merge_reference_sections only fires when ≥2 explicit reference
    headings are present; with ONE heading and a free-floating
    cluster (the Knudson2013 case: refs 46-58 above a `##
    References` heading that owns refs 1-32, 35-45), the orphan
    cluster is silently dropped from the consolidated output.

    Detection is conservative: a contiguous run of ≥3 lines whose
    bullet is immediately followed by an optional <span id=...></span>
    anchor and a digit. Marker emits this format for numbered
    reference entries even when the leading author name was joined
    to the number. A blank-line gap of up to 2 lines is tolerated
    within a run. Runs entirely inside an existing References-
    section line span are skipped.

    Returns (cleaned_md, n_clusters_marked). Idempotent: a clean
    document returns unchanged. Skipped silently when no orphan
    clusters are detected."""
    section_spans = _refs_section_line_spans(md)

    def _in_existing_section(line_idx: int) -> bool:
        return any(s <= line_idx < e for s, e in section_spans)

    lines = md.split("\n")
    n_lines = len(lines)

    # Walk the lines, accumulating contiguous runs of bulleted-
    # numbered ref lines. Allow up to 2 blank lines within a run
    # (paragraph separators between entries) without breaking.
    runs: list[tuple[int, int, int]] = []  # (start_line, end_line, hits)
    i = 0
    while i < n_lines:
        if not _BULLETED_NUMBERED_REF_RE.match(lines[i]):
            i += 1
            continue
        run_start = i
        last_hit = i
        hits = 1
        j = i + 1
        while j < n_lines:
            if _BULLETED_NUMBERED_REF_RE.match(lines[j]):
                hits += 1
                last_hit = j
                j += 1
                continue
            if lines[j].strip() == "":
                # Blank gap: tolerate up to 2 in a row, then break.
                blanks = 0
                while j < n_lines and lines[j].strip() == "":
                    blanks += 1
                    j += 1
                if blanks > 2:
                    break
                continue
            # Non-blank, non-matching line ends the run.
            break
        if hits >= 3:
            # Confidence filter: at least half of the cluster's lines
            # must contain a 4-digit year. Real references almost
            # always carry a publication year; enumerated body lists
            # ("1. Install Python", AASTEX feature lists) never do.
            cluster_lines = [lines[k] for k in range(run_start, last_hit + 1)
                             if _BULLETED_NUMBERED_REF_RE.match(lines[k])]
            year_hits = sum(1 for ln in cluster_lines
                            if _REF_YEAR_HINT_RE.search(ln))
            if year_hits * 2 >= hits:
                runs.append((run_start, last_hit + 1, hits))
        i = max(j, last_hit + 1)

    # Filter to runs entirely outside existing References sections.
    orphan_runs = [(s, e, h) for s, e, h in runs
                   if not _in_existing_section(s)
                   and not _in_existing_section(e - 1)]
    if not orphan_runs:
        return md, 0

    # Inject synthetic heading + blank line above each orphan
    # cluster, in reverse line order so earlier indices stay valid.
    for start, _, _ in sorted(orphan_runs, key=lambda r: r[0], reverse=True):
        lines = lines[:start] + ["## References", ""] + lines[start:]
    return "\n".join(lines), len(orphan_runs)


# Two citation-line shapes, both anchored at line start (after any
# leading <span id=...></span> page anchors and an optional list
# number):
#   shape A -- surname or collaboration first: "Lastname, F. M., ...",
#     "Astropy Collaboration, Robitaille, T. P., ...". The Capitalized
#     run before the first comma can be one or more space-separated
#     words (handles AAS-style collaboration leads). The post-comma
#     token is either an initial ("G.", "Yu.") or another surname-like
#     capitalized word ("Robitaille") -- some journals omit author
#     initials and just chain surnames. Names like "Zel'dovich" /
#     "O'Hara" rely on the apostrophe in [\w'.-]; hyphenated surnames
#     ("Smith-Jones") flow through it too.
#   shape B -- numbered + initials first: "1. F. W. Neilson and W. B.
#     Benedick, Bull. Am. Phys. Soc. ...". Common in older physics
#     papers (wackerle 1962). Requires the leading number; without
#     it the same shape is the corresponding-author address line
#     ("M. B. Boslough, Division 1131, ..."), caught by the addendum
#     regex below (which is checked first, so collaboration-style
#     "Astropy Collaboration, ..." citations don't get misclassified
#     as addresses by the multi-word allowance).
_CITATION_OPEN_RE = re.compile(
    r"^(?:<span\s+id=\"[^\"]*\"></span>\s*)*"
    r"(?:"
    r"(?:\d+\.\s+)?"
    r"[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)*"
    r",\s+(?:[A-Z][a-z]?\.|[A-Z][a-z]+)"
    r"|"
    r"\d+\.?\s+(?:[A-Z]\.?\s*){1,3}[A-Z][a-z]+"
    r")"
)

# Tail material that often gets bulleted alongside real references but
# isn't itself a citation: the corresponding-author address ("M. B.
# Boslough, Division 1131, ...") and the manuscript-history note
# ("(Received June 15, 1987; revised ...)"). The address pattern is
# "initials-first-then-surname-then-comma" -- the inverse of citation
# style -- so a surname-leading citation like "Boslough, M. B., A
# model for ..." will NOT match.
_REF_ADDENDUM_RE = re.compile(
    r"^(?:<span\s+id=\"[^\"]*\"></span>\s*)*"
    r"(?:"
    r"\([Rr]eceived\b"
    r"|[Mm]anuscript\s+received\b"
    r"|(?:[A-Z]\.\s+){1,3}[A-Z][a-z]+,"
    r")"
)


def normalise_references_section(md: str) -> tuple[str, int]:
    """Tidy the references list inside a `## References` (or peer alias)
    section so the formatting is internally consistent.

    Three transforms, applied in order, all idempotent:
      1. Merge column-break continuation lines into the entry they
         belong to. Marker emits split entries when a reference wraps
         from one column to the next, e.g.
             Lyzenga, G. A., T. J. Ahrens, and A. C. Mitchell, Shock temperatures

             - of $\\mathrm{SiO}_2$ and their geophysical implications, J. Geophys. Res., 88, 2431-2444, 1983.
         which gets joined into a single entry.
      2. Pull the corresponding-author address line and the manuscript-
         history "(Received ...)" parenthetical out of the bullet
         list -- they're frequently emitted as bullets even though
         they're not citations -- and re-emit them as plain
         paragraphs after the references list.
      3. Apply uniform formatting: every citation becomes a "- " list
         item, separated from neighbours by a blank line, addenda
         appear after a blank line as plain paragraphs. References
         that already match this layout are unchanged (idempotent).

    Skipped when the document has no `## References` heading.

    Returns (cleaned_md, n_changed) where n_changed counts the number
    of continuations merged plus addenda extracted from the bullet
    list. Zero on no-op runs and on already-clean sections."""
    m = _REFS_HEADING_RE.search(md)
    if m is None:
        return md, 0
    header_end = m.end()
    depth = len(m.group("hashes"))

    # Section ends at the next heading of equal or shallower depth, or
    # at end-of-document. (Don't stop at a deeper heading -- e.g. a
    # ##### appendix-style sub-heading inside the references.)
    next_h_re = re.compile(r"(?m)^(#{1,6})\s+\S")
    section_end = len(md)
    for nm in next_h_re.finditer(md, header_end):
        if len(nm.group(1)) <= depth:
            section_end = nm.start()
            break

    # Skip leading blank lines after the header.
    body_start = header_end
    while body_start < section_end and md[body_start] in "\n\r":
        body_start += 1
    body = md[body_start:section_end].rstrip()
    if not body.strip():
        return md, 0

    # Walk the section line-by-line. Each non-blank line is a unit:
    #   - Lines starting with a citation pattern open a new entry,
    #     whether they're bulleted or not.
    #   - Lines starting with the addendum pattern open a new addendum
    #     entry.
    #   - Anything else folds into the current entry (column-break
    #     continuations, mid-sentence wraps emitted as a separate
    #     line / bullet by marker).
    # Blank lines are paragraph hints but do NOT force a boundary --
    # marker emits blanks both between citations and within a single
    # citation's wrapped lines, so the citation regex is the only
    # reliable boundary signal.
    entries: list[tuple[str, str]] = []  # (text, kind) kind = "cite" | "add"
    n_continuations_merged = 0
    n_addenda_pulled = 0

    def _strip_bullet(s: str) -> tuple[str, bool]:
        if s.startswith("- ") or s.startswith("* "):
            return s[2:].lstrip(), True
        return s, False

    for raw_line in body.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        text, was_bulleted = _strip_bullet(stripped)
        if _REF_ADDENDUM_RE.match(text):
            entries.append((text, "add"))
            if was_bulleted:
                n_addenda_pulled += 1
        elif _CITATION_OPEN_RE.match(text):
            entries.append((text, "cite"))
        else:
            # Continuation of preceding entry. If there's no preceding
            # entry (stray opening text), keep it as a citation so the
            # content isn't lost; safer than dropping.
            if entries:
                prev_text, prev_kind = entries[-1]
                entries[-1] = (prev_text + " " + text, prev_kind)
                n_continuations_merged += 1
            else:
                entries.append((text, "cite"))

    citations = [t for (t, k) in entries if k == "cite"]
    addenda = [t for (t, k) in entries if k == "add"]
    if not citations:
        # All addenda or empty -- don't reformat (the user might be
        # looking at a non-references section that happened to match
        # the heading regex). No-op.
        return md, 0

    out_lines: list[str] = []
    for i, text in enumerate(citations):
        if i:
            out_lines.append("")
        out_lines.append(f"- {text}")
    for text in addenda:
        out_lines.append("")
        out_lines.append(text)
    new_body = "\n".join(out_lines).rstrip() + "\n"

    # Idempotency check: if the new body is byte-identical to the
    # current normalised body, return md unchanged so callers can
    # skip the log message and so multiple passes are stable.
    if md[body_start:section_end].rstrip() + "\n" == new_body:
        return md, 0

    rebuilt = md[:body_start] + new_body
    if section_end < len(md):
        # Preserve the trailing newline boundary before the next
        # heading / EOF so the structure isn't disturbed.
        rebuilt = rebuilt.rstrip() + "\n\n" + md[section_end:].lstrip("\n")
    return rebuilt, n_continuations_merged + n_addenda_pulled


def strip_running_footers(md: str) -> tuple[str, int]:
    """Remove "<AUTHOR> ET AL.  N of M" running-footer lines.

    Common in AGU / Wiley / many society journals; marker preserves
    them as standalone body lines. The match is bounded (≤ 60 chars
    before "et al") and pinned at end-of-line so legitimate prose
    that happens to contain "et al" + numbers is not affected.
    Skips lines inside fenced code blocks and markdown tables.
    Returns (cleaned_md, count_stripped).
    """
    out_lines: list[str] = []
    in_code = False
    stripped = 0
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code or s.startswith("|"):
            out_lines.append(line)
            continue
        if RUNNING_FOOTER_RE.match(s):
            stripped += 1
            continue
        out_lines.append(line)
    return "\n".join(out_lines), stripped


# LaTeX `\label{...}` and `\eqref{...}` are cross-reference commands
# that KaTeX (the in-browser math renderer used by every common
# Markdown viewer) does not implement. Surya's equation OCR sometimes
# preserves them verbatim from the PDF text layer, which then makes
# the markdown un-renderable. \label is purely a LaTeX-side handle for
# `\ref{}`/`\eqref{}` lookup; stripping it is lossless for the rendered
# math. \eqref is rarer in extracted text (the resolved number usually
# wins), but if seen we replace with the bare bracketed argument so a
# reader at least sees a hint of the reference target.
_LABEL_RE = re.compile(r"\\label\{[^}]*\}")
_EQREF_RE = re.compile(r"\\eqref\{([^}]*)\}")


def strip_math_labels(md: str) -> tuple[str, int]:
    """Strip LaTeX cross-reference commands KaTeX can't render.
    Returns (cleaned_md, count_stripped). Counts \\label removals; \\eqref
    rewrites are not counted (rare and lossy in either direction)."""
    new_md, n_label = _LABEL_RE.subn("", md)
    new_md = _EQREF_RE.sub(r"(\1)", new_md)
    return new_md, n_label


def strip_line_numbers(md: str) -> tuple[str, int]:
    """Remove stray line-number artifacts that some PDFs embed in the
    text layer (common in pre-submission manuscripts using the LaTeX
    `lineno` package -- AASTeX, MNRAS, APS templates all do this).

    A line number is a lone 1-4 digit integer on its own line, outside
    fenced code blocks and markdown tables. Returns (cleaned_md,
    count_stripped).

    Safe against legitimate content:
    - Ordered-list items keep their punctuation ("1.", "2)") so a bare
      "1" never matches a list marker.
    - Table rows start with "|" so they're skipped.
    - Numeric cells inside fenced code blocks are skipped.
    """
    out_lines: list[str] = []
    in_code = False
    stripped = 0
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue
        if s.startswith("|"):
            out_lines.append(line)
            continue
        if re.fullmatch(r"\d{1,4}", s):
            stripped += 1
            continue
        out_lines.append(line)
    return "\n".join(out_lines), stripped


def sparse_pages(doc: fitz.Document) -> list[int]:
    out = []
    for i, page in enumerate(doc):
        chars = len(page.get_text("text"))
        area = page.rect.width * page.rect.height
        if area > 0 and chars / area < SPARSE_CHARS_PER_PT2:
            out.append(i)
    return out


class PyMuPDFFinder:
    """Table candidates from PyMuPDF's page.find_tables(). Reliable on
    ruled tables; misses borderless ones."""
    name = "pymupdf"

    def candidates(self, doc: fitz.Document, page_idx: int):
        page = doc[page_idx]
        try:
            tables = page.find_tables().tables
        except Exception:
            return []
        out = []
        for t in tables:
            try:
                cells = t.extract()
            except Exception:
                continue
            text = " ".join((c or "") for row in cells for c in row)
            out.append((tuple(t.bbox), text))
        return out


class TATRFinder:
    """Table candidates from Microsoft Table Transformer (detection only).
    Catches borderless tables that PyMuPDF misses. Extra deps:
    `pip install transformers torch timm`."""
    name = "tatr"

    def __init__(self, detect_dpi: int = 150, threshold: float = 0.7):
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        import torch

        self.torch = torch
        forced = os.environ.get("TORCH_DEVICE", "").lower()
        if forced == "cpu":
            self.device = "cpu"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        log.info("Loading Table Transformer (detection) on %s", self.device)
        model_id = "microsoft/table-transformer-detection"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.detect_dpi = detect_dpi
        self.threshold = threshold
        self._cache: dict[int, list] = {}

    def candidates(self, doc: fitz.Document, page_idx: int):
        if page_idx in self._cache:
            return self._cache[page_idx]
        page = doc[page_idx]
        img = render_page(doc, page_idx, dpi=self.detect_dpi)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]
        scale = 72 / self.detect_dpi  # image pixels -> PDF points
        out = []
        for box in results["boxes"].tolist():
            bbox = tuple(c * scale for c in box)
            text = page.get_textbox(fitz.Rect(bbox)) or ""
            out.append((bbox, text))
        self._cache[page_idx] = out
        return out


class DoclingFinder:
    """Table candidates from IBM Docling (TableFormer). Strong on
    borderless and multi-row-header tables, including scanned papers
    where PyMuPDF and TATR both miss. One whole-PDF conversion is run
    on first access (~50 s for a 16-page scan) and cached per PDF
    path; subsequent per-page candidate queries are O(1) lookups.

    Optional dep since v0.3.x. Not installed by default in
    environment-mac.yml / environment-gpu.yml; install explicitly with
    `pip install docling==2.92.0` if you want this finder under
    --layout-source=marker. Under the default --layout-source=mineru,
    MinerU's own table extractor handles borderless / scanned tables,
    so docling adds nothing.
    """
    name = "docling"

    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise RuntimeError(
                "docling is not installed. The 'docling' table finder is "
                "optional since v0.3.x. To use it under "
                "--layout-source=marker, install it explicitly:\n"
                "    pip install docling==2.92.0\n"
                "Or pick a different finder: --table-finder pymupdf "
                "(default), tatr, or both."
            ) from e
        log.info("Loading docling (first run downloads ~500MB models)...")
        self.converter = DocumentConverter()
        # path -> {page_idx: [(bbox, text), ...]}
        self._cache: dict[str, dict[int, list]] = {}

    def candidates(self, doc: fitz.Document, page_idx: int):
        path = doc.name
        if path not in self._cache:
            self._cache[path] = self._convert(doc, path)
        return self._cache[path].get(page_idx, [])

    def _convert(self, doc: fitz.Document,
                 path: str) -> dict[int, list]:
        from docling_core.types.doc.base import CoordOrigin
        log.info("docling: converting %s", path)
        t0 = time.time()
        result = self.converter.convert(path)
        log.info("docling: convert took %.1fs (%d tables)",
                 time.time() - t0, len(result.document.tables))
        per_page: dict[int, list] = {}
        for tbl in result.document.tables:
            if not tbl.prov:
                continue
            page_idx = tbl.prov[0].page_no - 1
            bb = tbl.prov[0].bbox
            # PyMuPDF's Rect is top-left-origin in PDF points; docling
            # uses BOTTOMLEFT for native PDFs. Flip Y when needed using
            # the actual page height (not a page-size assumption).
            if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
                h = doc[page_idx].rect.height
                bbox = (bb.l, h - bb.t, bb.r, h - bb.b)
            else:
                bbox = (bb.l, bb.t, bb.r, bb.b)
            try:
                text = tbl.export_to_markdown(result.document)
            except Exception:
                text = ""
            per_page.setdefault(page_idx, []).append((bbox, text))
        return per_page


class ChainedFinder:
    """Unions candidates from multiple finders. First match wins in
    find_pdf_table, so cheaper/more-precise finders should come first."""
    name = "chained"

    def __init__(self, *finders):
        self.finders = finders

    def candidates(self, doc: fitz.Document, page_idx: int):
        out = []
        for f in self.finders:
            out.extend(f.candidates(doc, page_idx))
        return out


def build_table_finder(name: str):
    if name == "pymupdf":
        return PyMuPDFFinder()
    if name == "tatr":
        return TATRFinder()
    if name == "both":
        return ChainedFinder(PyMuPDFFinder(), TATRFinder())
    if name == "docling":
        return DoclingFinder()
    if name == "all":
        # PyMuPDF first (cheap & precise on ruled tables), then TATR
        # (image-detection for borderless), then Docling (scanned-doc
        # specialist). First-match-wins in find_pdf_table, so the
        # cheapest finder gets the first crack at each table.
        return ChainedFinder(PyMuPDFFinder(), TATRFinder(), DoclingFinder())
    raise ValueError(f"unknown table finder: {name}")


# Characters we treat as "LaTeX markup noise" when deciding whether a
# candidate anchor is too LaTeX-dense to be matchable.
_LATEX_CHARS = set(r"${}\[]^_")


def _normalize_for_match(s: str) -> str:
    """Strip marker's LaTeX markup + typographic artifacts so a markdown
    cell value can be matched against PyMuPDF's Unicode text layer. Both
    sides of the `token in text` check go through this so they meet in
    the middle."""
    if not s:
        return ""
    # Unwrap bold / mathrm / text / mathit / operatorname: \textbf{X} -> X.
    # Run repeatedly to handle nested usage.
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\(?:textbf|mathbf|mathrm|text|mathit|operatorname|boldsymbol)\s*\{([^{}]*)\}",
                   r"\1", s)
    # Drop array/equation/environment wrappers entirely.
    s = re.sub(r"\\begin\{[^{}]*\}", " ", s)
    s = re.sub(r"\\end\{[^{}]*\}", " ", s)
    # Known math symbols -> Unicode.
    s = s.replace(r"\pm", "±").replace(r"\mp", "∓")
    s = s.replace(r"\times", "×").replace(r"\cdot", "·")
    # +/-, + / -, +-, + - all get collapsed to ± .
    s = re.sub(r"\+\s*/\s*-", "±", s)
    s = re.sub(r"\+\s*-", "±", s)
    # Drop any remaining \command sequences (e.g. \sim, \sigma, \rho).
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    # Math delimiters, LaTeX script markers, and residual glue.
    s = s.replace("$", " ").replace("{", " ").replace("}", " ")
    s = s.replace("_", "").replace("^", "")
    s = s.replace("\\\\", " ").replace("\\", " ")
    # Unify minus-sign variants (Unicode U+2212 -> hyphen-minus) so that
    # "10−3" from the PDF and "10-3" from markdown match.
    s = s.replace("−", "-").replace("–", "-")
    # Drop citation brackets like [23], [10, 11].
    s = re.sub(r"\[\s*[\d,\s\-]+\s*\]", " ", s)
    # Collapse whitespace, then strip whitespace adjacent to ± and × so
    # that "X ± Y" and "X±Y" are equivalent (common asymmetry between
    # marker markdown and PyMuPDF text layer).
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*±\s*", "±", s)
    s = re.sub(r"\s*×\s*", "×", s)
    s = s.strip().lower()
    return s


def _is_matchable_anchor(token: str, normalized: str) -> bool:
    """Reject anchors that are pure LaTeX markup or too short/generic
    after normalization to be useful for locating a PDF region."""
    if len(normalized) < 5:
        return False
    latex_chars = sum(1 for c in token if c in _LATEX_CHARS)
    if latex_chars / max(1, len(token)) > 0.5:
        return False
    return True


def _extract_table_anchors(marker_table_md: str) -> list[tuple[str, str]]:
    """Pull up to 3 distinctive body-cell anchors from a marker markdown
    table, returned as (raw, normalized) pairs in descending length
    order. Empty list when the table has too few rows or no matchable
    cells. Used both by find_pdf_table (anchor scan across all pages)
    and by the multi-page caption-page sanity check (verify a chosen
    candidate's text really contains content from THIS markdown
    table, not the previous continuation's)."""
    rows = [r.strip().strip("|").split("|") for r in marker_table_md.splitlines()
            if r.strip().startswith("|")]
    if len(rows) < 3:
        return []
    body_cells = [c.strip() for r in rows[2:] for c in r if c.strip()]
    if not body_cells:
        return []
    sorted_cells = sorted(set(body_cells), key=len, reverse=True)
    anchors: list[tuple[str, str]] = []
    for cell in sorted_cells:
        norm = _normalize_for_match(cell)
        if _is_matchable_anchor(cell, norm):
            anchors.append((cell, norm))
        if len(anchors) >= 3:
            break
    return anchors


def _text_contains_table_anchor(anchors: list[tuple[str, str]],
                                 text: str) -> bool:
    """True when `text` (a finder-candidate's payload) contains any of
    the markdown table's anchor cells after normalization. Used as a
    cheap sanity check to confirm a caption-page candidate corresponds
    to THIS markdown table -- relevant on multi-page tables where the
    occurrence-by-position lookup could land on the wrong continuation
    page if marker's reading order disagrees with PDF page order."""
    if not anchors:
        return False
    norm_text = _normalize_for_match(text or "")
    return any(norm_anchor in norm_text for _raw, norm_anchor in anchors)


def find_pdf_table(doc: fitz.Document, marker_table_md: str, finder):
    """Return (page_idx, bbox, text) for the detected table whose content
    contains a distinctive cell from the given marker table, else None.

    `text` is the matching candidate's text payload (e.g. docling's
    export_to_markdown for that table; PyMuPDF's space-joined cell
    content; TATR's bbox text). Callers that don't need the text
    (most existing call sites) can ignore it; the --no-vlm-tables
    path uses it to substitute the detector's pre-extracted markdown
    in place of the VLM's crop rewrite.

    Tries up to the three longest body cells as anchors (in descending
    order of length), after LaTeX-stripping both the anchor and each
    finder-candidate's text. First match wins."""
    anchors = _extract_table_anchors(marker_table_md)
    if not anchors:
        return None

    # Single pass over candidates; normalize each candidate's text once
    # and test every anchor against it.
    for i in range(len(doc)):
        for bbox, text in finder.candidates(doc, i):
            if _text_contains_table_anchor(anchors, text):
                return i, bbox, text
    return None


# Serializes Marker on the GPU across batch workers. Uncontended in
# single-paper mode and with --workers 1; only relevant when batch mode
# runs multiple papers in parallel and we want the VLM hooks of paper N
# to overlap the Marker run of paper N+1 instead of fighting for the GPU.
_MARKER_LOCK = threading.Lock()


# Substrings (case-insensitive) found in PDF /Producer or /Creator
# metadata that strongly suggest the PDF was produced by an old OCR
# engine. Match against these is treated as a strong scan signal --
# any one hit is enough to flip --auto-force-ocr to ON for the paper.
# Generic "OCR" is included as a substring; it occasionally appears
# in legitimate digital-tool names but in practice the false-positive
# cost (a wasted ~30-60s of force-OCR work) is tolerable next to the
# false-negative cost (bad output for a real scan).
_OCR_TOOL_PATTERNS = (
    "FineReader",
    "OmniPage",
    "ScanSoft",
    "Adobe Capture",
    "AdobeOCR",
    "Capture Plug-in",
    "ABBYY",
    "Iris OCR",
    "Readiris",
    "Tesseract",
    "OCR",
)


def _scan_signals_str(signals: dict) -> str:
    """Compact one-line summary of detect_scan_ocr_pdf signals for
    log output. Field order chosen so the most action-relevant
    signals come first."""
    return (
        f"producer={signals.get('producer', '')!r} "
        f"creator={signals.get('creator', '')!r} "
        f"image_cover_median={signals.get('image_cover_median', 0.0):.2f} "
        f"font_count={signals.get('font_count', 0)} "
        f"bold_char_ratio={signals.get('bold_char_ratio', 0.0):.2f} "
        f"pages={signals.get('n_pages', 0)}"
    )


def _extract_scan_signals(doc: "fitz.Document",
                          sample_pages: int = 8) -> dict:
    """Pull the scan/old-OCR detection signals from a PyMuPDF document.
    CPU-only, ~100-300 ms on a typical paper. Samples up to
    `sample_pages` pages spread across the document so a long PDF
    doesn't pay full per-page cost while still catching late-document
    image floods (figure-heavy back-matter / scanned plates after a
    digital body)."""
    metadata = doc.metadata or {}
    producer = (metadata.get("producer") or "").strip()
    creator = (metadata.get("creator") or "").strip()

    n_pages = doc.page_count
    if n_pages == 0:
        return {
            "producer": producer, "creator": creator,
            "image_cover_median": 0.0, "font_count": 0,
            "bold_char_ratio": 0.0, "n_pages": 0,
        }

    # Sample evenly: first N/2 pages plus a stride-spread sample
    # over the rest. Guarantees coverage of the cover/title page
    # (where producer-tool padding tends to live) and of late
    # pages.
    if n_pages <= sample_pages:
        page_idxs = list(range(n_pages))
    else:
        head = list(range(sample_pages // 2))
        tail = [int(round(i * (n_pages - 1) / (sample_pages - 1)))
                for i in range(sample_pages // 2, sample_pages)]
        page_idxs = sorted(set(head + tail))

    image_cover_ratios: list[float] = []
    fonts: set[str] = set()
    bold_chars = 0
    total_chars = 0

    for i in page_idxs:
        try:
            page = doc[i]
        except Exception:
            continue
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            continue

        # Image-cover ratio: union the bbox areas of every embedded
        # image on the page. Doesn't deduplicate overlapping images
        # (rare in scans, where there's typically one full-page
        # image per page); the ratio is capped at 1.0 below so an
        # over-counted page can't blow the median.
        try:
            img_area = 0.0
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                for r in rects:
                    img_area += r.width * r.height
            image_cover_ratios.append(min(1.0, img_area / page_area))
        except Exception:
            pass

        # Font diversity + bold-char ratio from text spans. flags &
        # 16 is PyMuPDF's bold-marker bit. Some old-OCR PDFs set
        # bold on every span via font-weight metadata, which is
        # exactly the noise we want to detect.
        try:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font = span.get("font") or ""
                        text = span.get("text") or ""
                        flags = int(span.get("flags") or 0)
                        if font:
                            fonts.add(font)
                        nchars = len(text)
                        total_chars += nchars
                        if flags & 16:
                            bold_chars += nchars
        except Exception:
            pass

    if image_cover_ratios:
        srt = sorted(image_cover_ratios)
        mid = len(srt) // 2
        if len(srt) % 2:
            image_cover_median = srt[mid]
        else:
            image_cover_median = (srt[mid - 1] + srt[mid]) / 2.0
    else:
        image_cover_median = 0.0

    bold_ratio = (bold_chars / total_chars) if total_chars else 0.0

    return {
        "producer": producer,
        "creator": creator,
        "image_cover_median": image_cover_median,
        "font_count": len(fonts),
        "bold_char_ratio": bold_ratio,
        "n_pages": n_pages,
    }


def _decide_force_ocr(signals: dict) -> tuple[bool, str]:
    """Apply the conservative scan-detection rule to a signals dict.
    Returns (should_force, reason) where reason is a short string
    naming the trigger. The thresholds err toward false-negatives
    (a missed scan) over false-positives (a wasted ~30-60s OCR pass
    on a born-digital paper) -- per Sarah's request.

    Rules, first-match wins:
      1. STRONG: an OCR-tool fingerprint substring appears in /Producer
         or /Creator metadata. One hit is enough.
      2. STRONG-COMBO: median image-cover > 0.6 AND font count <= 3.
         A scanned paper is one full-page image per page with a
         skeletal text overlay using one or two fonts.
      3. MEDIUM-COMBO: any 2 of {image_cover>0.4, fonts<=4,
         bold_ratio>0.25}. Catches papers that scrape past rule 2 but
         have multiple weaker signals.
    Otherwise: not a scan."""
    producer = signals.get("producer", "") or ""
    creator = signals.get("creator", "") or ""
    p_lower = producer.lower()
    c_lower = creator.lower()
    for sig in _OCR_TOOL_PATTERNS:
        sig_lower = sig.lower()
        if sig_lower in p_lower:
            return True, f"OCR fingerprint in /Producer ({sig!r})"
        if sig_lower in c_lower:
            return True, f"OCR fingerprint in /Creator ({sig!r})"

    img_cov = float(signals.get("image_cover_median", 0.0) or 0.0)
    fonts = int(signals.get("font_count", 0) or 0)
    bold = float(signals.get("bold_char_ratio", 0.0) or 0.0)

    if img_cov > 0.6 and 0 < fonts <= 3:
        return True, f"image-heavy + low-font (img={img_cov:.2f}, fonts={fonts})"

    medium = []
    if img_cov > 0.4:
        medium.append(f"img={img_cov:.2f}")
    if 0 < fonts <= 4:
        medium.append(f"fonts={fonts}")
    if bold > 0.25:
        medium.append(f"bold={bold:.2f}")
    if len(medium) >= 2:
        return True, "medium-combo: " + ", ".join(medium)
    return False, "no scan signals"


def detect_scan_ocr_pdf(pdf_path: Path,
                        sample_pages: int = 8) -> tuple[bool, dict]:
    """Top-level scan/old-OCR detector. Open the PDF with PyMuPDF,
    extract signals, apply _decide_force_ocr, return (should_force,
    signals_with_decision).

    `signals_with_decision` is the raw signals dict augmented with
    a "force_ocr_decision" boolean and a "force_ocr_reason" string,
    so callers can log + serialize the verdict in one place.

    Errors during extraction are not fatal: the function returns
    (False, {...}) with whatever signals it could gather rather
    than blocking the whole conversion."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            signals = _extract_scan_signals(doc, sample_pages=sample_pages)
    except Exception as e:
        log.warning("scan-detect: could not open %s: %s", pdf_path, e)
        return False, {
            "producer": "", "creator": "", "image_cover_median": 0.0,
            "font_count": 0, "bold_char_ratio": 0.0, "n_pages": 0,
            "force_ocr_decision": False,
            "force_ocr_reason": f"open-failed: {e}",
        }
    decision, reason = _decide_force_ocr(signals)
    signals["force_ocr_decision"] = decision
    signals["force_ocr_reason"] = reason
    return decision, signals


def run_marker(pdf_path: Path, assets_dir: Path,
               asset_prefix: str = "",
               force_ocr: bool = False) -> tuple[str, dict]:
    """Run marker on the PDF. When `asset_prefix` is non-empty, every
    extracted image filename is prefixed before being written to
    `assets_dir`, and the corresponding image references inside the
    returned markdown are rewritten to match. This lets a supplement
    share the same assets directory as its main paper without
    filename collisions.

    `force_ocr=True` passes the same option to marker's PdfProvider,
    bypassing the embedded text layer entirely and forcing every page
    through surya OCR. Recommended for pre-2000 / scanned papers
    whose publisher-OCR text layer is poisoning marker's output --
    e.g. spurious bold paragraphs from stale font-weight metadata,
    line-broken table headers like "TABLE\\n2.". Costs full surya
    recognition on every page (~30-60 s on an 8-page paper); on
    clean modern PDFs it's pure overhead. Default off."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config = {"force_ocr": True} if force_ocr else None
    with _MARKER_LOCK:
        log.info("Loading marker models (first run downloads weights)...")
        if force_ocr:
            log.info("--force-ocr: bypassing PDF text layer; "
                     "every page will go through surya OCR")
        converter = PdfConverter(artifact_dict=create_model_dict(),
                                 config=config)
        log.info("Running marker on %s", pdf_path.name)
        rendered = converter(str(pdf_path))

    markdown = rendered.markdown
    images_written = {}
    original_names = list((rendered.images or {}).keys())
    for name, img in (rendered.images or {}).items():
        renamed = asset_prefix + name
        out_path = assets_dir / renamed
        if hasattr(img, "save"):
            img.save(out_path)
        else:
            out_path.write_bytes(img)
        images_written[renamed] = out_path
    if asset_prefix:
        # Rewrite marker's internal image references (bare filenames) to
        # match the prefixed files on disk, before the downstream
        # IMAGE_RE.sub in convert() normalizes them to assets/<name>.
        for name in original_names:
            markdown = markdown.replace(name, asset_prefix + name)
    log.info("marker produced %d chars of markdown, %d images",
             len(markdown), len(images_written))
    return markdown, images_written


def run_mineru_layout(pdf_path: Path, assets_dir: Path,
                      *,
                      asset_prefix: str = "",
                      vlm_rewrite_tables: bool = False,
                      rescue_orphan_captions: bool = True,
                      rescue_subpanels: bool = True,
                      strip_chart_details: bool = False,
                      mineru_backend: str = "pipeline",
                      mineru_extra_args: tuple = (),
                      skip_mineru_run: bool = False,
                      mineru_existing_dir: Optional[Path] = None,
                      report: "QualityReport",
                      mineru_dir: Optional[Path] = None,
                      vlm_force: bool = False,
                      table_workers: int = 1) -> str:
    """Body-emission alternative to run_marker.

    Shells out to MinerU (via layout_mineru.run_mineru) and emits a
    flat body markdown via wrap_mineru.emit_markdown. Figure / table /
    caption layout comes from MinerU's middle.json; body OCR is
    PaddleOCR (MinerU's pipeline backend). The orphan-caption rescue
    runs as a belt-and-suspenders pass before emit_markdown so mis-
    classified caption / footnote text is adopted into the parent
    figure / table block.

    Side effects:
      - Copies MinerU's images/ into `assets_dir`, prefixing each
        filename with `asset_prefix` (matches run_marker's contract
        for the supplement-bundling path).
      - Appends FigureScore / TableScore entries to `report` via
        emit_markdown.

    `mineru_dir` may be passed when the caller wants the MinerU
    intermediate artifacts persisted (e.g., paper_dir/mineru/);
    otherwise a tempdir is used and discarded after asset copy.

    `rescue_orphan_captions=False` skips the post-parse pass that
    adopts misclassified caption / footnote text into parent
    figure / table blocks. Useful for a fast clean run when you
    want to see MinerU's raw classification without paper2md's
    fixups.

    `mineru_backend` (default 'pipeline' = PaddleOCR) selects MinerU's
    internal extraction backend. 'hybrid-auto-engine' uses MinerU's
    bundled VLM for cell-level tables / chart-data reconstruction;
    its emitted <details> chart blocks can be stripped via
    `strip_chart_details=True`.

    `mineru_extra_args` are additional arguments passed through to the
    mineru CLI subprocess (e.g. ('--device-mode', 'cpu',)).

    `skip_mineru_run=True` consumes an existing MinerU output instead
    of running the subprocess. `mineru_existing_dir` MUST be provided
    in that case and point at either a flat layout dir or a nested
    <stem>/auto/ subdir produced by a prior MinerU invocation.

    Returns the body markdown.
    """
    import layout_mineru as _lm
    import wrap_mineru as _wm

    stem = pdf_path.stem
    do_rescue = rescue_orphan_captions
    do_subpanels = rescue_subpanels

    def _emit_from(produced_dir: Path) -> str:
        middle = produced_dir / f"{stem}_middle.json"
        if not middle.exists():
            raise FileNotFoundError(
                f"MinerU output missing {middle.name} under {produced_dir}")
        parsed = _lm.parse_middle_json(middle)
        if do_rescue:
            rescue_stats = _lm.rescue_orphan_captions(
                parsed.blocks, page_sizes=parsed.page_sizes)
            if rescue_stats:
                report.rescues["orphan_captions"] = rescue_stats
        else:
            log.info("MinerU layout: orphan-caption rescue skipped "
                     "(--no-rescue-orphan-captions)")
        if do_subpanels:
            sp_stats = _lm.rescue_subpanel_groups(parsed.blocks)
            if sp_stats and sp_stats.get("groups"):
                log.info("MinerU layout: consolidated %d multi-panel "
                         "figure group(s); %d panel(s) adopted",
                         sp_stats["groups"], sp_stats["panels_adopted"])
            if sp_stats:
                report.rescues["subpanel_groups"] = sp_stats
        else:
            log.info("MinerU layout: sub-panel consolidation skipped "
                     "(--no-rescue-subpanels)")
        n_assets = _wm.copy_assets(produced_dir / "images", assets_dir,
                                   asset_prefix=asset_prefix)
        log.info("Copied %d MinerU image asset(s) to %s",
                 n_assets, assets_dir)
        body = _wm.emit_markdown(parsed, "assets", assets_dir,
                                 vlm_rewrite_tables, report,
                                 strip_chart_details=strip_chart_details,
                                 asset_prefix=asset_prefix,
                                 vlm_force=vlm_force,
                                 table_workers=table_workers)
        log.info("MinerU layout produced %d chars of markdown "
                 "(figures=%d, tables=%d)",
                 len(body), len(report.figures), len(report.tables))
        return body

    with _stage_mineru_run(pdf_path,
                           mineru_dir=mineru_dir,
                           mineru_backend=mineru_backend,
                           mineru_extra_args=mineru_extra_args,
                           skip_mineru_run=skip_mineru_run,
                           mineru_existing_dir=mineru_existing_dir) as produced:
        return _emit_from(produced)


@contextmanager
def _stage_mineru_run(pdf_path: Path,
                      *,
                      mineru_dir: Optional[Path],
                      mineru_backend: str,
                      mineru_extra_args: tuple,
                      skip_mineru_run: bool,
                      mineru_existing_dir: Optional[Path]):
    """Stage a MinerU output directory and yield it.

    Three modes, all converging on a flat-layout `<dir>/<stem>_middle.json`:

    1. `skip_mineru_run=True`: copy/use an existing MinerU output via
       `layout_mineru.stage_existing_mineru_dir`. `mineru_existing_dir`
       must be set.
    2. `mineru_dir is not None`: run MinerU into the persistent dir.
    3. Otherwise: run MinerU into a tempdir that's cleaned up on exit.

    Used by both `run_mineru_layout` and `run_marker_plus_mineru_layout`
    so the staging contract is shared.
    """
    import tempfile

    import layout_mineru as _lm

    if skip_mineru_run:
        if mineru_existing_dir is None:
            raise ValueError(
                "--skip-mineru-run requires --mineru-dir to point at an "
                "existing MinerU output (auto/ subdir or flat layout dir).")
        target = mineru_dir if mineru_dir is not None else mineru_existing_dir
        if mineru_dir is not None:
            target.mkdir(parents=True, exist_ok=True)
        produced = _lm.stage_existing_mineru_dir(mineru_existing_dir, target)
        log.info("MinerU layout: skipping subprocess; using existing "
                 "output at %s", produced)
        yield produced
        return

    if mineru_dir is not None:
        mineru_dir.mkdir(parents=True, exist_ok=True)
        _lm.run_mineru(pdf_path, mineru_dir,
                       backend=mineru_backend,
                       extra_args=tuple(mineru_extra_args))
        yield mineru_dir
        return

    with tempfile.TemporaryDirectory(prefix="paper2md-mineru-") as td:
        produced = _lm.run_mineru(pdf_path, Path(td),
                                  backend=mineru_backend,
                                  extra_args=tuple(mineru_extra_args))
        yield produced


# --- Hybrid layout (--layout-source=hybrid) =================================
# Marker for body+caption text and reading order; MinerU for figure assets
# and table HTML. Bipartite-by-number splice: each MinerU image/table block
# pairs with marker's `Fig. N` / `Table N` caption line. Marker's caption is
# never replaced by PaddleOCR text.


def _render_image_only(primary, num: str, marker_caption_line: str,
                       assets_rel: str, asset_prefix: str,
                       extras: tuple = (),
                       assets_abs: Optional[Path] = None) -> str:
    """Image link(s) ONLY -- no italic-caption tail.

    Marker has the caption line in place; we splice this image emission
    above it. For multi-panel primaries (subpanel_paths populated by
    rescue_subpanel_groups), one image-link per panel with letter
    suffixes. For "extras" (multiple distinct mineru blocks sharing a
    fig number with no subpanel grouping -- ambiguous), each gets its
    own image-link.

    When `assets_abs` is provided, each emitted image file is
    renamed on disk to the semantic
    `{prefix}figure_{num}{letter?}_p{page}{ext}` convention (mirrors
    the table naming). Without `assets_abs` (legacy callers), the
    MinerU content-hash names are preserved.
    """
    alt_base = f"Figure {num}"
    if marker_caption_line:
        tail = re.sub(r"\s+", " ", marker_caption_line.strip()).strip()
        if len(tail) > 80:
            tail = tail[:77].rstrip() + "..."
        if tail:
            alt_base = tail

    import wrap_mineru as _wm
    page_idx = getattr(primary, "page_idx", None)

    def _maybe_rename(path: str, letter: Optional[str]) -> str:
        """Rename one panel file if we have assets_abs + a known id;
        return the basename to use in the link. Falls back to the
        original path on any error."""
        if assets_abs is None or not path:
            return path
        ext = Path(path).suffix or ".jpg"
        new_basename = _wm._figure_filename_for_panel(
            num, letter, page_idx, ext)
        resolved = _wm._rename_figure_image_file(
            path, new_basename, assets_abs, asset_prefix)
        return resolved if resolved else path

    out_lines: list[str] = []
    subpanel_paths = list(getattr(primary, "subpanel_paths", None) or [])
    subpanel_bboxes = list(getattr(primary, "subpanel_bboxes", None) or [])
    if subpanel_paths and primary.image_path:
        # Order panels alphabetically by their visual position so the
        # alt-text letters match the rendered layout. Without this the
        # primary block emitted first regardless of position, producing
        # b-before-a (canup Figs 1, 2) or sequentially-mismatched
        # letters (canup Fig 3, primary visually rightmost = panel c
        # but got labelled 'a').
        ordered = _ordered_panels_for_emission(
            primary.bbox, primary.image_path,
            subpanel_bboxes, subpanel_paths,
            getattr(primary, "panel_letter", None))
        for letter, path in ordered:
            renamed = _maybe_rename(path, letter)
            rel = f"{assets_rel}/{asset_prefix}{renamed}"
            out_lines.append(f"![Figure {num} panel {letter}]({rel})")
    elif primary.image_path:
        renamed = _maybe_rename(primary.image_path, None)
        rel = f"{assets_rel}/{asset_prefix}{renamed}"
        out_lines.append(f"![{alt_base}]({rel})")

    for extra in extras:
        if getattr(extra, "image_path", None):
            renamed = _maybe_rename(extra.image_path, None)
            rel = f"{assets_rel}/{asset_prefix}{renamed}"
            out_lines.append(f"![{alt_base}]({rel})")

    return "\n\n".join(out_lines)


def _ordered_panels_for_emission(primary_bbox, primary_path,
                                 subpanel_bboxes, subpanel_paths,
                                 primary_letter) -> "list[tuple[str, str]]":
    """Return (letter, image_path) tuples in alphabetical-letter order
    for emission.

    Two letter-assignment strategies depending on the inputs:

    1. **Have bboxes** (the normal path: rescue_subpanel_groups
       populated subpanel_bboxes alongside subpanel_paths): sort all
       panels by reading order (y_center, then x_center) and assign
       sequential a, b, c, ... in sorted order. The primary's caption-
       extracted letter is ignored as a label hint -- visual position
       is ground truth.

    2. **No bboxes** (legacy / test stubs that pre-date the bbox
       plumbing): fall back to `_subpanel_letter_assignments`, which
       honours the primary's caption-extracted letter, then sort the
       resulting (letter, path) pairs alphabetically.
    """
    have_bboxes = (
        subpanel_bboxes
        and len(subpanel_bboxes) == len(subpanel_paths)
        and primary_bbox is not None
    )
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if have_bboxes:
        all_panels = [(primary_bbox, primary_path)]
        all_panels.extend(zip(subpanel_bboxes, subpanel_paths))

        def _key(p):
            x0, y0, x1, y1 = p[0]
            return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)

        all_panels.sort(key=_key)
        return [(alphabet[i], path)
                for i, (_bbox, path) in enumerate(all_panels)]
    # Fallback: assign letters via the primary-letter-aware helper,
    # then sort alphabetically by letter for emission.
    from layout_mineru import _subpanel_letter_assignments
    n_panels = 1 + len(subpanel_paths)
    letters = _subpanel_letter_assignments(primary_letter, n_panels)
    all_paths = [primary_path] + list(subpanel_paths)
    pairs = list(zip(letters, all_paths))
    pairs.sort(key=lambda lp: lp[0])
    return pairs


_HYBRID_REFS_HEADING_RE = re.compile(
    # Matches references-section headings across the variants observed
    # in marker + MinerU output: `## References`, `#### References and
    # Notes`, `#### REFERENCES AND NOTES`, `## **REFERENCES AND NOTES**`,
    # `#### **References**`, plus heading-less paragraph forms
    # (`References and Notes` on its own line, used by MinerU for
    # Science papers like cuk).
    #
    # NOTE: there is a stricter module-level `_REFS_HEADING_RE` (line
    # ~1715) with named `hashes` / `title` groups used by the existing
    # refs-cleanup hooks (consolidate, merge, normalise). This loose
    # regex is hybrid-only and intentionally permissive enough to catch
    # MinerU's heading-less Science-paper layouts -- do not confuse the
    # two.
    r"^(?:#{1,6}\s+)?"           # optional 1-6 hash marks
    r"(?:\*\*\s*)?"              # optional bold open
    r"REFERENCES?(?:\s+AND\s+NOTES)?"
    r"(?:\s*\*\*)?"              # optional bold close
    r"\s*$",
    re.MULTILINE | re.IGNORECASE)


# Top-level `# Title` heading -- a level-1 markdown heading whose text
# begins with a non-whitespace character. Used by the refs-swap to
# detect cross-article contamination in MinerU's output.
_MINERU_TOP_TITLE_RE = re.compile(r"^# \S", re.MULTILINE)


def _extract_mineru_refs_section(mineru_md_path: Path) -> "Optional[str]":
    """Return MinerU's references section (heading + body) from its raw
    markdown, or None if no `^##? References?` heading is found.

    The None result is the load-bearing signal: it tells the hybrid
    refs-swap caller that MinerU silently dropped the references (e.g.
    Wackerle1962, where footnote-style refs were not recognised as a
    section heading) and marker's version should be kept instead.

    Cross-article contamination guard (alexander Science / Elliott
    News & Views): MinerU sometimes emits two papers concatenated in
    one .md (a perspective and the actual article). The first refs
    heading then belongs to the preceding article, and its section
    extends through the current article's TITLE + BODY + refs, which
    would pull a duplicate of the current article into marker's body
    on swap. Skip any refs heading whose section is followed by a
    level-1 `^# Title` heading; pick the first one that isn't. Falls
    back to the last refs heading if every candidate is followed by
    a title (defensive; the simple-paper case has 1 heading + 0
    titles after it, so the fallback never fires there).
    """
    if not mineru_md_path.exists():
        return None
    text = mineru_md_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(_HYBRID_REFS_HEADING_RE.finditer(text))
    if not matches:
        return None
    chosen = None
    for m in matches:
        if not _MINERU_TOP_TITLE_RE.search(text, m.end()):
            chosen = m
            break
    if chosen is None:
        chosen = matches[-1]
    # Truncate at the next non-ref top-level heading (`# ` / `## `) so
    # that MinerU's own back-matter duplication (canup pattern: ## Refs
    # ... ## Acknowledgements ... ## Methods ... ## Refs) doesn't leak
    # the duplicate Methods + second Refs into marker's body via the
    # swap. The refs section is a self-contained unit; anything after
    # a `## Acknowledgements` / `## Methods` / etc. is by definition
    # not part of refs. Headings that ARE another "References" /
    # "REFERENCES" instance don't terminate -- they're a continuation
    # of refs (e.g. supplementary refs lumped under the same canonical
    # section).
    section = text[chosen.start():]
    after_heading = section.find("\n") + 1
    nxt = re.search(
        r"^#{1,2}\s+(?!References?\b|REFERENCES?\b)",
        section[after_heading:],
        re.MULTILINE,
    )
    if nxt:
        section = section[:after_heading + nxt.start()]
    return section.rstrip() + "\n"


def _swap_hybrid_refs_with_mineru(body: str,
                                  mineru_md_path: Path) -> tuple[str, dict]:
    """Replace marker's references section in `body` with MinerU's.

    Both sides must have a recognisable `^##? References?` heading; if
    either is absent the body is returned unchanged. Per the audit at
    `docs/dev/HYBRID_REFERENCES_AUDIT_2026-05-12.md`, MinerU's references
    are strictly better than marker's on 3-col Science/Nature papers
    (cuk, young, canup2012, Knudson2013) where marker's column-serialiser
    drops or glues entries, and equivalent on OK papers. The only case
    where MinerU is worse (Wackerle1962) is caught by the heading guard.
    """
    stats: dict = {"mineru_refs_swapped": False, "reason": ""}
    marker_match = _HYBRID_REFS_HEADING_RE.search(body)
    if not marker_match:
        stats["reason"] = "no_marker_refs_heading"
        return body, stats
    mineru_refs = _extract_mineru_refs_section(mineru_md_path)
    if mineru_refs is None:
        stats["reason"] = "no_mineru_refs_heading"
        return body, stats
    new_body = body[:marker_match.start()].rstrip() + "\n\n" + mineru_refs
    stats["mineru_refs_swapped"] = True
    return new_body, stats


def _strip_marker_image_links(body: str,
                              marker_image_filenames: set) -> tuple[str, int]:
    """Remove marker-emitted image-link LINES from `body` whose path
    basename appears in `marker_image_filenames`. Files on disk are NOT
    touched (caller can still recover them from `assets/`).

    Marker often prefixes image links with empty `<span id="page-X-Y">
    </span>` cross-reference targets on the SAME line (PRL/APS papers,
    e.g. Hicks2006, Knudson2009). When the matching `![](...)` is
    removed, the leading span(s) are removed with it as a unit; an
    isolated span without an image is left in place.

    Used by the hybrid layout to drop marker's redundant figure crops
    after MinerU's images have been spliced in. Retained on disk during
    the hybrid evaluation phase so we can A/B compare without re-running.
    """
    if not marker_image_filenames:
        return body, 0
    pattern = re.compile(
        r"^(?:<span\s+id=[\"'][^\"']*[\"']>\s*</span>\s*)*"
        r"\!\[[^\]]*\]\(([^)]+)\)[^\S\n]*\n?",
        re.MULTILINE)
    removed = 0

    def _maybe_drop(m):
        nonlocal removed
        basename = m.group(1).rsplit("/", 1)[-1]
        if basename in marker_image_filenames:
            removed += 1
            return ""
        return m.group(0)

    return pattern.sub(_maybe_drop, body), removed


def _rename_table_asset_semantic(block, table_idx: int,
                                  assets_abs: Path,
                                  asset_prefix: str,
                                  table_id: Optional[str] = None) -> None:
    """Rename the MinerU table-image file in `assets/` from its content-
    hash name to match the sidecar `.md` convention so the JPG and
    sidecar `.md` share a stem:

      - `{prefix}table_{id}_p{page}_{idx}.jpg` when `table_id` is known
        (typical hybrid / mineru-only path; id has `.` -> `_` for
        filesystem safety).
      - `{prefix}table_p{page}_{idx}.jpg` when only page+idx is known
        (marker layout's `process_tables` convention -- legacy).

    Mutates `block.image_path` so the splice's inline image link and
    the TableScore both reference the new name -- and so the sidecar
    `.md` written by wrap_mineru._write_hybrid_table_sidecar shares
    the same stem.

    Idempotent: if the source isn't present (already renamed on a prior
    run) or the target already exists (re-run without `--clean`), bails
    and just points `block.image_path` at the target name. Failures fall
    back to the original hash name (logged WARNING).
    """
    if not block.image_path:
        return
    if getattr(block, "page_idx", None) is None:
        return
    suffix = Path(block.image_path).suffix or ".jpg"
    # Sanitize id for filesystem (e.g. "A.4" -> "A_4"). Reuse the
    # wrap_mineru helper so the .md sidecar and the .jpg use the
    # identical sanitization rule.
    import wrap_mineru as _wm
    safe_id = _wm._sanitize_table_id_for_filename(table_id)
    page_str = f"p{block.page_idx + 1}_{table_idx}"
    if safe_id is not None:
        new_basename = f"table_{safe_id}_{page_str}{suffix}"
    else:
        new_basename = f"table_{page_str}{suffix}"
    src = assets_abs / f"{asset_prefix}{block.image_path}"
    dst = assets_abs / f"{asset_prefix}{new_basename}"
    if src == dst:
        return  # already semantic-named upstream
    if dst.exists() and not src.exists():
        block.image_path = new_basename
        return
    if not src.exists():
        return  # nothing on disk to rename; leave image_path alone
    if dst.exists():
        block.image_path = new_basename
        return
    try:
        src.rename(dst)
        block.image_path = new_basename
    except OSError as e:
        log.warning("Hybrid: failed to rename table asset %s -> %s "
                    "(keeping hash name): %s", src, dst, e)


def _render_mineru_table(block, num: str, table_idx: int,
                         assets_rel: str, asset_prefix: str,
                         vlm_rewrite_tables: bool, assets_abs: Path,
                         report, pdf_doc=None,
                         vlm_force: bool = False) -> str:
    """Render a MinerU table block as body-only (no caption header).

    Reuses wrap_mineru.table_block_to_md and strips its leading
    `\\n**caption**\\n\\n` so marker's `Table N.` caption stays as the
    heading. Image link + footnote (if any) are preserved.

    `pdf_doc` (optional fitz.Document): when provided AND the block
    has a usable bbox + page_idx, the VLM table-rewrite path receives
    a fresh `render_crop(doc, page_idx, bbox)` PIL image instead of
    loading MinerU's saved JPG from disk. Hypothesis: MinerU's JPG
    intermediate carries compression artifacts that the vLLM vision
    encoder chokes on under hybrid; rendering from PDF avoids that.
    """
    import wrap_mineru as _wm
    # Rename MinerU's content-hash JPG to match the sidecar `.md`
    # naming (`table_{id}_p{page}_{idx}.jpg` when id is known,
    # else `table_p{page}_{idx}.jpg`).
    _rename_table_asset_semantic(block, table_idx, assets_abs,
                                 asset_prefix, table_id=num)
    pil_override = None
    if (pdf_doc is not None
            and vlm_rewrite_tables
            and getattr(block, "bbox", None)
            and getattr(block, "page_idx", None) is not None):
        try:
            pil_override = render_crop(pdf_doc, block.page_idx, block.bbox)
            log.info("Hybrid: rendered table %d from PDF page %d bbox=%s "
                     "for VLM (bypassing MinerU JPG)",
                     table_idx, block.page_idx + 1, block.bbox)
        except Exception as e:
            log.warning("Hybrid: render_crop failed for table %d "
                        "(falling back to MinerU JPG): %s", table_idx, e)
            pil_override = None
    full = _wm.table_block_to_md(
        block, table_idx=table_idx, assets_rel=assets_rel,
        vlm_tables=vlm_rewrite_tables, assets_abs=assets_abs,
        report=report, asset_prefix=asset_prefix,
        pil_image_override=pil_override,
        table_id=num,
        inline_in_body=False,
        vlm_force=vlm_force,
    )
    # `inline_in_body=False` already excludes the leading `**caption**`
    # header -- nothing to strip. The returned content is just
    # `<image_md><footnote_md><sidecar_suffix>`. Preserve as-is.
    return full.rstrip() + "\n"


def _render_unmatched_table_sidecar(block, num: str, table_idx: int,
                                    assets_rel: str, asset_prefix: str,
                                    vlm_rewrite_tables: bool,
                                    assets_abs: Path,
                                    report, pdf_doc=None,
                                    vlm_force: bool = False
                                    ) -> Optional[dict]:
    """Run the full table_block_to_md pipeline (renames the JPG, fires
    the VLM rewrite, writes the sidecar `.md`) for a MinerU table that
    has NO matching marker caption. We DISCARD the returned body
    string -- the caller will build a single start-of-doc index
    section from the sidecar info, not splice inline content.

    Returns a dict capturing the sidecar info for the index, or None
    if no sidecar was written (e.g., HTML-fallback path)."""
    import wrap_mineru as _wm
    _rename_table_asset_semantic(block, table_idx, assets_abs,
                                 asset_prefix, table_id=num)
    pil_override = None
    if (pdf_doc is not None
            and vlm_rewrite_tables
            and getattr(block, "bbox", None)
            and getattr(block, "page_idx", None) is not None):
        try:
            pil_override = render_crop(pdf_doc, block.page_idx, block.bbox)
        except Exception as e:
            log.warning("Hybrid (unmatched): render_crop failed for "
                        "table %d (falling back to MinerU JPG): %s",
                        table_idx, e)
            pil_override = None
    _wm.table_block_to_md(
        block, table_idx=table_idx, assets_rel=assets_rel,
        vlm_tables=vlm_rewrite_tables, assets_abs=assets_abs,
        report=report, asset_prefix=asset_prefix,
        pil_image_override=pil_override,
        table_id=num,
        inline_in_body=False,
        vlm_force=vlm_force,
    )
    # Locate the sidecar entry just appended to report.tables (the
    # latest TableScore is ours).
    if report.tables and getattr(report.tables[-1], "sidecar", None):
        return {
            "id": num,
            "sidecar": getattr(report.tables[-1], "sidecar"),
            "image_path": block.image_path,
            "page": block.page_idx + 1,
        }
    return None


def _emit_extracted_tables_index(unmatched: list, assets_rel: str,
                                 asset_prefix: str) -> str:
    """Build a `## Extracted tables` section listing every unmatched
    MinerU table sidecar. Returned string is meant to be prepended at
    the START of the body (right after the frontmatter `---`), where
    refs-walks-from-end hooks won't mis-detect it.

    Each row carries:
      - the table image (visual reference)
      - the [Table N — separate markdown](path) sidecar link (audit-
        recognizable via _TBL_SIDECAR_RE; MCP-discoverable)

    `unmatched` items are dicts from _render_unmatched_table_sidecar
    or tuples (num, idx, blk) from _run_one. Both shapes accepted."""
    if not unmatched:
        return ""
    rows = []
    for item in unmatched:
        if isinstance(item, dict):
            num = item.get("id", "?")
            sidecar = item.get("sidecar")
            image_path = item.get("image_path")
        else:
            # (num, idx, blk) tuple from _run_one
            num, _idx, blk = item
            sidecar = getattr(blk, "_sidecar_name", None)
            image_path = getattr(blk, "image_path", None)
        # Best-effort: skip rows where no sidecar landed (HTML-fallback
        # or no-html path) -- nothing useful to link to.
        if not sidecar:
            continue
        img_link = (
            f"![Table {num}]({assets_rel}/{asset_prefix}{image_path})"
            if image_path else ""
        )
        sep = " — " if img_link else ""
        rows.append(
            f"- {img_link}{sep}[Table {num} — separate markdown]"
            f"({assets_rel}/{sidecar})"
        )
    if not rows:
        return ""
    return (
        "## Extracted tables\n\n"
        "_Tables detected by MinerU that could not be matched to a "
        "body caption. The high-quality VLM transcription is in each "
        "sidecar `.md`._\n\n"
        + "\n".join(rows)
        + "\n\n---\n\n"
    )


def _prepend_extracted_tables_index(body: str, index_md: str) -> str:
    """Insert the `## Extracted tables` block at the START of the body.

    Body has no frontmatter yet at the splice stage (frontmatter is
    emitted later by QualityReport.to_frontmatter). So we just prepend
    -- no `---` parsing needed. A `---` separator at the END of the
    index visually distinguishes it from the paper body that follows.
    """
    if not index_md:
        return body
    return index_md + body.lstrip("\n")


def _blank_line_prefix(marker_md: str, insert_pos: int) -> str:
    """Return the number of newlines to prepend at `insert_pos` so the
    inserted content sits on its own paragraph (blank line BEFORE).

    `_find_marker_table_region` can return two different kinds of
    insert positions depending on which branch fired:

      - **Standard case**: `region_end` = position of the FIRST `\\n`
        of the trailing `\\n\\n` separator. `marker_md[:insert_pos]`
        does NOT end with `\\n` (the byte AT insert_pos is the `\\n`),
        so we need `\\n\\n` prefix to produce a blank line.

      - **Chunk-too-large / insert-only fallback**: `caption_line_end
        + 1`. `marker_md[:insert_pos]` DOES end with `\\n` (the
        caption's terminating newline). One additional `\\n` is enough
        to make a blank line; two would render an extra blank gap.

    Without this correction, a single-`\\n` prefix at the standard-case
    position produces `|cell|\\n![image](...)` which some markdown
    renderers continue as a wide table cell instead of closing the
    table and starting a new paragraph.
    """
    if insert_pos <= 0:
        return "\n\n"
    # If the char at insert_pos-1 is already a newline, one more makes
    # a blank line. Otherwise we need two newlines.
    return "\n" if marker_md[insert_pos - 1] == "\n" else "\n\n"


def _find_marker_table_region(marker_md: str, anchor) -> tuple[int, int]:
    """Locate the marker-side table-body extent following a `Table N.`
    caption anchor.

    Returns (region_start, region_end) -- callers replace marker_md
    [region_start:region_end] with mineru's table body. region_start
    is the first non-blank line after the caption; region_end is the
    end of the LAST consecutive pipe-md chunk that follows. When the
    candidate region would swallow another caption / heading, returns
    (insert_point, insert_point) for an insert-only splice.

    Marker (surya) sometimes emits the SAME table TWICE in marker_md
    -- e.g. cuk Table 1 produces a LaTeX-styled pipe-md immediately
    after the caption, then a `<sub>`-styled pipe-md separated by a
    blank line. The basic "stop at first \\n\\n" rule would consume
    only the first; this loop extends the region across blank lines
    while the next non-blank chunk is still a pipe-md table (its
    first non-blank line starts with `|`). Each chunk is gated by
    the same 40-line cap and caption/heading checks as the first.
    """
    caption_line_end = marker_md.find("\n", anchor.end())
    if caption_line_end < 0:
        return (len(marker_md), len(marker_md))
    cursor = caption_line_end + 1
    while cursor < len(marker_md) and marker_md[cursor] == "\n":
        cursor += 1
    if cursor >= len(marker_md):
        return (cursor, cursor)
    region_start = cursor
    blank = marker_md.find("\n\n", region_start)
    region_end = blank if blank >= 0 else len(marker_md)

    def _chunk_safe(text: str) -> bool:
        """Reject chunks that are too large or contain another caption /
        heading -- those signal we'd be swallowing unrelated content."""
        if text.count("\n") > 40:
            return False
        if (HYBRID_FIG_LINE_RE.search(text)
                or HYBRID_TABLE_LINE_RE.search(text)
                or any(ln.lstrip().startswith("#")
                       for ln in text.split("\n"))):
            return False
        return True

    if not _chunk_safe(marker_md[region_start:region_end]):
        return (caption_line_end + 1, caption_line_end + 1)

    # Extend across blank lines while the next chunk is still pipe-md.
    while region_end < len(marker_md):
        cur = region_end
        while cur < len(marker_md) and marker_md[cur] == "\n":
            cur += 1
        if cur >= len(marker_md):
            break
        line_end = marker_md.find("\n", cur)
        if line_end < 0:
            line_end = len(marker_md)
        if not marker_md[cur:line_end].lstrip().startswith("|"):
            break
        next_blank = marker_md.find("\n\n", cur)
        next_chunk_end = next_blank if next_blank >= 0 else len(marker_md)
        if not _chunk_safe(marker_md[cur:next_chunk_end]):
            break
        region_end = next_chunk_end

    return (region_start, region_end)


def _build_eod_figure_splice(blk, marker_md: str, num: str,
                             assets_rel: str, asset_prefix: str
                             ) -> tuple[int, int, str]:
    """Insert tuple for a MinerU figure with no marker caption anchor.

    Search marker_md for the nearest body cross-ref to "Fig. N" /
    "Figure N" (NOT line-anchored). Splice image + mineru's caption at
    the next blank line after the cross-ref; fall back to end-of-doc
    if no cross-ref. Renders mineru's caption (since marker has none
    to use). If the block has `subpanel_paths` (set by
    `rescue_subpanel_groups`), each panel is emitted with a letter
    suffix the same way `_render_image_only` does so subpanels aren't
    silently dropped on the unmatched-anchor path.
    """
    pat = re.compile(r"\b(?:Figure|Fig\.?)\s+" + re.escape(num) + r"\b",
                     re.IGNORECASE)
    m = pat.search(marker_md)
    if m:
        blank = marker_md.find("\n\n", m.end())
        insert_pos = blank if blank >= 0 else len(marker_md)
    else:
        insert_pos = len(marker_md)
    caption = (blk.text or f"Figure {num}").strip()
    subpanel_paths = list(getattr(blk, "subpanel_paths", None) or [])
    subpanel_bboxes = list(getattr(blk, "subpanel_bboxes", None) or [])
    if subpanel_paths and blk.image_path:
        # Sort panels into reading-order alphabetical alt-text (see
        # _ordered_panels_for_emission for rationale).
        ordered = _ordered_panels_for_emission(
            blk.bbox, blk.image_path, subpanel_bboxes, subpanel_paths,
            getattr(blk, "panel_letter", None))
        img_lines = []
        for letter, path in ordered:
            rel = f"{assets_rel}/{asset_prefix}{path}"
            img_lines.append(f"![Figure {num} panel {letter}]({rel})")
        emission = "\n\n" + "\n\n".join(img_lines) + "\n\n" + f"*{caption}*\n"
    elif blk.image_path:
        rel = f"{assets_rel}/{asset_prefix}{blk.image_path}"
        emission = (f"\n\n![Figure {num} | {caption[:80]}]({rel})\n\n"
                    f"*{caption}*\n")
    else:
        emission = f"\n\n*{caption}*\n"
    return (insert_pos, insert_pos, emission)


def _build_eod_table_splice(blk, marker_md: str, num: str, table_idx: int,
                            assets_rel: str, asset_prefix: str,
                            vlm_rewrite_tables: bool, assets_abs: Path,
                            report, pdf_doc=None,
                            vlm_force: bool = False) -> tuple[int, int, str]:
    """Insert tuple for a MinerU table with no marker caption anchor.

    `pdf_doc` (optional): same role as in `_render_mineru_table` --
    when provided, the VLM rewrite renders the table crop fresh from
    the PDF instead of using MinerU's saved JPG.
    """
    import wrap_mineru as _wm
    # Rename MinerU's content-hash JPG to match the sidecar `.md`
    # naming (`table_{id}_p{page}_{idx}.jpg` when id known).
    _rename_table_asset_semantic(blk, table_idx, assets_abs,
                                 asset_prefix, table_id=num)
    pat = re.compile(r"\bTable\s+" + re.escape(num) + r"\b", re.IGNORECASE)
    m = pat.search(marker_md)
    if m:
        blank = marker_md.find("\n\n", m.end())
        insert_pos = blank if blank >= 0 else len(marker_md)
    else:
        insert_pos = len(marker_md)
    pil_override = None
    if (pdf_doc is not None
            and vlm_rewrite_tables
            and getattr(blk, "bbox", None)
            and getattr(blk, "page_idx", None) is not None):
        try:
            pil_override = render_crop(pdf_doc, blk.page_idx, blk.bbox)
            log.info("Hybrid (EOD): rendered table %d from PDF page %d "
                     "bbox=%s for VLM (bypassing MinerU JPG)",
                     table_idx, blk.page_idx + 1, blk.bbox)
        except Exception as e:
            log.warning("Hybrid (EOD): render_crop failed for table %d "
                        "(falling back to MinerU JPG): %s", table_idx, e)
            pil_override = None
    full = _wm.table_block_to_md(
        blk, table_idx=table_idx, assets_rel=assets_rel,
        vlm_tables=vlm_rewrite_tables, assets_abs=assets_abs,
        report=report, asset_prefix=asset_prefix,
        pil_image_override=pil_override,
        table_id=num,
        inline_in_body=False,
        vlm_force=vlm_force,
    )
    return (insert_pos, insert_pos, "\n" + full.lstrip())


def _align_marker_to_mineru_layout(marker_md: str, parsed,
                                   assets_rel: str, assets_abs: Path,
                                   *,
                                   asset_prefix: str,
                                   vlm_rewrite_tables: bool,
                                   strip_chart_details: bool,
                                   report,
                                   table_workers: int = 1,
                                   pdf_doc=None,
                                   vlm_force: bool = False) -> str:
    """Splice MinerU figure/table blocks into marker's body, matched by
    figure/table number.

    Marker's caption text is preserved as-is (clean surya OCR). Each
    MinerU image/table block whose number matches a marker `Fig. N` /
    `Table N` line gets spliced above (figures) or replaces marker's
    table region (tables). Unmatched-on-either-side cases get logged
    counters and routed to fallbacks.

    Populates report.rescues['hybrid_splice'] when any counter is
    non-zero.
    """
    fig_anchors: dict[str, list] = {}
    for m in HYBRID_FIG_LINE_RE.finditer(marker_md):
        n = _normalise_fig_id(m.group(1))
        fig_anchors.setdefault(n, []).append(m)
    tbl_anchors: dict[str, list] = {}
    for m in HYBRID_TABLE_LINE_RE.finditer(marker_md):
        n = m.group("id")
        tbl_anchors.setdefault(n, []).append(m)

    mineru_figs: dict[str, list] = {}
    mineru_tbls: dict[str, list] = {}
    for blk in parsed.blocks:
        t = blk.type
        if t.startswith("_adopted"):
            continue
        if t in ("image", "chart"):
            fid = _hybrid_fig_id_from_block(blk)
            if fid is not None:
                mineru_figs.setdefault(_normalise_fig_id(fid), []).append(blk)
        elif t == "table":
            n = _extract_table_number(blk.text)
            if n is not None:
                mineru_tbls.setdefault(n, []).append(blk)

    counters = {
        "figures_spliced": 0,
        "tables_swapped": 0,                # legacy; tables_linked_inline supersedes
        "tables_linked_inline": 0,          # matched caption -> sidecar link inserted
        "tables_appended_index": 0,         # no caption anchor -> start-of-doc index
        "unmatched_mineru_figs": 0,
        "unmatched_mineru_tbls": 0,
        "marker_caption_no_mineru_asset": 0,
        "duplicate_mineru_figs": 0,
        "inferred_fig_number_pairings": 0,
        "inferred_tbl_number_pairings": 0,
    }
    splices: list[tuple[int, int, str]] = []

    def _record_fig_score(blk, num, caption_line, score, dropped=False):
        """Append a FigureScore entry mirroring what wrap_mineru.emit_markdown
        does for the MinerU-only layout. Hybrid had been incrementing the
        figures_spliced counter without populating report.figures, so
        QualityReport.overall() saw `figures=[]` and graded figure-only
        papers F regardless of how well the splice worked."""
        if not getattr(blk, "image_path", None):
            return
        cap = (caption_line or "").strip()
        cap_words = len(cap.split())
        report.figures.append(FigureScore(
            filename=asset_prefix + blk.image_path,
            caption_produced=bool(cap),
            caption_length=cap_words,
            score=score,
            matched_figure=num,
            dropped=dropped,
        ))

    for num, blks in mineru_figs.items():
        anchors = fig_anchors.get(num, [])
        if not anchors:
            for blk in blks:
                splices.append(_build_eod_figure_splice(
                    blk, marker_md, num, assets_rel, asset_prefix))
                counters["unmatched_mineru_figs"] += 1
                # EOD splice: no marker caption anchor matched, so score
                # is lower (0.6). Use MinerU's caption text for length.
                _record_fig_score(blk, num, blk.text or "", score=0.6)
            continue
        anchor = anchors[0]
        primary = next((b for b in blks if b.subpanel_paths), blks[0])
        extras = ([b for b in blks if b is not primary]
                  if not primary.subpanel_paths else [])
        line_start = marker_md.rfind("\n", 0, anchor.start()) + 1
        line_end = marker_md.find("\n", anchor.end())
        if line_end < 0:
            line_end = len(marker_md)
        marker_caption_line = marker_md[line_start:line_end]
        emission = _render_image_only(
            primary, num, marker_caption_line,
            assets_rel, asset_prefix, extras=tuple(extras),
            assets_abs=assets_abs)
        if emission:
            splices.append((line_start, line_start, emission + "\n\n"))
            counters["figures_spliced"] += 1
            # Clean match (marker caption + MinerU image both present):
            # score 1.0 when caption has >= 5 words, else 0.7 -- mirrors
            # _score_figure's caption-length gate.
            _record_fig_score(
                primary, num, marker_caption_line,
                score=1.0 if len(marker_caption_line.split()) >= 5 else 0.7,
            )
        if extras:
            counters["duplicate_mineru_figs"] += len(extras)
            for ex in extras:
                # Extras are alternate MinerU extractions for the same
                # fig number -- recorded so their existence is visible
                # in report.figures, but flagged dropped=True since they
                # don't add unique content.
                _record_fig_score(
                    ex, num, marker_caption_line,
                    score=0.6, dropped=True,
                )

    # Inferred-pairing fallback for figures whose MinerU caption text
    # was garbled or empty so `_hybrid_fig_id_from_block` returned None.
    # Two real-world cases that motivated this pass:
    #   alexander Fig 2: MinerU caption "Fig. 2." prefix got mid-text-
    #     merged into "FigNa" (the "2." disappeared).
    #   jacquet Fig 7: MinerU caption text was empty entirely.
    # In both, MinerU's image asset was extracted correctly but the
    # block dropped out of `mineru_figs` -> Sarah saw marker's `Fig. N`
    # caption sitting in the body with no image above it.
    #
    # Only fires when:
    #   1. there's at least one MinerU image/chart block with no fid AND
    #      at least one fig number in `fig_anchors` not in `mineru_figs`,
    #   2. the two lists have IDENTICAL lengths (ambiguity guard:
    #      mismatched counts could mislabel),
    #   3. the unmatched blocks are large enough to be real figures
    #      (`min(width, height) > 50` filters publisher logos / DOI
    #      badges / page-decoration thumbs).
    # Pairing is sequential: parsed.blocks already in reading order,
    # marker anchors sorted by document position -- for born-digital
    # papers these orders agree.
    unmatched_blocks = []
    for blk in parsed.blocks:
        if blk.type.startswith("_adopted"):
            continue
        if blk.type not in ("image", "chart"):
            continue
        if _hybrid_fig_id_from_block(blk) is not None:
            continue
        bbox = getattr(blk, "bbox", None)
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < 50 or (y1 - y0) < 50:
            continue
        unmatched_blocks.append(blk)

    unmatched_marker_nums = sorted(
        (n for n in fig_anchors if n not in mineru_figs),
        key=lambda n: fig_anchors[n][0].start(),
    )

    inferred_paired_nums: set = set()
    if (len(unmatched_blocks) >= 1
            and len(unmatched_blocks) == len(unmatched_marker_nums)):
        for blk, num in zip(unmatched_blocks, unmatched_marker_nums):
            anchor = fig_anchors[num][0]
            line_start = marker_md.rfind("\n", 0, anchor.start()) + 1
            line_end = marker_md.find("\n", anchor.end())
            if line_end < 0:
                line_end = len(marker_md)
            marker_caption_line = marker_md[line_start:line_end]
            emission = _render_image_only(
                blk, num, marker_caption_line,
                assets_rel, asset_prefix, extras=(),
                assets_abs=assets_abs)
            if emission:
                splices.append((line_start, line_start, emission + "\n\n"))
                counters["inferred_fig_number_pairings"] += 1
                inferred_paired_nums.add(num)
                # Inferred pairings have a marker caption + a real
                # MinerU image, but the match was via position order
                # rather than caption text -- score slightly below the
                # clean-match path (0.85 vs 1.0).
                _record_fig_score(blk, num, marker_caption_line, score=0.85)

    for num, anchors in fig_anchors.items():
        if num in mineru_figs or num in inferred_paired_nums:
            continue
        counters["marker_caption_no_mineru_asset"] += len(anchors)

    # Phase 1 (sequential): build per-table tasks. Cheap -- no VLM.
    table_tasks: list[tuple] = []
    table_idx_counter = 0
    for num, blks in mineru_tbls.items():
        anchors = tbl_anchors.get(num, [])
        for blk in blks:
            table_idx_counter += 1
            table_tasks.append((blk, num, table_idx_counter, anchors))

    # Phase 2 (potentially concurrent): each task invokes the VLM if
    # pipe-conversion fell through (rowspan, etc.). The VLM call is the
    # only network-bound step, so a ThreadPoolExecutor here gives the
    # same speedup as marker layout's process_tables. vLLM batches
    # concurrent requests server-side, so workers > 1 is the
    # recommended setting on vLLM; LM Studio users may want workers=1.
    def _run_one(args):
        blk, num, idx, anchors = args
        if not anchors:
            # Conservative splice: unmatched MinerU tables accumulate
            # for a single start-of-doc `## Extracted tables` block.
            # The sidecar is written; the index helper builds links
            # from the returned dict.
            info = _render_unmatched_table_sidecar(
                blk, num, idx,
                assets_rel, asset_prefix,
                vlm_rewrite_tables, assets_abs, report,
                pdf_doc=pdf_doc, vlm_force=vlm_force)
            return ("unmatched", info)
        anchor = anchors[0]
        region_start, region_end = _find_marker_table_region(
            marker_md, anchor)
        replacement = _render_mineru_table(
            blk, num, idx,
            assets_rel, asset_prefix,
            vlm_rewrite_tables, assets_abs, report,
            pdf_doc=pdf_doc, vlm_force=vlm_force)
        # Conservative splice: ALWAYS insert at region_end (the end of
        # marker's natural table block), never replace marker's
        # pipe-md. This avoids the corruption surface where the
        # range-replacement could swallow / mangle marker's content.
        prefix = _blank_line_prefix(marker_md, region_end)
        return ("matched",
                (region_end, region_end, prefix + replacement.lstrip()))

    if table_workers > 1 and len(table_tasks) > 1:
        log.info("Hybrid layout: dispatching %d table VLM call(s) "
                 "concurrently (workers=%d)",
                 len(table_tasks), table_workers)
        with ThreadPoolExecutor(max_workers=table_workers) as pool:
            results = list(pool.map(_run_one, table_tasks))
    else:
        results = [_run_one(t) for t in table_tasks]

    # Accumulator for unmatched tables -> start-of-doc `## Extracted
    # tables` index. Each entry is the dict returned by
    # `_render_unmatched_table_sidecar` (or None if no sidecar was
    # writable). Ordered by appearance in MinerU's reading order
    # (`_run_one` preserves).
    unmatched_for_index: list = []
    for kind, payload in results:
        if kind == "matched":
            splices.append(payload)
            counters["tables_linked_inline"] += 1
        elif kind == "unmatched":
            if payload is not None:
                unmatched_for_index.append(payload)
            counters["tables_appended_index"] += 1
        else:
            # Legacy kind == "eod" path no longer used; defensive.
            splices.append(payload)
            counters["unmatched_mineru_tbls"] += 1

    # report.tables may have been appended out-of-order by concurrent
    # workers. Sort by table index to restore deterministic order.
    if table_workers > 1 and len(table_tasks) > 1 and report.tables:
        report.tables.sort(key=lambda t: getattr(t, "index", 0))

    # Inferred-pairing fallback for tables whose MinerU caption text was
    # garbled or didn't start with "Table N." so `_extract_table_number`
    # returned None and the block dropped out of `mineru_tbls`. Mirror of
    # the figure-side inferred-pairing pass above (fudge #5).
    # Real-world case that motivated this:
    #   cuk Table 1: MinerU caption text starts with the table content
    #     ("Potential Moon-forming giant impacts...") instead of "Table
    #     1." prefix. Marker has `**Table 1.**` in the body but no
    #     MinerU asset matched -> splice silently dropped the table.
    # Conditions (mirror of fudge #5):
    #   1. >=1 MinerU table block with no extractable number AND >=1
    #      marker tbl_anchor with no MinerU match,
    #   2. counts identical (ambiguity guard),
    #   3. blocks have a usable bbox (filters caption-only fragments).
    unmatched_mineru_tbl_blocks: list = []
    for blk in parsed.blocks:
        if blk.type.startswith("_adopted"):
            continue
        if blk.type != "table":
            continue
        if _extract_table_number(blk.text) is not None:
            continue
        if not getattr(blk, "bbox", None):
            continue
        unmatched_mineru_tbl_blocks.append(blk)

    unmatched_marker_tbl_nums = sorted(
        (n for n in tbl_anchors if n not in mineru_tbls),
        key=lambda n: tbl_anchors[n][0].start(),
    )

    inferred_paired_tbl_nums: set = set()
    if (len(unmatched_mineru_tbl_blocks) >= 1
            and len(unmatched_mineru_tbl_blocks) ==
                len(unmatched_marker_tbl_nums)):
        for blk, num in zip(unmatched_mineru_tbl_blocks,
                            unmatched_marker_tbl_nums):
            anchor = tbl_anchors[num][0]
            region_start, region_end = _find_marker_table_region(
                marker_md, anchor)
            table_idx_counter += 1
            replacement = _render_mineru_table(
                blk, num, table_idx_counter,
                assets_rel, asset_prefix,
                vlm_rewrite_tables, assets_abs, report,
                pdf_doc=pdf_doc, vlm_force=vlm_force)
            # Same conservative-splice rule as _run_one: insert at
            # region_end, never replace marker's pipe-md.
            prefix = _blank_line_prefix(marker_md, region_end)
            splices.append((region_end, region_end,
                            prefix + replacement.lstrip()))
            counters["inferred_tbl_number_pairings"] += 1
            inferred_paired_tbl_nums.add(num)

    for num, anchors in tbl_anchors.items():
        if num in mineru_tbls or num in inferred_paired_tbl_nums:
            continue
        counters["marker_caption_no_mineru_asset"] += len(anchors)

    splices.sort(key=lambda s: s[0], reverse=True)
    out = marker_md
    for start, end, repl in splices:
        out = out[:start] + repl + out[end:]

    # Conservative-splice index: unmatched MinerU tables (no marker
    # caption to anchor against) get their sidecar links collected
    # into a single `## Extracted tables` block at the START of the
    # body. Refs-walks-from-end hooks (_detect_refs_section,
    # rescue_journal_refs, _rescue_append_api_refs) walk backward
    # from end of doc -- prepending at start avoids confusing them.
    if unmatched_for_index:
        index_md = _emit_extracted_tables_index(
            unmatched_for_index, assets_rel, asset_prefix)
        out = _prepend_extracted_tables_index(out, index_md)

    if any(counters.values()):
        report.rescues["hybrid_splice"] = counters
    return out


def _warmup_vlm_connection() -> None:
    """Wake vLLM's model worker before the first heavy table-image
    call by sending one real `/chat/completions` request with a tiny
    placeholder image. Failures are non-fatal: if the warmup can't
    reach the endpoint, the subsequent real calls will surface the
    same error with their own retry paths.

    Rationale (per docs/dev/HYBRID_IMPLEMENTATION.md fudge #7): under
    hybrid the marker + MinerU phase runs for several minutes with no
    VLM activity. The marker layout path naturally hits
    `trim_to_first_article` and `extract_citation` BEFORE
    `process_tables`, so the table calls are never the first
    `/chat/completions` request after idle. The hybrid path doesn't
    have this prelude -- the table splice fires inside
    `run_marker_plus_mineru_layout` as the first chat/completions
    call. vLLM's model worker apparently isn't responsive on the very
    first call after a long idle and rejects with APIConnectionError.

    An earlier attempt used `client.models.list()` (a `/v1/models` GET)
    as the warmup -- that succeeded but didn't fix the failure, because
    `/v1/models` is served separately and doesn't exercise the model
    worker. This version sends a real `/chat/completions` POST with a
    32x32 white image and a one-token prompt, mirroring exactly what
    the table calls will look like. Costs ~1-5s per paper but reliably
    warms the worker.
    """
    if _PROVIDER == "anthropic":
        # Anthropic SDK / provider not subject to the vLLM post-idle
        # rejection pattern. Skip the warmup.
        return
    try:
        warm_img = Image.new("RGB", (32, 32), color="white")
        # max_retries=5 + raise_on_error=False mirrors the table-call
        # retry behaviour. timeout=30 caps the wait so a genuinely
        # down endpoint doesn't stall the run.
        result = vlm("Reply OK.", warm_img, max_tokens=5,
                     max_retries=5, timeout=30.0)
        if result is not None:
            log.info("VLM warmup chat/completions succeeded")
        else:
            log.info("VLM warmup chat/completions returned no content "
                     "(model worker may not be ready -- table calls "
                     "will retry on their own)")
    except Exception as e:
        log.debug("VLM warmup failed (non-fatal): %s", e)


def run_marker_plus_mineru_layout(pdf_path: Path, assets_dir: Path,
                                  *,
                                  asset_prefix: str = "",
                                  force_ocr: bool = False,
                                  vlm_rewrite_tables: bool = False,
                                  rescue_orphan_captions: bool = True,
                                  rescue_subpanels: bool = True,
                                  strip_chart_details: bool = False,
                                  mineru_backend: str = "pipeline",
                                  mineru_extra_args: tuple = (),
                                  skip_mineru_run: bool = False,
                                  mineru_existing_dir: Optional[Path] = None,
                                  report: "QualityReport",
                                  mineru_dir: Optional[Path] = None,
                                  table_workers: int = 1,
                                  vlm_force: bool = False) -> str:
    """--layout-source=hybrid: marker for body+caption text, MinerU for
    figure assets and table HTML.

    Runs marker and MinerU sequentially (additive cost). Marker provides
    the canonical body markdown -- text, caption text, reading order.
    MinerU provides figure assets (cropped JPGs in `assets/`) and table
    HTML; the same `rescue_orphan_captions` + `rescue_subpanel_groups`
    passes used by `run_mineru_layout` populate `figure_number` /
    `subpanel_paths` on `_MBlock`s. The aligner then splices each
    MinerU image / table into marker's body matched by figure/table
    NUMBER (not text similarity), so marker's clean OCR caption stays
    in place.

    Side effects:
      - Marker writes its own image extracts to `assets_dir`.
      - MinerU images are copied to `assets_dir` with `asset_prefix`.
      - report.rescues gets `orphan_captions`, `subpanel_groups`,
        `hybrid_splice` blocks (when non-empty).

    `mineru_dir` is a persistent path (e.g. `<paper_dir>/mineru/`) that
    receives MinerU's flat output (mirrors run_mineru_layout's contract).
    """
    import layout_mineru as _lm
    import wrap_mineru as _wm

    marker_md, marker_images = run_marker(
        pdf_path, assets_dir, asset_prefix=asset_prefix, force_ocr=force_ocr)

    stem = pdf_path.stem
    with _stage_mineru_run(pdf_path,
                           mineru_dir=mineru_dir,
                           mineru_backend=mineru_backend,
                           mineru_extra_args=mineru_extra_args,
                           skip_mineru_run=skip_mineru_run,
                           mineru_existing_dir=mineru_existing_dir) as produced:
        middle = produced / f"{stem}_middle.json"
        if not middle.exists():
            raise FileNotFoundError(
                f"MinerU output missing {middle.name} under {produced}")
        parsed = _lm.parse_middle_json(middle)
        if rescue_orphan_captions:
            rc_stats = _lm.rescue_orphan_captions(
                parsed.blocks, page_sizes=parsed.page_sizes)
            if rc_stats:
                report.rescues["orphan_captions"] = rc_stats
        else:
            log.info("Hybrid layout: orphan-caption rescue skipped "
                     "(--no-rescue-orphan-captions)")
        if rescue_subpanels:
            sp_stats = _lm.rescue_subpanel_groups(parsed.blocks)
            if sp_stats and sp_stats.get("groups"):
                log.info("Hybrid layout: consolidated %d multi-panel "
                         "figure group(s); %d panel(s) adopted",
                         sp_stats["groups"], sp_stats["panels_adopted"])
            if sp_stats:
                report.rescues["subpanel_groups"] = sp_stats
        else:
            log.info("Hybrid layout: sub-panel consolidation skipped "
                     "(--no-rescue-subpanels)")
        n_assets = _wm.copy_assets(produced / "images", assets_dir,
                                   asset_prefix=asset_prefix)
        log.info("Hybrid layout: copied %d MinerU image asset(s) to %s",
                 n_assets, assets_dir)
        # Warmup the VLM connection pool when table VLM rewrites are
        # enabled. Under hybrid, marker + MinerU run for several minutes
        # without touching the VLM, and the FIRST table call frequently
        # hits a "Connection error" because the persistent HTTP
        # connection has gone stale. A cheap models.list() ping
        # re-establishes the connection before the splice fires.
        if vlm_rewrite_tables:
            _warmup_vlm_connection()
        # Open the PDF here so table VLM rewrites can render crops
        # fresh from the PDF (via render_crop(doc, page_idx, bbox))
        # instead of using MinerU's saved JPGs. Hypothesis: MinerU's
        # JPG intermediate carries compression artifacts that the
        # vLLM vision encoder chokes on under hybrid; PDF-render
        # bytes match what the marker+vlm-tables path always used
        # successfully. See docs/dev/HYBRID_IMPLEMENTATION.md fudge
        # #10. fitz.Document is thread-safe for reads, so concurrent
        # table workers can share this handle. If the PDF can't be
        # opened (e.g. synthetic test fixtures), fall back to None
        # and the splice will use MinerU's JPGs.
        _pdf_doc = None
        try:
            _pdf_doc = fitz.open(str(pdf_path))
        except Exception as e:
            log.debug("Hybrid: could not open PDF for VLM table "
                      "render (will use MinerU JPG fallback): %s", e)
        try:
            body = _align_marker_to_mineru_layout(
                marker_md, parsed, "assets", assets_dir,
                asset_prefix=asset_prefix,
                vlm_rewrite_tables=vlm_rewrite_tables,
                strip_chart_details=strip_chart_details,
                report=report,
                table_workers=table_workers,
                pdf_doc=_pdf_doc,
                vlm_force=vlm_force)
        finally:
            if _pdf_doc is not None:
                _pdf_doc.close()
    body, n_marker_dropped = _strip_marker_image_links(
        body, set(marker_images.keys()))
    if n_marker_dropped:
        report.rescues.setdefault("hybrid_splice", {})[
            "marker_image_links_removed"] = n_marker_dropped
        log.info("Hybrid layout: removed %d marker image-link line(s) "
                 "from body (files retained in assets/ for evaluation)",
                 n_marker_dropped)
    log.info("Hybrid layout produced %d chars of markdown "
             "(figures=%d, tables=%d)",
             len(body), len(report.figures), len(report.tables))
    return body


# --- Quality rating ---------------------------------------------------------

@dataclass
class TableScore:
    index: int
    page: Optional[int]
    located: bool
    jpeg_saved: bool
    pre_redo_reason: Optional[str]
    vlm_redone: bool
    post_redo_reason: Optional[str]
    score: float
    sidecar: Optional[str] = None
    # The paper-relative table id ("1", "S1", "I", "A.4") -- mirrors
    # FigureScore.matched_figure. Populated by hybrid (from the splice's
    # `num` variable) and by mineru-only (derived from the block's
    # caption text via _extract_table_number). Left None by marker
    # layout where the table-finder doesn't carry table-id semantics.
    matched_table: Optional[str] = None


@dataclass
class FigureScore:
    filename: str
    caption_produced: bool
    caption_length: int
    score: float
    matched_figure: Optional[str] = None  # e.g. "1", "S1", "12"
    dropped: bool = False


@dataclass
class PageScore:
    """Diagnostic per-page record. Tracks character density and the
    sparse-page-rescue trigger state; does NOT contribute to
    `QualityReport.overall()`. The density signal is too weak to
    serve as a quality score (see comment on score_pages) -- it
    discriminates content-type (text-heavy vs figure-heavy) more
    than extraction quality. Kept for diagnostics and as the trigger
    input for the opt-in sparse-page rescue (Hook 3)."""
    page: int
    char_density: float
    sparse: bool
    rescued: bool


@dataclass
class RefsLayout:
    """Layout fingerprint of the references section in the PDF. Used by
    Phase-2 rescue dispatch: the Science 3-column rescue requires
    columns==3, otherwise the rescue is skipped. Recorded as a sidecar
    field on ReferencesScore for diagnostics regardless of whether a
    rescue fires."""
    columns: int                # 1, 2, 3, or 0 (unknown / noisy)
    marker_style: str           # numbered | superscript | bracketed | none
    page_idx: Optional[int]     # 0-indexed page where the refs section starts


@dataclass
class ReferencesScore:
    """Heuristic quality score for the final References section.

    Phase-1 instrumentation for journal-aware reference reconstruction
    (no behavior change). Computed by score_references() after all
    deterministic reference passes have run, recorded in QualityReport
    as a sidecar field, and surfaced in frontmatter / .meta.json /
    manifest. NOT included in QualityReport.overall() yet -- the
    signal is uncalibrated.

    Phase-2 adds the layout / rescue-bookkeeping fields below the
    Phase-1 fields. They stay None unless a rescue is dispatched.
    """
    section_count: int           # number of `## References` headings seen
    entry_count: int             # detected entries inside the canonical section
    numbered_continuous: bool    # 1..N with no gaps when style is numbered
    year_hit_ratio: float        # share of entries containing 19xx/20xx
    longest_over_median: float   # longest_entry_chars / median_entry_chars
    style: str                   # numbered | author-year | superscript | footnote | unknown
    journal_slug: Optional[str]  # carried so consumers can dispatch later
    score: float                 # 0.0..1.0 weighted composite
    # Phase-2 additions (default None so existing call sites stay valid):
    last_well_bounded: bool = True  # last entry doesn't bleed into Acks/Funding
    layout: Optional["RefsLayout"] = None
    score_pre_rescue: Optional[float] = None    # set when a rescue ran
    score_post_rescue: Optional[float] = None   # set when a rescue ran
    rescue_applied: Optional[str] = None        # e.g. "science-3col", "aps-bleed"
    rescue_decision: Optional[str] = None       # spliced | kept-pre-rescue-* | dry-run
    # API-fallback bookkeeping (always-on rescue that appends a refs
    # section sourced from Crossref/OpenAlex when local transcription
    # is poor or absent).
    api_source: Optional[str] = None            # "crossref" | "openalex" | None
    api_refs_appended: int = 0                  # count of refs appended
    # Set to True when --offline (or allow_network=False) prevented
    # the API fallback rescue from running on a paper whose trigger
    # gate would otherwise have fired (low score / gaps / no section).
    # Disambiguates "no fetch attempted" from "fetch returned nothing".
    api_skipped_offline: bool = False


def _score_table(located: bool, pre_reason: Optional[str],
                 vlm_redone: bool, post_reason: Optional[str]) -> float:
    if located:
        if vlm_redone:
            return 1.0 if post_reason is None else 0.6
        return 0.8 if pre_reason is None else 0.4
    # not located
    if vlm_redone:
        # text-only cleanup path: the VLM reformatted marker's markdown
        # without a PDF crop. Slight credibility discount vs crop-based.
        return 0.85 if post_reason is None else 0.55
    return 0.7 if pre_reason is None else 0.3


def _score_figure(caption_produced: bool, caption_length: int,
                  vlm_enabled: bool, file_missing: bool) -> float:
    if file_missing:
        return 0.3
    if not vlm_enabled:
        return 0.8
    if not caption_produced:
        return 0.7
    return 1.0 if caption_length >= 5 else 0.7


_PAGE_SPAN_RE = re.compile(r'<span\s+id="page-(\d+)-\d+"></span>')


def _refs_page_idx_from_md(md: str, doc: "fitz.Document") -> Optional[int]:
    """Best-effort PDF page index for the references section.

    Strategy:
    1. PDF text search for a line-leading "References" heading (works
       for any journal with an explicit ## References).
    2. Fallback: look at marker's <span id="page-N-M"></span> anchors
       inside the canonical refs body and take the most common N (the
       page where the bulk of the references render).
    3. Final fallback: the last page of the PDF.

    All paths return a 0-indexed page; None if the document is empty.
    """
    if doc.page_count == 0:
        return None
    page = _find_caption_page(doc, "References")
    if page is not None:
        return page
    sc, span = _detect_refs_section(md)
    if span is not None:
        body = md[span[0]:span[1]]
        page_hits = [int(m.group(1)) for m in _PAGE_SPAN_RE.finditer(body)]
        if page_hits:
            from collections import Counter as _Counter
            return _Counter(page_hits).most_common(1)[0][0]
    return doc.page_count - 1


def _detect_refs_marker_style(md: str) -> str:
    """Inspect the first few entries inside the canonical refs section
    to classify the in-list marker style. Distinct from the broader
    style classifier in score_references: this returns a finer-grained
    label suitable for the layout fingerprint."""
    sc, span = _detect_refs_section(md)
    if span is None:
        return "none"
    body = md[span[0]:span[1]][:5000]
    # Take a scan of the first 5 candidate entries.
    if re.search(r"<sup>\s*\d+\s*</sup>", body):
        return "superscript"
    if re.search(
        r'^\s*[-*]?\s*(?:<span\s+id="[^"]*"></span>\s*)*\[\d+\]',
        body, re.MULTILINE,
    ):
        return "bracketed"
    if re.search(
        r'^\s*[-*]?\s*(?:<span\s+id="[^"]*"></span>\s*)*\d{1,3}\.\s+\S',
        body, re.MULTILINE,
    ):
        return "numbered"
    return "none"


def _detect_refs_layout(doc: "fitz.Document",
                        md: str) -> "RefsLayout":
    """Best-effort layout fingerprint of the references section.

    columns is detected by histogramming text-block left edges on the
    references page. For a single-column page, blocks share one
    cluster of x0 values; for a 2-column page, two well-separated
    clusters; for a 3-column page (older Science layout), three. Bins
    are 50pt wide; "well-separated" means the gap between mode bins
    must be at least one bin's worth of empty space.

    Returns columns=0 when the histogram is too noisy or the page
    can't be located.

    Lightweight (no rendering, just block-coordinate sampling). Safe
    on any backend; uses only PyMuPDF block geometry."""
    page_idx = _refs_page_idx_from_md(md, doc)
    if page_idx is None:
        return RefsLayout(columns=0,
                          marker_style=_detect_refs_marker_style(md),
                          page_idx=None)
    try:
        page = doc[page_idx]
        blocks = page.get_text("blocks") or []
        page_w = float(page.rect.width) or 1.0
    except Exception as e:
        log.debug("_detect_refs_layout: page read failed (%s); fallback",
                  e)
        return RefsLayout(columns=0,
                          marker_style=_detect_refs_marker_style(md),
                          page_idx=page_idx)

    # Filter to substantive text blocks; small / image / empty ones
    # add noise to the histogram.
    BLOCK_TYPE_TEXT = 0
    text_blocks = [b for b in blocks
                   if len(b) >= 7 and b[6] == BLOCK_TYPE_TEXT
                   and isinstance(b[4], str) and len(b[4].strip()) >= 20]
    if len(text_blocks) < 5:
        return RefsLayout(columns=0,
                          marker_style=_detect_refs_marker_style(md),
                          page_idx=page_idx)

    bin_width = 50.0
    n_bins = max(2, int(page_w / bin_width) + 1)
    hist = [0] * n_bins
    for b in text_blocks:
        x0 = float(b[0])
        idx = max(0, min(n_bins - 1, int(x0 / bin_width)))
        hist[idx] += 1

    # Identify "mode" bins: a bin counts as a mode iff it has >= 2
    # blocks AND it's a local peak (>= each immediate neighbour).
    modes: list[int] = []
    for i, c in enumerate(hist):
        if c < 2:
            continue
        left = hist[i - 1] if i > 0 else 0
        right = hist[i + 1] if i < n_bins - 1 else 0
        if c >= left and c >= right:
            modes.append(i)
    # Merge adjacent modes (a wide column might span two bins).
    merged: list[int] = []
    for m in modes:
        if merged and m - merged[-1] <= 1:
            continue
        merged.append(m)

    columns = min(3, len(merged))
    return RefsLayout(columns=columns,
                      marker_style=_detect_refs_marker_style(md),
                      page_idx=page_idx)


# Boundary markers used to detect last-ref bleed into Acknowledgments
# / Funding / Author Contributions paragraphs. Word-boundary anchored;
# case-insensitive. Match returns the start position of the boundary
# inside the entry text.
_VERIFY_CONTENT_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]{4,}")


def _missing_numbered_gaps(entries: list[str]) -> list[int]:
    """Return the list of missing reference numbers in a numbered
    entries sequence. E.g. entries with leading numbers
    [1, 2, ..., 32, 35, 36, ...] yields [33, 34].

    Returns [] when the sequence is empty, not numbered, or already
    continuous. Caps at 5 missing numbers per gap to avoid
    pathological recovery on heavily broken refs lists."""
    nums: list[int] = []
    for e in entries:
        m = _NUMBERED_ENTRY_RE.match(e)
        if not m:
            return []
        raw = m.group("num1") or m.group("num2")
        try:
            nums.append(int(raw))
        except (TypeError, ValueError):
            return []
    if len(nums) < 2:
        return []
    missing: list[int] = []
    for i in range(len(nums) - 1):
        gap = nums[i + 1] - nums[i]
        if gap > 1:
            missing.extend(range(nums[i] + 1, nums[i + 1]))
    if not missing:
        return []
    if len(missing) > 5:
        return []
    return missing


# PRB / PRL old-format references use a superscript-prefix style:
# "33M. D. Knudson..." (the digits are <sup>33</sup> in the PDF, but
# get_text("text") flattens them to inline digits adjacent to the
# author surname). We scan PDF page text for this shape when looking
# for refs marker dropped from the markdown. The lookahead requires
# either an uppercase letter (author surname) or "S." (author
# initial-letter-period prefix) so we don't false-match volume
# numbers buried inside other refs.
_PDF_SUPERSCRIPT_REF_RE_FMT = (
    r"(?m)^{n}([A-Z][a-z]?\.?\s|[A-Z][a-z]+\s|S\.\s).*?(?=\n\s*\n|"
    r"\n\d{{1,3}}[A-Z]|\Z)"
)


def _rescue_missing_numbered_refs(md: str, doc: "fitz.Document",
                                  report: "QualityReport",
                                  out_dir: Optional[Path] = None) -> str:
    """Phase-2 deterministic rescue: when a numbered references list
    has small gaps in its numbering (e.g. 32 then 35, missing 33 and
    34), search the PDF text layer for the missing entries in the
    superscript-prefix format common to PRB / PRL papers
    ('33M. D. Knudson and R. W. Lemke ...'), and inject them into the
    markdown at the correct sequence position.

    Knudson2013 in the corpus has this exact pattern: refs 33 and 34
    are short and unusual ('to be published', 'unpublished'), so
    marker's layout heuristic drops them, leaving the rendered refs
    list discontinuous (32 -> 35).

    Trigger gates (checked by the dispatcher AND defensively here):
      - style == 'numbered'
      - 1 <= len(missing) <= 5
      - PDF text layer contains the missing-number entries

    No VLM. Returns md unchanged on any failure."""
    if doc is None:
        return md
    sc, span = _detect_refs_section(md)
    if span is None:
        return md
    body_start, body_end = span
    body = md[body_start:body_end]
    style = _classify_ref_style(body)
    if style != "numbered":
        return md
    entries = _split_ref_entries(body, style)
    missing = _missing_numbered_gaps(entries)
    if not missing:
        return md

    # Search the PDF text for each missing-number ref. Most likely
    # location is the page where the surrounding refs sit; cheaper to
    # scan all pages than guess.
    found: dict[int, str] = {}
    for page in doc:
        try:
            page_text = page.get_text("text") or ""
        except Exception:
            continue
        for n in missing:
            if n in found:
                continue
            pat = re.compile(_PDF_SUPERSCRIPT_REF_RE_FMT.format(n=n),
                             re.DOTALL)
            m = pat.search(page_text)
            if m:
                # Capture the full ref text starting with the number,
                # collapsed to a single line.
                start = m.start()
                # Walk forward to find the start of the next ref or
                # end-of-paragraph: boundary is "\n\d+[A-Z]" (next
                # superscript ref) or "\n\s*\n" (paragraph break) or
                # end of page text.
                end_pat = re.compile(r"\n\s*\n|\n\d{1,3}[A-Z]")
                end_m = end_pat.search(page_text, pos=start + 1)
                end = end_m.start() if end_m else len(page_text)
                ref_text = page_text[start:end].strip()
                # Collapse internal whitespace so the markdown bullet
                # is one logical line; preserve hyphenated breaks.
                ref_text = re.sub(r"\s+", " ", ref_text)
                # Reformat from "33M. D. Knudson..." to
                # "33. M. D. Knudson..." so the bullet matches the
                # rest of the markdown's numbered style.
                ref_text = re.sub(rf"^{n}", f"{n}. ", ref_text, count=1)
                found[n] = ref_text

    if not found:
        return md

    # Inject the recovered refs into the body at the correct
    # sequence position. For each missing number, find the FIRST
    # existing entry whose number > missing_n -- this handles
    # consecutive gaps (Knudson2013 has 33 AND 34 missing; entry 34
    # doesn't exist so we'd lose 33 if we only looked for n+1).
    # Process missing numbers in reverse so earlier insertions don't
    # invalidate later offsets.
    edits: list[tuple[int, int, str]] = []
    for missing_n in sorted(found.keys(), reverse=True):
        ref_text = found[missing_n]
        target_idx: Optional[int] = None
        for i, e in enumerate(entries):
            m = _NUMBERED_ENTRY_RE.match(e)
            if not m:
                continue
            raw = m.group("num1") or m.group("num2")
            if raw and int(raw) > missing_n:
                target_idx = i
                break
        if target_idx is None:
            continue
        target_in_body = body.find(entries[target_idx])
        if target_in_body < 0:
            continue
        injection = f"- {ref_text}\n\n"
        edits.append((body_start + target_in_body,
                      body_start + target_in_body, injection))

    if not edits:
        return md

    new_md = md
    for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        new_md = new_md[:start] + rep + new_md[end:]

    log.info("Rescue (missing-numbered-refs): recovered %d ref(s) "
             "[numbers: %s] from PDF text layer",
             len(found), sorted(found.keys()))
    return new_md


# Boundary inside an author-year mega-entry: marker accidentally
# joined two adjacent refs into one bulleted line. The boundary is
# the whitespace between the previous ref's terminator (period,
# close-angle-bracket from a `<DOI>` autolink, OR close-paren from
# a `[label](url)` markdown link) and the next ref's surname-comma-
# initial pattern. Lookbehind needs to allow `)` because modern AGU
# / Elsevier outputs encode DOIs as `[10.xxxx/...](https://doi.org/...)`
# whose close-paren is what immediately precedes the inter-ref
# whitespace. Lookbehind avoids matching author-list separators
# within a single ref (those are preceded by `,` or `&`, never
# `.`/`>`/`)`). Surname allows hyphenated forms (Liu-Wang) using
# ASCII hyphen plus the U+2010/U+2012 Unicode variants common in
# scholarly text.
_AUTHOR_YEAR_INTERNAL_BOUNDARY_RE = re.compile(
    r"(?<=[\.>)])\s+"
    r"(?=[A-Z][a-z]+(?:[‐\-‒][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?,\s+[A-Z]\.?)"
)


# PRB / PRL old-style references in superscript-prefix form. When
# marker fails to split refs across a page boundary, multiple refs
# get mashed onto one bullet that begins with a digit-letter pair
# (e.g., "- <span...></span>46From manufacturers website:..."). The
# transitions between refs inside such a mashed bullet show as:
# "<previous ref's terminator>" + space + "<next ref's number><name>".
# The terminator is `.` or `)` (close-paren of a markdown link).
# The next ref's leading "number" can be either a bare digit prefix
# (`46From...`) or a preserved superscript (`<sup>51</sup>...`).
_PRB_SUPERSCRIPT_BULLET_LINE_RE = re.compile(
    r"^- (?:<span\b[^>]*></span>\s*)*(?:<sup>)?\d{1,3}(?:</sup>)?"
    r"[A-Za-z*]",
    re.MULTILINE,
)
_PRB_INTERNAL_BOUNDARY_RE = re.compile(
    r"(?<=[\.\)])\s+(?=(?:<sup>)?\d{1,3}(?:</sup>)?[A-Za-z*])"
)
_PRB_NUM_PREFIX_RE = re.compile(
    r"^(?:<sup>)?(\d{1,3})(?:</sup>)?(?=[A-Za-z*])"
)


def _rescue_numbered_pageboundary_mash(md: str, doc: "fitz.Document",
                                       report: "QualityReport",
                                       out_dir: Optional[Path] = None) -> str:
    """Phase-3 deterministic rescue for PRB / PRL papers where marker
    emitted a single mashed bullet containing multiple refs (typically
    a continuation page of the references list). Knudson2013 in the
    silica-shock corpus has refs 46-58 mashed onto one bullet placed
    AT THE TOP of the references section, before ref 1 -- a column-
    reading-order failure on the second refs page.

    The rescue:
      1. Finds the first bullet line whose content begins with a
         PRB-superscript-prefix shape (`<digit><Capital>` directly
         after the bullet/anchors, no period, no bracket).
      2. Splits at internal boundaries (`(?<=[.)])\\s+(?=<digit><Capital>)`).
      3. Reformats each split part as a proper `- N. <rest>` bullet.
      4. If all split numbers exceed the highest existing entry's
         number (the page-boundary case), moves the split bullets
         to AFTER the existing highest-numbered entry so the list
         reads in numeric order. Otherwise splits in place.

    No PDF read; pure regex. Fires only on numbered-style sections."""
    sc, span = _detect_refs_section(md)
    if span is None:
        return md
    body_start, body_end = span
    body = md[body_start:body_end]
    style = _classify_ref_style(body)
    if style != "numbered":
        return md

    mm = _PRB_SUPERSCRIPT_BULLET_LINE_RE.search(body)
    if mm is None:
        return md
    line_start = mm.start()
    line_end = body.find("\n", line_start)
    if line_end < 0:
        line_end = len(body)
    line = body[line_start:line_end]

    # Strip the bullet prefix and any leading span anchors so the
    # boundary regex sees raw "<digit><Capital>" content.
    rest = re.sub(
        r"^- (?:<span\b[^>]*></span>\s*)*", "", line)
    boundaries = list(_PRB_INTERNAL_BOUNDARY_RE.finditer(rest))
    if not boundaries:
        return md

    cuts = [0] + [b.end() for b in boundaries] + [len(rest)]
    parts = [rest[cuts[i]:cuts[i + 1]].strip()
             for i in range(len(cuts) - 1)]
    bullets: list[str] = []
    nums: list[int] = []
    for part in parts:
        num_m = _PRB_NUM_PREFIX_RE.match(part)
        if not num_m:
            continue
        try:
            n = int(num_m.group(1))
        except (TypeError, ValueError):
            continue
        content = part[num_m.end():].lstrip()
        if not content:
            continue
        bullets.append(f"- {n}. {content}")
        nums.append(n)

    if len(bullets) < 2:
        return md

    # Find the highest existing numbered entry (excluding the mashed
    # one). If all split numbers exceed it, move split bullets to
    # after that entry's tail.
    entries = _split_ref_entries(body, "numbered")
    existing_max = 0
    last_entry_text = ""
    for e in entries:
        m = _NUMBERED_ENTRY_RE.match(e)
        if not m:
            continue
        raw = m.group("num1") or m.group("num2")
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        if n > existing_max:
            existing_max = n
            last_entry_text = e

    replacement = "\n\n".join(bullets)
    new_body: str
    if (existing_max > 0 and last_entry_text and nums
            and min(nums) > existing_max):
        # Reorder: remove the mashed line, then insert split bullets
        # after the existing highest-numbered entry's tail.
        # Step 1: remove mashed line (and a trailing newline if present
        # so we don't leave a blank).
        cut_end = line_end + 1 if line_end < len(body) else line_end
        body_no_mash = body[:line_start] + body[cut_end:]
        # Step 2: locate the highest-numbered entry's end in the
        # post-removal body. The removal is BEFORE the entry (Knudson
        # case: mash at top, entries below), so we need to shift.
        last_pos = body_no_mash.find(last_entry_text)
        if last_pos < 0:
            # Fallback: split in place if we can't locate the target.
            new_body = body[:line_start] + replacement + body[line_end:]
        else:
            insert_at = last_pos + len(last_entry_text)
            # Ensure surrounding blank lines so splits read cleanly.
            new_body = (body_no_mash[:insert_at]
                        + "\n\n" + replacement + "\n"
                        + body_no_mash[insert_at:])
    else:
        # In-place split: keep the mash position, replace with bullets.
        new_body = body[:line_start] + replacement + body[line_end:]

    new_md = md[:body_start] + new_body + md[body_end:]
    log.info("Rescue (numbered-pageboundary-mash): split %d refs "
             "(numbers=%s) from a mashed bullet%s",
             len(bullets), nums,
             "; reordered to after existing entries"
             if (existing_max > 0 and nums and min(nums) > existing_max)
             else " (in-place)")
    return new_md


def _rescue_author_year_resplit(md: str, doc: "fitz.Document",
                                report: "QualityReport",
                                out_dir: Optional[Path] = None) -> str:
    """Phase-3 deterministic rescue: re-split author-year mega-
    entries that marker accidentally joined into one bullet.

    Pattern (feng 2024 in the silica-shock corpus): two adjacent
    author-year refs end up on one bulleted line because marker's
    paragraph segmentation joined them on a column boundary. The
    transition is recognizable as 'period-or-angle-bracket + space
    + Surname, Initial.' inside the entry text.

    Trigger gates (re-checked here defensively):
      - style == 'author-year' (otherwise the boundary regex doesn't
        apply).
      - At least one entry exceeds 2x the median entry length
        (the mega-entry signal).
      - The boundary regex matches inside that mega-entry.

    No PDF read, no VLM. Pure regex on the markdown body."""
    sc, span = _detect_refs_section(md)
    if span is None:
        return md
    body_start, body_end = span
    body = md[body_start:body_end]
    style = _classify_ref_style(body)
    if style != "author-year":
        return md
    entries = _split_ref_entries(body, style)
    if len(entries) < 3:
        return md
    lengths = sorted(len(e) for e in entries)
    median = lengths[len(lengths) // 2] or 1

    edits: list[tuple[int, int, str]] = []
    for e in entries:
        if len(e) <= median * 2:
            continue
        boundaries = list(_AUTHOR_YEAR_INTERNAL_BOUNDARY_RE.finditer(e))
        if not boundaries:
            continue
        # Split the mega-entry at each boundary; the boundary regex
        # consumes the inter-ref whitespace, so each part is the raw
        # ref text. The first part already has the leading "- "
        # bullet prefix from the original entry; subsequent parts
        # need their own.
        parts: list[str] = []
        last = 0
        for m in boundaries:
            parts.append(e[last:m.start()].rstrip())
            last = m.end()
        parts.append(e[last:].rstrip())
        if len(parts) < 2:
            continue
        # Normalise: strip any leading bullet from each part, then
        # uniformly re-prefix with "- " so the resplit entries match
        # the rest of the section's bullet style.
        normalised: list[str] = []
        for piece in parts:
            piece = re.sub(r"^[-*]\s+", "", piece).strip()
            if piece:
                normalised.append(f"- {piece}")
        if len(normalised) < 2:
            continue
        new_text = "\n\n".join(normalised)
        offset = body.find(e)
        if offset < 0:
            continue
        edits.append((body_start + offset,
                      body_start + offset + len(e), new_text))

    if not edits:
        return md
    new_md = md
    for start, end, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        new_md = new_md[:start] + rep + new_md[end:]
    log.info("Rescue (author-year-resplit): re-split %d mega-entry/ies "
             "into multiple refs", len(edits))
    return new_md


def _verify_content_preserved(input_text: str,
                              output_text: str) -> bool:
    """Confirm the VLM didn't hallucinate bibliographic content.

    Looser than a strict reorder-only check: the VLM is allowed to
    add numbering ('- 1.', '[2]'), italic/bold markup, punctuation,
    whitespace, and 4-digit year fill-ins for OCR-truncated entries.
    What it MUST NOT do is invent new author names, journal names,
    place names, or other substantive content words.

    Implementation: extract alphabetic words of length >= 4 (Latin +
    accented Latin range) from both input and output, lowercased,
    compared as multisets. Output's multiset must be a subset of
    input's plus a small synthetic whitelist ('unplaced',
    'fragments'). Numbers, single/double-letter words, and pure
    markdown markup are excluded from comparison so they don't
    trigger false positives.

    Catches:
    - Hallucinated authors ('Smithers' if not in input).
    - Substituted journal names ('Nature' -> 'Cell').
    - Inserted body-paragraph commentary the model wrote on its own.

    Allows:
    - Numbering inserted by the rescue (digits aren't "content").
    - Cosmetic markdown additions (*italics*, **bold**, periods).
    - Year fill-ins from prior knowledge on truncated entries.
    - Single-character diacritic restorations on existing names
      (Lognonne -> Lognonné stays a 4+ char word; the multiset still
      sees a new lowercase word, so this is the one false-positive
      class to watch -- mitigated by checking lowercased ASCII-fold
      via .lower() being stable across diacritics for matching, and
      flagged for monitoring rather than relaxed further).
    """
    from collections import Counter as _Counter
    in_words = _Counter(m.group(0).lower()
                        for m in _VERIFY_CONTENT_WORD_RE.finditer(input_text))
    out_words = _Counter(m.group(0).lower()
                         for m in _VERIFY_CONTENT_WORD_RE.finditer(output_text))
    # Synthetic words the prompt explicitly permits in the output.
    for w in ("unplaced", "fragments"):
        if w in out_words:
            in_words[w] = in_words.get(w, 0) + out_words[w]
    extras = out_words - in_words
    return not extras


_ACKS_BOUNDARY_RE = re.compile(
    r"\b(?:Acknowledg(?:e?)ments?|Funding|Author\s+[Cc]ontributions?|"
    r"Competing\s+[Ii]nterests?)\b",
    re.IGNORECASE,
)


_APS_BRACKETED_TAIL_ENTRY_RE = re.compile(
    r"^\s*[-*]?\s*(?:<span\s+id=\"[^\"]*\"></span>\s*)*"
    r"\[(\d{1,3})\]\s+\S",
)


def _rescue_aps_missing_refs_heading(md: str,
                                     report: "QualityReport") -> str:
    """Always-on rescue (NOT gated behind --use-journal-rescue):
    insert a `## References` heading for APS papers (Phys. Rev.
    family) that emit a `[N]`-style numbered reference list at the
    end of the document with no heading line. Hicks2006 / Knudson2009
    in the user's corpus hit this case: PRL/PRB papers with refs as
    `[1] Author...` `[2] Author...` etc. Without a heading, section-
    chunked embedding pipelines cannot locate the refs section.

    Triggers when ALL hold:
      * report.metadata.journal_slug starts with "aps" (incl. "aps")
      * No `## References` (or alias) heading anywhere in `md`
      * A trailing run of >=5 contiguous `[N]\\s+...` lines exists,
        with N starting at 1 and strictly monotonically increasing.

    Body content is unchanged; only a single `## References` heading
    line plus a blank line are inserted before the first `[1]` entry.
    Idempotent: a second invocation finds the just-inserted heading
    and returns `md` unchanged.
    """
    if report is None or getattr(report, "metadata", None) is None:
        return md
    slug = getattr(report.metadata, "journal_slug", None) or ""
    if not (slug == "aps" or slug.startswith("aps-")):
        return md
    if _REFS_HEADING_FOR_SCORE_RE.search(md):
        return md
    lines = md.split("\n")
    entries: list[tuple[int, int]] = []
    for i, ln in enumerate(lines):
        m = _APS_BRACKETED_TAIL_ENTRY_RE.match(ln)
        if m:
            entries.append((i, int(m.group(1))))
    if len(entries) < 5:
        return md
    one_indices = [idx for idx, n in entries if n == 1]
    if not one_indices:
        return md
    start_pos = one_indices[-1]
    tail = [(idx, n) for (idx, n) in entries if idx >= start_pos]
    if len(tail) < 5:
        return md
    last_n = 0
    for _, n in tail:
        if n <= last_n:
            return md
        last_n = n
    line_idx = tail[0][0]
    new_lines = lines[:line_idx] + ["## References", ""] + lines[line_idx:]
    log.info("Rescue (aps-missing-heading): inserted '## References' "
             "before line %d (entries=%d, slug=%s)",
             line_idx, len(tail), slug)
    return "\n".join(new_lines)


# Generic refs-heading rescue (ported from wrap_mineru's
# rescue_missing_refs_heading, 2026-05-10). Catches papers whose
# trailing references list has no `## References` heading at all,
# regardless of journal. Runs AFTER _rescue_aps_missing_refs_heading
# so APS papers get the journal-specific path first; the heading
# check below makes both passes idempotent.
#
# Three styles supported. The Knudson-style footnote (1A.) detector +
# reformatter from wrap_mineru is intentionally NOT ported (Q3.3:
# minimize niche corrections in the codebase). Knudson papers will
# fall back to numbered detection if the run is dense enough; if not,
# the API-refs append path covers them.
_GENERIC_NUMBERED_REF_PAT = re.compile(r"(?m)^[ \t]*(\d{1,3})[\.\)]\s+\S")
_GENERIC_BRACKETED_REF_PAT = re.compile(r"(?m)^[ \t]*\[(\d{1,3})\]\s+\S")
_GENERIC_AUTHOR_YEAR_REF_PAT = re.compile(
    r"(?m)^[A-Z][A-Za-z\-\.,& ]{1,120}"
    r"(?:\([12]\d{3}[a-z]?\)|,\s+[12]\d{3}[a-z]?[\.\,])"
)


def _generic_refs_detect_run(matches: list,
                             max_gap_chars: int = 250
                             ) -> Optional[tuple[int, int]]:
    """Find the LAST contiguous run where adjacent matches start
    within `max_gap_chars`. Returns (start_offset, count) or None
    when no run >= 5 entries."""
    if not matches:
        return None
    runs: list[tuple[int, int, int]] = []
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
    last = runs[-1]
    if last[1] < 5:
        return None
    return (last[0], last[1])


def _detect_generic_numbered_or_bracketed(md: str, pat: re.Pattern,
                                          *, label: str) -> Optional[int]:
    """Inject offset for `## References` if a trailing numbered or
    bracketed run is detected; else None. Requires >=5 matches in
    the last 30% AND at least one anchor with N in {1,2,3} so we
    don't trigger on body-text "1." footnotes."""
    cut = int(len(md) * 0.7)
    tail_matches = list(pat.finditer(md, cut))
    if len(tail_matches) < 5:
        return None
    anchored = [m for m in tail_matches if int(m.group(1)) in (1, 2, 3)]
    if not anchored:
        return None
    anchor = anchored[0]
    log.info("Rescue (generic-missing-heading)[%s]: anchor at offset %d "
             "(N=%s, %d tail matches)", label, anchor.start(),
             anchor.group(1), len(tail_matches))
    return anchor.start()


def _detect_generic_author_year(md: str) -> Optional[int]:
    """Strict author-year detector: >=10 contiguous matches in last 25%."""
    cut = int(len(md) * 0.75)
    tail_matches = list(_GENERIC_AUTHOR_YEAR_REF_PAT.finditer(md, cut))
    run = _generic_refs_detect_run(tail_matches, max_gap_chars=400)
    if run is None or run[1] < 10:
        return None
    log.info("Rescue (generic-missing-heading)[author-year]: %d-entry "
             "run starting at offset %d", run[1], run[0])
    return run[0]


def _rescue_generic_missing_refs_heading(md: str,
                                         report: "QualityReport") -> str:
    """Always-on rescue: when the doc has no `## References` heading
    AND a trailing run of refs is detectable, inject the heading.

    Detection order: bracketed (`[1] Author...`), numbered (`1. Author...`),
    author-year (`Author A. (2020)`). First match wins.

    Idempotent: a doc with an existing References / Bibliography heading
    is returned unchanged (the heading regex matches `## References`,
    `# Bibliography`, `## Methods References`, etc.).

    Skipped when `_rescue_aps_missing_refs_heading` already inserted a
    heading, which is checked via the same `_REFS_HEADING_FOR_SCORE_RE`.
    """
    if not md:
        return md
    if _REFS_HEADING_FOR_SCORE_RE.search(md):
        return md
    # 1. Bracketed.
    pos = _detect_generic_numbered_or_bracketed(
        md, _GENERIC_BRACKETED_REF_PAT, label="bracketed")
    if pos is not None:
        return md[:pos].rstrip() + "\n\n## References\n\n" + md[pos:]
    # 2. Numbered.
    pos = _detect_generic_numbered_or_bracketed(
        md, _GENERIC_NUMBERED_REF_PAT, label="numbered")
    if pos is not None:
        return md[:pos].rstrip() + "\n\n## References\n\n" + md[pos:]
    # 3. Author-year (strictest).
    pos = _detect_generic_author_year(md)
    if pos is not None:
        return md[:pos].rstrip() + "\n\n## References\n\n" + md[pos:]
    return md


_API_REFS_RESCUE_THRESHOLD = 0.5


def _rescue_append_api_refs(md: str,
                            report: "QualityReport",
                            *,
                            allow_network: bool = True) -> str:
    """Always-on fallback rescue: when the locally extracted reference
    section is poor or absent AND a DOI is known, fetch the paper's
    deposited reference list from Crossref (with OpenAlex fallback)
    and APPEND it as a new `## References (from <source>)` section at
    the end of the body. Never modifies existing content.

    Triggers when DOI is set AND any of:
      * score < _API_REFS_RESCUE_THRESHOLD
      * section_count == 0 (no References heading detected)
      * style == "numbered" AND not numbered_continuous AND
        entry_count >= 3 (visible gaps in a numbered list signal
        missing entries even when the composite score is decent --
        Wackerle1962 / Hicks2006-with-gaps pattern)

    Fail-safe: silent skip (returns md unchanged) on any of:
      * no DOI
      * network unreachable
      * APIs return no refs
      * response was empty / malformed

    Cache: results memoized by DOI on disk; offline reruns hit cache.
    """
    if report is None or getattr(report, "metadata", None) is None:
        return md
    doi = getattr(report.metadata, "doi", None)
    if not doi:
        return md
    if report.references is None:
        return md
    rs = report.references
    score = rs.score if rs.score is not None else 0.0
    numbered_gaps = (rs.style == "numbered"
                     and not rs.numbered_continuous
                     and rs.entry_count >= 3)
    healthy = (score >= _API_REFS_RESCUE_THRESHOLD
               and rs.section_count > 0
               and not numbered_gaps)
    if healthy:
        return md
    try:
        from metadata_frontend import fetch_references
        result = fetch_references(doi, allow_network=allow_network)
    except Exception as e:
        log.debug("_rescue_append_api_refs: fetch_references raised: %s", e)
        return md
    if not result or not result.get("refs"):
        # Trigger fired but fetch returned nothing. When the cause is
        # offline mode (allow_network=False AND no cache hit), record
        # the skip so the frontmatter consumer can distinguish
        # "skipped due to offline" from "fetched but empty".
        if not allow_network:
            rs.api_skipped_offline = True
        return md
    src = result.get("source") or "external"
    refs = result["refs"]
    heading_label = {"crossref": "Crossref", "openalex": "OpenAlex"} \
        .get(src, src.title() if src else "external")
    section = (f"\n\n## References (from {heading_label})\n\n"
               + "\n".join(refs) + "\n")
    rs.api_source = src
    rs.api_refs_appended = len(refs)
    log.info("Rescue (api-append): appended %d refs from %s for doi=%s",
             len(refs), src, doi)
    return md.rstrip() + section


def _rescue_aps_last_ref_bleed(md: str, doc: "fitz.Document",
                               report: "QualityReport",
                               out_dir: Optional[Path] = None) -> str:
    """Phase-2 deterministic rescue: trim Acknowledgments / Funding
    paragraph that bled into the last reference entry on APS papers
    (e.g. millot, tracy in the user's silica-shock corpus).

    No VLM. Idempotent on already-clean papers (the boundary marker
    is absent from the last entry; rescue is a no-op).

    Edge case: if the trimmed Acks content has no peer block earlier
    in the document, the trimmed text is promoted to an
    `## Acknowledgments` heading placed immediately before the refs
    heading so the content isn't lost.
    """
    sc, span = _detect_refs_section(md)
    if span is None:
        return md
    body_start, body_end = span
    body = md[body_start:body_end]
    style = _classify_ref_style(body)
    entries = _split_ref_entries(body, style)
    if not entries:
        return md
    last = entries[-1]
    bm = _ACKS_BOUNDARY_RE.search(last)
    if not bm:
        return md
    cut = bm.start()
    # Reject false positives: marker has to sit far enough into the
    # entry that the leading text is plausible reference content
    # (~20 chars covers "1. Author A, J 1 (2020)." minimum).
    if cut < 20:
        return md
    keep_entry = last[:cut].rstrip()
    drop_text = last[cut:].strip()
    if not keep_entry or not drop_text:
        return md

    # Rewrite the body so the last entry is the trimmed version.
    last_in_body = body.rfind(last)
    if last_in_body < 0:
        log.debug("_rescue_aps_last_ref_bleed: last-entry offset miss")
        return md
    new_body = (body[:last_in_body] + keep_entry + "\n"
                + body[last_in_body + len(last):])

    # Determine whether an Acks block already exists earlier.
    pre_refs = md[:body_start]
    has_acks_earlier = bool(re.search(
        r"^#+\s+(?:Acknowledg(?:e?)ments?|Funding|Author\s+Contributions?)",
        pre_refs, re.IGNORECASE | re.MULTILINE))

    # Find the heading line for the refs section so we can insert
    # before it. body_start sits at the END of the heading line; the
    # heading itself begins at the start of that line.
    # Accept optional bold-wrapped headings like
    # `## **REFERENCES AND NOTES**` (Science / NPG style) -- the
    # leading `\**\s*` allows 0+ asterisks plus optional whitespace
    # between the hash prefix and the title.
    heading_match = re.search(
        r"(?im)^#+\s+\**\s*(?:Notes\s+and\s+|Methods?\s+)?References?"
        r"(?:\s+(?:and|&)\s+Notes?)?|^#+\s+\**\s*Bibliography|"
        r"^#+\s+\**\s*Cited\s+References?",
        md[:body_start])
    if heading_match is None:
        heading_start = body_start
    else:
        heading_start = heading_match.start()

    if has_acks_earlier:
        new_md = (md[:body_start] + new_body + md[body_end:])
    else:
        injection = f"## Acknowledgments\n\n{drop_text}\n\n"
        new_md = (md[:heading_start]
                  + injection
                  + md[heading_start:body_start]
                  + new_body
                  + md[body_end:])

    log.info("Rescue (aps-bleed): trimmed %d chars from last reference "
             "(promoted_to_acks=%s)",
             len(drop_text), not has_acks_earlier)
    return new_md


def _journal_rescue_dispatch_table() -> dict:
    """Dispatch table from journal_slug -> rescue function. Built on
    demand because the rescue functions are defined later in this
    module; building lazily avoids forward-reference problems while
    keeping the registration explicit (every supported slug has its
    own line)."""
    # Dispatch values are TUPLES of rescue functions; the dispatcher
    # iterates and applies each in order so multiple per-slug rescues
    # compose (e.g. trim-bleed then recover-missing-refs). Single-
    # rescue slugs use a 1-tuple for shape uniformity.
    return {
        # ^^ DISABLED 2026-05-04. SUPERSEDED 2026-05-05.
        # The VLM-based 3-column reorder rescue has been REPLACED as
        # the canonical solution for old-Science (and similar 3-col
        # pre-2010 layouts) by the always-on Crossref/OpenAlex API
        # reference fallback (`_rescue_append_api_refs`). Validated
        # on the v0.3.1 corpus run: the API path produces a clean,
        # correctly-ordered reference list at zero VLM cost, and the
        # original mangled local section is preserved alongside it.
        # The VLM rescue is retained in the module as dead code in
        # case a future use-case wants in-place rewrite (the API
        # path appends rather than replaces); do not re-enable in
        # dispatch without a specific reason.
        # APS slugs get TWO rescues, applied in order:
        # 1. trim Acknowledgments bleed from the last entry
        # 2. recover refs that marker dropped via numbered-gap scan
        #    (Knudson2013 PRB pattern: short / unusual entries marker
        #    skipped on the second page of the refs list).
        "aps-prl": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps-prb": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps-prx": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps-pra": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps-pre": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps-prd": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps-prresearch": (_rescue_aps_last_ref_bleed,
                           _rescue_numbered_pageboundary_mash,
                           _rescue_missing_numbered_refs),
        "aps-rmp": (_rescue_aps_last_ref_bleed,
                    _rescue_numbered_pageboundary_mash,
                    _rescue_missing_numbered_refs),
        "aps": (_rescue_aps_last_ref_bleed,
                _rescue_numbered_pageboundary_mash,
                _rescue_missing_numbered_refs),
        # The bleed rescue is journal-agnostic -- it just trims text
        # that begins with Acknowledg(e?)ments? / Funding / Author
        # Contributions from the last reference entry. Adding Science
        # family slugs catches the alexander / millot / tracy bleed
        # pattern reported in collections/refs-score.txt.
        "science": (_rescue_aps_last_ref_bleed,),
        "science-advances": (_rescue_aps_last_ref_bleed,),
        # Author-year journals: re-split mega-entries (marker
        # accidentally joined adjacent refs onto one bullet). Pure
        # regex; gated on style=="author-year" + a mega-entry
        # signal so it no-ops on numbered-style or healthy papers.
        # Bleed-trim is also applicable when the author-year refs
        # have an Acks bleed at the end, so chain it before the
        # resplit.
        "agu": (_rescue_aps_last_ref_bleed,
                _rescue_author_year_resplit),
        "elsevier": (_rescue_aps_last_ref_bleed,
                     _rescue_author_year_resplit),
        "elsevier-icarus": (_rescue_aps_last_ref_bleed,
                            _rescue_author_year_resplit),
        "elsevier-epsl": (_rescue_aps_last_ref_bleed,
                          _rescue_author_year_resplit),
        "elsevier-gca": (_rescue_aps_last_ref_bleed,
                         _rescue_author_year_resplit),
        "elsevier-pepi": (_rescue_aps_last_ref_bleed,
                          _rescue_author_year_resplit),
        "elsevier-chemgeo": (_rescue_aps_last_ref_bleed,
                             _rescue_author_year_resplit),
        "copernicus": (_rescue_aps_last_ref_bleed,
                       _rescue_author_year_resplit),
        "royal-society": (_rescue_aps_last_ref_bleed,
                          _rescue_author_year_resplit),
    }


def rescue_journal_refs(md: str, doc: "fitz.Document",
                        report: "QualityReport",
                        use_vlm: bool,
                        out_dir: Optional[Path] = None) -> str:
    """Phase-2 dispatcher: route papers to a journal-specific rescue
    keyed by ReferencesScore.journal_slug.

    Universal gates (all must pass for any rescue):
      1. RESCUE_JOURNAL_REFS module toggle is on.
      2. report.references is populated AND has a journal_slug.
      3. journal_slug is in the dispatch table.

    Per-rescue gates:
      - aps-bleed (signal-driven, journal-agnostic): fires when
        longest_over_median > REF_RESCUE_LOR AND
        last_well_bounded == False. Overall score is NOT a gate --
        the bleed pattern can occur on otherwise-healthy refs
        sections (tracy at 0.80) so a score-threshold filter
        would silently skip real fixes.
      - science-3col (currently disabled in dispatch table): score
        below REF_RESCUE_THRESHOLD AND use_vlm AND layout.columns==3.

    Splicing rule: even after a rescue runs, the proposed output is
    spliced ONLY IF the post-rescue score strictly exceeds the
    pre-rescue score. A regression keeps the pre-rescue body and
    logs a warning. In dry-run mode the body is never spliced; both
    scores are recorded for review.
    """
    if not RESCUE_JOURNAL_REFS:
        return md
    if report.references is None:
        return md
    pre = report.references
    slug = pre.journal_slug
    if not slug:
        return md
    dispatch = _journal_rescue_dispatch_table()
    rescue_entry = dispatch.get(slug)
    if rescue_entry is None:
        return md
    # Normalize to a tuple of rescues so we can iterate uniformly.
    rescue_fns = (rescue_entry if isinstance(rescue_entry, tuple)
                  else (rescue_entry,))

    def _gate_for(rfn) -> tuple[bool, str]:
        """Return (should_run, rescue_name). Per-rescue trigger:
        signal-driven rescues use their own signals; score-gated
        rescues use the composite threshold."""
        if rfn is _rescue_science_3col:
            if pre.score >= REF_RESCUE_THRESHOLD:
                return False, "science-3col"
            if not use_vlm:
                return False, "science-3col"
            if pre.layout is None or pre.layout.columns != 3:
                return False, "science-3col"
            return True, "science-3col"
        if rfn is _rescue_aps_last_ref_bleed:
            if (pre.longest_over_median <= REF_RESCUE_LOR
                    or pre.last_well_bounded):
                return False, "aps-bleed"
            return True, "aps-bleed"
        if rfn is _rescue_missing_numbered_refs:
            # Signal-driven: fires only when there's actually a gap
            # in the numbered sequence (numbered_continuous == False)
            # AND the style is numbered (so a gap is meaningful).
            # Score-agnostic.
            if pre.style != "numbered":
                return False, "missing-refs"
            if pre.numbered_continuous:
                return False, "missing-refs"
            return True, "missing-refs"
        if rfn is _rescue_numbered_pageboundary_mash:
            # Signal-driven: fires when style is numbered. The
            # rescue itself checks whether a PRB-superscript mashed
            # bullet exists and is a no-op otherwise. Score-agnostic.
            if pre.style != "numbered":
                return False, "numbered-mash"
            return True, "numbered-mash"
        if rfn is _rescue_author_year_resplit:
            # Signal-driven: author-year style + a mega-entry signal
            # (one entry > 5x the median or longest_over_median > 5).
            # Score-agnostic. Restrictive so we don't false-positive
            # on healthy papers with one slightly long ref.
            if pre.style != "author-year":
                return False, "ay-resplit"
            if pre.longest_over_median <= 5.0:
                return False, "ay-resplit"
            return True, "ay-resplit"
        # Unknown rescue: default to score-gated.
        if pre.score >= REF_RESCUE_THRESHOLD:
            return False, "unknown"
        return True, "unknown"

    # Apply rescues in sequence. Each can either splice (improving
    # the score), be a no-op, or be skipped by its gate.
    # `dry_best_score` tracks the highest "would-have-spliced" score
    # any rescue produced in dry-run mode, so the manifest's
    # score_post_rescue reflects the proposed delta even when nothing
    # is actually spliced.
    cur_md = md
    cur_score = pre.score
    dry_best_score: Optional[float] = None
    applied: list[tuple[str, float, float, str]] = []  # (name, pre, post, decision)
    for rfn in rescue_fns:
        ok, rescue_name = _gate_for(rfn)
        if not ok:
            continue
        log.info("Rescue dispatch: slug=%s, rescue=%s, pre_score=%.2f, "
                 "threshold=%.2f, dry_run=%s",
                 slug, rescue_name, cur_score, REF_RESCUE_THRESHOLD,
                 REF_RESCUE_DRY_RUN)
        try:
            proposed = rfn(cur_md, doc, report, out_dir=out_dir)
        except Exception as e:
            log.warning("Rescue %s raised (%s); pre-rescue kept",
                        rescue_name, e)
            applied.append((rescue_name, cur_score, cur_score,
                            "kept-pre-rescue-error"))
            continue
        if proposed == cur_md:
            log.info("Rescue %s: no change", rescue_name)
            applied.append((rescue_name, cur_score, cur_score,
                            "no-change"))
            continue
        post_score_obj = score_references(proposed, slug)
        if REF_RESCUE_DRY_RUN:
            log.info("Rescue %s (dry-run): score %.2f -> %.2f (no splice)",
                     rescue_name, cur_score, post_score_obj.score)
            applied.append((rescue_name, cur_score, post_score_obj.score,
                            "dry-run"))
            if (dry_best_score is None
                    or post_score_obj.score > dry_best_score):
                dry_best_score = post_score_obj.score
            # In dry-run, don't propagate the change to subsequent
            # rescues either -- they should each see the pre-rescue body.
            continue
        if post_score_obj.score >= cur_score:
            # Accept splices that maintain or improve the score.
            # Rescues whose effect ISN'T captured by the score (e.g.,
            # the numbered-mash split — score is blind to mashed
            # bullets that don't match _NUMBERED_ENTRY_RE) would
            # otherwise be silently rejected here. A regression
            # (post < pre) is still rejected as a safety net.
            log.info("Rescue %s: score %.2f -> %.2f -> spliced",
                     rescue_name, cur_score, post_score_obj.score)
            applied.append((rescue_name, cur_score, post_score_obj.score,
                            "spliced"))
            cur_md = proposed
            cur_score = post_score_obj.score
        else:
            log.warning("Rescue %s regressed score %.2f -> %.2f; "
                        "pre-rescue kept", rescue_name, cur_score,
                        post_score_obj.score)
            applied.append((rescue_name, cur_score, post_score_obj.score,
                            "kept-pre-rescue-regressed"))

    # Record bookkeeping. Only count rescues that actually did
    # something: a splice, a dry-run that would have helped, or an
    # error that's worth recording. No-op rescues (the rescue ran
    # but its internal gates declined to act) don't show up in
    # rescue_applied -- they're noise for the manifest.
    interesting = [a for a in applied
                   if a[3] not in ("no-change",)]
    if interesting:
        names = ",".join(a[0] for a in interesting)
        pre.rescue_applied = names
        pre.score_pre_rescue = pre.score
        # In dry-run mode, surface the BEST proposed score so the
        # manifest reflects what the rescue would have achieved.
        # Otherwise surface the actual current score (post-splices).
        pre.score_post_rescue = (dry_best_score
                                 if REF_RESCUE_DRY_RUN
                                 and dry_best_score is not None
                                 else cur_score)
        decisions = [a[3] for a in interesting]
        if "spliced" in decisions:
            pre.rescue_decision = "spliced"
        elif "dry-run" in decisions:
            pre.rescue_decision = "dry-run"
        else:
            pre.rescue_decision = decisions[-1]
    return cur_md


SCIENCE_3COL_REORDER_PROMPT = (
    "You are given fragments of a numbered reference list extracted "
    "from a 3-column scientific paper. The marker pipeline mis-ordered "
    "them because the columns on the reference page were read out of "
    "sequence.\n\n"
    "Your ONLY task is to reassemble the fragments into a single "
    "continuous list ordered 1, 2, 3, ... N.\n\n"
    "STRICT RULES:\n"
    "- Do NOT rewrite, paraphrase, translate, or 'fix' any text.\n"
    "- Do NOT change punctuation, italics, bold, DOI links, or "
    "author names.\n"
    "- Do NOT merge fragments that don't belong together.\n"
    "- If a fragment has no leading reference number, place it "
    "immediately after the entry whose continuation it appears to be.\n"
    "- If you cannot place a fragment with confidence, output it at "
    "the end under `## Unplaced fragments` rather than guessing.\n\n"
    "Output: one entry per line, prefixed by `- `. No commentary, "
    "no preamble.\n\n"
    "INPUT FRAGMENTS:\n"
    "{fragments}\n"
)


def _rescue_science_3col(md: str, doc: "fitz.Document",
                         report: "QualityReport",
                         out_dir: Optional[Path] = None) -> str:
    """Phase-2 Science 3-column VLM rescue. Renders the references
    page and asks the VLM to reorder the existing fragments into 1..N
    sequence. Output is verified to contain only characters present
    in the input (allowing for the synthetic '## Unplaced fragments'
    heading); failure to verify means the pre-rescue body is kept.

    When `out_dir` is provided AND the verifier rejects the VLM output,
    the raw VLM response is dumped to
    `<out_dir>/rescue_science_3col_rejected.md` so the rejection can
    be diffed against the pre-rescue body for diagnosis.

    Returns the proposed rewritten markdown -- the dispatcher decides
    whether to splice based on score delta. Returns md unchanged when
    any gate fails (no refs section, no page, VLM error, verifier
    rejects, etc.)."""
    sc, span = _detect_refs_section(md)
    if span is None:
        log.debug("science-3col: no refs section detected")
        return md
    body_start, body_end = span
    body = md[body_start:body_end].strip()
    if not body:
        return md

    page_idx = _refs_page_idx_from_md(md, doc)
    if page_idx is None:
        log.debug("science-3col: refs page not located")
        return md

    # Lower DPI than the default PAGE_DPI=170: the VLM only needs the
    # column layout to decide ordering, not glyph-level detail (the
    # text fragments are passed in the prompt). 110 dpi cuts pixel
    # count ~2.4x vs 170, reducing prefill cost and the 8+ minute
    # wall times observed on the cuk / young pages.
    try:
        page_img = render_page(doc, page_idx, dpi=110)
    except Exception as e:
        log.warning("science-3col: render_page failed (%s); skipping", e)
        return md

    # max_tokens: 26 entries x ~250 chars each ~= 6500 chars ~= 2200
    # tokens. 3000 gives ~30% headroom for the largest expected refs
    # sections while capping wall-clock time at ~1 min on a healthy
    # vLLM. The earlier 6000 setting let the model wander into
    # generation loops on rejected outputs (8+ min observed).
    prompt = SCIENCE_3COL_REORDER_PROMPT.format(fragments=body)
    # Cap the per-call wait at 180s and disable the SDK's default
    # 2-retry policy: a single timeout shouldn't fan out into
    # 3x180s = 9min of wasted wall time per paper. On healthy vLLM
    # this is plenty for a 3000-token output; on a hang we fail
    # definitively at 180s and keep the pre-rescue body.
    try:
        result = vlm(prompt, page_img, max_tokens=3000,
                     timeout=180.0, max_retries=0)
    except Exception as e:
        log.warning("science-3col: VLM call raised (%s); skipping", e)
        return md
    if not result or result.strip().upper() in ("SKIP", "NO_TABLE_HERE"):
        log.info("science-3col: VLM returned empty/SKIP; pre-rescue kept")
        return md

    if not _verify_content_preserved(body, result):
        log.warning("science-3col: VLM output failed content-preserved "
                    "verifier (likely hallucinated author / journal / "
                    "place name); pre-rescue kept")
        if out_dir is not None:
            try:
                dump_path = out_dir / "rescue_science_3col_rejected.md"
                dump_path.write_text(result)
                log.info("science-3col: dumped rejected VLM output to %s",
                         dump_path)
            except Exception as e:
                log.debug("science-3col: failed to dump rejected output: %s", e)
        return md

    # Splice: replace the body of the refs section with the VLM output.
    # Preserve a leading and trailing newline so the surrounding
    # heading + post-section flow remain well-formed.
    new_body = "\n" + result.rstrip() + "\n"
    new_md = md[:body_start] + new_body + md[body_end:]
    return new_md
    """Confirm the VLM only reordered text, didn't rewrite or invent.

    Every non-whitespace character in `output_text` must appear (in
    at least the same count) in `input_text`. Uses character multiset
    comparison so reordering at any granularity (lines, paragraphs,
    fragments) passes, while hallucinated names / DOIs / digits fail.

    A synthetic "## Unplaced fragments" heading is the only
    permitted output text not present in the input; it's stripped
    before counting.
    """
    from collections import Counter as _Counter
    # Strip the explicitly permitted synthetic heading.
    cleaned_out = re.sub(r"##\s*Unplaced\s+fragments\s*", "", output_text,
                         flags=re.IGNORECASE)
    in_chars = _Counter(re.sub(r"\s+", "", input_text))
    out_chars = _Counter(re.sub(r"\s+", "", cleaned_out))
    extras = out_chars - in_chars
    return not extras


# Patterns used by score_references(). Defined lazily inside the
# function so they don't add to the module-level namespace; cached
# once via re.compile.
_REFS_HEADING_FOR_SCORE_RE = re.compile(
    r"(?im)^(?P<hashes>#{1,4})\s*(?:\*{0,2}\s*)?"
    r"(?P<title>(?:Notes\s+and\s+|Methods?\s+)?References?"
    r"(?:\s+(?:and|&)\s+Notes?)?|Bibliography|Cited\s+References?)"
    r"\s*(?:\*{0,2}\s*)?[\.:]?\s*$"
)
_NUMBERED_ENTRY_RE = re.compile(
    r"^\s*[-*]?\s*(?:<span\s+id=\"[^\"]*\"></span>\s*)*"  # multiple anchors OK
    r"(?:(?P<num1>\d{1,3})\.\s+\S"           # "1. Author..."
    r"|\[(?P<num2>\d{1,3})\]\s*\S)",         # "[1] Author..." (APS PRL/PRB)
    re.MULTILINE,
)
_AUTHOR_YEAR_ENTRY_RE = re.compile(
    r"^\s*[-*]\s+(?:<span\s+id=\"[^\"]*\"></span>\s*)*"   # multiple anchors OK
    r"[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)*,\s+(?:[A-Z][a-z]?\.|[A-Z][a-z]+)",
    re.MULTILINE,
)
_SUP_ENTRY_RE = re.compile(r"<sup>\s*\d{1,3}\s*</sup>")
_REF_YEAR_FOR_SCORE_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")


def _detect_refs_section(md: str) -> tuple[int, Optional[tuple[int, int]]]:
    """Return `(section_count, span_of_canonical_section)`.

    `section_count` is the number of `## References` (or alias) headings
    found anywhere in the document. The canonical section is the LAST one
    -- after `merge_reference_sections` has run, the merged consolidated
    body lives at the end. Span is `(body_start, body_end)` excluding the
    heading line itself; `body_end` is the next equal-or-shallower heading,
    or `len(md)`.

    When no heading exists at all, falls back to detecting a trailing
    bulleted/numbered cluster (APS PRL/PRB pattern: refs land at the
    end with no `## References` heading). The fallback span starts at
    the first matching cluster line and runs to end-of-document.
    section_count stays 0 so the scorer can still penalize the missing
    heading via its weight.
    """
    matches = list(_REFS_HEADING_FOR_SCORE_RE.finditer(md))
    if matches:
        last = matches[-1]
        body_start = last.end()
        depth = len(last.group("hashes"))
        next_pat = re.compile(rf"(?m)^#{{1,{depth}}}\s+\S")
        nxt = next_pat.search(md, pos=body_start + 1)
        body_end = nxt.start() if nxt else len(md)
        return len(matches), (body_start, body_end)

    # No-heading fallback: scan for a contiguous numbered cluster
    # near the end of the document. Reuses _NUMBERED_ENTRY_RE which
    # accepts both "1." and "[1]" styles. Requires >= 5 contiguous
    # entry-like lines (with up to 2 blank-line gaps tolerated) so a
    # short numbered list elsewhere in the body doesn't false-trigger.
    lines = md.split("\n")
    line_offsets = [0]
    for ln in lines:
        line_offsets.append(line_offsets[-1] + len(ln) + 1)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if not _NUMBERED_ENTRY_RE.match(lines[i]):
            i += 1
            continue
        run_start = i
        last_hit = i
        j = i + 1
        while j < len(lines):
            if _NUMBERED_ENTRY_RE.match(lines[j]):
                last_hit = j
                j += 1
                continue
            if lines[j].strip() == "":
                blanks = 0
                while j < len(lines) and lines[j].strip() == "":
                    blanks += 1
                    j += 1
                if blanks > 2:
                    break
                continue
            break
        if (last_hit - run_start + 1) >= 5:
            runs.append((run_start, last_hit + 1))
        i = max(j, last_hit + 1)
    if not runs:
        return 0, None
    # Pick the LAST cluster (refs land at end of document).
    rs, re_ = runs[-1]
    return 0, (line_offsets[rs], line_offsets[re_])


def _classify_ref_style(body: str) -> str:
    """Detect the entry style inside the references body. Order matters:
    numbered before author-year, since numbered entries can also start
    with a capitalized author name."""
    if _NUMBERED_ENTRY_RE.search(body):
        return "numbered"
    if _SUP_ENTRY_RE.search(body):
        return "superscript"
    if _AUTHOR_YEAR_ENTRY_RE.search(body):
        return "author-year"
    if "<sup>" in body and re.search(r"<sup>\d", body):
        return "footnote"
    return "unknown"


def _split_ref_entries(body: str, style: str) -> list[str]:
    """Return the list of entries inside `body`, split by style. Falls
    back to blank-line-separated paragraphs for unknown styles."""
    if style == "numbered":
        boundaries = [m.start() for m in _NUMBERED_ENTRY_RE.finditer(body)]
        if not boundaries:
            return []
        boundaries.append(len(body))
        return [body[boundaries[i]:boundaries[i + 1]].strip()
                for i in range(len(boundaries) - 1)
                if body[boundaries[i]:boundaries[i + 1]].strip()]
    if style == "author-year":
        boundaries = [m.start() for m in _AUTHOR_YEAR_ENTRY_RE.finditer(body)]
        if not boundaries:
            return []
        boundaries.append(len(body))
        return [body[boundaries[i]:boundaries[i + 1]].strip()
                for i in range(len(boundaries) - 1)
                if body[boundaries[i]:boundaries[i + 1]].strip()]
    # Fallback: paragraphs separated by blank lines.
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def _numbered_continuous(entries: list[str]) -> bool:
    """True iff the leading numbers on each entry form 1..N with no gap.
    Returns False on missing numbers, gaps, duplicates, or fewer than
    two entries (a single entry can't establish continuity)."""
    if len(entries) < 2:
        return False
    nums: list[int] = []
    for e in entries:
        m = _NUMBERED_ENTRY_RE.match(e)
        if not m:
            return False
        raw = m.group("num1") or m.group("num2")
        try:
            nums.append(int(raw))
        except (TypeError, ValueError):
            return False
    return nums == list(range(nums[0], nums[0] + len(nums))) and nums[0] in (0, 1)


def score_references(md: str,
                     journal_slug: Optional[str]) -> ReferencesScore:
    """Pure heuristic scorer for the final References section.

    Stable across re-runs (no VLM, no network, no nondeterminism).
    Surfaces six signals that rank-correlate with the user's manual
    annotations in collections/refs-score.txt: section count, entry
    count, numbering continuity, plausibility of year hits, entry-
    length consistency, and a soft "last entry doesn't run into
    Acknowledgments" check. Weighted composite returned in `score`."""
    section_count, span = _detect_refs_section(md)
    if span is None:
        return ReferencesScore(
            section_count=0, entry_count=0, numbered_continuous=False,
            year_hit_ratio=0.0, longest_over_median=0.0,
            style="unknown", journal_slug=journal_slug, score=0.0,
            last_well_bounded=True,
        )

    body = md[span[0]:span[1]]
    style = _classify_ref_style(body)
    entries = _split_ref_entries(body, style)
    entry_count = len(entries)

    if entry_count == 0:
        return ReferencesScore(
            section_count=section_count, entry_count=0,
            numbered_continuous=False, year_hit_ratio=0.0,
            longest_over_median=0.0, style=style,
            journal_slug=journal_slug, score=0.0,
            last_well_bounded=True,
        )

    year_hits = sum(1 for e in entries if _REF_YEAR_FOR_SCORE_RE.search(e))
    year_hit_ratio = year_hits / entry_count

    lengths = sorted(len(e) for e in entries)
    median = lengths[len(lengths) // 2] or 1
    longest = lengths[-1]
    lor = longest / median

    numbered_continuous = (style == "numbered"
                           and _numbered_continuous(entries))

    # Last-entry boundary heuristic: the final entry shouldn't end with
    # body-text that looks like the start of an Acknowledgments
    # paragraph. The boundary failure mode (millot, tracy) leaves
    # "...references continued. Acknowledgments. We thank ..." in the
    # final entry. Cheap probe: look for the literal word in the last
    # entry text.
    last_entry = entries[-1]
    last_well_bounded = not re.search(
        r"\b(?:Acknowledg(?:e?ments?|ments)|Funding|Author "
        r"contributions?)\b", last_entry, re.IGNORECASE)

    score = 0.0
    if section_count == 1:
        score += 0.20
    elif section_count == 0:
        score += 0.0
    else:  # >=2 means merge_reference_sections didn't fire / failed
        score += 0.05
    if entry_count >= 3:
        score += 0.10
    elif entry_count <= 1:
        # Single (or zero) entries usually means refs were run together
        # into one mega-paragraph; clearly broken regardless of other
        # signals. Penalize.
        score -= 0.20
    if style == "numbered" and numbered_continuous:
        score += 0.30
    elif style in ("author-year", "superscript"):
        score += 0.20  # numbering continuity not applicable
    if year_hit_ratio >= 0.5:
        score += 0.20
    elif year_hit_ratio >= 0.25:
        score += 0.10
    else:
        # Real reference lists almost always carry publication years;
        # zero-or-near-zero year hits is a strong "garbled / truncated"
        # signal (Elliott / young / lyzenga 1980 cluster).
        score -= 0.20
    if lor <= 5.0:
        score += 0.10
    elif lor <= 10.0:
        score += 0.05
    if last_well_bounded:
        score += 0.10
    score = max(0.0, min(1.0, score))

    return ReferencesScore(
        section_count=section_count,
        entry_count=entry_count,
        numbered_continuous=numbered_continuous,
        year_hit_ratio=round(year_hit_ratio, 3),
        longest_over_median=round(lor, 2),
        style=style,
        journal_slug=journal_slug,
        score=round(score, 3),
        last_well_bounded=last_well_bounded,
    )


def _mean(vals) -> float:
    xs = list(vals)
    return sum(xs) / len(xs) if xs else 0.0


def _yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _yaml_bool(b: bool) -> str:
    return "true" if b else "false"


# Packages whose versions get recorded in run.packages. Chosen for
# reproducibility. Split into three sets so the recorded list reflects
# what actually ran:
#   - SHARED: torch / transformers (drive ML stack regardless of layout
#     source); pymupdf / pillow (crop + PDF reading on every run); SDK
#     packages (VLM clients); h5py / python-dotenv (bundle + env).
#   - MARKER: marker-pdf / surya-ocr / docling. Loaded only when
#     layout_source == "marker"; load-bearing pins for marker's MPS
#     behaviour and the docling table finder.
#   - MINERU: mineru / mineru-vl-utils. Loaded only when
#     layout_source == "mineru"; pinned for the MinerU subprocess and
#     the optional VLM extras.
# Omits transitive deps; `pip freeze` is the right tool for full closure.
_RECORDED_PACKAGES_SHARED = (
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
_RECORDED_PACKAGES_MARKER = (
    "marker-pdf",
    "surya-ocr",
    "docling",
)
_RECORDED_PACKAGES_MINERU = (
    "mineru",
    "mineru-vl-utils",
)


def _collect_package_versions(layout_source: str = "mineru") -> dict:
    """Best-effort: read installed versions of the load-bearing packages.

    `layout_source` selects which engine-specific packages to include:
    'mineru' adds mineru + mineru-vl-utils; 'marker' adds marker-pdf +
    surya-ocr + docling. Shared packages are recorded either way.

    Skips any package not present in the environment (e.g. anthropic
    when --provider isn't anthropic and the SDK isn't installed).
    """
    from importlib.metadata import PackageNotFoundError, version
    if layout_source == "mineru":
        names = _RECORDED_PACKAGES_SHARED + _RECORDED_PACKAGES_MINERU
    elif layout_source == "marker":
        names = _RECORDED_PACKAGES_SHARED + _RECORDED_PACKAGES_MARKER
    else:
        # 'hybrid' (or any unknown) records both engines: marker
        # produces body, mineru produces figure/table layout.
        names = (_RECORDED_PACKAGES_SHARED + _RECORDED_PACKAGES_MARKER
                 + _RECORDED_PACKAGES_MINERU)
    out: dict = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _collect_pipeline_state(table_finder: str,
                            vlm_rewrite_tables: bool,
                            rescue_sparse_pages: bool,
                            force_ocr: bool,
                            layout_source: str = "mineru",
                            rescue_mineru_orphan_captions: bool = True,
                            rescue_mineru_subpanels: bool = True) -> dict:
    """Snapshot of the deterministic pipeline toggles + active strategy
    knobs. Records the state at the moment convert() was called so the
    YAML can be replayed by anyone reading the front-matter.

    Marker-only fields (table_finder, force_ocr, figmatch_strategy,
    rotate_tables / table_rotation_*, rescue_orphan_tables,
    rescue_orphan_figures) are omitted when layout_source == "mineru"
    because none of those code paths run. Mineru-only fields
    (rescue_mineru_orphan_captions) are omitted when layout_source ==
    "marker". Body-text and refs hooks apply to both backends and are
    always recorded.
    """
    is_mineru = (layout_source == "mineru")
    is_hybrid = (layout_source == "hybrid")
    state: dict = {
        "layout_source": layout_source,
        # vlm_rewrite_tables + rescue_sparse_pages apply to both
        # backends (mineru's emit_markdown also has a VLM-table
        # rewrite path; sparse-page rescue is body-level).
        "vlm_rewrite_tables": vlm_rewrite_tables,
        "rescue_sparse_pages": rescue_sparse_pages,
        # Body-text hooks apply regardless of layout source.
        "trim_articles": TRIM_ARTICLES,
        "fix_references": FIX_REFERENCES,
        "consolidate_footnotes": CONSOLIDATE_FOOTNOTES,
        "normalise_refs": NORMALISE_REFS,
        "strip_running_footers": STRIP_RUNNING_FOOTERS,
        "strip_page_headers": STRIP_PAGE_HEADERS,
        "strip_publisher_stamps": STRIP_PUBLISHER_STAMPS,
        "strip_math_labels": STRIP_MATH_LABELS,
        "merge_references": MERGE_REFERENCES,
        "inject_orphan_refs": INJECT_ORPHAN_REFS,
        "tidy_refs": TIDY_REFS,
        "rescue_journal_refs": RESCUE_JOURNAL_REFS,
        "ref_rescue_threshold": REF_RESCUE_THRESHOLD,
        "ref_rescue_lor": REF_RESCUE_LOR,
        "ref_rescue_dry_run": REF_RESCUE_DRY_RUN,
        "data_repos": DATA_REPOS,
        "fetch_data_repos": FETCH_DATA_REPOS,
        "offline": OFFLINE,
    }
    if is_hybrid:
        # Hybrid runs marker (body+caption text) AND mineru (figure
        # assets + table HTML); the splitting is captured in two new
        # keys so downstream filters can identify hybrid output even
        # without parsing the layout_source field.
        state["figure_layout_source"] = "mineru"
        state["caption_text_source"] = "marker"
        state["force_ocr"] = force_ocr
        state["rescue_mineru_orphan_captions"] = rescue_mineru_orphan_captions
        state["rescue_mineru_subpanels"] = rescue_mineru_subpanels
    elif is_mineru:
        state["rescue_mineru_orphan_captions"] = rescue_mineru_orphan_captions
        state["rescue_mineru_subpanels"] = rescue_mineru_subpanels
    else:
        # Marker path: docling table finder + caption_figures + the
        # marker-side orphan rescues + rotation hooks all run.
        state.update({
            "table_finder": table_finder,
            "force_ocr": force_ocr,
            "figmatch_strategy": FIGMATCH_STRATEGY,
            "rotate_tables": ROTATE_TABLES,
            "table_rotation_aspect_threshold": TABLE_ROTATION_ASPECT_THRESHOLD,
            "table_rotation_direction": TABLE_ROTATION_DIRECTION,
            "rescue_orphan_tables": RESCUE_ORPHAN_TABLES,
            "rescue_orphan_figures": RESCUE_ORPHAN_FIGURES,
        })
    return state


@dataclass
class UserAnnotations:
    """Optional user-curated metadata about the paper. Emitted as a
    peer `user:` YAML block alongside `copyright:` / `run:` etc. None
    of these fields are derived from the PDF; they come from CLI
    flags (--user / --collection / --note) or the matching env vars
    (PAPER2MD_USER, PAPER2MD_COLLECTION). The `note` field is free-
    form prose; multi-line notes survive YAML round-trip via
    block-scalar emission."""
    user: Optional[str] = None
    collection: Optional[str] = None
    note: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.user or self.collection or self.note)

    def to_yaml_lines(self) -> list[str]:
        L: list[str] = ["user:"]
        if self.user:
            L.append(f"  user: {_yaml_str(self.user)}")
        if self.collection:
            L.append(f"  collection: {_yaml_str(self.collection)}")
        if self.note:
            # Multi-line notes get block-scalar form so newlines
            # survive YAML parse without escape-soup; single-line
            # notes use the inline string for compactness.
            if "\n" in self.note:
                L.append("  note: |")
                for ln in self.note.splitlines():
                    L.append(f"    {ln}")
            else:
                L.append(f"  note: {_yaml_str(self.note)}")
        return L

    def to_dict(self) -> dict:
        return {
            "user": self.user,
            "collection": self.collection,
            "note": self.note,
        }


@dataclass
class EditsAnnotation:
    """User-authored marker that the markdown artifact has been
    hand-corrected after extraction. Emitted as a peer `edits:` YAML
    block alongside `user:` / `run:` / `copyright:` / `quality:`.

    There are no CLI flags that set these fields today: the user
    hand-writes the block in the .md frontmatter after editing the
    body, e.g.:

        edits:
          edited: true
          note: "Manually inserted refs 33-34 from PDF text layer;
                 rewrote Figure 3 caption."

    Downstream consumers (vector embedder, MCP) can read this block
    to discriminate human-verified content from auto-extracted
    content -- useful for retrieval ranking and provenance.

    CAVEAT (re-run round-trip): if you re-run paper2md on a PDF
    whose output already has an `edits:` block, the new run will
    OVERWRITE the .md file and the block will be lost. Until a
    preservation pass is added, the workflow is: edit, then don't
    re-run; or re-run, then re-add the block. The pipeline does NOT
    populate this field automatically.
    """
    edited: bool = False
    note: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.edited and not self.note

    def to_yaml_lines(self) -> list[str]:
        if self.is_empty():
            return []
        L: list[str] = ["edits:"]
        L.append(f"  edited: {_yaml_bool(self.edited)}")
        if self.note:
            if "\n" in self.note:
                L.append("  note: |")
                for ln in self.note.splitlines():
                    L.append(f"    {ln}")
            else:
                L.append(f"  note: {_yaml_str(self.note)}")
        return L

    def to_dict(self) -> dict:
        return {
            "edited": self.edited,
            "note": self.note,
        }


@dataclass
class RunInfo:
    """Reproducibility metadata for a single conversion run. Emitted as a
    peer `run:` YAML block alongside `quality:` and `copyright:`.

    `started_at` is an ISO 8601 UTC timestamp set at the top of convert();
    `elapsed_sec` is the wall-clock duration of that same convert() call
    (marker + hooks + scoring + frontmatter emit). The metadata pre-pass
    and any HDF5 bundle write that wrap convert() are excluded from
    elapsed_sec -- both are typically sub-second for single PDFs.

    paper2md_version, python_version, compute_backend, layout_source,
    text_engine, vlm_endpoint, pipeline, and packages are static for
    the whole CLI invocation; they're set once on the template in
    main() and inherited by every per-paper RunInfo via dataclass
    replace()."""
    command: str           # full CLI invocation (sys.argv joined + shell-quoted)
    hostname: str          # socket.gethostname() at run start
    vlm_provider: str      # resolved provider after auto-pairing with backend
    vlm_model: str         # resolved $VLM_MODEL after configure_client
    paper2md_version: str = __version__
    paper2md_license: str = __license__
    paper2md_doi: str = __doi__   # "10.5281/zenodo.RESERVED" if not yet wired
    python_version: str = ""    # populated in main()
    compute_backend: str = ""   # cuda | mps | cpu (was: 'backend')
    layout_source: str = ""     # mineru | marker (top-level mirror of pipeline.layout_source)
    text_engine: str = ""       # mineru-paddleocr | marker-surya | (future) mineru-vlm | marker-surya-hybrid
    vlm_endpoint: str = ""      # base URL of the VLM API in use
    # AI / VLM disclosure fields for publication reproducibility. paper2md
    # does NOT override sampling parameters (temperature, top_p, seed) --
    # those default to whatever the server/provider sets. The system
    # prompt and inference-params note are RECORDED so a downstream user
    # has the full picture of what paper2md did vs what the server did.
    # See USAGE.md §18 for the full disclosure template.
    vlm_system_prompt: str = ""    # SYSTEM constant text used on every call
    vlm_inference_params: str = ""  # human-readable note about who controls sampling
    vlm_temperature: Optional[float] = None  # sampling temp sent on every call
    vlm_seed: Optional[int] = None  # sampling seed sent on every call (None = server picks)
    started_at: str = ""        # set inside convert() right after t0 = time.time()
    pipeline: dict = field(default_factory=dict)
    packages: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0  # set by convert() right before to_frontmatter

    def to_yaml_lines(self) -> list[str]:
        L = [
            "run:",
            f"  paper2md_version: {self.paper2md_version}",
            f"  paper2md_license: {self.paper2md_license}",
        ]
        if self.paper2md_doi and not self.paper2md_doi.endswith("RESERVED"):
            L.append(f"  paper2md_doi: {_yaml_str(self.paper2md_doi)}")
        if self.python_version:
            L.append(f"  python_version: {_yaml_str(self.python_version)}")
        L += [
            f"  command: {_yaml_str(self.command)}",
            f"  hostname: {_yaml_str(self.hostname)}",
        ]
        if self.started_at:
            L.append(f"  started_at: {self.started_at}")
        L.append(f"  elapsed_sec: {self.elapsed_sec:.1f}")
        if self.compute_backend:
            L.append(f"  compute_backend: {self.compute_backend}")
        if self.layout_source:
            L.append(f"  layout_source: {self.layout_source}")
        if self.text_engine:
            L.append(f"  text_engine: {_yaml_str(self.text_engine)}")
        L += [
            f"  vlm_provider: {self.vlm_provider}",
            f"  vlm_model: {_yaml_str(self.vlm_model)}",
        ]
        if self.vlm_endpoint:
            L.append(f"  vlm_endpoint: {_yaml_str(self.vlm_endpoint)}")
        if self.vlm_system_prompt:
            L.append(f"  vlm_system_prompt: {_yaml_str(self.vlm_system_prompt)}")
        if self.vlm_inference_params:
            L.append(
                f"  vlm_inference_params: "
                f"{_yaml_str(self.vlm_inference_params)}")
        if self.vlm_temperature is not None:
            L.append(f"  vlm_temperature: {self.vlm_temperature}")
        if self.vlm_seed is not None:
            L.append(f"  vlm_seed: {self.vlm_seed}")
        if self.pipeline:
            L.append("  pipeline:")
            for k, v in self.pipeline.items():
                if isinstance(v, bool):
                    L.append(f"    {k}: {_yaml_bool(v)}")
                elif isinstance(v, dict):
                    # Nested dict (triage_signals, scan_signals) emits
                    # as a YAML mapping rather than Python repr.
                    L.append(f"    {k}:")
                    for kk, vv in v.items():
                        if isinstance(vv, bool):
                            L.append(f"      {kk}: {_yaml_bool(vv)}")
                        elif isinstance(vv, str):
                            L.append(f"      {kk}: {_yaml_str(vv)}")
                        else:
                            L.append(f"      {kk}: {vv}")
                else:
                    L.append(f"    {k}: {_yaml_str(v) if isinstance(v, str) else v}")
        if self.packages:
            L.append("  packages:")
            for name in sorted(self.packages):
                L.append(f"    {name}: {_yaml_str(self.packages[name])}")
        return L

    def to_dict(self) -> dict:
        out = {
            "paper2md_version": self.paper2md_version,
            "paper2md_license": self.paper2md_license,
        }
        if self.paper2md_doi and not self.paper2md_doi.endswith("RESERVED"):
            out["paper2md_doi"] = self.paper2md_doi
        out.update({
            "python_version": self.python_version,
            "command": self.command,
            "hostname": self.hostname,
            "started_at": self.started_at,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "compute_backend": self.compute_backend,
            "layout_source": self.layout_source,
            "text_engine": self.text_engine,
            "vlm_provider": self.vlm_provider,
            "vlm_model": self.vlm_model,
            "vlm_endpoint": self.vlm_endpoint,
            "vlm_system_prompt": self.vlm_system_prompt,
            "vlm_inference_params": self.vlm_inference_params,
            "vlm_temperature": self.vlm_temperature,
            "vlm_seed": self.vlm_seed,
            "pipeline": dict(self.pipeline),
            "packages": dict(self.packages),
        })
        return out


@dataclass
class QualityReport:
    tables: list[TableScore] = field(default_factory=list)
    figures: list[FigureScore] = field(default_factory=list)
    pages: list[PageScore] = field(default_factory=list)
    vlm_enabled: bool = True
    # Optional copyright/OA metadata; populated by metadata_frontend.resolve()
    # in the pre-pass. Emitted as a peer 'copyright:' YAML block.
    metadata: object = None
    # Optional run/reproducibility info; populated in convert() right before
    # frontmatter emission. Emitted as a peer 'run:' YAML block.
    run_info: Optional[RunInfo] = None
    # Optional data-repository links extracted from the body. Populated in
    # convert() / convert_text_only() right before frontmatter emission.
    # Emitted as a peer 'data:' YAML block.
    data_repos: list = field(default_factory=list)
    # Optional user-curated annotations: --user / --collection / --note.
    # Emitted as a peer 'user:' YAML block (between copyright: and run:)
    # whenever any of the three fields is non-empty.
    user_annotations: Optional["UserAnnotations"] = None
    # Optional user-authored marker that the artifact has been hand-
    # corrected. Emitted as a peer `edits:` YAML block when set.
    # Pipeline never populates this -- the user writes it by hand in
    # the .md frontmatter after editing the body.
    edits: Optional["EditsAnnotation"] = None
    # Optional references-section quality score (Phase 1 instrumentation).
    # Surfaced in frontmatter / .meta.json / manifest but NOT included in
    # overall() until the signal is calibrated.
    references: Optional["ReferencesScore"] = None
    # Per-paper rescue stats (which auto-rescues fired with what effect).
    # Populated by rescue_orphan_captions / rescue_subpanel_groups under
    # --layout-source=mineru; emitted as a peer 'rescues:' YAML block so
    # users can audit what changed.  Each entry is a dict from the
    # rescue's name to its stats sub-dict.
    rescues: dict = field(default_factory=dict)

    def overall(self) -> float:
        """Weighted composite of the per-asset quality buckets.

        Only tables (weight 0.3) and figures (weight 0.2) contribute.
        Page-level scores were dropped from this composite as of
        2026-05: they were derived from char-density alone, which
        is a content-type proxy (figure-heavy vs text-heavy) rather
        than an extraction-quality signal, and were dragging
        figure-heavy documents (notably Sci-Adv-style supplements)
        down unfairly. References scoring (Phase-1 instrumentation)
        is still uncalibrated and likewise excluded for now.

        Edge case: if neither tables nor figures were detected, the
        composite is 0.0. Theory-style papers with no figures and
        no tables fall in this bucket; the grade is informative
        rather than a hard quality assertion.
        """
        buckets = []
        if self.tables:
            buckets.append((0.3, _mean(t.score for t in self.tables)))
        if self.figures:
            buckets.append((0.2, _mean(f.score for f in self.figures)))
        if not buckets:
            return 0.0
        total = sum(w for w, _ in buckets)
        return sum(w * s for w, s in buckets) / total

    def grade(self) -> str:
        s = self.overall()
        if s >= 0.90:
            return "A"
        if s >= 0.80:
            return "B"
        if s >= 0.70:
            return "C"
        if s >= 0.60:
            return "D"
        return "F"

    def to_frontmatter(self) -> str:
        L = ["---"]
        if self.metadata is not None:
            L.extend(self.metadata.to_yaml_lines())
        if self.user_annotations is not None and not self.user_annotations.is_empty():
            L.extend(self.user_annotations.to_yaml_lines())
        if self.edits is not None and not self.edits.is_empty():
            L.extend(self.edits.to_yaml_lines())
        if self.run_info is not None:
            L.extend(self.run_info.to_yaml_lines())
        if self.data_repos:
            L.append("data:")
            for d in self.data_repos:
                L.extend(d.to_yaml_lines())
        L.append("quality:")
        L.append(f"  overall: {self.overall():.2f}")
        L.append(f"  grade: {self.grade()}")
        L.append(f"  vlm_enabled: {_yaml_bool(self.vlm_enabled)}")
        L.append('  note: "Scores reflect pipeline confidence, not factual correctness."')

        if self.tables:
            L.append("  tables:")
            for t in self.tables:
                L.append(f"    - index: {t.index}")
                if t.page is not None:
                    L.append(f"      page: {t.page}")
                L.append(f"      score: {t.score:.2f}")
                L.append(f"      located: {_yaml_bool(t.located)}")
                L.append(f"      jpeg_saved: {_yaml_bool(t.jpeg_saved)}")
                if t.pre_redo_reason:
                    L.append(f"      pre_redo_reason: {_yaml_str(t.pre_redo_reason)}")
                L.append(f"      vlm_redone: {_yaml_bool(t.vlm_redone)}")
                if t.post_redo_reason:
                    L.append(f"      post_redo_reason: {_yaml_str(t.post_redo_reason)}")
                if t.sidecar:
                    L.append(f"      sidecar: {_yaml_str(t.sidecar)}")
                if t.matched_table is not None:
                    L.append(f"      matched_table: {_yaml_str(t.matched_table)}")

        if self.figures:
            L.append("  figures:")
            for f_ in self.figures:
                L.append(f"    - filename: {_yaml_str(f_.filename)}")
                L.append(f"      score: {f_.score:.2f}")
                L.append(f"      caption_produced: {_yaml_bool(f_.caption_produced)}")
                L.append(f"      caption_length: {f_.caption_length}")
                if f_.matched_figure is not None:
                    L.append(f"      matched_figure: {_yaml_str(f_.matched_figure)}")
                if f_.dropped:
                    L.append(f"      dropped: true")

        # Per-page diagnostics are an audit trail for Hook 3
        # (--rescue-sparse-pages, opt-in). They do NOT contribute to
        # quality.overall (see QualityReport.overall docstring). Emit
        # only when at least one page was rescued; otherwise the block
        # is N lines per PDF page for no consumer (the rescue is rare;
        # most papers have zero rescues and the block was just noise).
        if any(p.rescued for p in self.pages):
            L.append("  pages:")
            for p in self.pages:
                L.append(f"    - page: {p.page}")
                L.append(f"      char_density: {p.char_density:.4f}")
                L.append(f"      sparse: {_yaml_bool(p.sparse)}")
                L.append(f"      rescued: {_yaml_bool(p.rescued)}")

        if self.references is not None:
            r = self.references
            L.append("  references:")
            L.append(f"    score: {r.score:.2f}")
            L.append(f"    style: {r.style}")
            L.append(f"    section_count: {r.section_count}")
            L.append(f"    entry_count: {r.entry_count}")
            L.append(f"    numbered_continuous: {_yaml_bool(r.numbered_continuous)}")
            L.append(f"    year_hit_ratio: {r.year_hit_ratio:.2f}")
            L.append(f"    longest_over_median: {r.longest_over_median:.2f}")
            L.append(f"    last_well_bounded: {_yaml_bool(r.last_well_bounded)}")
            if r.journal_slug:
                L.append(f"    journal_slug: {r.journal_slug}")
            if r.layout is not None:
                L.append("    layout:")
                L.append(f"      columns: {r.layout.columns}")
                L.append(f"      marker_style: {r.layout.marker_style}")
                if r.layout.page_idx is not None:
                    L.append(f"      page: {r.layout.page_idx + 1}")
            if r.rescue_applied is not None:
                L.append(f"    rescue_applied: {r.rescue_applied}")
            if r.rescue_decision is not None:
                L.append(f"    rescue_decision: {r.rescue_decision}")
            if r.score_pre_rescue is not None:
                L.append(f"    score_pre_rescue: {r.score_pre_rescue:.2f}")
            if r.score_post_rescue is not None:
                L.append(f"    score_post_rescue: {r.score_post_rescue:.2f}")
            if r.api_source is not None:
                L.append(f"    api_source: {r.api_source}")
                L.append(f"    api_refs_appended: {r.api_refs_appended}")
            if r.api_skipped_offline:
                L.append("    api_skipped_offline: true")

        # rescues: -- per-paper record of which auto-rescue passes fired
        # and what they did.  Lets the user audit which papers had MinerU
        # layout decisions overridden by paper2md so manual review can
        # target only the affected files.  Emitted only when at least
        # one rescue logged a non-zero stat.
        if self.rescues:
            rescue_lines = self._rescues_yaml_lines()
            if rescue_lines:
                L.extend(rescue_lines)

        L.append("---")
        return "\n".join(L) + "\n\n"

    def _rescues_yaml_lines(self) -> list:
        """Emit a `rescues:` block for the YAML frontmatter.  Returns []
        when no rescue produced a non-zero stat (avoid noise on clean
        single-column papers).
        """
        lines: list = []
        # rescue_orphan_captions stats:
        orph = self.rescues.get("orphan_captions") or {}
        orph_interesting = any(orph.get(k, 0) for k in (
            "misnested_dropped", "captions_adopted", "footnotes_adopted",
            "captions_adopted_3col", "ambiguous_3col_warnings"))
        # rescue_subpanel_groups stats:
        sp = self.rescues.get("subpanel_groups") or {}
        sp_interesting = bool(sp.get("groups"))
        # hybrid_splice stats (--layout-source=hybrid):
        hsp = self.rescues.get("hybrid_splice") or {}
        hsp_keys = ("figures_spliced", "tables_swapped",
                    "unmatched_mineru_figs", "unmatched_mineru_tbls",
                    "marker_caption_no_mineru_asset",
                    "duplicate_mineru_figs",
                    "marker_image_links_removed",
                    "mineru_refs_swapped",
                    "inferred_fig_number_pairings",
                    "inferred_tbl_number_pairings")
        hsp_interesting = any(hsp.get(k, 0) for k in hsp_keys)
        if not orph_interesting and not sp_interesting and not hsp_interesting:
            return []
        lines.append("rescues:")
        if orph_interesting:
            lines.append("  orphan_captions:")
            for k in ("image_table_blocks", "misnested_dropped",
                      "captions_adopted", "footnotes_adopted",
                      "captions_adopted_3col", "ambiguous_3col_warnings"):
                if orph.get(k, 0):
                    lines.append(f"    {k}: {int(orph[k])}")
            pages_3col = orph.get("pages_3col") or []
            if pages_3col:
                lines.append("    pages_3col: "
                             f"[{', '.join(str(p) for p in pages_3col)}]")
            if orph.get("ambiguous_3col_warnings", 0):
                lines.append('    review_note: "Ambiguous 3-column '
                             'caption candidates were left unchanged; '
                             'check the log for warnings to resolve '
                             'manually."')
        if sp_interesting:
            lines.append("  subpanel_groups:")
            for k in ("groups", "panels_adopted"):
                if sp.get(k, 0):
                    lines.append(f"    {k}: {int(sp[k])}")
        if hsp_interesting:
            lines.append("  hybrid_splice:")
            for k in hsp_keys:
                if hsp.get(k, 0):
                    lines.append(f"    {k}: {int(hsp[k])}")
        return lines

    def to_dict(self) -> dict:
        """Structured mirror of to_frontmatter() for the .meta.json sidecar
        and HDF5 bundle. The .md frontmatter remains the canonical source;
        this dict is regenerated from the in-memory report on every run."""
        from dataclasses import asdict as _asdict
        d: dict = {
            "quality": {
                "overall": round(self.overall(), 4),
                "grade": self.grade(),
                "vlm_enabled": self.vlm_enabled,
                "tables": [_asdict(t) for t in self.tables],
                "figures": [_asdict(f) for f in self.figures],
                # `pages` mirrors the YAML gate: emit only when at
                # least one page was rescued (Hook 3 audit trail).
                # Otherwise the block is per-PDF-page noise with no
                # consumer. See to_frontmatter for rationale.
                "pages": ([_asdict(p) for p in self.pages]
                          if any(p.rescued for p in self.pages) else []),
                "references": (_asdict(self.references)
                               if self.references is not None else None),
            }
        }
        if self.metadata is not None and hasattr(self.metadata, "to_dict"):
            d["copyright"] = self.metadata.to_dict()
        if self.user_annotations is not None and not self.user_annotations.is_empty():
            d["user"] = self.user_annotations.to_dict()
        if self.edits is not None and not self.edits.is_empty():
            d["edits"] = self.edits.to_dict()
        if self.run_info is not None:
            d["run"] = self.run_info.to_dict()
        if self.data_repos:
            d["data"] = [dl.to_dict() for dl in self.data_repos]
        if self.rescues:
            # Mirror of _rescues_yaml_lines: omit when no stat fired.
            rescue_dict: dict = {}
            orph = self.rescues.get("orphan_captions") or {}
            if any(orph.get(k, 0) for k in (
                    "misnested_dropped", "captions_adopted",
                    "footnotes_adopted", "captions_adopted_3col",
                    "ambiguous_3col_warnings")):
                rescue_dict["orphan_captions"] = {
                    k: orph[k] for k in orph if orph.get(k)
                }
            sp = self.rescues.get("subpanel_groups") or {}
            if sp.get("groups"):
                rescue_dict["subpanel_groups"] = {
                    k: sp[k] for k in sp if sp.get(k)
                }
            hsp = self.rescues.get("hybrid_splice") or {}
            if any(hsp.get(k, 0) for k in (
                    "figures_spliced", "tables_swapped",
                    "unmatched_mineru_figs", "unmatched_mineru_tbls",
                    "marker_caption_no_mineru_asset",
                    "duplicate_mineru_figs",
                    "marker_image_links_removed",
                    "mineru_refs_swapped",
                    "inferred_fig_number_pairings",
                    "inferred_tbl_number_pairings")):
                rescue_dict["hybrid_splice"] = {
                    k: hsp[k] for k in hsp if hsp.get(k)
                }
            if rescue_dict:
                d["rescues"] = rescue_dict
        return d


def score_pages(doc: fitz.Document, report: QualityReport,
                rescued_idxs: set[int]) -> None:
    """Record per-page char density + sparse flag + rescue flag for
    diagnostics. As of 2026-05 these are NOT used in the quality
    overall() composite -- char-density is a content-type proxy
    (figure-heavy vs text-heavy), not an extraction-quality signal,
    and was systematically dragging figure-heavy supplements down.

    The `sparse` flag is still load-bearing as the trigger input for
    the opt-in sparse-page rescue (Hook 3, --rescue-sparse-pages).
    SPARSE_CHARS_PER_PT2 stays calibrated for that purpose."""
    for i, page in enumerate(doc):
        chars = len(page.get_text("text"))
        area = page.rect.width * page.rect.height
        density = chars / area if area > 0 else 0.0
        sparse = density < SPARSE_CHARS_PER_PT2
        rescued = i in rescued_idxs
        report.pages.append(PageScore(
            page=i + 1,
            char_density=density,
            sparse=sparse,
            rescued=rescued,
        ))


def _emit_run_summary(report: "QualityReport", out_md: Path,
                      vlm_rewrite_tables: bool,
                      use_vlm: bool) -> None:
    """End-of-run banner: total time, grade, table-mode, and optional
    re-run suggestion when the detector-text path was used. Uses
    log.info so it lands in the same stream as the runtime attribution
    banner at startup."""
    elapsed = (report.run_info.elapsed_sec
               if report.run_info is not None else 0.0)
    n_tables = len(report.tables)
    n_with_sidecar = sum(1 for t in report.tables if t.sidecar)
    log.info("=" * 60)
    log.info("paper2md complete: %s", out_md.name)
    log.info("  Total time: %.1f s", elapsed)
    log.info("  Grade:      %s (overall: %.2f)",
             report.grade(), report.overall())
    if report.references is not None:
        r = report.references
        log.info("  References: %.2f (style=%s, entries=%d%s)",
                 r.score, r.style, r.entry_count,
                 f", journal={r.journal_slug}" if r.journal_slug else "")
        if r.rescue_applied is not None:
            pre = (r.score_pre_rescue if r.score_pre_rescue is not None
                   else float('nan'))
            post = (r.score_post_rescue if r.score_post_rescue is not None
                    else float('nan'))
            log.info("  Rescue:     %s (%.2f -> %.2f, decision=%s)",
                     r.rescue_applied, pre, post, r.rescue_decision)
    if n_tables:
        mode = "VLM-rewritten" if vlm_rewrite_tables else "detector text"
        log.info("  Tables:     %d located, %d with sidecar (%s)",
                 n_tables, n_with_sidecar, mode)
    if use_vlm and not vlm_rewrite_tables and n_with_sidecar:
        log.info("")
        log.info("  Tables were extracted from the detector's text "
                 "(default fast mode).")
        log.info("  For higher fidelity (subscripts, math, footnote "
                 "markers), re-run with:")
        log.info("    --vlm-tables --table-workers N")
        log.info("  (N depends on the VLM server: ~4-6 on a 32B model + "
                 "vLLM with continuous batching; 1-2 on LM Studio or a "
                 "small model on a single GPU. See USAGE.md §6.2e.)")
    log.info("=" * 60)


# --- Hook 1: tables ---------------------------------------------------------

def _caption_for_table_match(md: str, m: re.Match) -> Optional[str]:
    """Return the caption ('Table II', 'Table 3', ...) associated with the
    markdown table matched by `m`. Searches a window around the table:
    800 chars before (Nature-style: caption above) and 200 chars after
    (Elsevier-style: caption below). The closest caption to the table
    boundary wins; ties prefer the one above.

    Window sizes are sized for journal captions that include a long
    descriptor sentence ("Supplementary Table S1. Experimental results
    for shock compressed MgSiO3 bridgmanite. We used the Hugoniot fit
    from Brygoo et al. (2015)..." -- ~330 chars) plus a paragraph break.
    The earlier 300/100 window missed the caption itself for any table
    with a multi-sentence descriptor."""
    before_start = max(0, m.start() - 800)
    after_end = min(len(md), m.end() + 200)
    before = md[before_start:m.start()]
    after = md[m.end():after_end]
    above = list(TABLE_CAPTION_RE.finditer(before))
    below = list(TABLE_CAPTION_RE.finditer(after))
    # Prefer the LAST above-match (closest caption preceding the table,
    # Nature style). Only fall back to the FIRST below-match (Elsevier
    # style) when no above-match exists. The previous "closer wins"
    # heuristic broke on consecutive tables because the after-window
    # always picks up the NEXT table's caption (e.g. "Table II"
    # appearing right after Table I's body), which is the wrong one.
    chosen = above[-1] if above else (below[0] if below else None)
    if not chosen:
        return None
    return f"Table {chosen.group(1).upper()}"


def _find_caption_page(doc: fitz.Document, caption: str,
                       occurrence: int = 0) -> Optional[int]:
    """Find which PDF page contains `caption` as a real caption (i.e.,
    line-leading), not as an inline cross-reference. Returns the 0-based
    page index, or None if no line-leading match exists. Falls back to
    the first inline match (only when occurrence == 0) if nothing is
    line-leading -- better than nothing for the page-image VLM call.

    `occurrence` selects which line-leading match to return when the
    caption appears on multiple pages, the multi-page-table case where
    each continuation page starts with "Table X. (Continued.)". The
    "(Continued.)" header is line-leading "Table X" too, so each
    continuation page contributes one entry. Marker renders each
    continuation as a separate markdown table; process_tables passes
    occurrence=0,1,2,... so the i-th task routes to the i-th
    continuation page. Default 0 = first match (back-compat for callers
    that don't have the multi-page concern, e.g. orphan-caption
    rescue).

    The line-leading pattern accepts an optional 'Supplementary'/'Supp.'
    prefix because supplementary captions in the PDF text layer commonly
    look like 'Supplementary Table S1.' even though our internal caption
    string is just 'Table S1' (the prefix gets stripped during caption
    matching against the markdown body).

    Internal whitespace inside `caption` is matched as "\s+" so the
    regex tolerates the line-broken text that older publisher-OCR'd
    PDFs commonly emit -- e.g. Boslough 1988 has page 5 storing the
    Table 2 header as three runs ("TABLE\n2.\nPostshock Spectral
    Radiance Data"), which a literal space wouldn't match."""
    cap_re = re.escape(caption).replace(r"\ ", r"\s+")
    line_lead_pat = re.compile(
        rf"^\s*(?:Supplementary\s+|Supplement(?:ary)?\s+|Supp\.\s+)?{cap_re}\b",
        re.IGNORECASE | re.MULTILINE,
    )
    inline_pat = re.compile(
        rf"\b{cap_re}\b",
        re.IGNORECASE,
    )
    line_lead_pages: list[int] = []
    inline_fallback: Optional[int] = None
    for i in range(doc.page_count):
        text = doc[i].get_text("text") or ""
        if line_lead_pat.search(text):
            line_lead_pages.append(i)
        elif inline_fallback is None and inline_pat.search(text):
            inline_fallback = i
    if line_lead_pages:
        # Stricter filter: keep only pages where the caption text
        # appears at the top-left of a PyMuPDF text block (a real
        # caption block). The regex above can match body line-wraps
        # like "...and\nFigure 11 below..." because PyMuPDF inserts
        # soft newlines on word-wrapped paragraphs, and re.MULTILINE
        # treats those as line starts. Block-level verification
        # rules out those false positives.
        block_lead_pages = _filter_block_leading_pages(
            doc, line_lead_pages, caption)
        pages_to_use = block_lead_pages if block_lead_pages else line_lead_pages
        # If occurrence overshoots the available line-leading pages
        # (markdown / PDF order misaligned, or marker double-rendered
        # without a corresponding "(Continued.)" header), return None
        # so the caller falls through to the anchor-based scan rather
        # than silently re-using an earlier page's crop.
        if occurrence < len(pages_to_use):
            return pages_to_use[occurrence]
        return None
    return inline_fallback if occurrence == 0 else None


def _filter_block_leading_pages(doc: "fitz.Document",
                                candidate_pages: list[int],
                                caption: str) -> list[int]:
    """Return the subset of `candidate_pages` where the caption text
    sits at the top-left of a PyMuPDF text block on the page (i.e.,
    the caption starts its own block, not buried as a line-wrapped
    inline reference inside a body paragraph).

    Tolerances:
      - rect.x0 within 30pt of containing block's bx0 (allows captions
        with small indents)
      - rect.y0 within 5pt of containing block's by0 (caption is the
        first line of the block)
    """
    INDENT_TOLERANCE = 30.0
    Y_TOLERANCE = 5.0
    out: list[int] = []
    for i in candidate_pages:
        try:
            rects = doc[i].search_for(caption) or []
        except Exception:
            continue
        if not rects:
            continue
        try:
            blocks = doc[i].get_text("blocks") or []
        except Exception:
            continue
        found = False
        for r in rects:
            for b in blocks:
                if len(b) < 7 or b[6] != 0:
                    continue
                bx0, by0, bx1, by1 = b[:4]
                if not (bx0 - 1 <= r.x0 and bx1 + 1 >= r.x1
                        and by0 - 1 <= r.y0 and by1 + 1 >= r.y1):
                    continue
                if ((r.x0 - bx0) < INDENT_TOLERANCE
                        and (r.y0 - by0) < Y_TOLERANCE):
                    found = True
                    break
            if found:
                break
        if found:
            out.append(i)
    return out


def _pick_candidate_near_caption(page: "fitz.Page", caption: str,
                                 cands: list) -> tuple:
    """Of the table-finder candidates on a page where `caption` appears,
    return the one whose top edge sits just below the caption's bottom
    edge (Nature/Elsevier convention: caption above, table below). Falls
    back to the first candidate if the caption text isn't searchable on
    the page or if no candidate appears below it.

    Tolerance: candidates whose top is within 5pt above the caption
    bottom are still considered "below" (handles fp/anti-alias jitter
    in the bbox detector)."""
    if len(cands) == 1:
        return cands[0]
    rects = page.search_for(caption) or []
    if not rects:
        return cands[0]
    # The line-leading caption is usually the topmost match on the page;
    # inline references in body text typically sit below.
    rects.sort(key=lambda r: r.y0)
    cap_bottom = rects[0].y1
    below = [c for c in cands if c[0][1] >= cap_bottom - 5]
    pool = below if below else cands
    return min(pool, key=lambda c: abs(c[0][1] - cap_bottom))


@dataclass
class _TableTask:
    """Per-table state assembled during process_tables's location pass and
    consumed by its finalize pass. The VLM call (if any) sits in the
    middle and can be dispatched concurrently across tasks because nothing
    in here touches the shared fitz.Document handle once the task is
    built."""
    i: int                            # 0-based markdown-table index
    span: tuple                       # (start, end) in source markdown
    body: str                         # initially marker's; replaced if redone
    located: bool
    page_idx: Optional[int] = None
    full_page: bool = True
    detector_text: str = ""
    crop: Optional[object] = None     # PIL.Image when full_page=False
    prefix: str = ""                  # leading image link if JPEG saved
    jpeg_saved: bool = False
    pre_reason: Optional[str] = None
    vlm_redone: bool = False
    post_reason: Optional[str] = None
    # VLM dispatch (set in phase 1 if a call is needed, consumed in phase 2):
    vlm_prompt: Optional[str] = None
    vlm_image: Optional[object] = None
    vlm_max_tokens: int = 0
    vlm_kind: str = ""                # "crop" | "page-image" | ""
    # VLM result (set in phase 2):
    vlm_result: Optional[str] = None
    # Sideways-table support (set in phase 1):
    # `rotated` flips the route to whole-page-image even when the
    # detector found a tight bbox, since rotated tables tend to lose
    # their headers in the bbox.
    rotated: bool = False
    rotation_caption: Optional[str] = None


# Letter-footnote marker: a single lowercase letter at the start of a
# line, optionally followed by a period or close-paren. Excludes plain
# digit prefixes (those tend to be numbered list items in body text).
_FOOTNOTE_MARKER_RE = re.compile(
    r'^\s*(?:[a-z][.)]?\s|[*†‡§¶#]\s)'
)


def _expand_bbox_for_footnotes(page: "fitz.Page",
                               bbox: tuple) -> tuple:
    """Heuristically extend a table bbox downward to capture footnote
    rows (`a`, `b`, `c`, ... or `*`, dagger, etc.) that often sit
    immediately below a scientific table.

    Strategy:
      - Walk PyMuPDF text blocks below `bbox.y1`.
      - For each block whose top is within 30pt of the current bottom
        AND whose horizontal span overlaps the table (with 20pt slack),
        check if its first line starts with a footnote marker. If so,
        extend bbox.y1 to include that block, then repeat.
      - Cap total extension at 25% of page height to stop runaway
        capture on dense pages.
      - Best-effort: any error returns the original bbox.

    Args/Returns: bbox is a (x0, y0, x1, y1) tuple in PDF user space."""
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        page_h = float(page.rect.height)
        max_extension = page_h * 0.25
        original_y1 = y1
        # Get text blocks; PyMuPDF returns
        # (x0, y0, x1, y1, text, block_no, block_type).
        blocks = page.get_text("blocks") or []
        # Order by top y, ascending.
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
        # Horizontal slack: a footnote may run wider than the table.
        slack = 20.0
        cur_bottom = y1
        for blk in blocks_sorted:
            bx0, by0, bx1, by1, btext, *_ = blk
            if by0 <= cur_bottom + 1:
                # block ends at or above current bottom: skip
                continue
            if by0 - cur_bottom > 30.0:
                # block too far below -- gap means we're past the
                # footnote zone
                break
            # Horizontal overlap test: block must intersect [x0-slack,
            # x1+slack].
            if bx1 < x0 - slack or bx0 > x1 + slack:
                continue
            first_line = (btext or "").lstrip().split("\n", 1)[0]
            if not _FOOTNOTE_MARKER_RE.match(first_line):
                # First non-overlapping non-footnote block -> stop
                # (we don't want to accidentally include body text).
                break
            # Extend bbox downward to include this block.
            new_y1 = max(cur_bottom, float(by1))
            if new_y1 - original_y1 > max_extension:
                break
            cur_bottom = new_y1
        return (x0, y0, x1, cur_bottom)
    except Exception:
        log.debug("bbox-footnote-expansion failed; using original bbox",
                  exc_info=True)
        return tuple(float(v) for v in bbox)


def _prepare_table_task(md: str, m, i: int, doc: fitz.Document, finder,
                        assets_dir: Path, asset_prefix: str,
                        use_vlm: bool, vlm_rewrite_tables: bool,
                        tables_will_rewrite: bool,
                        caption_occurrence: int = 0) -> _TableTask:
    """Phase 1 of process_tables: everything that touches the fitz
    Document or the finder. Decides the route (located+vlm-crop vs
    located+detector-text vs page-image-fallback vs marker-only) and
    queues the VLM call params if the chosen route needs one.

    `caption_occurrence` is the 0-based index of this markdown table
    among earlier markdown tables that share the same caption -- the
    multi-page-table case where marker emits N markdown tables for an
    N-page "TABLE X. (Continued.)" sequence. Drives _find_caption_page
    so each task routes to its own continuation page rather than all
    collapsing onto page 1."""
    task = _TableTask(i=i, span=(m.start(), m.end()), body=m.group(0),
                      located=False)
    caption = (_caption_for_table_match(md, m)
               if tables_will_rewrite else None)

    # Caption-page bypass.
    #
    # Used to require exactly 1 candidate on the caption page; that
    # constraint dropped real tables when (a) the page had 2+ tables
    # (e.g. Table III + IV both on root-supplemental p4) or (b) the
    # detector found a spurious extra candidate. Now we accept any
    # non-empty candidate set and pick the one closest to (and below)
    # the caption's vertical position via _pick_candidate_near_caption.
    #
    # On multi-page tables, caption_occurrence > 0 routes to the i-th
    # "(Continued.)" page in PDF order. The anchor sanity-check below
    # catches misalignment (markdown order != PDF order) by confirming
    # the candidate's text contains a body-cell anchor from THIS
    # markdown table; on failure we drop the bypass result and let
    # the anchor scan via find_pdf_table find the right page.
    #
    # The check is GATED on caption_occurrence > 0. Reason:
    # caption_occurrence == 0 covers the dominant single-occurrence
    # case where the caption is line-leading on exactly one page;
    # that page IS the table, full stop. A failed anchor check there
    # only means marker's body cells (surya-OCR'd on --force-ocr
    # runs) and docling's candidate text payload (rapidocr on
    # scanned PDFs) disagree on a few characters per cell -- a
    # tolerable OCR-engine mismatch, not a misrouted-page bug.
    # Trust the caption-page lookup unconditionally for occ=0, and
    # keep the anchor check active for continuations where the
    # markdown caption alone is ambiguous and the anchor is the
    # only real disambiguator. This matched the diagnostic finding
    # on bosloughPostshockTemperaturesSilica1988.pdf where a flaky
    # anchor check on a clean Table 1 caption-page misrouted the
    # crop to a different page.
    loc = None
    if caption:
        cap_page = _find_caption_page(doc, caption,
                                      occurrence=caption_occurrence)
        if cap_page is not None:
            cands_on_cap = list(finder.candidates(doc, cap_page))
            if cands_on_cap:
                bb, txt = _pick_candidate_near_caption(
                    doc[cap_page], caption, cands_on_cap)
                if caption_occurrence == 0:
                    anchor_ok = True
                else:
                    anchors = _extract_table_anchors(m.group(0))
                    # Empty anchor list (very short table) skips the
                    # check and trusts the bypass -- same outcome as
                    # the original bypass logic in that edge case.
                    anchor_ok = (not anchors
                                 or _text_contains_table_anchor(anchors, txt))
                if anchor_ok:
                    loc = (cap_page, bb, txt)
                    log.info("Table %d: caption-page bypass "
                             "(caption=%r, page=%d, occ=%d, "
                             "candidates=%d)",
                             i + 1, caption, cap_page + 1,
                             caption_occurrence, len(cands_on_cap))
                else:
                    log.info("Table %d: caption-page candidate on "
                             "page %d (occ=%d) failed anchor check; "
                             "falling back to anchor scan",
                             i + 1, cap_page + 1, caption_occurrence)

    # Anchor-substring scan if bypass didn't apply.
    if loc is None:
        loc = find_pdf_table(doc, m.group(0), finder)

    task.located = loc is not None
    if task.located:
        task.page_idx, bbox, task.detector_text = loc
        page = doc[task.page_idx]
        # Expand bbox downward to capture footnote rows below the
        # table body (common in dense physics tables, e.g. wackerle).
        # No-op when no footnote-marked block sits immediately below.
        bbox = _expand_bbox_for_footnotes(page, bbox)
        page_area = page.rect.width * page.rect.height
        bw = max(0.0, bbox[2] - bbox[0])
        bh = max(0.0, bbox[3] - bbox[1])
        task.full_page = (page_area <= 0
                          or (bw * bh) / page_area > FULL_PAGE_TABLE_FRAC)
        if not task.full_page:
            initial_crop = render_crop(doc, task.page_idx, bbox)
            if _crop_looks_sideways(initial_crop):
                # Sideways table: bbox crops on landscape-rotated
                # tables tend to be too tight (the rotated header row
                # sits at the LEFT page edge, often outside the table
                # bbox). Render the whole page and rotate it upright
                # instead -- includes the headers, caption, and any
                # surrounding context the VLM needs. The rotated page
                # becomes both the saved JPG asset (so the human sees
                # the table upright) and the VLM input.
                aspect = initial_crop.size[1] / max(1, initial_crop.size[0])
                page_img = render_page(doc, task.page_idx, dpi=PAGE_DPI)
                task.crop = _rotate_upright(page_img)
                task.rotated = True
                task.rotation_caption = caption
                log.info("Table %d: detected sideways crop on page %d "
                         "(h/w=%.2f > %.2f); rendering whole page "
                         "rotated %s and routing to whole-page VLM",
                         i + 1, task.page_idx + 1, aspect,
                         TABLE_ROTATION_ASPECT_THRESHOLD,
                         "CCW" if TABLE_ROTATION_DIRECTION == "cw"
                         else "CW")
            else:
                task.crop = initial_crop
            filename = f"{asset_prefix}table_p{task.page_idx + 1}_{i + 1}.jpg"
            task.crop.save(assets_dir / filename, format="JPEG",
                           quality=TABLE_JPEG_QUALITY)
            log.info("Saved table JPEG: %s", filename)
            task.prefix = (f"![Table {i + 1} (page {task.page_idx + 1})]"
                           f"(assets/{filename})\n\n")
            task.jpeg_saved = True
        else:
            log.debug("Table %d ~fills page %d; skipping JPEG",
                      i + 1, task.page_idx + 1)

    task.pre_reason = table_is_suspicious(task.body)

    # Route selection. Either we apply the detector-markdown shortcut
    # synchronously, queue a VLM call for phase 2, or do nothing.
    if task.located and not vlm_rewrite_tables:
        stripped = (task.detector_text or "").lstrip()
        if stripped.startswith("|") or "\n|" in stripped[:200]:
            task.body = task.detector_text.strip() + "\n"
            task.vlm_redone = True
            task.post_reason = table_is_suspicious(task.body)
            log.info("Table %d: detector-text mode (default); using "
                     "detector markdown (page %d)",
                     i + 1, task.page_idx + 1)
        else:
            log.info("Table %d: detector-text mode (default); detector "
                     "returned non-markdown text, keeping marker body",
                     i + 1)
    elif task.located and use_vlm:
        if task.crop is None:
            task.crop = render_crop(doc, task.page_idx, bbox)
        task.vlm_image = task.crop
        if task.rotated:
            # The crop is the WHOLE page rotated upright. Use the
            # page-image prompt (it has rotation hint baked in plus
            # the "extract the named table from this page; ignore
            # other content" framing), captioned when we have one.
            task.vlm_kind = "page-image"
            task.vlm_prompt = TABLE_PAGE_PROMPT.format(
                caption=task.rotation_caption or caption or f"Table {i + 1}")
        else:
            task.vlm_kind = "crop"
            task.vlm_prompt = TABLE_PROMPT
        # Load-bearing: 2000 is the empirical ceiling for stable vLLM
        # generation on this hardware (DGX Spark + Qwen3-VL-32B BF16,
        # ~1400x1200 dense tables). At max_tokens >= 2500, vLLM hangs
        # and the call eventually times out -- whole sidecar lost.
        # Trade-off: very dense tables (e.g. wackerle Table II:
        # 28 rows x 11 cols) will truncate the body and miss the
        # **Notes:** block. Documented in USAGE.md.
        task.vlm_max_tokens = 2000
    elif (not task.located) and use_vlm and vlm_rewrite_tables:
        # Page-image fallback: detector missed AND bypass didn't apply.
        # Render the caption's page (touches doc -- must be in phase 1)
        # and queue the VLM call. Multi-page-table aware via
        # caption_occurrence: continuations get their own page image.
        cap_page = (_find_caption_page(doc, caption,
                                       occurrence=caption_occurrence)
                    if caption is not None else None)
        if cap_page is not None:
            log.info("Table %d not located; page-image fallback "
                     "(caption=%r, page=%d)",
                     i + 1, caption, cap_page + 1)
            task.vlm_kind = "page-image"
            task.vlm_prompt = TABLE_PAGE_PROMPT.format(caption=caption)
            task.vlm_image = render_page(doc, cap_page, dpi=PAGE_DPI)
            # Modest bump from the original 1000: enough room for a
            # **Notes:** block on typical pages (a few short lines)
            # without inviting the vLLM-hang behaviour we hit at >=2500
            # tokens with the dense-page-image input. See crop branch
            # comment for the full rationale.
            task.vlm_max_tokens = 2000
            task.page_idx = cap_page  # for sidecar naming + score
        else:
            log.info("Table %d not located in PDF and no caption "
                     "found nearby; keeping marker markdown", i + 1)
    return task


def _run_table_vlm(task: _TableTask) -> None:
    """Phase 2: invoke the queued VLM call and store the result on the
    task. Pure I/O wrapper around vlm(); thread-safe because the task
    is the only mutable target and each task is touched by exactly one
    thread."""
    if task.vlm_prompt is None:
        return
    if task.vlm_kind == "crop":
        log.info("VLM-rewriting table %d (page %d, pre_reason=%s)",
                 task.i + 1, task.page_idx + 1, task.pre_reason or "clean")
    task.vlm_result = vlm(task.vlm_prompt, task.vlm_image,
                          max_tokens=task.vlm_max_tokens)


def _finalize_table_task(task: _TableTask, assets_dir: Path,
                         asset_prefix: str,
                         report: QualityReport) -> Optional[tuple]:
    """Phase 3: process the VLM result (if any), write the sidecar,
    append the TableScore. Returns (start, end, replacement) for the
    markdown edit, or None if the body wasn't changed."""
    # Apply VLM result if we made a call.
    if task.vlm_kind == "crop":
        if task.vlm_result:
            task.body = task.vlm_result.strip() + "\n"
            task.vlm_redone = True
            task.post_reason = table_is_suspicious(task.body)
        else:
            log.warning("VLM table rewrite failed for table %d; "
                        "keeping marker's markdown", task.i + 1)
    elif task.vlm_kind == "page-image":
        if (task.vlm_result
                and task.vlm_result.strip(".!").upper() != "SKIP"):
            task.body = task.vlm_result.strip() + "\n"
            task.vlm_redone = True
            task.post_reason = table_is_suspicious(task.body)
            log.info("  VLM extracted table %d from page image",
                     task.i + 1)
        else:
            log.info("  VLM declined / returned SKIP; keeping marker "
                     "markdown")

    sidecar_name: Optional[str] = None
    suffix = ""
    if task.vlm_redone:
        if task.located:
            sidecar_name = (f"{asset_prefix}table_p"
                            f"{task.page_idx + 1}_{task.i + 1}.md")
        elif task.page_idx is not None:
            sidecar_name = (f"{asset_prefix}table_page"
                            f"{task.page_idx + 1}_{task.i + 1}.md")
        else:
            sidecar_name = f"{asset_prefix}table_{task.i + 1}.md"
        (assets_dir / sidecar_name).write_text(task.body)
        log.info("Wrote table sidecar: %s", sidecar_name)
        suffix = (f"\n[Table {task.i + 1} — separate markdown]"
                  f"(assets/{sidecar_name})\n")

    # Emit an edit iff something changed: JPEG link prefixed or body
    # was redone (sidecar suffix follows from vlm_redone). When neither
    # holds, marker's body stays in place untouched.
    edit = None
    if task.prefix or task.vlm_redone:
        edit = (task.span[0], task.span[1],
                task.prefix + task.body + suffix)

    report.tables.append(TableScore(
        index=task.i + 1,
        page=(task.page_idx + 1) if task.page_idx is not None else None,
        located=task.located,
        jpeg_saved=task.jpeg_saved,
        pre_redo_reason=task.pre_reason,
        vlm_redone=task.vlm_redone,
        post_redo_reason=task.post_reason,
        score=_score_table(task.located, task.pre_reason,
                           task.vlm_redone, task.post_reason),
        sidecar=sidecar_name,
    ))
    return edit


def process_tables(md: str, doc: fitz.Document, finder,
                   assets_dir: Path, report: QualityReport,
                   use_vlm: bool, asset_prefix: str = "",
                   vlm_rewrite_tables: bool = False,
                   table_workers: int = 1) -> str:
    """For each markdown table in md, locate its PDF region once and do:
      - save a JPEG crop to assets/ (skipping tables that ~fill the page)
        and prepend an image link above the markdown table;
      - if use_vlm is True and `vlm_rewrite_tables` is True, rewrite the
        table body via VLM on the crop and save as a sidecar .md;
      - if `vlm_rewrite_tables` is False (the default) and the detector
        returned a markdown-formatted text payload (e.g. docling's
        export_to_markdown), substitute that payload as the table body
        without calling the VLM. Faster on dense-table corpora at the
        cost of subscript / math fidelity (no semantic re-rendering);
      - append a [Table N -- separate markdown] link after the inline
        table pointing at the sidecar;
      - append a TableScore to the report regardless.
    `asset_prefix` is prepended to every filename written under
    assets/, so a supplement run can share the assets dir with its
    main paper without filename collisions.
    """
    matches = list(TABLE_RE.finditer(md))
    if not matches:
        return md
    # Tables flow tries to rewrite (via VLM crop, detector markdown, or
    # page-image VLM) whenever any VLM hook is on: --no-vlm leaves
    # marker's tables untouched; otherwise the detector-text path runs
    # by default, with --vlm-tables switching to the VLM-crop path.
    tables_will_rewrite = use_vlm

    # Compute the caption occurrence for each markdown table -- the
    # 0-based count of earlier matches that share the same caption.
    # Drives multi-page-table handling: marker emits a separate
    # markdown table for each page of a "TABLE X. (Continued.)"
    # sequence, so the captions are identical and only the occurrence
    # index distinguishes them. _find_caption_page uses this to route
    # the i-th occurrence to the i-th line-leading "Table X" page in
    # the PDF (each "(Continued.)" header is line-leading too) instead
    # of always returning page 1.
    caption_occurrences: list[int] = []
    seen_captions: dict[str, int] = {}
    for m in matches:
        cap = (_caption_for_table_match(md, m)
               if tables_will_rewrite else None)
        occ = seen_captions.get(cap, 0) if cap else 0
        caption_occurrences.append(occ)
        if cap:
            seen_captions[cap] = occ + 1

    # Phase 1 (sequential — touches fitz Document, which is not
    # thread-safe): locate each table, render JPEG/page-image, queue
    # any VLM call params on the per-task record.
    tasks = [_prepare_table_task(md, m, i, doc, finder,
                                 assets_dir, asset_prefix,
                                 use_vlm, vlm_rewrite_tables,
                                 tables_will_rewrite,
                                 caption_occurrence=caption_occurrences[i])
             for i, m in enumerate(matches)]

    # Phase 2 (concurrent — only the network-bound VLM call). Each
    # _run_table_vlm() touches one task and the shared vLLM endpoint.
    # vLLM batches concurrent requests server-side, so 4 workers is a
    # comfortable default; users on smaller backends (LM Studio with
    # one GPU) can drop to 1.
    pending = [t for t in tasks if t.vlm_prompt is not None]
    if pending:
        if table_workers > 1 and len(pending) > 1:
            log.info("Dispatching %d table VLM call(s) concurrently "
                     "(workers=%d)", len(pending), table_workers)
            with ThreadPoolExecutor(max_workers=table_workers) as pool:
                futs = {pool.submit(_run_table_vlm, t): t for t in pending}
                for fut in as_completed(futs):
                    fut.result()  # propagates exceptions; result on task
        else:
            for t in pending:
                _run_table_vlm(t)

    # Phase 3 (sequential): apply VLM results, write sidecars, build
    # markdown edits, append TableScores in original order.
    edits: list[tuple[int, int, str]] = []
    for t in tasks:
        edit = _finalize_table_task(t, assets_dir, asset_prefix, report)
        if edit:
            edits.append(edit)

    for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        md = md[:start] + rep + md[end:]
    return md


# --- Hook 1.5: orphan-caption table rescue ---------------------------------


def _has_table_follow_up(md: str, after_pos: int,
                         lookahead_chars: int = 2500) -> bool:
    """True if within `lookahead_chars` after `after_pos` the body
    contains either a markdown-table separator row or a Table-N
    image-link / sidecar reference -- i.e. process_tables already
    handled this caption.

    Stops scanning at the next Table-N caption line: any follow-up
    signal that appears AFTER the next caption belongs to that next
    table, not this one. Without this boundary, a paragraph-rendered
    "Table I" preceding a normal "Table II" with its image link
    would falsely register as having a follow-up."""
    chunk = md[after_pos:after_pos + lookahead_chars]
    sep_re = re.compile(r"^\s*\|[-:|\s]+\|\s*$")
    for line in chunk.split("\n", 30)[:30]:
        # A new line-leading "Table N." caption ends this caption's
        # follow-up window: anything after it owns the next caption.
        if TABLE_CAPTION_LINE_RE.match(line):
            return False
        if sep_re.match(line):
            return True
        if "![Table " in line and "(page " in line:
            return True
        if "— separate markdown]" in line or "-- separate markdown]" in line:
            return True
    return False


def rescue_orphan_table_captions(md: str, doc: "fitz.Document",
                                 assets_dir: Path, asset_prefix: str,
                                 report: QualityReport,
                                 use_vlm: bool) -> str:
    """Post-pass after process_tables: catch table captions that have
    no markdown table or sidecar following them within ~30 lines (the
    "marker rendered the table as a paragraph" failure mode -- e.g.
    root-supplemental Table I).

    For each orphan caption, find its PDF page via _find_caption_page,
    render the page image, and ask the VLM to extract the captioned
    table directly using the existing TABLE_PAGE_PROMPT (which already
    has the right "ignore other tables on the page" + SKIP escape
    behavior).

    A successful rescue writes a sidecar .md, saves the page image as
    a JPEG, splices an image-link + table body + sidecar-link into the
    markdown right after the orphan caption line, and appends a
    TableScore to `report.tables` so the front-matter records it.

    No-op when use_vlm is False, when RESCUE_ORPHAN_TABLES is off, or
    when no orphan captions exist. Best-effort: any per-caption
    failure logs and continues."""
    if not use_vlm or not RESCUE_ORPHAN_TABLES:
        return md

    edits: list[tuple[int, int, str]] = []
    next_idx = len(report.tables) + 1
    rescued_count = 0

    for cm in TABLE_CAPTION_LINE_RE.finditer(md):
        if _has_table_follow_up(md, cm.end()):
            continue
        full_caption = cm.group("full").strip()
        try:
            cap_page = _find_caption_page(doc, full_caption)
        except Exception as e:
            log.debug("orphan-rescue: caption-page lookup failed for %r: %s",
                      full_caption, e)
            continue
        if cap_page is None:
            continue

        log.info("Orphan-rescue: caption %r unmatched in markdown; "
                 "rendering page %d", full_caption, cap_page + 1)
        try:
            page_img = render_page(doc, cap_page, dpi=PAGE_DPI)
        except Exception as e:
            log.warning("orphan-rescue: render_page failed (%s); skipping",
                        e)
            continue

        result = vlm(TABLE_PAGE_PROMPT.format(caption=full_caption),
                     page_img, max_tokens=2000)
        if not result or result.strip(".!").upper() in ("SKIP", "NO_TABLE_HERE"):
            log.info("Orphan-rescue: VLM SKIP/empty for %r on page %d",
                     full_caption, cap_page + 1)
            continue
        body = result.strip() + "\n"

        idx = next_idx
        next_idx += 1
        img_filename = f"{asset_prefix}table_p{cap_page + 1}_{idx}.jpg"
        sidecar_filename = f"{asset_prefix}table_p{cap_page + 1}_{idx}.md"
        try:
            page_img.save(assets_dir / img_filename, format="JPEG",
                          quality=TABLE_JPEG_QUALITY)
            (assets_dir / sidecar_filename).write_text(body)
        except Exception as e:
            log.warning("orphan-rescue: failed to write artifacts for %r: %s",
                        full_caption, e)
            continue

        replacement = (
            f"\n\n![Table {idx} (page {cap_page + 1})]"
            f"(assets/{img_filename})\n\n"
            f"{body}\n"
            f"[Table {idx} — separate markdown](assets/{sidecar_filename})\n"
        )
        edits.append((cm.end(), cm.end(), replacement))
        report.tables.append(TableScore(
            index=idx,
            page=cap_page + 1,
            located=True,
            jpeg_saved=True,
            pre_redo_reason="orphan-caption rescue",
            vlm_redone=True,
            post_redo_reason=None,
            score=1.0,
            sidecar=sidecar_filename,
        ))
        rescued_count += 1
        log.info("Orphan-rescue: extracted table for %r -> %s",
                 full_caption, sidecar_filename)

    if rescued_count:
        for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
            md = md[:start] + rep + md[end:]
        log.info("Orphan-rescue: recovered %d additional table(s)", rescued_count)
    return md


# --- Hook 2.5: orphan-figure-caption rescue --------------------------------

# Line-leading "Fig. N." caption matcher used to locate the splice
# point for an orphan figure. Mirrors the leading-anchor / bold-wrapper
# tolerance of FIG_CAPTION_RE_PLAIN (and the bold-block variants) but
# anchored on a specific figure id substituted at call time. The id
# is interpolated as a literal regex (re.escape) since callers pass
# extracted ids verbatim ("5", "12", "S3", "B.13").
def _orphan_fig_caption_line_re(fig_id: str) -> re.Pattern:
    """Match a 'Figure N. ...' or 'Fig. N. ...' caption line in the
    markdown body. Used by the orphan-figure rescue to locate where
    to splice the recovered image. Recognizes four caption styles:

      1. 'Figure 1. Title text...'           (plain)
      2. 'Figure 1 Title text...'            (period dropped, title
                                              starts with capital/paren)
      3. '**Figure 1** . Title...' / '**Figure 1** | Title...'
                                              (bold then separator)
      4. '**Figure 1.** Title...'            (period inside bold;
                                              Wiley / AGU style, e.g. Luo)
    """
    fid = re.escape(fig_id)
    return re.compile(
        r'(?im)^\s*'
        r'(?:<span\s+id="page-\d+-\d+"></span>\s*)?'
        r'\*{0,2}\s*'
        r'(?:Figure|Fig\.?)\s+' + fid +
        r'(?:\s+\([^)\n]{1,40}\))?'
        r'(?:'
        r'\s*\.\s+'                          # 1: ". Title"
        r'|\s+(?=(?-i:[A-Z(]))'              # 2: " Title" (capital follows)
        r'|\s*\*{1,2}\s*[\.|]\s*'            # 3: "** ." or "** |"
        r'|\s*\.\s*\*{1,2}\s+'               # 4: ".** Title" (period in bold)
        r')'
    )


def _has_image_for_fig(md: str, fig_id: str) -> bool:
    """True if the markdown already contains an image-markdown line
    whose alt-text references this figure id -- i.e. caption_figures
    matched a marker-extracted crop to this caption. Pattern matches
    both the matched-path alt-text (`![Figure N | ...](...)`) and
    common variants (`![Figure N. ...]`, `![Fig. N | ...]`)."""
    fid = re.escape(fig_id)
    pat = re.compile(
        r'!\[\s*(?:Extended Data\s+)?(?:Figure|Fig\.?)\s+' + fid +
        r'\s*[|.\]:]',
        re.IGNORECASE,
    )
    return bool(pat.search(md))


def _crop_around_caption(doc: "fitz.Document", page_idx: int,
                         caption_search: str,
                         page_img: "Image.Image") -> "Image.Image":
    """Render-style crop for the orphan-figure rescue. Decides where
    the figure sits relative to the caption and crops to that region.

    Three layouts handled:

    1. **Single-column / wide caption** (caption width >= 60% of page
       width): vertical splitting only. If text-density above the
       caption >> below, figure is BELOW (caption-above-figure, e.g.
       Science). If below >> above, figure is ABOVE (figure-above-
       caption, e.g. Elsevier / Nature). Otherwise return whole page.

    2. **Multi-column with figure in the same column** (caption is
       column-confined AND substantial text exists in the OPPOSITE
       column at the same y-range): typical 2-column journal page
       (PRB / JGR / etc.). Crop horizontally to the caption's column
       (with a small margin) and vertically per the same above/below
       text-density rule. Avoids including unrelated body text from
       the adjacent column.

    3. **Side-by-side caption** (caption is column-confined AND the
       opposite column has LOW text-density at the caption's
       y-range): figure occupies the column opposite the caption,
       caption alongside it (e.g. young 2016 Fig. 1 in Science).
       Crop the opposite column at full page height.

    Falls back to the full page when the caption can't be located,
    block text isn't available, or the resulting crop is < 10% of
    page area.
    """
    page = doc[page_idx]
    rects = page.search_for(caption_search) or []
    if not rects:
        return page_img
    page_w_pdf = page.rect.width
    page_h_pdf = page.rect.height
    if page_h_pdf <= 0 or page_w_pdf <= 0:
        return page_img
    img_w, img_h = page_img.size

    blocks: list = []
    try:
        blocks = page.get_text("blocks") or []
    except Exception as e:
        log.debug("_crop_around_caption: block extraction failed (%s); "
                  "falling back to whole page", e)
        return page_img
    BLOCK_TYPE_TEXT = 0

    # Filter rects to those that look like LINE-LEADING captions
    # rather than inline cross-references. A real caption is at the
    # left edge of its containing text block (e.g. "FIG. 7. (Color
    # online) ..." starts at the column's left edge with at most a
    # small indent). An inline reference like "...illustrated in
    # Fig. 7. The measured..." sits mid-paragraph, so its x0 is far
    # from the block's left edge.
    INDENT_TOLERANCE = 30.0
    leading_rects = []
    for r in rects:
        for b in blocks:
            if len(b) < 7 or b[6] != BLOCK_TYPE_TEXT:
                continue
            bx0, by0, bx1, by1 = b[:4]
            if not (bx0 - 1 <= r.x0 and bx1 + 1 >= r.x1
                    and by0 - 1 <= r.y0 and by1 + 1 >= r.y1):
                continue
            if (r.x0 - bx0) < INDENT_TOLERANCE:
                leading_rects.append(r)
            break
    if leading_rects:
        rects = leading_rects
    rects.sort(key=lambda r: r.y0)
    cap = rects[0]

    # Expand caption rect to its containing text block so we know the
    # caption's full horizontal and vertical extent (the search hit
    # only matches the literal "Fig. N." prefix).
    cap_top = cap.y0
    cap_bot = cap.y1
    cap_l = cap.x0
    cap_r = cap.x1
    for b in blocks:
        if len(b) < 7 or b[6] != BLOCK_TYPE_TEXT:
            continue
        bx0, by0, bx1, by1 = b[:4]
        if (bx0 - 1 <= cap.x0 and bx1 + 1 >= cap.x1
                and by0 - 1 <= cap.y0 and by1 + 1 >= cap.y1):
            cap_l, cap_top, cap_r, cap_bot = bx0, by0, bx1, by1
            break
    cap_w = cap_r - cap_l
    cap_cx = (cap_l + cap_r) / 2

    def _to_img_box(x0_pdf: float, y0_pdf: float,
                    x1_pdf: float, y1_pdf: float):
        x0 = int(max(0, x0_pdf) / page_w_pdf * img_w)
        x1 = int(min(page_w_pdf, x1_pdf) / page_w_pdf * img_w)
        y0 = int(max(0, y0_pdf) / page_h_pdf * img_h)
        y1 = int(min(page_h_pdf, y1_pdf) / page_h_pdf * img_h)
        return x0, y0, x1, y1

    def _safe_crop(x0_pdf, y0_pdf, x1_pdf, y1_pdf):
        x0, y0, x1, y1 = _to_img_box(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
        # Reject crops that are below ~5% of page area as
        # implausibly small (a legitimate figure column on a 2-col
        # page is ~50% page width * ~25% page height = ~12.5%; a
        # tight band can be smaller, e.g. 9-10%).
        if (x1 - x0) * (y1 - y0) < img_w * img_h * 0.05:
            return page_img
        return page_img.crop((x0, y0, x1, y1))

    is_full_width = cap_w >= 0.6 * page_w_pdf

    # Bucket text into 4 regions relative to caption:
    #   same_col_above / same_col_below: blocks in caption's column
    #   opp_at_y: blocks in opposite column at caption's y-range
    # For same-column we ALSO track each block's y-extent so we can
    # find the largest empty band adjacent to the caption (where the
    # figure actually lives — text-density alone fails when the
    # caption sits at the column's top or bottom edge).
    chars_same_above = 0
    chars_same_below = 0
    chars_opp_at_y = 0
    n_opp_at_y_blocks = 0
    max_opp_at_y_block = 0
    nearest_above_y1 = 0.0          # bottom edge of nearest block above cap
    nearest_below_y0 = page_h_pdf   # top edge of nearest block below cap
    # Detect another Fig/Table caption block in the same column above
    # or below. Helps disambiguate gap-based decisions: if there's a
    # second caption WITHIN the gap, that gap belongs to the OTHER
    # figure, not ours.
    other_caption_above_y0 = None
    other_caption_below_y0 = None
    other_caption_re = re.compile(
        r"^\s*\*{0,2}\s*(?:Figure|Fig\.?|Table)\s+\S", re.IGNORECASE)
    for b in blocks:
        if len(b) < 7 or b[6] != BLOCK_TYPE_TEXT:
            continue
        bx0, by0, bx1, by1, btext, *_ = b
        # Skip caption's own block.
        if (bx0 >= cap_l - 1 and bx1 <= cap_r + 1
                and by0 >= cap_top - 1 and by1 <= cap_bot + 1):
            continue
        text_len = len(str(btext or ""))
        if text_len == 0:
            continue
        x_overlap = max(0.0, min(bx1, cap_r) - max(bx0, cap_l))
        in_same_col = x_overlap > 0.3 * cap_w if cap_w > 0 else True
        looks_like_caption = bool(
            other_caption_re.match(str(btext or "").lstrip()))
        if in_same_col:
            if by1 <= cap_top:
                chars_same_above += text_len
                if by1 > nearest_above_y1:
                    nearest_above_y1 = by1
                if looks_like_caption:
                    # Track LATEST caption above (closest to ours).
                    if (other_caption_above_y0 is None
                            or by0 > other_caption_above_y0):
                        other_caption_above_y0 = by0
            elif by0 >= cap_bot:
                chars_same_below += text_len
                if by0 < nearest_below_y0:
                    nearest_below_y0 = by0
                if looks_like_caption:
                    # Track EARLIEST caption below (closest to ours).
                    if (other_caption_below_y0 is None
                            or by0 < other_caption_below_y0):
                        other_caption_below_y0 = by0
        else:
            y_overlap = max(0.0, min(by1, cap_bot) - max(by0, cap_top))
            if y_overlap > 0:
                chars_opp_at_y += text_len
                n_opp_at_y_blocks += 1
                if text_len > max_opp_at_y_block:
                    max_opp_at_y_block = text_len

    # ----- Vertical-only branch for wide / single-column captions -----
    if is_full_width:
        chars_above = chars_same_above
        chars_below = chars_same_below
        if chars_above > 2 * chars_below and chars_above >= 200:
            return _safe_crop(0, cap_bot, page_w_pdf, page_h_pdf)
        if chars_below > 2 * chars_above and chars_below >= 200:
            return _safe_crop(0, 0, page_w_pdf, cap_top)
        return page_img

    # Caption is column-confined. Two layouts to distinguish:
    #
    # (a) Side-by-side caption: figure occupies the column OPPOSITE
    #     the caption, at similar y. Recognized by MANY (>=5) SMALL
    #     blocks (max < 200 chars) in that opposite region — figure
    #     annotations (axis tick labels, model names, contour
    #     numbers). young 2016 Fig. 1 has 23 small blocks max=58.
    # (b) Same-column figure: figure is in the SAME column as the
    #     caption, either immediately above or immediately below.
    #     Recognized by either no opposite content or a small number
    #     of larger blocks (e.g. a body paragraph or short table).
    side_by_side = (n_opp_at_y_blocks >= 5
                    and max_opp_at_y_block < 200)

    if side_by_side:
        if cap_cx > page_w_pdf / 2:
            crop_x0, crop_x1 = 0.0, cap_l
        else:
            crop_x0, crop_x1 = cap_r, page_w_pdf
        # Use full page height — figure may extend above and/or below
        # the caption's y-range (young Fig 1 has chart B above, chart
        # A below the caption text on the opposite side).
        return _safe_crop(crop_x0, 0, crop_x1, page_h_pdf)

    # Same-column case. The figure occupies an empty band IMMEDIATELY
    # adjacent to the caption. Use the gap to the nearest text block
    # above and the nearest text block below to pick which side the
    # figure is on. Text-density alone fails when the caption sits at
    # the column's top or bottom edge (Knudson Fig 4 / Fig 10:
    # caption at page bottom, figure in the empty gap above it; lots
    # of body text further above, almost nothing below).
    margin = 6.0
    col_x0 = max(0.0, cap_l - margin)
    col_x1 = min(page_w_pdf, cap_r + margin)
    gap_above = cap_top - nearest_above_y1
    gap_below = nearest_below_y0 - cap_bot
    MIN_FIGURE_GAP = 50.0  # below this, the gap is too small to plausibly hold a figure

    # When ANOTHER Fig/Table caption sits in the same column below
    # ours, that "below" region belongs to the other figure -- our
    # figure must be ABOVE the caption (figure-above-caption layout,
    # the most common journal style). This catches cases where the
    # PDF text layer is sparse on figure regions (vector graphics
    # without text annotations, e.g. Luo 2004 Fig 4: 3D plot above
    # caption has zero text blocks; the gap-to-nearest-text logic
    # otherwise picks the bigger empty span which leads to the next
    # figure's region).
    if (other_caption_below_y0 is not None
            and gap_above >= MIN_FIGURE_GAP):
        return _safe_crop(col_x0, nearest_above_y1, col_x1, cap_top)
    # Mirror: another caption above suggests our figure is BELOW.
    if (other_caption_above_y0 is not None
            and gap_below >= MIN_FIGURE_GAP):
        return _safe_crop(col_x0, cap_bot, col_x1, nearest_below_y0)

    if gap_above > gap_below and gap_above >= MIN_FIGURE_GAP:
        return _safe_crop(col_x0, nearest_above_y1, col_x1, cap_top)
    if gap_below > gap_above and gap_below >= MIN_FIGURE_GAP:
        return _safe_crop(col_x0, cap_bot, col_x1, nearest_below_y0)

    # No clear gap. Fall back to text-density direction (preserves
    # the previous behavior on full-column pages).
    if chars_same_above > 2 * chars_same_below and chars_same_above >= 200:
        return _safe_crop(col_x0, cap_bot, col_x1, page_h_pdf)
    if chars_same_below > 2 * chars_same_above and chars_same_below >= 200:
        return _safe_crop(col_x0, 0, col_x1, cap_top)
    return _safe_crop(col_x0, 0, col_x1, page_h_pdf)


def rescue_orphan_figure_captions(md: str, doc: "fitz.Document",
                                  assets_dir: Path, asset_prefix: str,
                                  report: QualityReport,
                                  use_vlm: bool) -> str:
    """Post-pass after caption_figures: catch figure captions whose
    figure crop was never extracted by marker (e.g. surya layout
    misclassified the caption block as body text and skipped the
    figure region above it -- jacquet 2026 Fig. 5).

    Locates each orphan caption's PDF page via _find_caption_page and
    crops the page-region above the caption as the figure asset. The
    crop reaches from the top of the page down to the caption's top
    edge; when that area is too small (caption near the top of the
    page or caption text not searchable), falls back to a full-page
    render. No VLM call -- this is deterministic.

    A successful rescue saves a JPEG and splices an
    `![Figure N | title](assets/_page_K_orphan_Figure_N.jpeg)` line
    immediately above the orphan caption. Skipped silently when
    RESCUE_ORPHAN_FIGURES is off or no orphan captions exist."""
    if not RESCUE_ORPHAN_FIGURES:
        return md

    captions = extract_figure_captions(md)
    if not captions:
        return md

    edits: list[tuple[int, int, str]] = []
    rescued_count = 0
    for fig_id, title in captions:
        if _has_image_for_fig(md, fig_id):
            continue
        cap_line_re = _orphan_fig_caption_line_re(fig_id)
        cm = cap_line_re.search(md)
        if not cm:
            continue

        cap_search = f"Fig. {fig_id}"
        cap_page = _find_caption_page(doc, cap_search)
        if cap_page is None:
            cap_search = f"Figure {fig_id}"
            cap_page = _find_caption_page(doc, cap_search)
        if cap_page is None:
            log.debug("orphan-figure-rescue: caption page not found for "
                      "Fig. %s", fig_id)
            continue

        try:
            page_img = render_page(doc, cap_page, dpi=PAGE_DPI)
        except Exception as e:
            log.warning("orphan-figure-rescue: render_page failed for "
                        "Fig. %s on page %d (%s); skipping", fig_id,
                        cap_page + 1, e)
            continue

        crop = _crop_around_caption(doc, cap_page, cap_search, page_img)

        img_filename = (f"{asset_prefix}_page_{cap_page}_orphan_"
                        f"Figure_{fig_id}.jpeg")
        try:
            crop.save(assets_dir / img_filename, format="JPEG", quality=85)
        except Exception as e:
            log.warning("orphan-figure-rescue: failed to save %s (%s); "
                        "skipping", img_filename, e)
            continue

        replacement = (f"\n\n![Figure {fig_id} | {title}]"
                       f"(assets/{img_filename})\n\n")
        edits.append((cm.start(), cm.start(), replacement))
        rescued_count += 1
        log.info("Orphan-figure-rescue: rendered Fig. %s on page %d -> %s",
                 fig_id, cap_page + 1, img_filename)

    if rescued_count:
        for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
            md = md[:start] + rep + md[end:]
        log.info("Orphan-figure-rescue: recovered %d additional figure(s)",
                 rescued_count)
    return md


# --- Hook 2: figure captions ------------------------------------------------

def caption_figures(md: str, assets_dir: Path, report: QualityReport,
                    use_vlm: bool,
                    doc: "Optional[fitz.Document]" = None) -> str:
    """Dispatch: if the markdown contains author `Figure N | ...` captions
    AND VLM is enabled, match each extracted image to a caption via the
    classifier path; drop unmatched images. Otherwise fall back to the
    freeform captioning path.

    `doc` is optional. When provided, the matched path uses caption
    page locations (from PyMuPDF text search) to validate VLM
    classifications: assignments where the image's source page is
    far from the caption's PDF page get demoted and re-classified.
    Catches Hook 2 mis-classifications like the Luo 2004 case
    (image extracted from doc[10] classified as Fig 10 whose
    caption is on doc[9])."""
    captions = extract_figure_captions(md)
    if captions and use_vlm:
        log.info("Extracted %d figure caption(s) from markdown", len(captions))
        return _caption_figures_matched(md, captions, assets_dir, report,
                                        use_vlm, doc=doc)
    return _caption_figures_freeform(md, assets_dir, report, use_vlm)


_PAGE_FROM_FILENAME_RE = re.compile(r"_page_(\d+)_(?:Picture|Figure|Table)_\d+", re.IGNORECASE)


def _parse_page_idx(filename: str) -> Optional[int]:
    """Extract the 0-indexed PDF page from marker's
    `_page_N_Figure_M.jpeg` filename convention. Returns None on miss."""
    m = _PAGE_FROM_FILENAME_RE.search(filename)
    return int(m.group(1)) if m else None


def _classify_one_image(prompt: str, img: "Image.Image",
                        strategy: str, page_idx: Optional[int]) -> Optional[int]:
    """Single classification call honouring the strategy's per-call knobs:
    page-prior (prompt augmentation) and vote (3× majority). Returns
    the VLM-chosen index into the captions list (1..N), 0 for NONE,
    or None if the call failed / produced no digit."""
    full_prompt = prompt
    if "page-prior" in strategy and page_idx is not None:
        full_prompt = (
            prompt
            + f"\nThis image was extracted from page {page_idx + 1} of the PDF "
            f"(1-indexed). Captions for figures appearing on or near that "
            f"page are most likely to match.\n"
        )

    def _one_call() -> Optional[int]:
        # max_tokens=24, not 5: well-aligned models (Qwen3-VL, gpt-4.1)
        # reply with a bare digit and 5 tokens fits, but verbose-default
        # instruction-tuned models (Gemma 4 IT, larger Llamas) sometimes
        # pad with leading prose ("**3**", "Caption 3", "The image shows
        # ...") and 5 tokens truncates before any digit appears, leaving
        # the parser with nothing -- the image then drops as "no match".
        # 24 tokens covers a short bold-formatted answer or single-clause
        # preamble; output is still tiny so the latency cost is zero.
        reply = vlm(full_prompt, img, max_tokens=24)
        if not reply:
            return None
        digits = re.findall(r"\d+", reply)
        return int(digits[0]) if digits else None

    if "vote" in strategy:
        votes = [v for v in (_one_call() for _ in range(3)) if v is not None]
        if not votes:
            return None
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]

    return _one_call()


def _reclassify_excluding(prompt: str, img: "Image.Image",
                          excluded_idx, n_captions: int,
                          strategy: str, page_idx: Optional[int]) -> Optional[int]:
    """Re-prompt the classifier to pick a caption OUTSIDE `excluded_idx`.

    `excluded_idx` may be either an int (single exclusion -- the
    original dup-detect callers) or a set/list of ints (page-locality
    iteration accumulating multiple rejected candidates). Returns
    the new choice index, or None if the VLM ignores the exclusion
    or answers 0 / picks an excluded option.

    Honours strategy's per-call augmentations (page-prior, vote)."""
    if isinstance(excluded_idx, int):
        excluded = {excluded_idx}
    else:
        excluded = set(excluded_idx)
    if not excluded:
        return _classify_one_image(prompt, img, strategy, page_idx)
    remaining = ", ".join(
        str(i) for i in range(1, n_captions + 1) if i not in excluded
    )
    excluded_list = ", ".join(str(i) for i in sorted(excluded))
    augmented = prompt + (
        f"\nIMPORTANT: option(s) {excluded_list} are already assigned "
        f"to other images and are no longer available. Pick the BEST "
        f"remaining option for THIS image from: 0, {remaining}. "
        f"Do not pick any of {excluded_list}.\n"
    )
    new_choice = _classify_one_image(augmented, img, strategy, page_idx)
    if new_choice in excluded:
        return None  # VLM ignored the constraint; treat as no match
    return new_choice


def _resolve_duplicates(items: list[tuple[str, Optional[int], "Image.Image", Optional[int]]],
                        captions: list[tuple[str, str]],
                        prompt: str,
                        strategy: str,
                        rejected_history: dict) -> list[tuple[str, Optional[int], "Image.Image", Optional[int]]]:
    """For any caption index claimed by ≥2 images, run a YES/NO
    confirmation per claiming image. NO answers are reclassified
    with the contested option removed (so a demoted image can still
    land on its correct caption — common when two figures sit on the
    same page and look alike to the VLM). If ≥2 images still confirm
    YES, keep the one whose page index is closest to the caption's
    position in the list and reclassify the others.

    items: list of (filename, choice_idx_or_None, img, page_idx_or_None)
    captions: same list passed to _caption_figures_matched.
    prompt: the full PAIR_PROMPT_HEADER+lines prompt, reused for the
            reclassify-excluding pass.
    strategy: current FIGMATCH_STRATEGY (so reclassification honours
              page-prior / vote if set).
    rejected_history: filename -> set of choice_idx that the VLM said
            NO to in this or any earlier _resolve_duplicates call.
            Used to break two-caption oscillation loops: if
            reclassify-excluding sends an image back to a caption it
            was already rejected from, drop it (set choice=0) instead
            of cycling. Caller maintains across loop iterations.
    Returns a possibly-mutated copy of items."""
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i, (_, choice, _, _) in enumerate(items):
        if choice and 1 <= choice <= len(captions):
            groups[choice].append(i)

    out = list(items)
    for choice_idx, indices in groups.items():
        if len(indices) < 2:
            continue
        fig_id, title = captions[choice_idx - 1]
        re_prompt = (
            f"This image was extracted from a scientific article. {len(indices)} "
            f"different images from the same article were initially classified "
            f"as the same figure: 'Figure {fig_id}: {title[:80]}'. Examine THIS "
            f"image specifically and confirm whether it depicts that figure. "
            f"Respond with the single word YES or NO."
        )
        confirmed: list[int] = []
        for idx in indices:
            filename, _, img, _ = out[idx]
            reply = vlm(re_prompt, img, max_tokens=5)
            yes = bool(reply and reply.strip().upper().startswith("Y"))
            log.info("dup-detect re-classify %s vs Fig %s: %r → %s",
                     filename, fig_id, reply, "KEEP" if yes else "DEMOTE")
            if yes:
                confirmed.append(idx)
            else:
                rejected_history.setdefault(filename, set()).add(choice_idx)

        # If the re-classifier still leaves ≥2 confirmed, fall back to
        # page-distance: caption at list-index choice_idx-1 corresponds
        # roughly to that ordinal position in the document, so prefer
        # the image whose page is closest.
        if len(confirmed) > 1:
            confirmed.sort(
                key=lambda i: abs((out[i][3] or 0) - (choice_idx - 1))
            )
            confirmed = confirmed[:1]
            log.info("dup-detect tiebreak Fig %s → keep %s",
                     fig_id, out[confirmed[0]][0])

        # If no image confirmed, fall back to keeping the original first-seen.
        if not confirmed:
            confirmed = indices[:1]
            log.info("dup-detect: 0 confirmed for Fig %s; falling back to first-seen %s",
                     fig_id, out[confirmed[0]][0])

        # Demoted images get a second classification pass with the
        # contested option explicitly excluded — so a real figure that
        # happens to look like the contested one can still find its
        # actual caption (e.g. two side-by-side figures on the same
        # page, both initially classified as the same fig_id).
        for idx in indices:
            if idx in confirmed:
                continue
            fname, _, img, page_idx = out[idx]
            new_choice = _reclassify_excluding(
                prompt, img, choice_idx, len(captions), strategy, page_idx,
            )
            # Cycle break: if the new choice is a caption the VLM has
            # already rejected for this image, drop it instead of
            # forcing a re-assignment. Without this, papers with N
            # images and < N captions (often caused by an upstream
            # caption-extraction miss) loop the demoted image
            # between the two contested captions until the iteration
            # cap, leaving it on a caption the VLM said NO to.
            prior_rejects = rejected_history.get(fname, set())
            if (new_choice
                    and 1 <= new_choice <= len(captions)
                    and new_choice in prior_rejects):
                log.info("dup-detect reclassify-excluding %s (was Fig %s, "
                         "excluded): → Fig %s previously rejected, "
                         "dropping as unmatched",
                         fname, fig_id, captions[new_choice - 1][0])
                new_choice = 0
            elif new_choice and 1 <= new_choice <= len(captions):
                new_fig_id = captions[new_choice - 1][0]
                log.info("dup-detect reclassify-excluding %s (was Fig %s, "
                         "excluded): → Fig %s",
                         fname, fig_id, new_fig_id)
            else:
                log.info("dup-detect reclassify-excluding %s (was Fig %s, "
                         "excluded): → DROP",
                         fname, fig_id)
            out[idx] = (fname, new_choice, img, page_idx)
    return out


_PAGE_LOCALITY_TOLERANCE = 1  # |image_page - caption_page| > this is a mismatch
_PAGE_LOCALITY_MAX_ITERS = 4  # cap reclassify attempts per image


def _is_page_mismatch(choice_idx: int, page_idx: int,
                      caption_page_map: dict,
                      page_to_cap_idxs: dict) -> bool:
    """A caption assignment is a 'page mismatch' when:
      (a) |image_page - caption_page| exceeds tolerance, OR
      (b) caption_page differs from image_page AND another caption
          lives on the image's source page (a co-located alternative
          must be preferred).
    Returns False (no mismatch) when caption_page is unknown."""
    cap_page = caption_page_map.get(choice_idx)
    if cap_page is None:
        return False
    diff = abs(page_idx - cap_page)
    if diff > _PAGE_LOCALITY_TOLERANCE:
        return True
    same_page_alts = page_to_cap_idxs.get(page_idx, set()) - {choice_idx}
    if diff > 0 and len(same_page_alts) > 0:
        return True
    return False


def _apply_page_locality(
    items: list,
    captions: list[tuple[str, str]],
    caption_page_map: dict,
    prompt: str,
    strategy: str,
) -> list:
    """Demote and re-classify any image whose VLM-assigned caption
    lives on a PDF page far from the image's source page.

    Iterates per image: each rejected choice is added to a cumulative
    'excluded' set, then `_reclassify_excluding` is called with the
    full set; the new choice is re-validated by the same rule. Caps
    at _PAGE_LOCALITY_MAX_ITERS reclassify calls to bound cost. If
    no co-located caption can be found, drops the image (sets
    choice to None).

    Two failure modes addressed:
      (a) diff > 1 (e.g. Luo image on doc[10] assigned to caption on
          doc[8]): exceeds tolerance, reject.
      (b) diff == 1 BUT a co-located caption exists (Luo image on
          doc[10] assigned to Fig 10 on doc[9], when Fig 11's caption
          is on doc[10]): tolerance lets it through but co-locality
          rejects it.
    """
    out: list = []
    page_to_cap_idxs: dict[int, set] = {}
    for cap_idx, cap_page in caption_page_map.items():
        if cap_page is not None:
            page_to_cap_idxs.setdefault(cap_page, set()).add(cap_idx)
    for filename, choice_idx, img, page_idx in items:
        if not (choice_idx is not None and choice_idx > 0
                and page_idx is not None):
            out.append((filename, choice_idx, img, page_idx))
            continue
        excluded: set = set()
        cur = choice_idx
        iters = 0
        while iters < _PAGE_LOCALITY_MAX_ITERS:
            if not _is_page_mismatch(cur, page_idx,
                                     caption_page_map, page_to_cap_idxs):
                break
            excluded.add(cur)
            cap_page = caption_page_map.get(cur, None)
            log.info("Page-locality reject (iter %d): %s -> Fig %s "
                     "(image_page=%d, caption_page=%s); reclassifying "
                     "with excluded=%s",
                     iters + 1, filename, captions[cur - 1][0],
                     page_idx + 1,
                     (cap_page + 1) if cap_page is not None else "?",
                     sorted(excluded))
            new_choice = _reclassify_excluding(
                prompt, img, excluded, len(captions),
                strategy, page_idx)
            if new_choice is None or new_choice == 0 or new_choice in excluded:
                cur = None
                break
            cur = new_choice
            iters += 1
        else:
            # Hit iteration cap with mismatch unresolved. Drop.
            if cur is not None and _is_page_mismatch(
                    cur, page_idx, caption_page_map, page_to_cap_idxs):
                cur = None
        out.append((filename, cur, img, page_idx))
    return out


def _caption_figures_matched(md: str, captions: list[tuple[str, str]],
                             assets_dir: Path, report: QualityReport,
                             use_vlm: bool,
                             doc: "Optional[fitz.Document]" = None) -> str:
    # Build a compact numbered list once; 0 is the NONE option. The list
    # index (1..N) is what the VLM returns; the figure id (e.g. "3" or
    # "S1") is what we display in alt-text and record in scoring.
    lines = ["0. NONE -- not a scientific figure, or does not match any listed caption"]
    for idx, (fig_id, title) in enumerate(captions, start=1):
        lines.append(f"{idx}. Figure {fig_id}: {title[:80]}")
    prompt = PAIR_PROMPT_HEADER + "\n".join(lines) + "\n"
    strategy = FIGMATCH_STRATEGY

    # Resolve each caption's PDF page (best-effort). Used post-
    # classification to catch image/caption page mismatches: e.g.,
    # marker extracts an image from doc[10] but the VLM assigns it
    # to a caption whose actual PDF page is doc[9]. The mismatched
    # assignment gets demoted and re-classified with the offending
    # caption excluded. Skip caption_page lookups when no doc
    # handle is available (text-only path or doc not threaded).
    caption_page_map: dict[int, Optional[int]] = {}
    if doc is not None:
        for idx, (fig_id, _title) in enumerate(captions, start=1):
            cp = _find_caption_page(doc, f"Fig. {fig_id}")
            if cp is None:
                cp = _find_caption_page(doc, f"Figure {fig_id}")
            caption_page_map[idx] = cp
        n_known = sum(1 for v in caption_page_map.values() if v is not None)
        log.info("Page-locality: resolved caption page for %d/%d figures",
                 n_known, len(captions))

    scored: list[FigureScore] = []
    drops: list[Path] = []

    # Pre-classification flags per filename so the markdown rewrite can
    # use post-resolution decisions. Values:
    #   "skip"       — table image inserted by process_tables; leave alone
    #   "missing"    — image referenced in markdown but missing from disk
    #   "load_error" — image present but Image.open raised
    #   choice_idx   — int in 0..len(captions); 0 / None means "no match"
    decisions: dict[str, object] = {}
    items: list[tuple[str, Optional[int], "Image.Image", Optional[int]]] = []
    item_imgs: dict[str, "Image.Image"] = {}

    # Pass 1: enumerate images and classify each one.
    for match in IMAGE_RE.finditer(md):
        alt, path = match.group(1), match.group(2)
        filename = os.path.basename(path)
        if filename in decisions:
            continue
        if alt.startswith("Table ") and "(page" in alt:
            decisions[filename] = "skip"
            continue
        img_path = assets_dir / filename
        if not img_path.exists():
            decisions[filename] = "missing"
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            decisions[filename] = "load_error"
            continue
        log.debug("Classifying %s (%dx%d, %d bytes)",
                  filename, img.width, img.height, img_path.stat().st_size)
        page_idx = _parse_page_idx(filename)
        choice_idx = _classify_one_image(prompt, img, strategy, page_idx)
        items.append((filename, choice_idx, img, page_idx))
        item_imgs[filename] = img

    # Pass 1.5: page-locality validation. Demote and re-classify
    # any image whose VLM-assigned caption lives on a far-away PDF
    # page. See _apply_page_locality.
    if caption_page_map:
        items = _apply_page_locality(
            items, captions, caption_page_map, prompt, strategy)

    # Pass 2: optional duplicate resolution. Iterate until stable —
    # reclassify-excluding can produce new duplicates (e.g. an image
    # demoted from Fig 10 lands on Fig 8 which was already taken),
    # so a single pass is insufficient on papers with cascading
    # mis-classifications. Capped at 3 iterations to bound cost; in
    # practice 2 are usually enough. The shared rejected_history
    # dict accumulates per-image VLM rejections across iterations so
    # _resolve_duplicates can break two-caption oscillation loops
    # (image bouncing between Fig A and Fig B with VLM rejecting both).
    if "dup-detect" in strategy:
        rejected_history: dict[str, set] = {}
        for _ in range(3):
            prev_choices = [it[1] for it in items]
            items = _resolve_duplicates(items, captions, prompt, strategy,
                                        rejected_history)
            new_choices = [it[1] for it in items]
            if new_choices == prev_choices:
                break

    # Materialise final per-filename decisions from items.
    for filename, choice_idx, _, _ in items:
        decisions[filename] = choice_idx if choice_idx is not None else 0

    # Pass 3: rewrite markdown using the resolved decisions.
    def repl(m):
        alt, path = m.group(1), m.group(2)
        filename = os.path.basename(path)
        d = decisions.get(filename)

        if d == "skip":
            return m.group(0)
        if d == "missing":
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=_score_figure(False, 0, use_vlm, True),
            ))
            return m.group(0)
        if d == "load_error":
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=0.3,
            ))
            return m.group(0)

        choice_idx = d if isinstance(d, int) else 0
        if choice_idx and 1 <= choice_idx <= len(captions):
            fig_id, title = captions[choice_idx - 1]
            new_alt = f"Figure {fig_id} | {title}"
            log.info("Matched %s -> Figure %s", filename, fig_id)
            scored.append(FigureScore(
                filename=filename,
                caption_produced=True,
                caption_length=len(title.split()),
                score=1.0,
                matched_figure=fig_id,
            ))
            return f"![{new_alt}](assets/{filename})"

        log.info("Dropped %s (no caption match, deleting)", filename)
        img_path = assets_dir / filename
        drops.append(img_path)
        scored.append(FigureScore(
            filename=filename,
            caption_produced=False,
            caption_length=0,
            score=0.8,
            dropped=True,
        ))
        return ""

    new_md = IMAGE_RE.sub(repl, md)
    for p in drops:
        try:
            p.unlink()
        except OSError as e:
            log.warning("Could not delete %s: %s", p, e)
    report.figures.extend(scored)
    return new_md


def _caption_figures_freeform(md: str, assets_dir: Path, report: QualityReport,
                              use_vlm: bool) -> str:
    scored: list[FigureScore] = []

    def repl(m):
        alt, path = m.group(1), m.group(2)
        filename = os.path.basename(path)

        # Don't touch the table JPEGs we inserted in process_tables.
        if alt.startswith("Table ") and "(page" in alt:
            return m.group(0)

        img_path = assets_dir / filename
        file_missing = not img_path.exists()

        # Already has meaningful alt-text from marker; keep it, score generously.
        if alt and alt.strip().lower() not in {"", "image", "figure"}:
            words = len(alt.split())
            scored.append(FigureScore(
                filename=filename,
                caption_produced=True,
                caption_length=words,
                score=_score_figure(True, words, use_vlm, file_missing),
            ))
            return m.group(0)

        if file_missing:
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=_score_figure(False, 0, use_vlm, True),
            ))
            return m.group(0)

        if not use_vlm:
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=_score_figure(False, 0, False, False),
            ))
            return m.group(0)

        file_bytes = img_path.stat().st_size
        if file_bytes < CAPTION_MIN_BYTES:
            log.info("Skipped %s (%d bytes < %d, likely decoration)",
                     filename, file_bytes, CAPTION_MIN_BYTES)
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=_score_figure(False, 0, use_vlm, False),
            ))
            return m.group(0)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=0.3,
            ))
            return m.group(0)

        log.info("Captioning %s (%dx%d, %d bytes)",
                 filename, img.width, img.height, file_bytes)
        caption = vlm(CAPTION_PROMPT, img, max_tokens=80)
        if not caption:
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=_score_figure(False, 0, use_vlm, False),
            ))
            return m.group(0)

        caption = caption.splitlines()[0].replace("[", "").replace("]", "").strip()
        if caption.strip(".!").upper() == "SKIP":
            log.info("VLM declined to caption %s (returned SKIP)", filename)
            scored.append(FigureScore(
                filename=filename,
                caption_produced=False,
                caption_length=0,
                score=_score_figure(False, 0, use_vlm, False),
            ))
            return m.group(0)
        words = len(caption.split())
        log.info("Captioned %s: %s", filename, caption)
        scored.append(FigureScore(
            filename=filename,
            caption_produced=True,
            caption_length=words,
            score=_score_figure(True, words, use_vlm, False),
        ))
        return f"![{caption}]({path})"

    new_md = IMAGE_RE.sub(repl, md)
    report.figures.extend(scored)
    return new_md


# --- Hook 0: citation synthesis ---------------------------------------------

def extract_citation(doc: fitz.Document) -> Optional[str]:
    """Ask the VLM to synthesize a bibliographic citation from the first
    page of the PDF in the journal's own reference style, with a DOI /
    publisher URL appended if visible. Returns a single-line markdown
    string ready to prepend to the body, or None if the VLM could not
    produce a usable citation."""
    if len(doc) == 0:
        return None
    try:
        img = render_page(doc, 0, dpi=PAGE_DPI)
    except Exception as e:
        log.warning("Could not render page 1 for citation: %s", e)
        return None
    log.info("Extracting citation from page 1 (%dx%d)", img.width, img.height)
    reply = vlm(CITATION_PROMPT, img, max_tokens=500)
    if not reply:
        log.warning("VLM returned no citation response")
        return None
    citation = reply.splitlines()[0].strip()
    if citation.strip(".!").upper() == "SKIP":
        log.info("VLM declined to synthesize citation (returned SKIP)")
        return None
    log.info("Cite: %s", citation[:120] + ("..." if len(citation) > 120 else ""))
    return citation


# --- Hook 3: sparse-page rescue ---------------------------------------------

def rescue_sparse_pages(md: str, doc: fitz.Document) -> tuple[str, set[int]]:
    idxs = sparse_pages(doc)
    if not idxs:
        return md, set()
    log.info("Rescuing %d sparse page(s): %s", len(idxs), [i + 1 for i in idxs])
    parts = [md, "\n\n---\n\n## VLM page rescues\n"]
    rescued: set[int] = set()
    for i in idxs:
        out = vlm(PAGE_PROMPT, render_page(doc, i), max_tokens=4000)
        if not out:
            continue
        parts.append(f"\n### Page {i + 1}\n\n{out}\n")
        rescued.add(i)
    return "".join(parts), rescued


def _populate_data_repos(md: str, report: "QualityReport") -> None:
    """Scan `md` for data-repository links; attach to `report.data_repos`.
    Optionally enriches each entry with a one-shot API summary when
    FETCH_DATA_REPOS is on. Best-effort: any failure logs and continues
    with whatever links were already detected."""
    if not DATA_REPOS:
        return
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
    log.info("Detected %d data-repository link(s): %s", len(links),
             ", ".join(f"{l.repository} ({l.confidence})" for l in links))
    if FETCH_DATA_REPOS and not OFFLINE:
        try:
            import requests as _requests
            with _requests.Session() as session:
                for l in links:
                    fetch_summary(l, session=session,
                                  timeout=FETCH_DATA_REPOS_TIMEOUT)
                    log.info("  %s -> %s",
                             l.url,
                             l.fetch_status or "(no fetch)")
        except Exception as e:
            log.warning("Data-repo fetch failed (%s); recording links "
                        "without summaries", e)
    elif FETCH_DATA_REPOS and OFFLINE:
        log.info("--offline: data-repo enrichment skipped; %d link(s) "
                 "recorded without API summaries", len(links))
    report.data_repos = links


def _populate_references_score(md: str, report: "QualityReport",
                               doc: "Optional[fitz.Document]" = None) -> None:
    """Compute the references-section quality score for the final body
    markdown and attach to `report.references`. Pure CPU, no VLM.

    When `doc` is provided, also computes the Phase-2 layout
    fingerprint (column count, marker style, refs page index) and
    attaches it to `report.references.layout`. Layout detection is
    cheap (PyMuPDF block coordinates only) and useful as a diagnostic
    even when no rescue fires.

    Best-effort: any failure logs at DEBUG and leaves
    report.references unset."""
    journal_slug = (getattr(report.metadata, "journal_slug", None)
                    if report.metadata is not None else None)
    try:
        report.references = score_references(md, journal_slug)
        if doc is not None:
            try:
                report.references.layout = _detect_refs_layout(doc, md)
            except Exception as e:
                log.debug("_detect_refs_layout failed: %s", e)
        layout = report.references.layout
        log.info("References: score=%.2f, style=%s, entries=%d, "
                 "section_count=%d, journal_slug=%s, columns=%s",
                 report.references.score,
                 report.references.style,
                 report.references.entry_count,
                 report.references.section_count,
                 journal_slug or "unknown",
                 layout.columns if layout else "n/a")
    except Exception as e:
        log.debug("score_references failed: %s", e)


def _write_meta_json(out_dir: Path, stem: str,
                     report: "QualityReport") -> Path:
    """Write a JSON sidecar mirroring the markdown frontmatter at
    <out_dir>/<stem>.meta.json. The sidecar is a derived artifact: the .md
    frontmatter is canonical, but the JSON is what batch / corpus consumers
    actually want (jq-friendly, no YAML parser needed)."""
    out_path = out_dir / (stem + ".meta.json")
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str,
                   ensure_ascii=False)
    )
    return out_path


# --- HDF5 bundle ------------------------------------------------------------

def write_hdf5_bundle(bundle_path: Path,
                      main_md: Path, main_report: "QualityReport",
                      main_pdf: Path,
                      si_md: Optional[Path] = None,
                      si_report: Optional["QualityReport"] = None,
                      si_pdf: Optional[Path] = None) -> None:
    """Pack the main markdown, optional supplement markdown, and every
    file under the shared assets/ directory into a single compressed
    HDF5 bundle.

    Layout:
        /                   attrs: schema_version, tool, created_at, vlm_model
        /main/              attrs: source_pdf, markdown_path, meta_json_path,
                                   overall, grade
            markdown        utf-8 string, gzipped
            meta_json       utf-8 string, gzipped (structured frontmatter mirror)
        /supplement/        (only when si_md is provided)
            markdown        utf-8 string, gzipped
            meta_json       utf-8 string, gzipped
            + same attrs as /main
        /assets/
            <name>.jpg      uint8 byte array (already compressed, no gzip)
            <name>.md       utf-8 string, gzipped
            ...
    """
    import h5py
    import numpy as np
    from datetime import datetime, timezone

    assets_dir = main_md.parent / "assets"

    with h5py.File(bundle_path, "w") as h5:
        h5.attrs["schema_version"] = 1
        h5.attrs["tool"] = "paper2md.py"
        h5.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["vlm_model"] = VLM_MODEL

        str_dtype = h5py.string_dtype("utf-8")

        def _doc_group(name: str, md_path: Path, report: "QualityReport",
                       src_pdf: Path):
            g = h5.create_group(name)
            # Wrap strings in a 1-element list to make them 1D, so HDF5
            # can chunk + gzip them. Consumers read with `.asstr()[0]`.
            g.create_dataset("markdown", data=[md_path.read_text()],
                             dtype=str_dtype,
                             compression="gzip", compression_opts=9)
            # Structured frontmatter mirror. Read fresh from disk so a
            # hand-edit of the .meta.json is preserved through repack;
            # if missing (e.g. --no-quality), fall back to regenerating
            # from the in-memory report.
            meta_name = md_path.stem + ".meta.json"
            meta_path = md_path.parent / meta_name
            if meta_path.is_file():
                meta_text = meta_path.read_text()
            else:
                meta_text = json.dumps(report.to_dict(), indent=2,
                                       default=str, ensure_ascii=False)
            g.create_dataset("meta_json", data=[meta_text],
                             dtype=str_dtype,
                             compression="gzip", compression_opts=9)
            g.attrs["source_pdf"] = src_pdf.name
            g.attrs["markdown_path"] = md_path.name
            g.attrs["meta_json_path"] = meta_name
            g.attrs["overall"] = float(report.overall())
            g.attrs["grade"] = report.grade()

        _doc_group("main", main_md, main_report, main_pdf)
        if si_md is not None and si_report is not None and si_pdf is not None:
            _doc_group("supplement", si_md, si_report, si_pdf)

        g_assets = h5.create_group("assets")
        kept = 0
        asset_iter = sorted(assets_dir.iterdir()) if assets_dir.is_dir() else []
        for asset in asset_iter:
            if not asset.is_file():
                continue
            suffix = asset.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png"}:
                data = np.frombuffer(asset.read_bytes(), dtype=np.uint8)
                g_assets.create_dataset(asset.name, data=data)
                kept += 1
            elif suffix == ".md":
                g_assets.create_dataset(asset.name, data=[asset.read_text()],
                                        dtype=str_dtype,
                                        compression="gzip", compression_opts=9)
                kept += 1
            else:
                log.debug("Skipping non-standard asset: %s", asset.name)
        g_assets.attrs["count"] = kept

    size_kb = bundle_path.stat().st_size / 1024
    log.info("Wrote HDF5 bundle: %s (%.1f KB, %d assets)",
             bundle_path, size_kb, kept)


# --- Top-level driver -------------------------------------------------------


def _clean_or_check_outputs(out_dir: Path, stem: str, asset_prefix: str,
                            clean: bool) -> None:
    """Detect outputs from a prior run on the same (out_dir, stem,
    asset_prefix) tuple. With clean=True, delete them so the new run
    writes into a pristine directory. Otherwise log a warning so the
    user knows stale files remain alongside the fresh output -- e.g.
    after re-running with different flags, where this run's table
    JPEG is `table_p5_2.jpg` but the prior run's `table_p4_2.jpg`
    sits in assets/ confusing downstream comparison.

    Targets, conservatively scoped so stray user files in out_dir are
    untouched:
      - <out_dir>/<stem>.md
      - <out_dir>/<stem>.meta.json
      - <out_dir>/<stem>.h5
      - <out_dir>/<stem>_oa_source.pdf  (--prefer-oa-source artifact)
      - <out_dir>/assets/<asset_prefix>_page_*       (marker images)
      - <out_dir>/assets/<asset_prefix>table_*       (hook 1 crops + sidecars)
      - <out_dir>/assets/<asset_prefix>page_*        (orphan-rescue artifacts)

    The asset_prefix scoping means a supplement's --clean only touches
    its own si_-prefixed files and leaves the main paper's assets
    intact (and vice versa)."""
    targets: list[Path] = []
    for ext in (".md", ".meta.json", ".h5"):
        p = out_dir / (stem + ext)
        if p.exists():
            targets.append(p)
    oa = out_dir / (stem + "_oa_source.pdf")
    if oa.exists():
        targets.append(oa)
    assets_dir = out_dir / "assets"
    if assets_dir.is_dir():
        for p in assets_dir.iterdir():
            if not p.is_file():
                continue
            n = p.name
            if not n.startswith(asset_prefix):
                continue
            tail = n[len(asset_prefix):]
            if (tail.startswith("_page_")
                    or tail.startswith("table_")
                    or tail.startswith("page_")):
                targets.append(p)
    if not targets:
        return
    if clean:
        for p in targets:
            try:
                p.unlink()
            except OSError as e:
                log.warning("--clean: could not delete %s: %s", p, e)
        log.info("--clean: removed %d pre-existing output file(s) "
                 "for stem=%r prefix=%r",
                 len(targets), stem, asset_prefix)
    else:
        sample = ", ".join(p.name for p in targets[:3])
        more = "" if len(targets) <= 3 else f", +{len(targets) - 3} more"
        log.warning("Output dir already has %d file(s) from a prior run "
                    "(%s%s). They'll persist alongside this run's output "
                    "and may confuse comparison. Pass --clean to remove "
                    "pre-existing outputs automatically.",
                    len(targets), sample, more)


def convert_text_only(pdf_path: Path, out_dir: Path,
                      emit_quality: bool = True,
                      emit_citation: bool = True,
                      metadata: object = None,
                      output_stem: Optional[str] = None,
                      clean: bool = False,
                      run_info: Optional[RunInfo] = None,
                      user_annotations: Optional[UserAnnotations] = None) -> tuple[Path, QualityReport]:
    """VLM-first conversion: render each PDF page and ask the VLM to
    produce clean markdown. No marker, no assets folder, no per-table
    or per-figure hooks. Best for quick text-only extraction / search
    and RAG ingestion where visual content isn't needed.

    Returns the same (Path, QualityReport) shape as `convert()` so
    downstream logic (HDF5 bundle, quality threshold) works unchanged.
    Each page gets a synthetic PageScore: 1.0 if the VLM returned
    content, 0.3 if the call failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_or_check_outputs(out_dir, output_stem or pdf_path.stem,
                            asset_prefix="", clean=clean)

    t0 = time.time()
    if run_info is not None:
        from datetime import datetime as _dt, timezone as _tz
        run_info.started_at = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = QualityReport(vlm_enabled=True, metadata=metadata,
                           run_info=run_info,
                           user_annotations=user_annotations)
    parts: list[str] = []

    with fitz.open(str(pdf_path)) as doc:
        if emit_citation:
            citation = extract_citation(doc)
            if citation:
                parts.append(citation + "\n")

        log.info("Text-only: VLM-converting %d page(s) of %s",
                 doc.page_count, pdf_path.name)
        for i in range(doc.page_count):
            log.info("  page %d/%d", i + 1, doc.page_count)
            img = render_page(doc, i)
            page_md = vlm(PAGE_PROMPT, img, max_tokens=4000)
            if page_md:
                parts.append(page_md + "\n")
            else:
                parts.append(f"\n<!-- page {i+1}: VLM returned no output -->\n")
            # Note: char_density is set to 0 in this VLM-direct path
            # because we never asked PyMuPDF for the text layer (the
            # whole point of --text-only is to skip surya/marker).
            # Density isn't a quality signal here either.
            report.pages.append(PageScore(
                page=i + 1, char_density=0.0,
                sparse=False, rescued=bool(page_md),
            ))

    md = "\n".join(parts)
    md, stripped = strip_line_numbers(md)
    if stripped:
        log.info("Stripped %d stray line-number lines", stripped)
    if STRIP_RUNNING_FOOTERS:
        md, footer_stripped = strip_running_footers(md)
        if footer_stripped:
            log.info("Stripped %d running-footer line(s) (et al. N of M)",
                     footer_stripped)
    if STRIP_PAGE_HEADERS:
        md, header_stripped = strip_journal_page_headers(md)
        if header_stripped:
            log.info("Stripped %d journal page-header line(s) (LETTERS, "
                     "NATURE ... DOI:)", header_stripped)
    if STRIP_PUBLISHER_STAMPS:
        md, stamp_stripped = strip_publisher_stamps(md)
        if stamp_stripped:
            log.info("Stripped %d per-page publisher download-watermark "
                     "line(s) (Wiley Online Library)", stamp_stripped)
    if STRIP_MATH_LABELS:
        md, n_labels = strip_math_labels(md)
        if n_labels:
            log.info("Stripped %d KaTeX-incompatible \\label{...} from math",
                     n_labels)
    if NORMALISE_REFS:
        md, ref_counts = normalise_span_anchored_refs(md)
        if any(ref_counts.values()):
            log.info("Normalised refs: %d re-bulletted, %d DOI links rejoined, "
                     "%d refhub wrappers stripped",
                     ref_counts["bulletted"], ref_counts["dois_rejoined"],
                     ref_counts["refhub_stripped"])
    if FIX_REFERENCES:
        md, repaired = fix_reference_numbering(md)
        if repaired:
            log.info("Repaired %d reference numbering(s)", repaired)
    if INJECT_ORPHAN_REFS:
        md, n_orphans = inject_orphan_ref_clusters(md)
        if n_orphans:
            log.info("Found %d orphan reference cluster(s) outside the "
                     "## References heading; synthesized a heading so "
                     "merge_reference_sections can consolidate", n_orphans)
    if MERGE_REFERENCES:
        md, n_merged = merge_reference_sections(md)
        if n_merged > 1:
            log.info("Merged %d reference section(s) into a single "
                     "## References at end of document", n_merged)
    if TIDY_REFS:
        md, n_tidied = normalise_references_section(md)
        if n_tidied:
            log.info("Tidied references section: merged/extracted %d "
                     "continuation/addendum line(s)", n_tidied)
    _populate_data_repos(md, report)
    # Always-on rescue: insert `## References` heading for headless
    # refs sections. Runs before scoring so the score reflects the
    # inserted heading. APS-specific path runs first (gated on
    # journal_slug); the generic path covers all journals as a
    # fallback. Both check _REFS_HEADING_FOR_SCORE_RE for idempotency
    # so re-runs and double-fires are safe.
    md = _rescue_aps_missing_refs_heading(md, report)
    md = _rescue_generic_missing_refs_heading(md, report)
    # text_only path: no doc handle needed (layout detection skipped),
    # and VLM-driven rescues silently no-op because use_vlm=False.
    # The deterministic APS-bleed rescue still runs if applicable.
    _populate_references_score(md, report, doc=None)
    md = rescue_journal_refs(md, None, report, use_vlm=False,
                             out_dir=out_dir)
    if (report.references is not None
            and report.references.rescue_decision == "spliced"):
        pre_score = report.references.score_pre_rescue
        pre_rescue_applied = report.references.rescue_applied
        pre_rescue_decision = report.references.rescue_decision
        pre_score_post = report.references.score_post_rescue
        _populate_references_score(md, report, doc=None)
        report.references.score_pre_rescue = pre_score
        report.references.score_post_rescue = pre_score_post
        report.references.rescue_applied = pre_rescue_applied
        report.references.rescue_decision = pre_rescue_decision
    # Always-on API-fallback append (Crossref -> OpenAlex). Skips
    # silently when DOI absent, score is healthy, or network fails.
    md = _rescue_append_api_refs(md, report, allow_network=not OFFLINE)
    if emit_quality:
        if report.run_info is not None:
            report.run_info.elapsed_sec = round(time.time() - t0, 1)
        md = report.to_frontmatter() + md

    out_md = out_dir / ((output_stem or pdf_path.stem) + ".md")
    out_md.write_text(md)
    if emit_quality:
        _write_meta_json(out_dir, output_stem or pdf_path.stem, report)
    log.info("Wrote %s (text-only, %d pages, quality %.2f / %s)",
             out_md, len(report.pages), report.overall(), report.grade())
    _emit_run_summary(report, out_md, vlm_rewrite_tables=False,
                      use_vlm=True)
    return out_md, report


def convert(pdf_path: Path, out_dir: Path, use_vlm: bool,
            table_finder: str = "docling",
            emit_quality: bool = True,
            emit_citation: bool = True,
            asset_prefix: str = "",
            metadata: object = None,
            output_stem: Optional[str] = None,
            vlm_rewrite_tables: bool = False,
            rescue_sparse_pages_flag: bool = False,
            table_workers: int = 1,
            force_ocr: bool = False,
            clean: bool = False,
            layout_source: str = "mineru",
            rescue_mineru_orphan_captions: bool = True,
            rescue_mineru_subpanels: bool = True,
            strip_chart_details: bool = False,
            mineru_backend: str = "pipeline",
            mineru_extra_args: tuple = (),
            skip_mineru_run: bool = False,
            mineru_existing_dir: Optional[Path] = None,
            run_info: Optional[RunInfo] = None,
            user_annotations: Optional[UserAnnotations] = None,
            vlm_tables_force: bool = False) -> tuple[Path, QualityReport]:
    """Convert one PDF to markdown.

    `asset_prefix` prepends every filename written under `out_dir/assets/`
    (marker-extracted images, table JPEGs, table sidecars). Use a non-empty
    prefix (e.g. "si_") when a secondary PDF (supplement) shares `out_dir`
    with its main paper so their assets don't collide.

    `force_ocr=True` forwards to marker as `force_ocr` config -- skips
    the embedded text layer entirely so every page goes through surya
    OCR. Use on pre-2000 / scanned papers whose publisher OCR is
    introducing artifacts (spurious bold paragraphs, line-broken
    table headers). See run_marker for the cost trade-off."""
    t0 = time.time()
    if run_info is not None:
        from datetime import datetime as _dt, timezone as _tz
        run_info.started_at = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_or_check_outputs(out_dir, output_stem or pdf_path.stem,
                            asset_prefix=asset_prefix, clean=clean)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    if layout_source not in ("marker", "mineru", "hybrid"):
        raise ValueError(
            f"layout_source must be 'marker', 'mineru', or 'hybrid', "
            f"got {layout_source!r}")
    use_mineru_layout = (layout_source == "mineru")
    use_hybrid_layout = (layout_source == "hybrid")
    # Hybrid produces marker-flavored markdown with MinerU figure/table
    # splices; the marker-only hooks (process_tables, caption_figures,
    # orphan-rescues) MUST be skipped for both mineru and hybrid because
    # those hooks would re-extract / re-pair what the layout step
    # already placed.
    use_mineru_or_hybrid_layout = use_mineru_layout or use_hybrid_layout

    # Build the QualityReport up front so the body emitter (mineru
    # path) can append FigureScore / TableScore entries during layout
    # extraction. The marker path populates these later in
    # process_tables / caption_figures, so the timing is harmless.
    report = QualityReport(vlm_enabled=use_vlm, metadata=metadata,
                           run_info=run_info,
                           user_annotations=user_annotations)

    if use_mineru_layout:
        if force_ocr:
            log.warning("--force-ocr has no effect under "
                        "--layout-source=mineru (MinerU runs its own "
                        "OCR pipeline)")
        # Persist MinerU intermediates next to the markdown for
        # diagnostic / re-extraction workflows; matches wrap_mineru's
        # convention of emitting a paper_dir/mineru/ subdir.
        md = run_mineru_layout(pdf_path, assets_dir,
                               asset_prefix=asset_prefix,
                               vlm_rewrite_tables=vlm_rewrite_tables,
                               rescue_orphan_captions=rescue_mineru_orphan_captions,
                               rescue_subpanels=rescue_mineru_subpanels,
                               strip_chart_details=strip_chart_details,
                               mineru_backend=mineru_backend,
                               mineru_extra_args=mineru_extra_args,
                               skip_mineru_run=skip_mineru_run,
                               mineru_existing_dir=mineru_existing_dir,
                               report=report,
                               mineru_dir=out_dir / "mineru",
                               vlm_force=vlm_tables_force,
                               table_workers=table_workers)
    elif use_hybrid_layout:
        # Hybrid: marker for body+caption text, MinerU for figure
        # assets and table HTML; matched by figure/table number.
        md = run_marker_plus_mineru_layout(
            pdf_path, assets_dir,
            asset_prefix=asset_prefix,
            force_ocr=force_ocr,
            vlm_rewrite_tables=vlm_rewrite_tables,
            rescue_orphan_captions=rescue_mineru_orphan_captions,
            rescue_subpanels=rescue_mineru_subpanels,
            strip_chart_details=strip_chart_details,
            mineru_backend=mineru_backend,
            mineru_extra_args=mineru_extra_args,
            skip_mineru_run=skip_mineru_run,
            mineru_existing_dir=mineru_existing_dir,
            report=report,
            mineru_dir=out_dir / "mineru",
            table_workers=table_workers,
            vlm_force=vlm_tables_force,
        )
    else:
        md, _ = run_marker(pdf_path, assets_dir, asset_prefix=asset_prefix,
                           force_ocr=force_ocr)

    # Strip stray line-number lines (lineno-package artifacts) before the
    # hooks see the body. Runs after marker, before image-link normalize.
    md, stripped = strip_line_numbers(md)
    if stripped:
        log.info("Stripped %d stray line-number lines", stripped)
    if STRIP_RUNNING_FOOTERS:
        md, footer_stripped = strip_running_footers(md)
        if footer_stripped:
            log.info("Stripped %d running-footer line(s) (et al. N of M)",
                     footer_stripped)
    if STRIP_PAGE_HEADERS:
        md, header_stripped = strip_journal_page_headers(md)
        if header_stripped:
            log.info("Stripped %d journal page-header line(s) (LETTERS, "
                     "NATURE ... DOI:)", header_stripped)
    if STRIP_PUBLISHER_STAMPS:
        md, stamp_stripped = strip_publisher_stamps(md)
        if stamp_stripped:
            log.info("Stripped %d per-page publisher download-watermark "
                     "line(s) (Wiley Online Library)", stamp_stripped)
    if STRIP_MATH_LABELS:
        md, n_labels = strip_math_labels(md)
        if n_labels:
            log.info("Stripped %d KaTeX-incompatible \\label{...} from math",
                     n_labels)
    if NORMALISE_REFS:
        md, ref_counts = normalise_span_anchored_refs(md)
        if any(ref_counts.values()):
            log.info("Normalised refs: %d re-bulletted, %d DOI links rejoined, "
                     "%d refhub wrappers stripped",
                     ref_counts["bulletted"], ref_counts["dois_rejoined"],
                     ref_counts["refhub_stripped"])
    if FIX_REFERENCES:
        md, repaired = fix_reference_numbering(md)
        if repaired:
            log.info("Repaired %d reference numbering(s)", repaired)
    if CONSOLIDATE_FOOTNOTES:
        md, consolidated = consolidate_footnote_references(md)
        if consolidated:
            log.info("Consolidated %d footnote-reference(s) into ## References",
                     consolidated)
    if INJECT_ORPHAN_REFS:
        md, n_orphans = inject_orphan_ref_clusters(md)
        if n_orphans:
            log.info("Found %d orphan reference cluster(s) outside the "
                     "## References heading; synthesized a heading so "
                     "merge_reference_sections can consolidate", n_orphans)
    if MERGE_REFERENCES:
        md, n_merged = merge_reference_sections(md)
        if n_merged > 1:
            log.info("Merged %d reference section(s) into a single "
                     "## References at end of document", n_merged)
    if TIDY_REFS:
        md, n_tidied = normalise_references_section(md)
        if n_tidied:
            log.info("Tidied references section: merged/extracted %d "
                     "continuation/addendum line(s)", n_tidied)

    if use_hybrid_layout:
        # Hybrid only: marker's column-aware refs serialiser fails on
        # 3-col Science/Nature layouts (cuk, young, canup2012, etc.).
        # MinerU's refs section is strictly better on those papers and
        # equivalent on OK ones (see docs/dev/HYBRID_REFERENCES_AUDIT_*).
        # Guarded by a `MinerU has a refs heading?` check so we don't
        # regress on papers where MinerU silently drops the section.
        mineru_md_path = out_dir / "mineru" / f"{pdf_path.stem}.md"
        md, swap_stats = _swap_hybrid_refs_with_mineru(md, mineru_md_path)
        if swap_stats["mineru_refs_swapped"]:
            report.rescues.setdefault("hybrid_splice", {})[
                "mineru_refs_swapped"] = True
            log.info("Hybrid: swapped marker references with MinerU's "
                     "(marker's 3-col column-serialiser is unreliable)")

    # Normalize image links in marker's markdown to assets/<filename>.
    # Must happen BEFORE trim_to_first_article so the orphan-cleanup
    # regex (which expects assets/<name>) sees the same form on disk
    # references and in markdown.
    md = IMAGE_RE.sub(
        lambda m: f"![{m.group(1)}](assets/{os.path.basename(m.group(2))})",
        md,
    )

    # Article-boundary trim: PDFs that include the next article in the
    # journal issue (Nature N&V style) get truncated here. Needs the VLM,
    # so skipped under --no-vlm.
    if TRIM_ARTICLES and use_vlm:
        with fitz.open(str(pdf_path)) as _trim_doc:
            md, trimmed_pages = trim_to_first_article(
                md, _trim_doc, assets_dir)
        if trimmed_pages:
            log.info("Trimmed %d page(s) belonging to a different article",
                     trimmed_pages)

    # Marker-layout path uses paper2md's own table finder + caption
    # matcher; MinerU- and hybrid-layout paths produce figure / table
    # markdown via emit_markdown / _align_marker_to_mineru_layout
    # (already populated `report.figures` / `.tables`).
    finder = (build_table_finder(table_finder)
              if not use_mineru_or_hybrid_layout else None)

    with fitz.open(str(pdf_path)) as doc:
        if use_vlm and emit_citation:
            citation = extract_citation(doc)
            if citation:
                md = citation + "\n\n" + md
        if not use_mineru_or_hybrid_layout:
            md = process_tables(md, doc, finder, assets_dir, report, use_vlm,
                                asset_prefix=asset_prefix,
                                vlm_rewrite_tables=vlm_rewrite_tables,
                                table_workers=table_workers)
            md = rescue_orphan_table_captions(md, doc, assets_dir, asset_prefix,
                                              report, use_vlm)
            md = caption_figures(md, assets_dir, report, use_vlm, doc=doc)
            md = rescue_orphan_figure_captions(md, doc, assets_dir,
                                               asset_prefix, report, use_vlm)
        rescued: set[int] = set()
        if use_vlm and rescue_sparse_pages_flag:
            md, rescued = rescue_sparse_pages(md, doc)
        score_pages(doc, report, rescued)

    # Data-repository link extraction. Pure regex by default; opt-in
    # API enrichment under --fetch-data-repos. Runs after all body
    # mutations so it sees the final markdown.
    _populate_data_repos(md, report)
    # Phase-1 score + Phase-2 layout fingerprint + journal rescue.
    # Re-open the PDF so the rescue dispatcher can render the refs
    # page if it dispatches to a VLM-driven rescue. The earlier
    # `with fitz.open(...)` block has already closed.
    # Always-on rescue: insert `## References` heading for headless
    # refs sections. Runs before scoring so the score reflects the
    # inserted heading. APS-specific path runs first (gated on
    # journal_slug); the generic path covers all journals as a
    # fallback. Both check _REFS_HEADING_FOR_SCORE_RE for idempotency
    # so re-runs and double-fires are safe.
    md = _rescue_aps_missing_refs_heading(md, report)
    md = _rescue_generic_missing_refs_heading(md, report)
    with fitz.open(str(pdf_path)) as _ref_doc:
        _populate_references_score(md, report, doc=_ref_doc)
        md = rescue_journal_refs(md, _ref_doc, report, use_vlm=use_vlm,
                                 out_dir=out_dir)
        # When the rescue spliced new content, re-score so the final
        # frontmatter reflects the post-rescue state. The pre-rescue
        # score is preserved on report.references.score_pre_rescue.
        if (report.references is not None
                and report.references.rescue_decision == "spliced"):
            pre_score = report.references.score_pre_rescue
            pre_layout = report.references.layout
            pre_rescue_applied = report.references.rescue_applied
            pre_rescue_decision = report.references.rescue_decision
            pre_score_post = report.references.score_post_rescue
            _populate_references_score(md, report, doc=_ref_doc)
            # Restore rescue bookkeeping (the re-score above wiped it).
            report.references.score_pre_rescue = pre_score
            report.references.score_post_rescue = pre_score_post
            report.references.rescue_applied = pre_rescue_applied
            report.references.rescue_decision = pre_rescue_decision
            if pre_layout is not None and report.references.layout is None:
                report.references.layout = pre_layout

    # Always-on API-fallback append (Crossref -> OpenAlex). Skips
    # silently when DOI absent, score is healthy, or network fails.
    md = _rescue_append_api_refs(md, report, allow_network=not OFFLINE)

    if emit_quality:
        if report.run_info is not None:
            report.run_info.elapsed_sec = round(time.time() - t0, 1)
        md = report.to_frontmatter() + md

    out_md = out_dir / ((output_stem or pdf_path.stem) + ".md")
    out_md.write_text(md)
    if emit_quality:
        _write_meta_json(out_dir, output_stem or pdf_path.stem, report)
    log.info("Wrote %s (quality: %.2f / %s)", out_md,
             report.overall(), report.grade())
    _emit_run_summary(report, out_md, vlm_rewrite_tables, use_vlm)
    return out_md, report


# Default SI-suffix pattern for batch-mode supplement auto-pairing.
# Matches "foo_SI", "foo-Supplement", "foo supp", "foo_S1",
# "foo_sm" (supplementary material; common in chemistry/physics
# Zotero exports), "foo_SM", etc. Capture group "stem" is the
# (presumed) main paper's stem.
SI_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)[ _\-]+(?:SI|supp(?:lement(?:ary|al)?)?|S\d+|sm)$",
    re.IGNORECASE,
)

# Companion pattern: matches "X-maintext", "X_manuscript", "X-paper". Lets the
# pairer resolve "root-maintext.pdf" + "root-supplemental.pdf" to a common base
# even when neither stem is a prefix of the other.
MAIN_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)[ _\-]+(?:main(?:text)?|manuscript|article|paper)$",
    re.IGNORECASE,
)


def discover_pdfs(batch_arg: str) -> list[Path]:
    """Resolve --batch arg to a sorted list of PDF paths.

    Accepts a directory (lists *.pdf one level deep), a glob string with
    *, ?, or [, or a single file.
    """
    if any(c in batch_arg for c in "*?["):
        matches = sorted(Path().glob(batch_arg))
        if not matches:
            raise SystemExit(f"--batch: no PDFs match glob {batch_arg!r}")
        return [m for m in matches if m.suffix.lower() == ".pdf"]
    p = Path(batch_arg)
    if p.is_dir():
        pdfs = sorted(p.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"--batch: no *.pdf files in {p}")
        return pdfs
    if p.is_file():
        return [p]
    raise SystemExit(f"--batch: {batch_arg} is not a file, dir, or glob")


def pair_supplements(
    pdfs: list[Path], pair_regex: re.Pattern
) -> tuple[list[tuple[Path, Optional[Path]]], list[Path]]:
    """Group sibling SI PDFs into (main, si?) tuples.

    Matching rules:
      1. SI stem matches pair_regex with captured base B; if a PDF whose stem
         equals B exists, pair them.
      2. Otherwise, if a PDF whose stem matches MAIN_SUFFIX_RE has the same
         captured base B, pair the SI to that "main" PDF.
      3. Otherwise the SI runs standalone (returned in the second list with
         a warning logged by the caller).
    """
    by_stem = {p.stem: p for p in pdfs}
    main_by_base: dict[str, Path] = {}
    for p in pdfs:
        m = MAIN_SUFFIX_RE.match(p.stem)
        if m:
            main_by_base[m.group("stem").lower()] = p

    si_attached: dict[str, Path] = {}  # main_stem -> si path
    standalone_si: list[Path] = []
    for p in pdfs:
        m = pair_regex.match(p.stem)
        if not m:
            continue
        base = m.group("stem")
        if base in by_stem and by_stem[base] != p:
            si_attached[by_stem[base].stem] = p
        elif base.lower() in main_by_base and main_by_base[base.lower()] != p:
            si_attached[main_by_base[base.lower()].stem] = p
        else:
            standalone_si.append(p)

    consumed = set(si_attached.values())
    pairs: list[tuple[Path, Optional[Path]]] = []
    for p in pdfs:
        if p in consumed:
            continue
        pairs.append((p, si_attached.get(p.stem)))
    return pairs, standalone_si


def run_batch(pairs: list[tuple[Path, Optional[Path]]],
              args, do_convert) -> int:
    """Run the per-paper pipeline over a batch. Writes per-paper subdirs and
    a manifest.jsonl in args.out. Returns the number of failed papers."""
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"

    def _annotate_mineru_error(entry: dict, e: Exception,
                                key_prefix: str = "") -> None:
        """If `e` is a typed MineruRunError, surface the classified
        kind + stderr tail under `{prefix}error_kind` / `_pdf` / _tail
        keys. Used by both the main-paper and SI-paper error branches
        so `jq 'select(.error_kind=="cuda_oom")'` and the analogous
        `select(.si_error_kind=="cuda_oom")` both work."""
        try:
            import layout_mineru as _lm
            if isinstance(e, _lm.MineruRunError):
                entry[key_prefix + "error_kind"] = e.kind
                entry[key_prefix + "error_pdf"] = e.pdf_path.name
                if e.stderr_tail:
                    entry[key_prefix + "error_stderr_tail"] = e.stderr_tail[-2000:]
        except Exception:
            pass

    def _process_one(main: Path, si: Optional[Path]) -> dict:
        t0 = time.time()
        paper_dir = args.out / main.stem
        out_md = paper_dir / (main.stem + ".md")
        if out_md.exists() and not args.force:
            return {"pdf": str(main),
                    "supplement": str(si) if si else None,
                    "status": "skipped",
                    "elapsed_sec": 0.0}
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: main paper. A main-paper failure aborts the entry
        # entirely (there's no salvageable output without the main).
        try:
            main_md, report = do_convert(main, paper_dir)
        except Exception as e:
            log.exception("Failed: %s", main.name)
            entry = {"pdf": str(main),
                     "supplement": str(si) if si else None,
                     "status": "error",
                     "error": f"{type(e).__name__}: {e}",
                     "elapsed_sec": round(time.time() - t0, 1)}
            _annotate_mineru_error(entry, e)
            return entry

        # Phase 2: supplement. A supplement failure must NOT discard the
        # successful main result -- the main is independently useful
        # (it's the canonical paper; SI is auxiliary). Mark si_status
        # separately so `jq 'select(.si_status=="error")'` filters
        # supplement failures without conflating with full-paper
        # errors. The main entry stays status="ok" with a usable
        # grade / overall / sidecar trail.
        si_md = None
        si_report = None
        si_status: Optional[str] = None
        si_error: Optional[Exception] = None
        if si is not None:
            try:
                si_md, si_report = do_convert(
                    si, paper_dir, asset_prefix="si_")
                si_status = "ok"
            except Exception as e:
                log.warning(
                    "Supplement failed for %s (main paper still "
                    "produced): %s: %s",
                    main.name, type(e).__name__, e)
                si_status = "error"
                si_error = e

        # Phase 3: bundle + main entry. Bundle works with si_md=None.
        if args.hdf5:
            bundle_path = paper_dir / (main.stem + ".h5")
            write_hdf5_bundle(
                bundle_path,
                main_md=main_md, main_report=report, main_pdf=main,
                si_md=si_md, si_report=si_report, si_pdf=si,
            )
        overall = report.overall()
        status = "ok"
        if (args.quality_threshold is not None
                and overall < args.quality_threshold):
            status = "low_quality"
        entry = {"pdf": str(main),
                 "supplement": str(si) if si else None,
                 "status": status,
                 "grade": report.grade(),
                 "overall": round(overall, 3),
                 "elapsed_sec": round(time.time() - t0, 1)}
        if report.metadata is not None:
            entry["copyright"] = report.metadata.manifest_dict()
        # Reference-vs-insertion audit: scan the produced markdown for
        # body cross-refs to `Fig. N` / `Table N` and compare to the
        # actual image / table-caption insertions. When refs > inserts,
        # something dropped a figure or table on the floor and the
        # user should manually inspect. `figure_mismatch` /
        # `table_mismatch` are top-level booleans in the manifest entry
        # so `jq 'select(.figure_mismatch)'` filters problem papers.
        try:
            import audit_md as _audit
            a = _audit.audit_md(main_md)
            entry["figures_referenced"] = len(a["figures_referenced"])
            entry["figures_inserted"] = a["figures_inserted"]
            entry["tables_referenced"] = len(a["tables_referenced"])
            entry["tables_inserted"] = len(a["tables_inserted"])
            entry["figure_mismatch"] = a["figure_mismatch"]
            entry["table_mismatch"] = a["table_mismatch"]
            # If table_mismatch fired AND a sibling mineru/ output
            # exists, count how many of the missing tables MinerU
            # detected -- these are recoverable via
            # `paper2md --recover-from-mineru`. Filter the manifest
            # with `jq 'select(.recoverable_tables > 0)'` to drive
            # the manual-edit triage workflow.
            if a["table_mismatch"]:
                try:
                    import mineru_recovery as _mr
                    items = _mr.find_recoverable_tables(main_md)
                    entry["recoverable_tables"] = len(items)
                except Exception as e2:
                    log.debug("recoverable_tables count failed on "
                              "%s: %s", main_md, e2)
        except Exception as e:
            log.debug("audit_md failed on %s: %s", main_md, e)
        if si is not None:
            entry["si_status"] = si_status
            if si_error is not None:
                entry["si_error"] = f"{type(si_error).__name__}: {si_error}"
                _annotate_mineru_error(entry, si_error,
                                       key_prefix="si_")
            elif si_md is not None:
                # Same audit for the supplement; flag under si_*.
                try:
                    import audit_md as _audit
                    a = _audit.audit_md(si_md)
                    entry["si_figures_referenced"] = len(a["figures_referenced"])
                    entry["si_figures_inserted"] = a["figures_inserted"]
                    entry["si_tables_referenced"] = len(a["tables_referenced"])
                    entry["si_tables_inserted"] = len(a["tables_inserted"])
                    entry["si_figure_mismatch"] = a["figure_mismatch"]
                    entry["si_table_mismatch"] = a["table_mismatch"]
                    if a["table_mismatch"]:
                        try:
                            import mineru_recovery as _mr
                            items = _mr.find_recoverable_tables(si_md)
                            entry["si_recoverable_tables"] = len(items)
                        except Exception as e2:
                            log.debug("si recoverable_tables count "
                                      "failed on %s: %s", si_md, e2)
                except Exception as e:
                    log.debug("audit_md failed on SI %s: %s", si_md, e)
        return entry

    manifest_lock = threading.Lock()

    def _record(entry: dict) -> None:
        line = json.dumps(entry)
        with manifest_lock:
            with manifest_path.open("a") as f:
                f.write(line + "\n")
        log.info("[batch] %s", line)

    # Per-paper deadline: prevents a single hung paper (network hang on
    # the VLM endpoint, marker stuck on a pathological page, docling
    # detector hang) from blocking the rest of the batch indefinitely.
    # Implementation runs _process_one in a daemon thread and joins
    # with a timeout. On timeout the daemon thread is *abandoned* (it
    # continues running, marked daemon so it won't block process
    # exit) while the next paper proceeds. See --paper-timeout help
    # for the lock-holdover caveat.
    deadline_s = float(args.paper_timeout) if args.paper_timeout else 0.0

    def _process_one_bounded(main: Path, si: Optional[Path]) -> dict:
        if deadline_s <= 0:
            return _process_one(main, si)
        box: dict = {}

        def _go() -> None:
            try:
                box["result"] = _process_one(main, si)
            except BaseException as e:  # noqa: BLE001
                box["error"] = e

        t = threading.Thread(target=_go, daemon=True,
                             name=f"paper2md-batch-{main.stem}")
        t0 = time.time()
        t.start()
        t.join(deadline_s)
        if t.is_alive():
            log.error("[batch] paper-timeout (%.0fs): %s -- abandoning "
                      "thread (may hold marker lock / GPU memory) and "
                      "continuing with next paper",
                      deadline_s, main.name)
            return {
                "pdf": str(main),
                "supplement": str(si) if si else None,
                "status": "timeout",
                "error": f"timed out after {deadline_s:.0f} s",
                "elapsed_sec": round(time.time() - t0, 1),
            }
        if "error" in box:
            # KeyboardInterrupt / SystemExit propagated. _process_one's
            # internal try/except already converts ordinary Exceptions
            # to status='error' entries, so this branch is for
            # BaseException only.
            raise box["error"]
        return box["result"]

    failures = 0
    workers = max(1, args.workers)
    fail_statuses = {"error", "timeout"}
    if workers == 1:
        for main, si in pairs:
            entry = _process_one_bounded(main, si)
            _record(entry)
            if entry["status"] in fail_statuses:
                failures += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_process_one_bounded, m, s)
                    for m, s in pairs]
            for fut in as_completed(futs):
                entry = fut.result()
                _record(entry)
                if entry["status"] in fail_statuses:
                    failures += 1
    return failures


def _run_metadata_only(args, parser) -> None:
    """Implements --metadata-only: run only the copyright/OA front-end
    and exit. No marker, no VLM, no scoring. Single-paper mode prints a
    standalone YAML 'copyright:' block to stdout. Batch mode iterates
    the corpus, writes per-paper records to <out>/manifest.jsonl, and
    logs a one-line summary per paper.

    Honors --prefer-oa-source (downloads the OA copy alongside the
    metadata) and --require-license (exits 3 on unsafe license).
    Returns nothing (exits via sys.exit)."""
    try:
        from metadata_frontend import resolve as resolve_metadata
    except ImportError as e:
        parser.error(f"metadata_frontend not importable: {e}")

    def _resolve_one(pdf_path: Path, oa_dir: Optional[Path] = None):
        return resolve_metadata(
            pdf_path,
            allow_network=not OFFLINE,
            prefer_oa=args.prefer_oa_source,
            timeout_s=args.metadata_timeout,
            cache_path=args.metadata_cache,
            oa_download_dir=oa_dir,
        )

    def _exit_if_required(meta) -> int:
        # --require-license maps to exit 3 (matches _do_convert behavior).
        if args.require_license and (
                meta is None
                or meta.safe_to_distribute in {"false", "readable"}):
            log.error("--require-license: license is not safe to "
                      "redistribute (license=%s, safe_to_distribute=%s)",
                      meta.license if meta else "none-resolved",
                      meta.safe_to_distribute if meta else "n/a")
            return 3
        return 0

    if args.batch:
        pdfs = discover_pdfs(args.batch)
        log.info("--metadata-only batch: %d PDF(s)", len(pdfs))
        args.out.mkdir(parents=True, exist_ok=True)
        manifest_path = args.out / "manifest.jsonl"
        # Truncate any prior manifest so re-runs produce a clean record.
        manifest_path.write_text("")
        worst_exit = 0
        for pdf in pdfs:
            try:
                meta = _resolve_one(pdf, oa_dir=args.out)
            except Exception as e:
                log.exception("metadata lookup failed for %s", pdf.name)
                entry = {"pdf": str(pdf), "status": "error",
                         "error": f"{type(e).__name__}: {e}"}
                with manifest_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                worst_exit = max(worst_exit, 1)
                continue
            entry = {"pdf": str(pdf),
                     "status": "ok",
                     **meta.manifest_dict()}
            if meta.oa_pdf_used and meta.oa_pdf_local_path:
                entry["oa_pdf_local"] = str(meta.oa_pdf_local_path)
            if meta.title:
                entry["title"] = meta.title
            with manifest_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            log.info("[metadata-only] %s -> license=%s safe=%s "
                     "confidence=%s via=%s",
                     pdf.name,
                     meta.license or "unknown",
                     meta.safe_to_distribute,
                     meta.confidence,
                     meta.resolved_via or "none")
            worst_exit = max(worst_exit, _exit_if_required(meta))
        log.info("--metadata-only batch complete: %d papers, manifest: %s",
                 len(pdfs), manifest_path)
        sys.exit(worst_exit)

    # Single paper.
    pdf = args.pdf
    oa_dir = args.out if args.prefer_oa_source else None
    if oa_dir is not None:
        oa_dir.mkdir(parents=True, exist_ok=True)
    try:
        meta = _resolve_one(pdf, oa_dir=oa_dir)
    except Exception as e:
        log.exception("metadata lookup failed: %s", e)
        sys.exit(1)

    log.info("Resolved %s in %s; printing YAML to stdout",
             pdf.name, meta.resolved_via or "(no api succeeded)")
    # Emit a self-contained YAML doc so the output is pipeable into
    # `yq`, `python -m yaml`, etc.
    print("---")
    for line in meta.to_yaml_lines():
        print(line)
    print("---")
    sys.exit(_exit_if_required(meta))


def _run_manual_edit(args, parser):
    """Dispatch --replace-table / --replace-fig / --revert-edit. The
    table path needs the VLM client (configure_client) and reuses
    paper2md.vlm + TABLE_PROMPT; the figure and revert paths are
    pure file operations."""
    import manual_edit as me

    if args.replace_table:
        id_, img_str, paper_str = args.replace_table
        image = Path(img_str)
        paper = Path(paper_str)
        if not image.exists():
            parser.error(f"image not found: {image}")
        if not paper.is_file():
            parser.error(f"paper.md not found: {paper}")
        configure_client(provider=args.provider)
        try:
            result = me.do_replace_table(
                paper, id_, image, note=args.note)
        except Exception as e:
            log.error("Table replacement failed: %s", e)
            sys.exit(2)
        log.info("Replaced Table %s metadata; user paste required.", id_)
        print(result["paste_hint"])
        return

    if args.replace_fig:
        id_, img_str, paper_str = args.replace_fig
        image = Path(img_str)
        paper = Path(paper_str)
        if not image.exists():
            parser.error(f"image not found: {image}")
        if not paper.is_file():
            parser.error(f"paper.md not found: {paper}")
        try:
            result = me.do_replace_fig(paper, id_, image, note=args.note)
        except Exception as e:
            log.error("Figure replacement failed: %s", e)
            sys.exit(2)
        archived = result.get("archived_original")
        if archived is not None:
            log.info("Replaced Fig. %s -> %s (original archived as %s)",
                     id_, result["image"], archived.name)
        else:
            log.info("Replaced Fig. %s -> %s (original already archived)",
                     id_, result["image"])
        return

    if args.revert_edit:
        kind, id_, paper_str = args.revert_edit
        if kind not in ("table", "figure"):
            parser.error(
                f"--revert-edit KIND must be 'table' or 'figure', "
                f"got {kind!r}")
        paper = Path(paper_str)
        if not paper.is_file():
            parser.error(f"paper.md not found: {paper}")
        try:
            result = me.do_revert(paper, kind, id_)
        except LookupError as e:
            parser.error(str(e))
        except Exception as e:
            log.error("Revert failed: %s", e)
            sys.exit(2)
        if result["restored"]:
            log.info("Reverted: restored %s", result["restored"])
        else:
            log.info("Reverted: dropped edits entry for %s %s "
                     "(no asset restore needed)", kind, id_)
        return

    if args.recover_from_mineru:
        import mineru_recovery as mr
        paper = args.recover_from_mineru
        if not paper.is_file():
            parser.error(f"paper.md not found: {paper}")
        # VLM client is only needed when use_vlm resolves to True.
        # Resolve early so we configure_client before staging starts.
        if args.recovery_vlm is None:
            use_vlm = mr.detect_original_vlm_mode(paper)
        else:
            use_vlm = args.recovery_vlm
        if use_vlm:
            configure_client(provider=args.provider)
        try:
            result = mr.do_recover_from_mineru(
                paper, vlm_override=args.recovery_vlm)
        except Exception as e:
            log.error("Recovery failed: %s", e)
            sys.exit(2)
        log.info("%s", result["summary"])
        for r in result["recovered"]:
            if "error" in r:
                log.warning("  Table %s: %s", r["id"], r["error"])
            else:
                line = r.get("caption_line")
                if line is not None:
                    log.info(
                        "  Table %s staged via %s; paste %s at "
                        "line %d in %s",
                        r["id"], r["source"],
                        r["sidecar"].name, line, paper.name)
                else:
                    log.info(
                        "  Table %s staged via %s; no caption line "
                        "found in body -- search for `Table %s` in %s "
                        "and paste %s there",
                        r["id"], r["source"], r["id"], paper.name,
                        r["sidecar"].name)
        log.info("After pasting, mark applied: paper2md "
                 "--confirm-recovery table <ID> %s", paper)
        return

    if args.confirm_recovery:
        import mineru_recovery as mr
        kind, id_, paper_str = args.confirm_recovery
        if kind not in ("table", "figure"):
            parser.error(
                f"--confirm-recovery KIND must be 'table' or "
                f"'figure', got {kind!r}")
        paper = Path(paper_str)
        if not paper.is_file():
            parser.error(f"paper.md not found: {paper}")
        try:
            result = mr.confirm_recovery(paper, kind, id_)
        except LookupError as e:
            parser.error(str(e))
        except Exception as e:
            log.error("Confirm failed: %s", e)
            sys.exit(2)
        log.info("Confirmed: %s %s promoted to `edits:` "
                 "(recovered_from=mineru)", kind, id_)
        return


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  paper2md.py paper.pdf                      # default: docling tables + VLM
                                           # hooks for citation/figures/rescue
                                           # (no per-table VLM rewrite)
  paper2md.py paper.pdf --no-vlm             # marker only, no VLM
  paper2md.py paper.pdf --vlm-tables --table-workers 4
                                           # high-fidelity tables: VLM
                                           # re-transcribes each crop
  paper2md.py paper.pdf --table-finder all   # PyMuPDF + TATR + Docling chain
  paper2md.py paper.pdf --supplement SI.pdf -o out/    # main + SI bundle
  paper2md.py --batch corpus/ -o out/ --workers 4      # folder batch
  paper2md.py paper.pdf --hdf5                         # also write .h5 bundle
  paper2md.py paper.pdf --provider openai              # cloud VLM
  paper2md.py paper.pdf --prefer-oa-source             # pull OA copy if any
  paper2md.py paper.pdf --require-license              # gate on CC license
  paper2md.py paper.pdf --metadata-only                # copyright check only;
                                                     # no marker, no VLM

env vars:
  VLM_PROVIDER, VLM_MODEL, VLLM_BASE_URL, LM_STUDIO_URL,
  OPENAI_API_KEY, ANTHROPIC_API_KEY,
  UNPAYWALL_EMAIL, OPENALEX_MAILTO, CROSSREF_MAILTO  (metadata polite pool)

see USAGE.md for full reference.
""",
    )

    g_io = p.add_argument_group("input / output")
    g_io.add_argument("pdf", type=Path, nargs="?",
                      help="PDF to convert (single-paper mode). "
                           "Mutually exclusive with --batch.")
    g_io.add_argument("-o", "--out", type=Path, default=Path("./out"),
                      help="output directory (default: ./out)")
    g_io.add_argument("--supplement", type=Path, default=None, metavar="SI_PDF",
                      help="also convert a supplementary-information PDF. "
                           "Writes a separate <si-stem>.md next to the main "
                           ".md, sharing the same assets/ folder. Supplement-"
                           "origin assets are prefixed with 'si_' to avoid "
                           "filename collisions.")
    g_io.add_argument("--hdf5", action="store_true",
                      help="also write a compressed HDF5 bundle "
                           "(<main-stem>.h5) in the output directory "
                           "containing the main markdown, supplement markdown "
                           "(if any), and every file under assets/. Loose "
                           "files are written as usual.")

    g_pipe = p.add_argument_group("pipeline control")
    g_pipe.add_argument("--layout-source",
                        choices=["marker", "mineru", "hybrid"],
                        # Sentinel default lets us distinguish 'user
                        # explicitly passed --layout-source' from 'user
                        # let the default kick in', so the mutex with
                        # --auto-layout-source can fire only on the
                        # explicit case. Resolved to "mineru" below.
                        default=None,
                        help="select the figure / table / caption layout "
                             "engine. 'mineru' (DEFAULT) shells out to "
                             "MinerU for body OCR AND figure / table / "
                             "caption layout in a single pass; paper2md's "
                             "pre-pass strippers, refs hooks, metadata, "
                             "trim, scoring, and frontmatter all still "
                             "run on top. Faster than 'marker' and "
                             "produces better figure / table layout on "
                             "the corpus's born-digital papers. 'marker' "
                             "uses marker + surya for body OCR plus "
                             "paper2md's docling / pymupdf table-finders "
                             "and caption_figures matcher; route scan / "
                             "pre-2000 / sup-sub-heavy papers through "
                             "'marker'. 'hybrid' (preview) runs marker "
                             "AND MinerU and splices MinerU's figure "
                             "assets / table HTML into marker's body, "
                             "matched by figure / table number; marker's "
                             "caption text and reading order are "
                             "preserved (no PaddleOCR on the captions). "
                             "Costs both extraction passes. Best on "
                             "multi-figure 3-column papers (Science, "
                             "Nature) where MinerU caption text is "
                             "garbled; not auto-routed. See triage_"
                             "pdfs.py for the auto-routing helper. "
                             "Needs `pip install mineru` for mineru/"
                             "hybrid. Mutually exclusive with "
                             "--auto-layout-source.")
    g_pipe.add_argument("--mineru-backend",
                        choices=["pipeline", "hybrid-auto-engine"],
                        default="pipeline",
                        help="MinerU's internal backend (only consulted "
                             "when --layout-source=mineru). 'pipeline' "
                             "(DEFAULT) uses PaddleOCR for body text, "
                             "fast (~30 s/paper). 'hybrid-auto-engine' "
                             "uses MinerU's bundled 2.3 GB VLM for "
                             "cell-level tables and chart-data "
                             "reconstruction (~3-17 min/paper); emits "
                             "<details> blocks with VLM-estimated "
                             "values (see --strip-chart-details to "
                             "drop them on output).")
    g_pipe.add_argument("--mineru-arg", action="append", default=[],
                        metavar="ARG",
                        help="Repeatable; passed through to mineru's "
                             "CLI as additional arguments. E.g. "
                             "--mineru-arg --device-mode --mineru-arg "
                             "cpu. Only used under --layout-source=mineru.")
    g_pipe.add_argument("--skip-mineru-run", action="store_true",
                        help="Skip the MinerU subprocess; consume an "
                             "existing auto/ output directory specified "
                             "via --mineru-dir. Intended for offline "
                             "replay / testing / re-running paper2md's "
                             "post-passes after editing MinerU output.")
    g_pipe.add_argument("--mineru-dir", type=Path, default=None,
                        metavar="PATH",
                        help="Existing MinerU auto/ directory containing "
                             "<stem>_middle.json and images/. Required "
                             "when --skip-mineru-run is passed; ignored "
                             "otherwise.")
    g_pipe.add_argument("--strip-chart-details", action="store_true",
                        help="Strip MinerU hybrid-mode <details><summary>"
                             "line</summary> chart-data blocks from the "
                             "body markdown. Off by default; use this "
                             "when downstream RAG / vector-DB ingestion "
                             "might quote the VLM-estimated curve "
                             "readings as facts. Only relevant under "
                             "--mineru-backend hybrid-auto-engine.")
    g_pipe.add_argument("--auto-layout-source", action="store_true",
                        help="auto-route per paper via the triage "
                             "detector instead of forcing one layout "
                             "engine for the whole batch. The 'good "
                             "fast route' for mixed corpora. Composes "
                             "two PyMuPDF probes per paper (~600 ms): "
                             "(1) triage_pdfs detects whether the text "
                             "layer is rich enough for MinerU's layout "
                             "extraction; (2) detect_scan_ocr_pdf "
                             "detects scan-style PDFs (producer "
                             "metadata, image-cover, bold-char ratio). "
                             "A paper routes to 'marker' if EITHER "
                             "detector flags it; this catches both "
                             "Lyzenga-style flat-OCR (caught by "
                             "triage's super-flag check) AND "
                             "Wackerle-style scans whose OCR happens "
                             "to preserve enough text-layer artifacts "
                             "to fool triage (caught by the scan "
                             "detector). When the scan detector fires "
                             "AND the paper routes to marker, "
                             "force-OCR auto-applies (override with "
                             "--no-auto-force-ocr). Mutually exclusive "
                             "with --layout-source. Per-paper triage "
                             "signals + scan signals + routing "
                             "decision are recorded in "
                             "run.pipeline.triage_signals / "
                             "scan_signals for reproducibility.")
    g_pipe.add_argument("--no-auto-force-ocr", action="store_true",
                        help="under --auto-layout-source, suppress the "
                             "automatic force-OCR application on "
                             "scan-detected marker-routed papers. "
                             "Routing still runs; force-OCR just "
                             "doesn't fire. Default is off (auto "
                             "force-OCR is on under "
                             "--auto-layout-source). Use this when "
                             "you want raw marker output on a "
                             "scan-routed paper (rare).")
    g_pipe.add_argument("--no-vlm", action="store_true",
                        help="skip ALL VLM post-passes (marker only; JPEGs "
                             "and scoring still run). Tables that need VLM "
                             "rewrite stay as marker's markdown.")
    g_pipe.add_argument("--text-only", action="store_true",
                        help="quick mode: skip marker/surya entirely, ask the "
                             "VLM to convert each page directly to markdown. "
                             "No tables/figures hooks, no assets folder, no "
                             "layout analysis (VLM handles layout). Best for "
                             "text-indexing / RAG ingestion. Incompatible "
                             "with --no-vlm.")
    g_pipe.add_argument("--force-ocr", action="store_true",
                        help="bypass the embedded PDF text layer; force "
                             "marker to run surya OCR on every page. Use "
                             "this on pre-2000 / scanned papers whose "
                             "publisher OCR poisons the output -- spurious "
                             "bold paragraphs from stale font-weight "
                             "metadata, line-broken table headers like "
                             "'TABLE\\n2.', mis-located figure captions. "
                             "Adds full surya recognition cost on every "
                             "page (~30-60s on an 8-page paper). On clean "
                             "modern PDFs it's pure overhead -- leave off "
                             "by default. No effect under --text-only "
                             "(which doesn't run marker). Mutually "
                             "exclusive with --auto-force-ocr.")
    g_pipe.add_argument("--auto-force-ocr", action="store_true",
                        help="auto-detect scan + old-OCR PDFs per paper "
                             "and apply --force-ocr only to those. Cheap "
                             "PyMuPDF-based pre-check (~100-300 ms / "
                             "paper) inspects /Producer and /Creator "
                             "metadata, image-coverage ratio, font "
                             "diversity, and bold-character ratio. "
                             "Conservative: a single OCR-tool fingerprint "
                             "in metadata, OR an image-heavy + "
                             "low-font-diversity combo, OR two weaker "
                             "signals together is enough to flip the "
                             "force-OCR switch ON for that paper. Most "
                             "useful in batch mode for mixed old/new "
                             "corpora; works in single-paper mode too. "
                             "Mutually exclusive with --force-ocr.")
    g_pipe.add_argument("--scan-detect-only", action="store_true",
                        help="dry-run mode: print the scan-detection "
                             "verdict and signals for each PDF (single "
                             "paper or whole batch) without converting. "
                             "Useful to pre-check a corpus and validate "
                             "the --auto-force-ocr decisions before "
                             "kicking off a multi-hour run. No marker, "
                             "no VLM, no GPU; ~100-300 ms / paper.")
    g_pipe.add_argument("--clean", action="store_true",
                        help="delete pre-existing output files in --out "
                             "before running. Targets just this paper's "
                             "<stem>.md / .meta.json / .h5 plus matching "
                             "files in assets/ (marker-extracted images, "
                             "table_p* JPEGs and sidecars). Other files "
                             "in the directory are left alone, and a "
                             "supplement run only clears its own "
                             "si_-prefixed assets. Without --clean, "
                             "stale files from a prior run on the same "
                             "paper persist alongside the fresh output "
                             "(common when re-running with different "
                             "flags) and a one-line warning is logged.")
    g_pipe.add_argument("--no-quality", action="store_true",
                        help="skip YAML quality-rating front-matter")
    g_pipe.add_argument("--quality-threshold", type=float, default=None,
                        help="exit non-zero if overall quality < this value")
    g_pipe.add_argument("--no-citation", action="store_true",
                        help="skip the VLM citation-extraction pass (hook 0)")
    g_pipe.add_argument("--no-fix-refs", action="store_true",
                        help="skip the deterministic reference-renumbering "
                             "pass (default on; uses surviving span anchors "
                             "to back-fill missing numbers in bibliography "
                             "lists)")
    g_pipe.add_argument("--no-trim-articles", action="store_true",
                        help="skip the article-boundary trim pass (default "
                             "on; asks the VLM whether page 2 is a different "
                             "article and truncates the markdown there if so). "
                             "Costs one extra VLM call per PDF; needs the "
                             "VLM, so already skipped under --no-vlm.")
    g_pipe.add_argument("--no-consolidate-footnotes", action="store_true",
                        help="skip the deterministic footnote-reference "
                             "consolidation pass (default on; lifts "
                             "<sup>N</sup>-prefixed footnote lines out of "
                             "the body into a single ## References section "
                             "at the end). No VLM, no GPU; pure regex.")
    g_pipe.add_argument("--no-normalise-refs", action="store_true",
                        help="skip the Elsevier/Icarus span-anchored "
                             "reference normalisation pass (default on; "
                             "re-bullets plain-<span id='page-X-N'>...</span> "
                             "ref lines, rejoins DOI links split across "
                             "markdown links by column reflow, strips refhub "
                             "Elsevier cross-reference link wrappers). No VLM.")
    g_pipe.add_argument("--no-strip-footers", action="store_true",
                        help="skip the running-footer strip pre-pass "
                             "(default on; removes '<AUTHOR> ET AL.  N of M' "
                             "lines that AGU / Wiley / society journals "
                             "interleave between body paragraphs). No VLM, "
                             "pure regex.")
    g_pipe.add_argument("--no-strip-page-headers", action="store_true",
                        help="skip the journal page-header strip pre-pass "
                             "(default on; removes 'LETTERS', 'ARTICLES', "
                             "'NATURE GEOSCIENCE DOI: [...]' and similar "
                             "page-header artifacts that marker preserves "
                             "between body paragraphs). No VLM, pure regex.")
    g_pipe.add_argument("--no-strip-publisher-stamps", action="store_true",
                        help="skip the per-page download-watermark strip "
                             "pre-pass (default on; removes Wiley Online "
                             "Library's 'See the Terms and Conditions / "
                             "OA articles are governed by the applicable "
                             "Creative Commons' boilerplate that's stamped "
                             "on every page of subscription downloads from "
                             "onlinelibrary.wiley.com and bleeds into the "
                             "body markdown). No VLM, pure regex; skips "
                             "fenced code and table rows so DOI URLs in "
                             "real reference lists aren't dropped.")
    g_pipe.add_argument("--no-merge-references", action="store_true",
                        help="skip the reference-section consolidation pass "
                             "(default on; merges multiple '## References' / "
                             "'# **References**' sections into one '## "
                             "References' at the end of the document, "
                             "preserving body / Methods sections in their "
                             "original positions). Useful for chunked-by-"
                             "section embedding pipelines that expect one "
                             "consolidated reference list. No VLM.")
    g_pipe.add_argument("--no-inject-orphan-refs", action="store_true",
                        help="skip the orphan-reference-cluster injector "
                             "(default on; runs immediately before --merge-"
                             "references). Detects contiguous runs of >=3 "
                             "bulleted-and-numbered reference lines living "
                             "outside any '## References' heading -- e.g. "
                             "marker split a multi-page references list "
                             "across a column boundary and some entries "
                             "ended up above the heading -- and prepends "
                             "a synthetic '## References' heading so the "
                             "merge pass can consolidate them. Skips runs "
                             "already inside an existing References "
                             "section. No VLM, pure regex.")
    g_pipe.add_argument("--no-tidy-refs", action="store_true",
                        help="skip the deterministic in-section reference "
                             "tidy-up (default on; runs after merge-"
                             "references). Merges column-break continuation "
                             "lines into their parent entry, pulls the "
                             "corresponding-author address and 'Received "
                             "...' parenthetical out of the bullet list, "
                             "and applies uniform '- ' bulleting. Common "
                             "after --force-ocr where surya's column-"
                             "boundary handling can split entries. No "
                             "VLM; pure regex.")
    g_pipe.add_argument("--no-strip-math-labels", action="store_true",
                        help="skip the KaTeX-incompatible math-label strip "
                             "pre-pass (default on; removes \\label{...} "
                             "and rewrites \\eqref{...} -> (...) so that "
                             "in-browser KaTeX renderers don't throw on "
                             "display-math equations preserved verbatim "
                             "from LaTeX source). No VLM, pure regex.")
    g_pipe.add_argument("--no-rescue-orphan-tables", action="store_true",
                        help="skip the orphan-caption table rescue post-"
                             "pass (default on; after hook 1, scans the "
                             "body for line-leading 'Table N.' captions "
                             "that aren't followed by a markdown table "
                             "or sidecar reference, renders the caption's "
                             "PDF page, and asks the VLM to extract the "
                             "table directly). Catches the case where "
                             "marker rendered a small or unusually-shaped "
                             "table as paragraph text. Needs the VLM, so "
                             "already skipped under --no-vlm.")
    g_pipe.add_argument("--no-rescue-orphan-figures", action="store_true",
                        help="skip the orphan-figure-caption rescue post-"
                             "pass (default on; after hook 2, scans the "
                             "markdown for 'Fig. N.' captions that have "
                             "no preceding ![Figure N | ...] image, "
                             "locates the caption's PDF page, and crops "
                             "the area above the caption as the figure "
                             "asset). Catches the case where marker's "
                             "layout misclassified the caption block "
                             "and dropped the figure region (e.g. "
                             "jacquet 2026 Fig. 5). No VLM call.")
    g_pipe.add_argument("--no-rescue-orphan-captions", action="store_true",
                        help="skip the MinerU orphan-caption / footnote "
                             "adoption pass (default on under "
                             "--layout-source=mineru; no-op otherwise). "
                             "The pass adopts text blocks that MinerU "
                             "misclassified as body but match figure / "
                             "table caption / footnote signatures into "
                             "the parent figure / table block, so they "
                             "render in the right place. Disable for a "
                             "fast clean MinerU run when you want to "
                             "see the raw layout output without any "
                             "block reclassification.")
    g_pipe.add_argument("--no-rescue-subpanels", action="store_true",
                        help="skip the multi-panel figure consolidation "
                             "pass (default on under "
                             "--layout-source=mineru; no-op otherwise). "
                             "The pass detects figure groups where "
                             "MinerU emitted one image block per "
                             "sub-panel (canup-style Figs 1-3 in Nature "
                             "/ Science: real caption attached to one "
                             "panel, stub 'Figure N' or panel-letter "
                             "captions on siblings) and consolidates "
                             "them into a single figure with one "
                             "shared italic caption and per-panel "
                             "alt-text. Figures only -- tables are "
                             "out of scope. Skips pages where the "
                             "primary's caption contains 2+ figure "
                             "anchors (avoids regressing on cross-"
                             "figure caption-absorption pathology).")
    g_pipe.add_argument("--use-journal-rescue", action="store_true",
                        help="enable Phase-2 journal-aware reference "
                             "rescue (off by default while we calibrate). "
                             "Routes papers whose references-score falls "
                             "below PAPER2MD_REF_RESCUE_THRESHOLD into a "
                             "journal-specific rescue keyed by the "
                             "DOI-derived journal_slug. Currently ships "
                             "two rescues: (a) Science 3-column VLM "
                             "reorder (fragments-only reorder; verifier "
                             "rejects any text mutation), and (b) APS "
                             "last-ref-into-acks deterministic trim "
                             "(triggered when longest-over-median > "
                             "PAPER2MD_REF_RESCUE_LOR and the last entry "
                             "isn't well-bounded). See plan in "
                             "docs/USAGE.md.")
    g_pipe.add_argument("--ref-rescue-threshold", type=float, default=None,
                        help="composite-score gate for journal rescue "
                             "(default 0.65, env "
                             "PAPER2MD_REF_RESCUE_THRESHOLD). Papers "
                             "scoring at or above this don't enter the "
                             "rescue dispatcher.")
    g_pipe.add_argument("--ref-rescue-dry-run", action="store_true",
                        help="run the journal-rescue dispatcher in "
                             "dry-run mode (no body changes, but pre/"
                             "post scores recorded so calibration is "
                             "observable). Env: "
                             "PAPER2MD_REF_RESCUE_DRY_RUN=1.")
    g_pipe.add_argument("--no-rotate-tables", action="store_true",
                        help="skip the sideways-table rotation hook "
                             "(default on). When a cropped table image "
                             "looks rotated 90 deg (height/width ratio "
                             "exceeds PAPER2MD_TABLE_ROTATION_THRESHOLD, "
                             "default 2.5), hook 1 rotates the crop "
                             "(direction PAPER2MD_TABLE_ROTATION_DIRECTION, "
                             "default 'ccw') so the VLM sees column "
                             "headers running across the top, and saves "
                             "the rotated JPG as the human-visible "
                             "asset. If the rotated transcription comes "
                             "back broken (suspicious shape), a fallback "
                             "VLM call on the full page with a rotation-"
                             "aware prompt tries to recover. Disable "
                             "this hook if your corpus has tall narrow "
                             "non-rotated tables that the heuristic "
                             "wrongly flips.")
    g_pipe.add_argument("--rescue-sparse-pages", action="store_true",
                        help="opt INTO the sparse-page rescue (hook 3). "
                             "Default OFF since v0.2: the density-only "
                             "trigger fires on legitimate-but-short pages "
                             "(end-of-references tails, figure-dominant "
                             "pages) and the rescue output is appended as "
                             "a tail '## VLM page rescues' section rather "
                             "than spliced into the body. Useful on "
                             "scanned PDFs / Adobe-Paper-Capture-vintage "
                             "OCR corpora where the text layer is "
                             "genuinely broken and you want a VLM-rendered "
                             "backup of every low-density page. Adds one "
                             "VLM call per sparse page (~30s-2min each on "
                             "a 32B model). Always skipped under --no-vlm.")
    g_pipe.add_argument("--no-data-repos", action="store_true",
                        help="skip the data-repository link extraction "
                             "post-pass (default on; scans the body for "
                             "DOIs/URLs pointing at Zenodo, Dryad, "
                             "Dataverse, figshare, OSF, PANGAEA, ESS-DIVE, "
                             "Mendeley Data, ICPSR, CaltechDATA and emits "
                             "a 'data:' YAML block + JSON sidecar entry). "
                             "Pure regex; no network unless paired with "
                             "--fetch-data-repos.")
    g_pipe.add_argument("--fetch-data-repos", action="store_true",
                        help="for each detected data-repository link, "
                             "fetch a one-shot summary from the repo's "
                             "public API (title, description, license, "
                             "file list). One HTTP GET per unique deposit. "
                             "Off by default to keep runs hermetic. "
                             "Failures fall through silently -- the link "
                             "is still recorded.")
    g_pipe.add_argument("--data-repos-timeout", type=float, default=8.0,
                        metavar="SEC",
                        help="per-API timeout for --fetch-data-repos "
                             "(default: 8).")
    g_pipe.add_argument("--figmatch-strategy", default=None,
                        metavar="STRATEGY",
                        help="hook-2 figure-caption matching strategy. "
                             "Default: 'dup-detect' (validated as the "
                             "best-performing setting in the project's test "
                             "corpus). Other tokens: 'single' (1 VLM call/"
                             "image, no cross-image post-pass), 'page-prior' "
                             "(append 'extracted from page N' to the prompt), "
                             "'vote' (3× majority). Combine with '+' "
                             "(e.g. 'page-prior+dup-detect'). See "
                             "docs/FIGURE_MATCH_TEST_PLAN.md.")
    g_pipe.add_argument("--skip-vlm-check", action="store_true",
                        help="skip the preflight VLM reachability check "
                             "(default on; pings the configured provider "
                             "before starting the marker run so a "
                             "misconfigured provider fails fast instead of "
                             "after a long extraction)")

    g_tab = p.add_argument_group("table extraction")
    g_tab.add_argument("--table-finder",
                       choices=["pymupdf", "tatr", "both", "docling", "all"],
                       default="pymupdf",
                       help="region detector for hook 1 (only used under "
                            "--layout-source=marker; ignored when "
                            "--layout-source=mineru since MinerU has its "
                            "own table extractor). 'pymupdf' (DEFAULT "
                            "since v0.3.x) is the lightweight built-in "
                            "detector; ruled-line tables only. 'tatr' "
                            "uses Microsoft Table Transformer (catches "
                            "borderless tables in vector-text PDFs; needs "
                            "transformers+torch). 'both' unions pymupdf + "
                            "tatr. 'docling' uses IBM Docling's "
                            "TableFormer for borderless / multi-row-"
                            "header / scanned tables -- one-shot whole-"
                            "PDF scan, ~7-10 s on a 16-page paper. "
                            "Optional dep since v0.3.x: install with "
                            "`pip install docling==2.92.0`. 'all' chains "
                            "pymupdf + tatr + docling (first-match-wins, "
                            "cheapest first); needs docling installed.")
    g_tab.add_argument("--table-workers", type=int, default=1,
                       metavar="N",
                       help="run hook 1's per-table VLM calls concurrently "
                            "(default 1 = serial, current behavior). "
                            "Each table's VLM call is independent; vLLM and "
                            "LM Studio handle concurrent requests fine, so "
                            "values like 4 typically halve the table phase "
                            "wall time on multi-table papers. Setting too "
                            "high risks queue contention with the "
                            "citation/figure/rescue VLM calls; 4-6 is a "
                            "safe upper bound on a single 32B-model server.")
    g_tab.add_argument("--vlm-tables", action="store_true",
                       help="opt INTO the per-table VLM rewrite. Default "
                            "(off) is fast: located tables get docling's "
                            "pre-extracted markdown directly as the table "
                            "body, no per-table VLM call. Pass --vlm-tables "
                            "to ask the VLM to re-transcribe each table "
                            "JPEG crop -- adds 1-8 minutes per dense table "
                            "but recovers subscript / Greek / footnote-"
                            "marker fidelity that detector text loses. "
                            "Pair with --table-workers 4 for concurrent "
                            "VLM calls when running on a multi-table paper.")
    g_tab.add_argument("--no-vlm-tables", action="store_true",
                       help=argparse.SUPPRESS)   # deprecated, see --vlm-tables
    g_tab.add_argument("--vlm-tables-force", action="store_true",
                       help="force VLM rewrite on EVERY table for the "
                            "sidecar `.md`, even tables whose cheap "
                            "HTML/detector conversion is already clean. "
                            "Body keeps the cheap conversion when "
                            "available (so what readers see inline is "
                            "unchanged); the sidecar contains the "
                            "higher-fidelity VLM output. Implies "
                            "--vlm-tables. Under marker layout this is "
                            "a no-op (marker's --vlm-tables already "
                            "rewrites every located table). Under "
                            "mineru / hybrid this triggers extra VLM "
                            "calls on clean tables -- expect ~30s per "
                            "additional table on Spark + Qwen3-VL.")

    g_vlm = p.add_argument_group("VLM provider / compute backend")
    g_vlm.add_argument("--provider",
                       choices=["lmstudio", "vllm", "openai", "anthropic"],
                       default=None,
                       help="VLM provider. Default: $VLM_PROVIDER, else "
                            "auto-paired with --backend ('vllm' for cuda, "
                            "'lmstudio' for mps/cpu). 'vllm' hits "
                            "$VLLM_BASE_URL (default http://localhost:8000/v1); "
                            "'openai' requires $OPENAI_API_KEY; 'anthropic' "
                            "requires $ANTHROPIC_API_KEY. Override model "
                            "per-provider with $VLM_MODEL (defaults: "
                            "qwen3-vl-32b-instruct-mlx, "
                            "Qwen/Qwen3-VL-32B-Instruct, gpt-4o, "
                            "claude-sonnet-4-6).")
    g_vlm.add_argument("--backend", choices=["auto", "cuda", "mps", "cpu"],
                       default="auto",
                       help="compute backend for marker/surya and TATR. "
                            "'auto' (default) picks cuda > mps > cpu by "
                            "torch availability. 'cpu' is a safety net for "
                            "MPS bugs on Apple Silicon and for CUDA-less "
                            "machines.")
    g_vlm.add_argument("--cpu", action="store_true",
                       help="deprecated alias for --backend cpu")
    g_vlm.add_argument("--vlm-temperature", type=float, default=None,
                       metavar="FLOAT",
                       help="sampling temperature sent on EVERY VLM "
                            "request. Default 0.0 (deterministic); "
                            "lower = less random, 0 = greedy. "
                            "Overrides $VLM_TEMPERATURE if set. "
                            "Recorded in run.vlm_temperature in the "
                            "output frontmatter for reproducibility. "
                            "vLLM / LM Studio / OpenAI / Anthropic all "
                            "honor this parameter.")
    g_vlm.add_argument("--vlm-seed", type=int, default=None,
                       metavar="INT",
                       help="seed sent on EVERY VLM request to make "
                            "sampling reproducible at temperature > 0. "
                            "Default 42. Set to a negative value to "
                            "DISABLE seed (server picks a random one "
                            "per request). Overrides $VLM_SEED if set. "
                            "Recorded in run.vlm_seed in the output "
                            "frontmatter. Anthropic ignores this "
                            "parameter (no seed in the Messages API).")

    g_batch = p.add_argument_group("batch mode")
    g_batch.add_argument("--batch", default=None, metavar="PATH_OR_GLOB",
                         help="folder, glob, or list of PDFs to process. "
                              "Each paper lands in its own subdir under -o, "
                              "with a manifest.jsonl summarizing the run. "
                              "Mutually exclusive with the positional pdf "
                              "arg and with --supplement.")
    g_batch.add_argument("--workers", type=int, default=1,
                         help="parallel papers in --batch mode (default 1). "
                              "Marker stays serialized on the GPU; workers "
                              "> 1 lets the VLM hooks of one paper overlap "
                              "Marker on the next.")
    g_batch.add_argument("--force", action="store_true",
                         help="re-run papers in --batch mode even if their "
                              "output already exists (default: skip = "
                              "resume).")
    g_batch.add_argument("--no-pair", action="store_true",
                         help="disable supplement auto-pairing in --batch "
                              "mode (every PDF runs standalone).")
    g_batch.add_argument("--pair-regex", default=None, metavar="REGEX",
                         help="override the default SI-detection regex used "
                              "in --batch mode. Must define a 'stem' named "
                              "group capturing the main paper's stem.")
    g_batch.add_argument("--paper-timeout", type=float, default=2700.0,
                         metavar="SECONDS",
                         help="abandon any single paper that exceeds this "
                              "wall-time limit (default 2700 = 45 min). "
                              "Bumped from 1800 in 2026-05 after hybrid "
                              "runs on wide-table papers (Wackerle1962, "
                              "cuk SI) finished cleanly just past the 30 "
                              "min mark. Pass 0 to disable. Applies in "
                              "--batch mode only. The timed-out paper is "
                              "recorded in manifest.jsonl with "
                              "status='timeout' and the batch continues "
                              "with the next paper. CAVEAT: the "
                              "abandoned worker thread keeps running in "
                              "the background and may hold GPU memory "
                              "or the marker lock; subsequent papers in "
                              "the same process can be slowed "
                              "or blocked. For very long batches with "
                              "frequent hangs, split into multiple "
                              "paper2md invocations and let each restart "
                              "with a clean Python process.")

    g_meta = p.add_argument_group("metadata / copyright front-end")
    g_meta.add_argument("--no-metadata-lookup", action="store_true",
                        help="skip the copyright/open-access metadata pre-"
                             "pass. By default each PDF is checked against "
                             "OpenAlex / Unpaywall / Europe PMC / OSTI (DOE "
                             "PAGES) / arXiv to fill the YAML 'copyright:' "
                             "block. Polite-pool emails come from "
                             "$UNPAYWALL_EMAIL, $OPENALEX_MAILTO, "
                             "$CROSSREF_MAILTO.")
    g_meta.add_argument("--metadata-only", action="store_true",
                        help="run ONLY the metadata pre-pass and exit; do "
                             "not run marker, the VLM, or any extraction. "
                             "Useful for screening copyright/OA status "
                             "across a corpus before deciding which PDFs to "
                             "fully extract. Single-paper: prints the YAML "
                             "'copyright:' block to stdout. Batch: writes "
                             "one JSON record per PDF to <out>/manifest.jsonl "
                             "and prints a one-line summary per paper. "
                             "Honors --prefer-oa-source (downloads OA copy "
                             "if found) and --require-license (exit 3 on "
                             "unsafe license).")
    g_meta.add_argument("--prefer-oa-source", action="store_true",
                        help="when a green/yellow-licensed open-access PDF "
                             "is found on a public repository, download it "
                             "and run the pipeline on that copy instead of "
                             "the local PDF. Provenance recorded in front-"
                             "matter (oa_pdf_used / oa_pdf_source).")
    g_meta.add_argument("--metadata-timeout", type=float, default=10.0,
                        metavar="SEC",
                        help="per-API request timeout for the metadata pre-"
                             "pass (default: 10).")
    g_meta.add_argument("--metadata-cache", type=Path, default=None,
                        metavar="PATH",
                        help="JSON cache for DOI->metadata lookups (default: "
                             "~/.cache/paper2md/metadata.json).")
    g_meta.add_argument("--require-license", action="store_true",
                        help="exit non-zero (code 3) if the resolved license "
                             "is not safe to redistribute. Passes for "
                             "safe_to_distribute in {true, restricted}; "
                             "fails for readable (publicly readable but not "
                             "redistributable, e.g. PMC author manuscripts) "
                             "and false (closed / unknown). Distinct from "
                             "--quality-threshold (code 1).")
    g_meta.add_argument("--offline", action="store_true",
                        help="run with NO scholarly-API network access. "
                             "Skips copyright/preprint metadata resolution, "
                             "Crossref/OpenAlex references fallback, OA PDF "
                             "download, and data-repository enrichment. "
                             "Local VLMs (lmstudio, vllm) still work; "
                             "--provider openai/anthropic will emit a "
                             "warning. Conflicting flags (--prefer-oa-"
                             "source, --fetch-data-repos) are auto-disabled "
                             "with a warning. Skipped steps are recorded "
                             "in the YAML frontmatter "
                             "(copyright.provenance: network-disabled, "
                             "run.pipeline.offline: true, "
                             "quality.references.api_skipped_offline) so "
                             "downstream consumers can see the gaps. "
                             "Cached metadata / refs from prior online "
                             "runs are still used.")

    g_user = p.add_argument_group("user annotations (recorded in front-matter)")
    g_user.add_argument("--user", default=None, metavar="NAME",
                        help="user name to record under user.user in the "
                             "YAML front-matter. Falls back to "
                             "$PAPER2MD_USER if unset. Useful for "
                             "attributing curated extractions in a "
                             "shared corpus.")
    g_user.add_argument("--collection", default=None, metavar="NAME",
                        help="collection / project label to record under "
                             "user.collection (e.g. 'moon-formation', "
                             "'shock-physics-2026'). Falls back to "
                             "$PAPER2MD_COLLECTION if unset. Applied to "
                             "every paper in --batch mode.")
    g_user.add_argument("--note", default=None, metavar="TEXT",
                        help="free-form note recorded under user.note. "
                             "Multi-line text is preserved via YAML "
                             "block-scalar emission. In --batch mode the "
                             "same note is applied to every paper; edit "
                             "per-paper notes in the output YAML afterward.")

    g_edit = p.add_argument_group(
        "manual edit (post-pipeline)",
        description="Operate on an existing paper2md output to "
                    "replace a garbled table or a mis-spliced figure. "
                    "These flags short-circuit the pipeline -- no PDF "
                    "needed. Logs the action in the paper's frontmatter "
                    "`edits:` block for auditability. See USAGE.md §16.")
    g_edit.add_argument("--replace-table", nargs=3, default=None,
                        metavar=("ID", "IMAGE", "PAPER_MD"),
                        help="VLM-transcribe an image crop of a single "
                             "table and write the result as a sidecar "
                             "`.md` next to PAPER_MD's assets/ folder. "
                             "The user pastes the sidecar contents over "
                             "the garbled table by hand; body is not "
                             "modified. Idempotent by ID -- re-running "
                             "overwrites the prior entry.")
    g_edit.add_argument("--replace-fig", nargs=3, default=None,
                        metavar=("ID", "IMAGE", "PAPER_MD"),
                        help="Swap a figure's asset in-place. Archives "
                             "the original as `<asset>.orig` (only on "
                             "first run; idempotent) and overwrites the "
                             "bound asset path. No VLM call. Body and "
                             "caption stay as-is.")
    g_edit.add_argument("--revert-edit", nargs=3, default=None,
                        metavar=("KIND", "ID", "PAPER_MD"),
                        help="Revert a prior --replace-table / "
                             "--replace-fig. For figure: restores "
                             "`<asset>.orig` over the replaced file and "
                             "deletes the archive. For table: drops "
                             "the frontmatter entry only (sidecar and "
                             "image crop stay on disk; manual body "
                             "edits are not unwound).")
    g_edit.add_argument("--recover-from-mineru", type=Path, default=None,
                        metavar="PAPER_MD",
                        help="After a hybrid run, scan PAPER_MD's audit "
                             "for tables referenced but not inserted, "
                             "and stage any that MinerU detected (via "
                             "the sibling mineru/<stem>_middle.json) "
                             "into assets/ + a `pending_recoveries:` "
                             "frontmatter block. Body is NOT modified; "
                             "user pastes from the sidecar(s). Preserves "
                             "the original run's vlm_rewrite_tables "
                             "setting unless --vlm / --no-vlm overrides.")
    g_edit.add_argument("--confirm-recovery", nargs=3, default=None,
                        metavar=("KIND", "ID", "PAPER_MD"),
                        help="Mark a pending recovery applied: move the "
                             "`pending_recoveries:` entry to `edits:` "
                             "with `recovered_from: mineru` audit trail. "
                             "Run after pasting the sidecar contents.")
    g_edit.add_argument("--recovery-vlm", dest="recovery_vlm",
                        action="store_const", const=True, default=None,
                        help="Only with --recover-from-mineru: force "
                             "VLM transcription even if the original "
                             "run was --no-vlm-tables. Useful for "
                             "spot-upgrading a few tables.")
    g_edit.add_argument("--recovery-no-vlm", dest="recovery_vlm",
                        action="store_const", const=False,
                        help="Only with --recover-from-mineru: force "
                             "MinerU HTML conversion (skip VLM). Useful "
                             "when the VLM server is offline.")

    g_log = p.add_argument_group("logging")
    g_log.add_argument("-v", "--verbose", action="store_true",
                       help="DEBUG-level logging (shows per-table, per-page, "
                            "per-figure decisions)")

    args = p.parse_args()

    # Manual edit / recovery short-circuit. These flags operate on an
    # existing paper2md output, not a PDF, so they bypass all the
    # pipeline validation/setup below. Mutex with each other and with
    # --batch / positional pdf. Dispatched after a minimal logging
    # setup.
    _manual_flags = sum(
        bool(x) for x in (args.replace_table, args.replace_fig,
                          args.revert_edit, args.recover_from_mineru,
                          args.confirm_recovery))
    if _manual_flags > 1:
        p.error("--replace-table, --replace-fig, --revert-edit, "
                "--recover-from-mineru, and --confirm-recovery "
                "are mutually exclusive")
    if _manual_flags == 1:
        if args.pdf is not None or args.batch is not None:
            p.error("manual-edit flags operate on an existing .md, "
                    "not a PDF; positional pdf and --batch are not "
                    "supported")
        if args.recovery_vlm is not None and not args.recover_from_mineru:
            p.error("--recovery-vlm / --recovery-no-vlm only apply to "
                    "--recover-from-mineru")
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG if args.verbose else logging.INFO,
        )
        return _run_manual_edit(args, p)

    if args.text_only and args.no_vlm:
        p.error("--text-only requires the VLM; incompatible with --no-vlm")
    if args.no_vlm_tables:
        log.warning("--no-vlm-tables is deprecated; the behavior it "
                    "selected is now the default. Pass --vlm-tables to "
                    "opt INTO the VLM-rewrite path.")
    if args.vlm_tables and args.no_vlm:
        p.error("--vlm-tables requires the VLM; incompatible with --no-vlm")
    if args.vlm_tables_force and args.no_vlm:
        p.error("--vlm-tables-force requires the VLM; incompatible "
                "with --no-vlm")
    if args.vlm_tables_force and not args.vlm_tables:
        # Auto-imply: --vlm-tables-force is a superset of --vlm-tables.
        # Erroring would be pedantic when intent is obvious.
        args.vlm_tables = True
        log.info("--vlm-tables-force implies --vlm-tables (auto-set)")
    if args.vlm_tables and args.table_finder == "pymupdf":
        log.warning("--vlm-tables works best with --table-finder docling; "
                    "PyMuPDF's text candidates are raw cell content and "
                    "rarely a clean substitute when VLM rewrite fails.")
    if args.metadata_only and args.no_metadata_lookup:
        p.error("--metadata-only and --no-metadata-lookup are contradictory")

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    # Attribution banner: 3 INFO lines so the authors, license, and a
    # citable reference are visible in every run's logs and in any
    # captured tee/log file. Comes BEFORE backend / VLM provider lines
    # so a user redirecting log output sees the citation in the first
    # few lines without having to scroll.
    log.info("paper2md v%s (%s License)", __version__, __license__)
    log.info("Authors: %s", __authors__)
    if __doi__ and not __doi__.endswith("RESERVED"):
        log.info("Cite: %s",
                 __citation_with_doi__.format(version=__version__,
                                              license=__license__,
                                              doi=__doi__))
    else:
        log.info("Cite: %s",
                 __citation__.format(version=__version__,
                                     license=__license__))

    # Typo guard: warn if the .env or shell sets LM_*-prefixed variables
    # that paper2md doesn't read. The codebase has both VLM_* and LM_*
    # variables (see USAGE §13) and the prefix mismatch is a well-known
    # footgun -- e.g., LM_MODEL silently falls back to the per-provider
    # default instead of overriding the model. List of expected
    # corrections; LM_STUDIO_URL is intentionally NOT flagged because
    # that's the canonical name from LM Studio's docs and is what
    # paper2md actually reads.
    _LM_TYPO_FIXES = {
        "LM_MODEL": "VLM_MODEL",
        "LM_PROVIDER": "VLM_PROVIDER",
        "LLM_BASE_URL": "VLLM_BASE_URL",
        "LLM_PROVIDER": "VLM_PROVIDER",
        "LLM_MODEL": "VLM_MODEL",
    }
    for _wrong, _right in _LM_TYPO_FIXES.items():
        if _wrong in os.environ and _right not in os.environ:
            log.warning("Environment variable %s is set but paper2md "
                        "reads %s -- the value is being ignored. "
                        "Rename in your .env (or shell) to %s.",
                        _wrong, _right, _right)

    backend = "cpu" if args.cpu else args.backend
    if backend == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                backend = "cuda"
            elif torch.backends.mps.is_available():
                # surya's MPS path produces silently corrupted body
                # text (OCR character drops, scrambled reading order,
                # lost equations) even when it doesn't crash. Float-
                # precision differences in PyTorch MPS kernels drift
                # the OCR decoder's argmax sampling away from the
                # CUDA / CPU reference. See USAGE.md §6.0. Default to
                # CPU on Apple Silicon and let users explicitly
                # --backend mps to opt in.
                backend = "cpu"
                log.info("Apple Silicon detected; defaulting to CPU "
                         "because surya's MPS output is unreliable. "
                         "Pass --backend mps to opt in (output may "
                         "contain corrupted text -- see USAGE.md "
                         "§6.0).")
            else:
                backend = "cpu"
        except ImportError:
            backend = "cpu"
    if backend == "mps":
        log.warning("--backend mps: surya's MPS path produces "
                    "silently corrupted body text on Apple Silicon "
                    "(OCR character drops, scrambled reading order, "
                    "lost equations). Compare against a CUDA / CPU "
                    "reference run before trusting the output. See "
                    "USAGE.md §6.0.")
    os.environ["TORCH_DEVICE"] = backend
    if backend == "cpu":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    log.info("Backend: %s (marker/surya/TATR)", backend)

    # Auto-pair provider with backend when the user didn't explicitly choose
    # one: vllm for cuda (the natural NVIDIA partner), lmstudio for mps/cpu
    # (the natural Apple Silicon partner). $VLM_PROVIDER, if set, still wins.
    if args.provider is None and not os.environ.get("VLM_PROVIDER"):
        args.provider = "vllm" if backend == "cuda" else "lmstudio"

    global FIX_REFERENCES, TRIM_ARTICLES, CONSOLIDATE_FOOTNOTES, NORMALISE_REFS, STRIP_RUNNING_FOOTERS, STRIP_PAGE_HEADERS, STRIP_PUBLISHER_STAMPS, STRIP_MATH_LABELS, MERGE_REFERENCES, INJECT_ORPHAN_REFS, TIDY_REFS, RESCUE_ORPHAN_TABLES, RESCUE_ORPHAN_FIGURES, ROTATE_TABLES, FIGMATCH_STRATEGY, DATA_REPOS, FETCH_DATA_REPOS, FETCH_DATA_REPOS_TIMEOUT, RESCUE_JOURNAL_REFS, REF_RESCUE_THRESHOLD, REF_RESCUE_DRY_RUN, OFFLINE
    if args.no_fix_refs:
        FIX_REFERENCES = False
    if args.no_trim_articles:
        TRIM_ARTICLES = False
    if args.no_consolidate_footnotes:
        CONSOLIDATE_FOOTNOTES = False
    if args.no_normalise_refs:
        NORMALISE_REFS = False
    if args.no_strip_footers:
        STRIP_RUNNING_FOOTERS = False
    if args.no_strip_page_headers:
        STRIP_PAGE_HEADERS = False
    if args.no_strip_publisher_stamps:
        STRIP_PUBLISHER_STAMPS = False
    if args.no_merge_references:
        MERGE_REFERENCES = False
    if args.no_inject_orphan_refs:
        INJECT_ORPHAN_REFS = False
    if args.no_tidy_refs:
        TIDY_REFS = False
    if args.no_strip_math_labels:
        STRIP_MATH_LABELS = False
    if args.no_rescue_orphan_tables:
        RESCUE_ORPHAN_TABLES = False
    if args.no_rescue_orphan_figures:
        RESCUE_ORPHAN_FIGURES = False
    if args.use_journal_rescue:
        RESCUE_JOURNAL_REFS = True
    if args.ref_rescue_threshold is not None:
        REF_RESCUE_THRESHOLD = float(args.ref_rescue_threshold)
    if args.ref_rescue_dry_run:
        REF_RESCUE_DRY_RUN = True
    if args.no_rotate_tables:
        ROTATE_TABLES = False
    if args.no_data_repos:
        DATA_REPOS = False
    if args.fetch_data_repos:
        FETCH_DATA_REPOS = True
    FETCH_DATA_REPOS_TIMEOUT = args.data_repos_timeout
    if args.offline:
        OFFLINE = True
        if args.prefer_oa_source:
            log.warning("--offline: --prefer-oa-source disabled "
                        "(would require network)")
            args.prefer_oa_source = False
        if FETCH_DATA_REPOS:
            log.warning("--offline: --fetch-data-repos enrichment "
                        "disabled (would require network); detected "
                        "data-repo links will still be recorded")
            FETCH_DATA_REPOS = False
            args.fetch_data_repos = False
        if args.provider in ("openai", "anthropic"):
            log.warning("--offline: --provider %s requires network "
                        "for VLM calls; expect failures. Switch to "
                        "lmstudio/vllm or pass --no-vlm.",
                        args.provider)
        log.info("Mode: OFFLINE -- copyright lookup, references API, "
                 "OA PDF download, and data-repo enrichment skipped")
    if args.figmatch_strategy:
        valid_tokens = {"single", "page-prior", "dup-detect", "vote"}
        tokens = set(args.figmatch_strategy.split("+"))
        unknown = tokens - valid_tokens
        if unknown:
            p.error(f"--figmatch-strategy: unknown token(s) {sorted(unknown)}; "
                    f"valid: {sorted(valid_tokens)}")
        FIGMATCH_STRATEGY = args.figmatch_strategy

    if args.batch and args.pdf is not None:
        p.error("pass either a positional pdf or --batch, not both")
    if args.batch is None and args.pdf is None:
        p.error("missing pdf path (or use --batch)")
    if args.force_ocr and args.auto_force_ocr:
        p.error("--force-ocr and --auto-force-ocr are mutually exclusive; "
                "pass --force-ocr to force OCR on every paper, "
                "--auto-force-ocr to auto-detect per paper")
    if args.auto_layout_source and args.layout_source is not None:
        p.error("--layout-source and --auto-layout-source are mutually "
                "exclusive; pass --layout-source to force one engine for "
                "every paper, --auto-layout-source to route per paper "
                "via the triage detector")
    if args.layout_source is None:
        args.layout_source = "mineru"  # resolve sentinel default
    if args.skip_mineru_run and args.mineru_dir is None:
        p.error("--skip-mineru-run requires --mineru-dir to point at an "
                "existing MinerU output (auto/ subdir or flat layout dir)")
    if args.skip_mineru_run and args.layout_source == "marker":
        p.error("--skip-mineru-run is incompatible with "
                "--layout-source=marker (no MinerU run to skip)")
    # Warn (don't error) when mineru-control flags are passed under
    # marker layout source; they're inert but the user may have
    # forgotten to pass --layout-source=mineru.
    if args.layout_source == "marker" and not args.auto_layout_source:
        if args.mineru_backend != "pipeline":
            log.warning("--mineru-backend has no effect under "
                        "--layout-source=marker")
        if args.mineru_arg:
            log.warning("--mineru-arg has no effect under "
                        "--layout-source=marker")
        if args.strip_chart_details:
            log.warning("--strip-chart-details has no effect under "
                        "--layout-source=marker (MinerU isn't running)")
    if args.batch and args.supplement is not None:
        p.error("--supplement is for single-paper mode; in --batch, "
                "supplements are auto-paired by filename")
    if args.pdf is not None and not args.pdf.is_file():
        p.error(f"pdf not found: {args.pdf}")
    if args.supplement is not None and not args.supplement.is_file():
        p.error(f"supplement pdf not found: {args.supplement}")

    # Validate the output path early so the user gets a clean error
    # before marker, the metadata pre-pass, or the VLM preflight burn
    # any time. Failure modes covered: path exists but is a regular
    # file (not a directory); parent doesn't exist and can't be
    # created (no permission); the path itself isn't writable.
    try:
        if args.out.exists() and not args.out.is_dir():
            p.error(f"output path exists but is not a directory: "
                    f"{args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        # Probe writability with a touch + unlink. Catches read-only
        # mounts and permission denials that mkdir's exist_ok=True
        # silently ignores when the directory already exists.
        _probe = args.out / ".paper2md_write_probe"
        try:
            _probe.touch()
        finally:
            try:
                _probe.unlink()
            except FileNotFoundError:
                pass
    except PermissionError as e:
        p.error(f"cannot write to output path {args.out}: {e}")
    except OSError as e:
        p.error(f"invalid output path {args.out}: {e}")

    # --metadata-only: short-circuit before any VLM/marker setup. We only
    # need the metadata pre-pass (network calls) and an OA download if
    # --prefer-oa-source is set. No GPU, no marker model load, no VLM.
    if args.metadata_only:
        return _run_metadata_only(args, p)

    # Apply --vlm-temperature / --vlm-seed BEFORE configure_client so the
    # startup log line reflects what the run will actually use. CLI > env
    # var > module default (0.0 / 42).
    global VLM_TEMPERATURE, VLM_SEED
    if args.vlm_temperature is not None:
        VLM_TEMPERATURE = args.vlm_temperature
    if args.vlm_seed is not None:
        VLM_SEED = args.vlm_seed if args.vlm_seed >= 0 else None

    configure_client(provider=args.provider)

    # Build the run-info record AFTER configure_client so we capture the
    # resolved provider/model (post-env-var, post-auto-pair). The CLI is
    # shell-quoted so the recorded command can be copy-pasted to reproduce.
    import shlex as _shlex
    import socket as _socket
    # vlm_endpoint: read directly off the active client when the OpenAI-
    # compat path is in use; the Anthropic SDK has no public base_url
    # attribute on its client, so record the canonical hostname instead.
    if _PROVIDER == "anthropic":
        _vlm_endpoint = "https://api.anthropic.com"
    else:
        _vlm_endpoint = str(getattr(client, "base_url", "") or "")
    # text_engine reflects which OCR produced the BODY text. Hybrid
    # uses marker (surya) for body even though MinerU contributes
    # figure assets and table HTML; the split is also recorded under
    # pipeline.figure_layout_source / pipeline.caption_text_source.
    if args.layout_source == "mineru":
        _text_engine = "mineru-paddleocr"
    else:
        # marker and hybrid both run marker for body text.
        _text_engine = "marker-surya"
    run_info_template = RunInfo(
        command=" ".join(_shlex.quote(a) for a in sys.argv),
        hostname=_socket.gethostname(),
        vlm_provider=_PROVIDER,
        vlm_model=VLM_MODEL,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        compute_backend=backend,
        layout_source=args.layout_source,
        text_engine=_text_engine,
        vlm_endpoint=_vlm_endpoint,
        vlm_system_prompt=SYSTEM,
        vlm_temperature=VLM_TEMPERATURE,
        vlm_seed=VLM_SEED,
        vlm_inference_params=(
            f"paper2md sets temperature={VLM_TEMPERATURE} and "
            f"seed={VLM_SEED if VLM_SEED is not None else 'unset'} on "
            "every VLM request (deterministic by default; override "
            "with --vlm-temperature / --vlm-seed or "
            "$VLM_TEMPERATURE / $VLM_SEED). top_p and other sampling "
            "params remain at server defaults. Per-call max_tokens "
            "varies from 5 (small probes) to 6000 (wide-table "
            "rewrites). Anthropic ignores seed (Messages API limit). "
            "Record server-side config (vLLM launch flags, "
            "quantization, model hash, hardware) separately for full "
            "publication-grade reproducibility (see USAGE.md §18)."
        ),
        pipeline=_collect_pipeline_state(
            args.table_finder,
            vlm_rewrite_tables=args.vlm_tables,
            rescue_sparse_pages=args.rescue_sparse_pages,
            force_ocr=args.force_ocr,
            layout_source=args.layout_source,
            rescue_mineru_orphan_captions=not args.no_rescue_orphan_captions,
            rescue_mineru_subpanels=not args.no_rescue_subpanels),
        packages=_collect_package_versions(args.layout_source),
        # started_at and elapsed_sec are filled in inside convert()
        # (started_at right at the top, elapsed_sec right before
        # frontmatter emit). Each per-paper convert() call gets a
        # fresh copy via dataclass replace so batch-mode papers don't
        # share a mutable state.
    )

    # User annotations: CLI flag wins, then env var, else None. Same
    # UserAnnotations instance is shared across batch papers (it's
    # immutable from the pipeline's perspective).
    _user_anno = UserAnnotations(
        user=args.user or os.environ.get("PAPER2MD_USER") or None,
        collection=args.collection or os.environ.get("PAPER2MD_COLLECTION") or None,
        note=args.note or None,
    )
    if _user_anno.is_empty():
        _user_anno = None

    # Preflight: VLM is used by hooks 0-3 (and is the entire pipeline in
    # --text-only mode). Ping the configured endpoint before kicking off the
    # multi-minute marker run, so a misconfigured provider fails in seconds
    # instead of after the whole extraction.
    needs_vlm = ((not args.no_vlm) or args.text_only) and not args.scan_detect_only
    if needs_vlm and not args.skip_vlm_check:
        err = check_vlm_reachable()
        if err is not None:
            log.error("VLM preflight failed: %s", err)
            log.error("To bypass this check, pass --skip-vlm-check (or "
                      "--no-vlm to skip the VLM hooks entirely).")
            sys.exit(2)
        log.info("VLM preflight OK")

    def _do_convert(pdf_path: Path, out_dir: Path, asset_prefix: str = ""):
        # Metadata pre-pass: CPU-only, runs before marker so OA substitution
        # (if requested) can swap the input PDF before the GPU is touched.
        # Best-effort: any exception falls back to no metadata.
        # When --prefer-oa-source is in play the OA PDF is downloaded into
        # out_dir as <stem>_oa_source.pdf so the user can inspect/archive it
        # alongside the produced markdown; the markdown filename is forced
        # to the original stem via output_stem to avoid renaming-by-swap.
        original_stem = pdf_path.stem
        metadata = None
        if not args.no_metadata_lookup:
            try:
                from metadata_frontend import resolve as resolve_metadata
                metadata = resolve_metadata(
                    pdf_path,
                    allow_network=not OFFLINE,
                    prefer_oa=args.prefer_oa_source,
                    timeout_s=args.metadata_timeout,
                    cache_path=args.metadata_cache,
                    oa_download_dir=out_dir,
                )
                if metadata.oa_pdf_used and metadata.oa_pdf_local_path:
                    log.info("Using OA copy from %s (license: %s); kept at %s",
                             metadata.oa_pdf_source, metadata.license,
                             metadata.oa_pdf_local_path)
                    pdf_path = metadata.oa_pdf_local_path
                else:
                    log.info("Copyright: license=%s, safe_to_distribute=%s, "
                             "confidence=%s, via=%s",
                             metadata.license or "unknown",
                             metadata.safe_to_distribute,
                             metadata.confidence,
                             metadata.resolved_via or "none")
            except Exception as e:
                log.warning("Metadata lookup failed (%s); continuing without "
                            "copyright block", e)
                metadata = None

        if args.require_license and (
                metadata is None
                or metadata.safe_to_distribute in {"false", "readable"}):
            log.error("--require-license: license is not safe to redistribute "
                      "(license=%s, safe_to_distribute=%s)",
                      metadata.license if metadata else "none-resolved",
                      metadata.safe_to_distribute if metadata else "n/a")
            sys.exit(3)

        # Fresh RunInfo per paper -- elapsed_sec is mutated inside
        # convert() and we don't want batch papers to share state.
        from dataclasses import replace as _replace
        per_paper_run_info = _replace(run_info_template)

        # Resolve --auto-layout-source per paper. Compose two
        # PyMuPDF detectors:
        #   (1) triage_pdfs.recommend_routing: text-layer richness
        #       (super flags, font sizes, char density). Catches
        #       Lyzenga-style flat OCR (super_flag_count==0).
        #   (2) detect_scan_ocr_pdf: scan signature (producer
        #       metadata, image cover, bold-char ratio). Catches
        #       Wackerle-style scans whose OCR happens to leave
        #       text-layer artifacts that fool triage.
        # Routing: marker if EITHER detector says so. Force-OCR:
        # auto-applies when scan_detect fires AND routing=marker
        # (avoids force-OCRing alexander_sm-style "no supers but
        # not a scan" papers). --no-auto-force-ocr opts out.
        layout_source_for_paper = args.layout_source
        triage_signals = None
        triage_routing = None
        triage_reason = None
        scan_signals = None
        scan_detected = False
        auto_force_ocr_applied = False
        if args.auto_layout_source:
            try:
                import triage_pdfs as _triage
                stats = _triage.analyze_pdf(pdf_path)
                triage_routing, triage_reason = _triage.recommend_routing(stats)
                triage_signals = {
                    "pages": stats.pages,
                    "chars_per_page": round(stats.chars_per_page, 1),
                    "distinct_font_sizes": stats.distinct_font_sizes,
                    "super_flag_count": stats.super_flag_count,
                    "style_flag_pct": round(stats.style_flag_pct, 3),
                    "unknown_char_pct": round(stats.unknown_char_pct, 3),
                }
                if stats.error:
                    triage_signals["error"] = stats.error
            except Exception as e:
                log.warning("Auto-layout-source triage failed for %s "
                            "(%s); treating as 'mineru' fallback",
                            pdf_path.name, e)
                triage_routing = "mineru-paddleocr"
                triage_reason = f"triage-error: {e}"
            try:
                scan_detected, scan_signals_full = detect_scan_ocr_pdf(pdf_path)
                # Trim signals_full to a YAML-friendly subset.
                scan_signals = {
                    "verdict": "scan" if scan_detected else "digital",
                    "reason": scan_signals_full.get("force_ocr_reason", ""),
                    "producer": scan_signals_full.get("producer", ""),
                    "creator": scan_signals_full.get("creator", ""),
                    "image_cover_median": round(
                        scan_signals_full.get("image_cover_median", 0.0), 3),
                    "font_count": scan_signals_full.get("font_count", 0),
                    "bold_char_ratio": round(
                        scan_signals_full.get("bold_char_ratio", 0.0), 3),
                    "n_pages": scan_signals_full.get("n_pages", 0),
                }
            except Exception as e:
                log.warning("Auto-layout-source scan-detect failed for %s "
                            "(%s); treating as 'not a scan'", pdf_path.name, e)
                scan_detected = False
                scan_signals = {"verdict": "error", "reason": str(e)}
            # Compose: marker if EITHER detector says so.
            triage_says_marker = (triage_routing == "marker-surya")
            if triage_says_marker or scan_detected:
                layout_source_for_paper = "marker"
            else:
                layout_source_for_paper = "mineru"
            # Auto force-OCR: only when scan detector confirmed AND
            # we routed to marker AND the user hasn't opted out.
            if (layout_source_for_paper == "marker"
                    and scan_detected
                    and not args.no_auto_force_ocr):
                auto_force_ocr_applied = True
            log.info("Auto-layout-source: %s for %s -- "
                     "triage=%s (%s); scan_detect=%s (%s); "
                     "force_ocr=%s",
                     layout_source_for_paper, pdf_path.name,
                     triage_routing, triage_reason,
                     "scan" if scan_detected else "digital",
                     scan_signals.get("reason", "") if scan_signals else "",
                     "auto" if auto_force_ocr_applied else "off")

        # Recompute per-paper pipeline state + top-level engine
        # fields when the resolved layout_source differs from the
        # template's, OR when auto-routing recorded triage signals
        # that need surfacing. Also flip the recorded packages list
        # so it reflects what THIS paper's engine actually used.
        if (layout_source_for_paper != args.layout_source
                or args.auto_layout_source):
            per_paper_run_info.layout_source = layout_source_for_paper
            per_paper_run_info.text_engine = (
                "mineru-paddleocr"
                if layout_source_for_paper == "mineru"
                else "marker-surya")
            # Per-paper pipeline reflects THIS paper's resolved
            # layout_source (so marker-only fields appear only when
            # this paper routed to marker, etc.).
            per_paper_run_info.pipeline = _collect_pipeline_state(
                args.table_finder,
                vlm_rewrite_tables=args.vlm_tables,
                rescue_sparse_pages=args.rescue_sparse_pages,
                force_ocr=args.force_ocr,
                layout_source=layout_source_for_paper,
                rescue_mineru_orphan_captions=not args.no_rescue_orphan_captions,
                rescue_mineru_subpanels=not args.no_rescue_subpanels,
            )
            # Per-paper packages list reflects this paper's engine
            # (mineru run records mineru-*, marker run records
            # marker-pdf / surya-ocr / docling). Important under
            # auto-routing where different papers in the same batch
            # may use different engines.
            per_paper_run_info.packages = _collect_package_versions(
                layout_source_for_paper)
            if args.auto_layout_source:
                per_paper_run_info.pipeline["auto_layout_source"] = True
                per_paper_run_info.pipeline["triage_routing"] = triage_routing
                per_paper_run_info.pipeline["triage_reason"] = triage_reason
                if triage_signals:
                    per_paper_run_info.pipeline["triage_signals"] = triage_signals
                if scan_signals is not None:
                    per_paper_run_info.pipeline["scan_signals"] = scan_signals
                per_paper_run_info.pipeline["auto_force_ocr_applied"] = (
                    auto_force_ocr_applied)

        # Resolve --force-ocr / --auto-force-ocr per paper. Manual
        # --force-ocr always wins. Under --auto-layout-source, the
        # scan detector already ran above and `auto_force_ocr_applied`
        # carries its decision; honor that here. Otherwise (legacy
        # path: --auto-force-ocr alone, no --auto-layout-source), run
        # the scan detector now.
        force_ocr_for_paper = bool(args.force_ocr)
        if args.auto_layout_source:
            # Auto-layout-source path: scan detect ran upstream; apply
            # the resolved force-OCR decision (already gated on layout
            # routing to marker + scan_detected + not opted out).
            if auto_force_ocr_applied:
                force_ocr_for_paper = True
        elif layout_source_for_paper == "mineru":
            # Mineru runs its own OCR pipeline; force-OCR is a no-op.
            pass
        elif args.auto_force_ocr:
            detected, signals = detect_scan_ocr_pdf(pdf_path)
            force_ocr_for_paper = detected
            verdict = "scan-detected" if detected else "born-digital"
            log.info("Auto-force-ocr: %s for %s -- %s [%s]",
                     verdict, pdf_path.name,
                     signals.get("force_ocr_reason", ""),
                     _scan_signals_str(signals))
            if per_paper_run_info is not None:
                per_paper_run_info.pipeline = {
                    **(per_paper_run_info.pipeline or {}),
                    "auto_force_ocr": True,
                    "scan_signals": signals,
                }
        elif per_paper_run_info is not None:
            per_paper_run_info.pipeline = {
                **(per_paper_run_info.pipeline or {}),
                "auto_force_ocr": False,
            }

        if args.text_only:
            return convert_text_only(
                pdf_path, out_dir,
                emit_quality=not args.no_quality,
                emit_citation=not args.no_citation,
                metadata=metadata,
                output_stem=original_stem,
                clean=args.clean,
                run_info=per_paper_run_info,
                user_annotations=_user_anno,
            )
        return convert(
            pdf_path, out_dir, use_vlm=not args.no_vlm,
            table_finder=args.table_finder,
            emit_quality=not args.no_quality,
            emit_citation=not args.no_citation,
            asset_prefix=asset_prefix,
            metadata=metadata,
            output_stem=original_stem,
            vlm_rewrite_tables=args.vlm_tables,
            rescue_sparse_pages_flag=args.rescue_sparse_pages,
            table_workers=args.table_workers,
            force_ocr=force_ocr_for_paper,
            clean=args.clean,
            layout_source=layout_source_for_paper,
            rescue_mineru_orphan_captions=not args.no_rescue_orphan_captions,
            rescue_mineru_subpanels=not args.no_rescue_subpanels,
            strip_chart_details=args.strip_chart_details,
            mineru_backend=args.mineru_backend,
            mineru_extra_args=tuple(args.mineru_arg),
            skip_mineru_run=args.skip_mineru_run,
            mineru_existing_dir=args.mineru_dir,
            run_info=per_paper_run_info,
            user_annotations=_user_anno,
            vlm_tables_force=args.vlm_tables_force,
        )

    # Dry-run scan-detection: walk the input(s), print one line per
    # PDF showing the scan-detection verdict, signals, and (in batch
    # mode) the supplement-pairing decision so the user can validate
    # both pieces in one survey before committing to a multi-hour
    # run. No marker, no VLM. The TSV format on stdout lets the user
    # pipe to a file for spreadsheet review.
    if args.scan_detect_only:
        # Compute pairings up front so each PDF row knows whether
        # it's a main paper, an attached SI, or a standalone SI.
        # Batch mode honors --pair-regex / --no-pair the same way
        # the actual run does; single-paper mode uses --supplement
        # if present (treated as an explicit pair).
        pair_role: dict[Path, str] = {}    # main | si | standalone-si | solo
        pair_main: dict[Path, str] = {}    # main paper's stem
        if args.batch:
            pdfs_to_scan = discover_pdfs(args.batch)
            if args.no_pair:
                pairs_dr: list[tuple[Path, Optional[Path]]] = [
                    (p, None) for p in pdfs_to_scan]
                standalone_dr: list[Path] = []
            else:
                pair_regex_dr = (re.compile(args.pair_regex, re.IGNORECASE)
                                 if args.pair_regex else SI_SUFFIX_RE)
                pairs_dr, standalone_dr = pair_supplements(
                    pdfs_to_scan, pair_regex_dr)
            for main, si in pairs_dr:
                if si is not None:
                    pair_role[main] = "main"
                    pair_main[main] = main.stem
                    pair_role[si] = "si"
                    pair_main[si] = main.stem
                else:
                    pair_role[main] = "solo"
                    pair_main[main] = main.stem
            for orphan in standalone_dr:
                pair_role[orphan] = "standalone-si"
                pair_main[orphan] = orphan.stem
        else:
            pdfs_to_scan = [args.pdf]
            pair_role[args.pdf] = "main" if args.supplement is not None else "solo"
            pair_main[args.pdf] = args.pdf.stem
            if args.supplement is not None:
                pdfs_to_scan.append(args.supplement)
                pair_role[args.supplement] = "si"
                pair_main[args.supplement] = args.pdf.stem

        print("filename\tpair_role\tpair_main\tverdict\treason\t"
              "producer\tcreator\timage_cover_median\tfont_count\t"
              "bold_char_ratio\tn_pages")
        n_scan = 0
        for pdf_path in pdfs_to_scan:
            detected, signals = detect_scan_ocr_pdf(pdf_path)
            if detected:
                n_scan += 1
            verdict = "scan" if detected else "digital"
            print(
                f"{pdf_path.name}\t"
                f"{pair_role.get(pdf_path, 'solo')}\t"
                f"{pair_main.get(pdf_path, pdf_path.stem)}\t"
                f"{verdict}\t"
                f"{signals.get('force_ocr_reason', '')}\t"
                f"{signals.get('producer', '')}\t"
                f"{signals.get('creator', '')}\t"
                f"{signals.get('image_cover_median', 0.0):.3f}\t"
                f"{signals.get('font_count', 0)}\t"
                f"{signals.get('bold_char_ratio', 0.0):.3f}\t"
                f"{signals.get('n_pages', 0)}"
            )
        # Summary line: scan counts + pair counts so the user can
        # eyeball the corpus shape before launching.
        n_pairs = sum(1 for v in pair_role.values() if v == "si")
        n_solos = sum(1 for v in pair_role.values() if v == "solo")
        n_orphan_si = sum(1 for v in pair_role.values() if v == "standalone-si")
        log.info("scan-detect-only: %d PDF(s); %d flagged as scan; "
                 "%d main+SI pair(s), %d solo paper(s), %d orphan SI(s)",
                 len(pdfs_to_scan), n_scan, n_pairs, n_solos, n_orphan_si)
        return

    if args.batch:
        pair_regex = (re.compile(args.pair_regex, re.IGNORECASE)
                      if args.pair_regex else SI_SUFFIX_RE)
        pdfs = discover_pdfs(args.batch)
        if args.no_pair:
            pairs = [(pdf, None) for pdf in pdfs]
            standalone_si = []
        else:
            pairs, standalone_si = pair_supplements(pdfs, pair_regex)
        for orphan in standalone_si:
            log.warning("SI-named PDF with no matching main, processing standalone: %s",
                        orphan.name)
        log.info("Batch: %d papers (%d SI auto-paired)",
                 len(pairs),
                 sum(1 for _, si in pairs if si is not None))
        failures = run_batch(pairs, args, _do_convert)
        log.info("Batch complete: %d papers, %d failures, manifest: %s",
                 len(pairs), failures, args.out / "manifest.jsonl")
        if failures and failures == len(pairs):
            sys.exit(1)
        return

    main_md_path, report = _do_convert(args.pdf, args.out)

    si_md_path: Optional[Path] = None
    si_report: Optional[QualityReport] = None
    if args.supplement is not None:
        log.info("--- processing supplement: %s ---", args.supplement.name)
        si_md_path, si_report = _do_convert(args.supplement, args.out,
                                            asset_prefix="si_")

    if args.hdf5:
        bundle_path = args.out / (args.pdf.stem + ".h5")
        write_hdf5_bundle(bundle_path,
                          main_md=main_md_path, main_report=report,
                          main_pdf=args.pdf,
                          si_md=si_md_path, si_report=si_report,
                          si_pdf=args.supplement)

    if args.quality_threshold is not None:
        worst = report.overall()
        if si_report is not None:
            worst = min(worst, si_report.overall())
        if worst < args.quality_threshold:
            log.warning("Overall quality %.2f below threshold %.2f",
                        worst, args.quality_threshold)
            sys.exit(1)


if __name__ == "__main__":
    main()
