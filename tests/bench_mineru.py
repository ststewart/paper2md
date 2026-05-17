#!/usr/bin/env python3
"""Benchmark MinerU on a fixed paper set for cross-machine comparison.

Runs MinerU's `pipeline` backend on each input PDF in turn (one
process per paper, so each run gets its own model-load timing) and
writes a CSV summarising wall time, peak RSS, model init cost, page
count, and MFR region count per paper. Use this to directly compare
NVIDIA CUDA, Apple Silicon MPS, and CPU-only numbers across
machines.

Cross-platform: detects Linux vs macOS and uses the right
`/usr/bin/time` flag (`-v` on Linux, `-l` on macOS BSD time). Wall
time is measured with Python's monotonic clock so it's identical on
both. Peak RSS is parsed from /usr/bin/time output and converted
to MB on both platforms.

Usage
-----

::

    # Default fixed corpus
    python tests/bench_mineru.py \\
        --pdfs examples/canup.pdf collections/moon/cuk.pdf \\
               collections/silica-shock/Knudson2013.pdf \\
               examples/wackerle.pdf \\
        -o mineru_bench_$(hostname).csv

    # Or scan a directory
    python tests/bench_mineru.py \\
        --input-dir ~/mineru-bench/in \\
        -o results.csv

    # Force CPU-only on a CUDA machine for an apples-to-apples
    # comparison vs an Apple Silicon CPU run:
    CUDA_VISIBLE_DEVICES="" python tests/bench_mineru.py ...

The script does NOT install MinerU; run it in an environment where
`mineru --version` works. On macOS you may need `brew install
gnu-time` if the BSD `/usr/bin/time` produces unparseable output --
the parser handles both BSD-style and GNU `-v` output, but
distributions vary.

CSV columns
-----------

paper, pages, mfr_regions, model_init_s, wall_s, peak_rss_mb,
exit_code, host, os, device, mineru_version, started_at

`device` reflects what MinerU is *most likely* to have used, based
on env vars + torch detection at script start. It does not parse
the run log; if MinerU silently fell back from MPS to CPU, the
device column won't reflect that. The wall_s vs CPU baseline is
still informative.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def detect_device() -> str:
    """Best-effort device label for the CSV. Looks at env vars first
    so a user running with CUDA_VISIBLE_DEVICES='' is correctly
    flagged as CPU."""
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv is not None and cv.strip() == "":
        return "cpu (CUDA_VISIBLE_DEVICES='')"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                return f"cuda ({name})"
            except Exception:
                return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
    except ImportError:
        pass
    return "cpu"


def detect_mineru_version() -> str:
    try:
        out = subprocess.check_output(
            ["mineru", "--version"], text=True,
            stderr=subprocess.STDOUT, timeout=10)
        m = re.search(r"version\s+([0-9.]+)", out)
        return m.group(1) if m else out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return "not_installed"


def count_pdf_pages(pdf: Path) -> int | None:
    """Quick page count via PyMuPDF (fitz) or pypdf, whichever loads."""
    try:
        import fitz  # type: ignore
        d = fitz.open(str(pdf))
        try:
            return d.page_count
        finally:
            d.close()
    except ImportError:
        pass
    try:
        from pypdf import PdfReader  # type: ignore
        return len(PdfReader(str(pdf)).pages)
    except ImportError:
        pass
    return None


_RE_MODEL_INIT = re.compile(r"model init cost:\s*([0-9.]+)")
# Final MFR line shape: "MFR Predict: 100%|...| N/N [00:07<00:00, ...]"
_RE_MFR = re.compile(
    r"MFR Predict:\s*100%[^|]*\|[^|]*\|\s*(\d+)/(\d+)")
_RE_PAGES = re.compile(r"Processed\s+(\d+)/\d+\s+pages")
_RE_RSS_MAC = re.compile(r"(\d+)\s+maximum resident set size")
_RE_RSS_LINUX = re.compile(
    r"Maximum resident set size \(kbytes\):\s+(\d+)")


def parse_log(log_text: str, is_mac: bool) -> dict:
    out: dict = {}
    m = _RE_MODEL_INIT.search(log_text)
    if m:
        out["model_init_s"] = round(float(m.group(1)), 1)
    m = _RE_MFR.search(log_text)
    if m:
        out["mfr_regions"] = int(m.group(2))
    m = _RE_PAGES.search(log_text)
    if m:
        out["log_pages"] = int(m.group(1))
    if is_mac:
        m = _RE_RSS_MAC.search(log_text)
        if m:
            out["peak_rss_mb"] = round(int(m.group(1)) / (1024 * 1024))
    else:
        m = _RE_RSS_LINUX.search(log_text)
        if m:
            out["peak_rss_mb"] = round(int(m.group(1)) / 1024)
    return out


def run_one(pdf: Path, out_dir: Path, lang: str, backend: str,
            log_dir: Path | None = None) -> dict:
    """Run mineru on one PDF, capturing wall time + peak RSS via
    /usr/bin/time. When `log_dir` is given, writes the captured
    stdout+stderr to <log_dir>/<paper-stem>.log so failures can be
    inspected after the run."""
    is_mac = platform.system() == "Darwin"
    time_flag = "-l" if is_mac else "-v"
    if not Path("/usr/bin/time").is_file():
        # Some distros / minimal containers may not have /usr/bin/time;
        # fall back to no-RSS measurement.
        cmd = ["mineru", "-p", str(pdf), "-o", str(out_dir),
               "-b", backend, "-l", lang]
    else:
        cmd = ["/usr/bin/time", time_flag,
               "mineru", "-p", str(pdf), "-o", str(out_dir),
               "-b", backend, "-l", lang]
    print(f"  running: {' '.join(cmd[:8])}...", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              check=False)
    except FileNotFoundError as e:
        return {"error": str(e), "exit_code": -1}
    wall = round(time.monotonic() - t0, 1)
    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{pdf.stem}.log").write_text(log_text)
        except OSError as e:
            print(f"  warn: couldn't write log for {pdf.stem}: {e}",
                  file=sys.stderr)
    info = parse_log(log_text, is_mac)
    info.update({
        "paper": pdf.stem,
        "wall_s": wall,
        "exit_code": proc.returncode,
    })
    info.setdefault("pages", count_pdf_pages(pdf) or info.get("log_pages"))
    return info


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--pdfs", nargs="+", type=Path,
                   help="explicit list of PDFs (mutually exclusive "
                        "with --input-dir)")
    p.add_argument("--input-dir", type=Path,
                   help="run on every *.pdf in this directory")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("mineru_bench.csv"),
                   help="CSV output path (default mineru_bench.csv)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("/tmp/mineru-bench-out"),
                   help="where MinerU writes its outputs "
                        "(default /tmp/mineru-bench-out)")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="save per-paper stdout+stderr logs here so "
                        "failures can be diagnosed after the run "
                        "(default: alongside --out CSV, named "
                        "<csv-stem>-logs/)")
    p.add_argument("--backend", default="pipeline",
                   choices=["pipeline", "vlm-auto-engine",
                            "hybrid-auto-engine"],
                   help="MinerU backend (default: pipeline)")
    p.add_argument("--lang", default="en",
                   help="MinerU OCR language hint (default en)")
    args = p.parse_args()

    if args.pdfs and args.input_dir:
        p.error("--pdfs and --input-dir are mutually exclusive")
    if args.pdfs:
        pdfs = [p_.resolve() for p_ in args.pdfs]
    elif args.input_dir:
        pdfs = sorted(args.input_dir.resolve().glob("*.pdf"))
    else:
        p.error("pass --pdfs <files> or --input-dir <dir>")

    if not shutil.which("mineru"):
        print("ERROR: `mineru` not in PATH. Activate the env where "
              "MinerU is installed and re-run.", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or args.out.with_name(
        args.out.stem + "-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    host = platform.node()
    osname = f"{platform.system()} {platform.release()}"
    device = detect_device()
    version = detect_mineru_version()
    print(f"# host={host}  os={osname}  device={device}  "
          f"mineru={version}  started={started}", file=sys.stderr)

    rows: list[dict] = []
    for pdf in pdfs:
        if not pdf.is_file():
            print(f"  skip: {pdf} not found", file=sys.stderr)
            continue
        print(f"\n## {pdf.stem} ({pdf.stat().st_size:,} bytes)",
              file=sys.stderr)
        info = run_one(pdf, args.out_dir, args.lang, args.backend,
                       log_dir=log_dir)
        info.update({
            "host": host, "os": osname, "device": device,
            "mineru_version": version, "started_at": started,
        })
        rows.append(info)
        print(f"  -> wall={info.get('wall_s')}s "
              f"rss={info.get('peak_rss_mb')}MB "
              f"init={info.get('model_init_s')}s "
              f"mfr={info.get('mfr_regions')} "
              f"pages={info.get('pages')} "
              f"exit={info.get('exit_code')}",
              file=sys.stderr, flush=True)

    cols = ["paper", "pages", "mfr_regions", "model_init_s",
            "wall_s", "peak_rss_mb", "exit_code",
            "host", "os", "device", "mineru_version", "started_at"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"Per-paper logs saved under {log_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
