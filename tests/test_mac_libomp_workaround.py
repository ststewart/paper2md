"""Tests for the macOS-only KMP_DUPLICATE_LIB_OK defensive set.

paper2md sets `KMP_DUPLICATE_LIB_OK=TRUE` at module import on Darwin
to prevent LLVM OpenMP from aborting when PyTorch and PaddlePaddle
(the latter brought in by `pip install mineru[core]`) load duplicate
libomp.dylib copies. The set must be:
  - gated on platform.system() == "Darwin" (no-op on Linux);
  - idempotent (setdefault, so an explicit user override wins);
  - applied before any C-extension imports that could trigger libomp.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "paper2md.py"


def _spawn(env_kmp: str | None) -> tuple[str, int]:
    """Spawn a subprocess that imports paper2md and prints the value
    of KMP_DUPLICATE_LIB_OK after import. Optionally pre-set the env
    var to simulate a user override. Returns (stdout, returncode).
    """
    code = (
        "import os; "
        "import sys; "
        f"sys.path.insert(0, {str(SRC.parent)!r}); "
        "import paper2md; "
        "print(os.environ.get('KMP_DUPLICATE_LIB_OK', 'UNSET'))"
    )
    env = {"PATH": "/usr/bin:/bin"}
    if env_kmp is not None:
        env["KMP_DUPLICATE_LIB_OK"] = env_kmp
    # Inherit HOME / minimal so dotenv / openai imports don't crash
    import os as _os
    for k in ("HOME", "LANG", "LC_ALL", "PYTHONPATH"):
        if k in _os.environ:
            env[k] = _os.environ[k]
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return r.stdout.strip(), r.returncode


def test_darwin_sets_workaround_by_default():
    """On macOS, importing paper2md sets KMP_DUPLICATE_LIB_OK=TRUE if
    unset. Sanity test only runs on Darwin (Linux CI skips)."""
    if platform.system() != "Darwin":
        # We can't actually validate the Darwin branch on Linux because
        # the guard short-circuits. But we can prove the env var stays
        # unset on Linux (see next test) and the source has the gate.
        return
    out, rc = _spawn(env_kmp=None)
    assert rc == 0, f"import crashed: {out}"
    assert out == "TRUE"


def test_non_darwin_does_not_set_workaround():
    """On Linux / Windows, paper2md must NOT touch KMP_DUPLICATE_LIB_OK
    (it has nothing to do with the libomp issue there, and changing
    env state for unrelated platforms is a leaky abstraction)."""
    if platform.system() == "Darwin":
        return  # skipped: Darwin branch is exercised by the previous test
    out, rc = _spawn(env_kmp=None)
    assert rc == 0, f"import crashed: {out}"
    assert out == "UNSET"


def test_setdefault_respects_user_override():
    """A user who already set KMP_DUPLICATE_LIB_OK (e.g. to FALSE to
    debug the underlying duplicate-libomp issue) must see their value
    preserved -- paper2md uses setdefault, not assignment."""
    out, rc = _spawn(env_kmp="FALSE")
    assert rc == 0, f"import crashed: {out}"
    assert out == "FALSE"


def test_source_has_darwin_guard():
    """Belt-and-suspenders source check: the set must be gated on
    platform.system() == 'Darwin' so the Linux test above is
    structurally sound, not just empirically lucky."""
    src = SRC.read_text()
    assert 'platform.system() == "Darwin"' in src
    assert 'os.environ.setdefault("KMP_DUPLICATE_LIB_OK"' in src
