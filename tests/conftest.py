"""Shared pytest fixtures for paper2md tests.

Tests that need a PDF fixture should accept the `test_pdf` fixture,
which resolves filenames against $TEST_FILES_DIR (default:
~/datasets/pdf2md/claude/test_files/). Tests skip cleanly when a
requested PDF isn't present, so a fresh clone without fixtures
still passes the unit-test suite (only smoke / eval tests skip).

Override the fixture root via:

    TEST_FILES_DIR=/path/to/dir python -m pytest tests/

The default location is the workspace-level test_files/ dir agreed
on in the 2026-05 cleanup; see docs/dev/EXAMPLES_FIXTURES.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


_DEFAULT_TEST_FILES = Path("~/datasets/pdf2md/claude/test_files").expanduser()


@pytest.fixture(scope="session")
def test_files_dir() -> Path:
    """Return the configured top-level test_files dir.

    Reads $TEST_FILES_DIR from the environment; defaults to the
    workspace location ~/datasets/pdf2md/claude/test_files/.
    """
    return Path(os.environ.get("TEST_FILES_DIR", _DEFAULT_TEST_FILES))


@pytest.fixture
def test_pdf(test_files_dir):
    """Resolve a PDF fixture by filename.

    Usage:

        def test_canup_smoke(test_pdf):
            pdf = test_pdf("canup.pdf")
            # Use pdf as a Path; pytest.skip if not present on disk.
    """
    def _resolve(filename: str) -> Path:
        p = test_files_dir / filename
        if not p.exists():
            pytest.skip(
                f"Fixture not present: {p} "
                f"(see docs/dev/EXAMPLES_FIXTURES.md for how to "
                f"populate test_files/ on this machine)"
            )
        return p
    return _resolve
