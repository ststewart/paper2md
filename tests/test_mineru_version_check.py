"""Tests for the MinerU version-pin runtime check.

paper2md's hybrid splice + layout-rescue hooks parse MinerU's
middle.json schema, which has drifted between minor versions in ways
that silently break parsing. A one-shot WARNING fires from run_mineru
when the installed MinerU diverges from the pinned `EXPECTED_-
MINERU_VERSION`. Soft-fails silently when MinerU isn't installed (the
subprocess will produce a clearer error). Fires at most once per
process so a batch run doesn't spam the log.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import layout_mineru as lm  # noqa: E402


def _reset_check_flag(monkeypatch):
    """Reset the module-level flag so each test runs the check fresh."""
    monkeypatch.setattr(lm, "_mineru_version_checked", False)


def test_matching_version_no_warning(monkeypatch, caplog):
    """When the installed MinerU matches EXPECTED_MINERU_VERSION, no
    warning is emitted -- just silent acknowledgment."""
    _reset_check_flag(monkeypatch)
    from importlib import metadata
    monkeypatch.setattr(
        metadata, "version",
        lambda pkg: lm.EXPECTED_MINERU_VERSION if pkg == "mineru" else None)
    with caplog.at_level("WARNING", logger="layout_mineru"):
        lm._check_mineru_version()
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warns == []


def test_mismatched_version_warns(monkeypatch, caplog):
    """When the installed version differs (e.g., 3.2.0 vs 3.1.7), a
    WARNING fires naming both versions and pointing at the upgrade
    procedure docs."""
    _reset_check_flag(monkeypatch)
    from importlib import metadata
    monkeypatch.setattr(
        metadata, "version",
        lambda pkg: "3.2.0" if pkg == "mineru" else None)
    with caplog.at_level("WARNING", logger="layout_mineru"):
        lm._check_mineru_version()
    msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert "3.2.0" in msg
    assert lm.EXPECTED_MINERU_VERSION in msg
    # User-actionable signposts in the message
    assert "USAGE.md" in msg or "environment-gpu.yml" in msg
    assert "HYBRID_IMPLEMENTATION" in msg


def test_check_fires_at_most_once_per_process(monkeypatch, caplog):
    """Multiple invocations (batch with N papers) must NOT spam the
    log. Once warned, subsequent calls are silent."""
    _reset_check_flag(monkeypatch)
    from importlib import metadata
    monkeypatch.setattr(
        metadata, "version",
        lambda pkg: "3.2.0" if pkg == "mineru" else None)
    with caplog.at_level("WARNING", logger="layout_mineru"):
        lm._check_mineru_version()
        lm._check_mineru_version()
        lm._check_mineru_version()
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1


def test_package_not_found_is_silent(monkeypatch, caplog):
    """If MinerU isn't installed, the check soft-fails silently. The
    subprocess call further down will fail with a clearer error than
    a missing-package warning would."""
    _reset_check_flag(monkeypatch)
    from importlib import metadata
    def _raise_not_found(pkg):
        raise metadata.PackageNotFoundError(pkg)
    monkeypatch.setattr(metadata, "version", _raise_not_found)
    with caplog.at_level("WARNING", logger="layout_mineru"):
        lm._check_mineru_version()
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warns == []


def test_unexpected_metadata_error_is_silent(monkeypatch, caplog):
    """A bizarre metadata-read error (corrupt PKG-INFO, etc.) must
    not abort the run. Check soft-fails silently and the subprocess
    proceeds; if MinerU is actually broken the subprocess errors
    surface the real problem."""
    _reset_check_flag(monkeypatch)
    from importlib import metadata
    def _raise(pkg):
        raise OSError("disk on fire")
    monkeypatch.setattr(metadata, "version", _raise)
    with caplog.at_level("WARNING", logger="layout_mineru"):
        lm._check_mineru_version()
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warns == []


# -- Tests for _assert_mineru_installed preflight ----------------------------


def test_assert_raises_when_module_and_cli_missing(monkeypatch):
    """Common failure mode: user ran `--layout-source mineru` without
    `pip install mineru[core]`. The preflight must raise a
    MineruNotInstalledError with the install hint, not a bare
    FileNotFoundError from subprocess."""
    from importlib import metadata
    def _raise_not_found(pkg):
        raise metadata.PackageNotFoundError(pkg)
    monkeypatch.setattr(metadata, "version", _raise_not_found)
    monkeypatch.setattr(lm.shutil, "which", lambda _: None)
    with pytest.raises(lm.MineruNotInstalledError) as exc:
        lm._assert_mineru_installed()
    msg = str(exc.value)
    assert "pip install" in msg
    assert lm.EXPECTED_MINERU_VERSION in msg
    assert "--layout-source marker" in msg  # alternative-route hint


def test_assert_raises_when_module_present_but_cli_missing(monkeypatch):
    """Rarer: package installed but `mineru` CLI not on PATH (broken
    env activation, broken entry-point). Distinct error message points
    at the diagnostic."""
    from importlib import metadata
    monkeypatch.setattr(metadata, "version",
                        lambda pkg: lm.EXPECTED_MINERU_VERSION)
    monkeypatch.setattr(lm.shutil, "which", lambda _: None)
    with pytest.raises(lm.MineruNotInstalledError) as exc:
        lm._assert_mineru_installed()
    msg = str(exc.value)
    assert "not on PATH" in msg
    assert "which mineru" in msg


def test_assert_passes_when_both_present(monkeypatch):
    """Happy path: module installed AND CLI on PATH -- no exception,
    run_mineru proceeds to the subprocess invocation."""
    from importlib import metadata
    monkeypatch.setattr(metadata, "version",
                        lambda pkg: lm.EXPECTED_MINERU_VERSION)
    monkeypatch.setattr(lm.shutil, "which",
                        lambda _: "/opt/conda/envs/paper2md/bin/mineru")
    lm._assert_mineru_installed()  # no exception
