"""Tests for deterministic VLM sampling.

paper2md sets `temperature` and `seed` on every VLM request (was
previously unset -> server defaults of 1.0 / random). These tests
verify the params land in the API call, are overridable, and get
recorded in the frontmatter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import paper2md as p2m  # noqa: E402


class _FakeChat:
    """Captures the kwargs passed to chat.completions.create."""
    captured: dict = {}

    def create(self, **kw):
        type(self).captured = kw

        class _R:
            class choices:
                pass
        r = _R()
        msg = type("M", (), {"content": "OK"})
        r.choices = [type("C", (), {"message": msg})]
        return r


class _FakeCli:
    """Stand-in for the OpenAI client (vLLM / LM Studio path)."""

    def __init__(self):
        self.chat = type("CChat", (), {"completions": _FakeChat()})()

    def with_options(self, **kw):
        return self


@pytest.fixture
def vllm_client(monkeypatch):
    """Patch the module-level client + provider to the OpenAI-compat
    path so vlm() doesn't try to hit a real server."""
    fake = _FakeCli()
    monkeypatch.setattr(p2m, "client", fake)
    monkeypatch.setattr(p2m, "_PROVIDER", "vllm")
    _FakeChat.captured = {}
    return fake


def _tiny_image():
    return Image.new("RGB", (10, 10), color="white")


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


def test_default_temperature_is_deterministic():
    """Module default is 0.0 (greedy / deterministic)."""
    assert p2m.VLM_TEMPERATURE == 0.0


def test_default_seed_is_42():
    """Module default is 42."""
    assert p2m.VLM_SEED == 42


# ---------------------------------------------------------------------
# API call payload
# ---------------------------------------------------------------------


def test_vlm_call_includes_temperature_and_seed(vllm_client, monkeypatch):
    monkeypatch.setattr(p2m, "VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(p2m, "VLM_SEED", 42)
    p2m.vlm("test", _tiny_image())
    captured = _FakeChat.captured
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42


def test_vlm_call_respects_custom_temperature(vllm_client, monkeypatch):
    monkeypatch.setattr(p2m, "VLM_TEMPERATURE", 0.7)
    monkeypatch.setattr(p2m, "VLM_SEED", 12345)
    p2m.vlm("test", _tiny_image())
    captured = _FakeChat.captured
    assert captured["temperature"] == 0.7
    assert captured["seed"] == 12345


def test_vlm_call_omits_seed_when_none(vllm_client, monkeypatch):
    """`VLM_SEED = None` means 'let the server pick'. The seed kwarg
    must NOT appear in the API call (sending seed=None would force the
    server to interpret it as a literal value on some implementations)."""
    monkeypatch.setattr(p2m, "VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(p2m, "VLM_SEED", None)
    p2m.vlm("test", _tiny_image())
    captured = _FakeChat.captured
    assert "seed" not in captured
    # Temperature still present.
    assert captured["temperature"] == 0.0


# ---------------------------------------------------------------------
# Frontmatter recording
# ---------------------------------------------------------------------


def test_run_info_emits_vlm_temperature_and_seed():
    ri = p2m.RunInfo(
        command="x", hostname="h",
        vlm_provider="vllm", vlm_model="m",
        vlm_temperature=0.0, vlm_seed=42,
        elapsed_sec=1.0,
    )
    yaml_text = "\n".join(ri.to_yaml_lines())
    assert "vlm_temperature: 0.0" in yaml_text
    assert "vlm_seed: 42" in yaml_text


def test_run_info_emits_dict_with_sampling_fields():
    ri = p2m.RunInfo(
        command="x", hostname="h",
        vlm_provider="vllm", vlm_model="m",
        vlm_temperature=0.3, vlm_seed=7,
        elapsed_sec=1.0,
    )
    d = ri.to_dict()
    assert d["vlm_temperature"] == 0.3
    assert d["vlm_seed"] == 7


def test_run_info_seed_none_emits_as_no_seed_line():
    """When seed is None (user disabled it), the YAML line is omitted
    so downstream consumers can tell determinism was opt-out."""
    ri = p2m.RunInfo(
        command="x", hostname="h",
        vlm_provider="vllm", vlm_model="m",
        vlm_temperature=0.0, vlm_seed=None,
        elapsed_sec=1.0,
    )
    yaml_text = "\n".join(ri.to_yaml_lines())
    assert "vlm_temperature: 0.0" in yaml_text
    assert "vlm_seed:" not in yaml_text


# ---------------------------------------------------------------------
# Anthropic path
# ---------------------------------------------------------------------


class _FakeAnthropic:
    captured: dict = {}

    class messages:
        @staticmethod
        def create(**kw):
            _FakeAnthropic.captured = kw

            class _R:
                content = [type("X", (), {"text": "OK"})()]
            return _R()


def test_vlm_anthropic_sends_temperature_not_seed(monkeypatch):
    """Anthropic Messages API accepts temperature but NOT seed; vlm()
    must omit seed under --provider anthropic to avoid an SDK error."""
    monkeypatch.setattr(p2m, "_anthropic_client", _FakeAnthropic())
    monkeypatch.setattr(p2m, "_PROVIDER", "anthropic")
    monkeypatch.setattr(p2m, "VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(p2m, "VLM_SEED", 42)
    _FakeAnthropic.captured = {}
    p2m.vlm("test", _tiny_image())
    captured = _FakeAnthropic.captured
    assert captured["temperature"] == 0.0
    assert "seed" not in captured
