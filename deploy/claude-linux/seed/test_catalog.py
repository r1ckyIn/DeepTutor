"""seed/catalog.py against temp settings files: right shape, and a re-run changes nothing.

Run from the repo root: python -m pytest -q deploy/claude-linux/seed/test_catalog.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deeptutor.services.config.model_catalog import ModelCatalogService
from deeptutor.services.config.runtime_settings import RuntimeSettingsService

_spec = importlib.util.spec_from_file_location(
    "seed_catalog", Path(__file__).with_name("catalog.py")
)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)


def _config_apply(catalog: dict) -> None:
    # The LLM profile `config apply` leaves behind (deeptutor_cli/setup_config.py).
    catalog["services"]["llm"]["profiles"].append(
        {
            "id": seed.LLM_PROFILE_ID,
            "name": "Agent Setup",
            "binding": "claude_agent_sdk",
            "base_url": "",
            "api_key": "",
            "models": [
                {
                    "id": "llm-model-agent-setup",
                    "name": "claude-sonnet-5-5",
                    "model": "claude-sonnet-5-5",
                }
            ],
        }
    )


def test_catalog_seed_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setitem(seed.IMAGEGEN_PROVIDERS, seed.CODEX_IMAGEGEN_BINDING, object())
    service = ModelCatalogService(tmp_path / "model_catalog.json")
    service.update(_config_apply)

    first = service.update(seed.seed_catalog)
    assert service.update(seed.seed_catalog) == first

    llm = first["services"]["llm"]
    assert llm["active_model_id"] == "llm-model-claude-sonnet-5-5"
    (profile,) = llm["profiles"]
    assert [m["model"] for m in profile["models"]] == list(seed.CLAUDE_MODELS)
    for model in profile["models"]:
        assert model["context_window"] == 1_000_000
        assert model["context_window_source"] == "manual"
        assert model["capabilities"] == {"tools": True, "vision": False, "json_output": False}
        assert model["codex_supported_reasoning_levels"] == [
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]

    tts = first["services"]["tts"]
    assert tts["active_model_id"] == "tts-model-kokoro-zh"
    assert [m["voice"] for m in tts["profiles"][0]["models"]] == ["zf_xiaobei", "af_heart"]
    assert first["services"]["stt"]["profiles"][0]["base_url"] == seed.SPEACHES_URL
    assert first["services"]["imagegen"]["profiles"][0]["binding"] == seed.CODEX_IMAGEGEN_BINDING
    assert first["services"]["videogen"]["profiles"] == []


def test_imagegen_waits_for_the_binding(tmp_path, monkeypatch):
    monkeypatch.delitem(seed.IMAGEGEN_PROVIDERS, seed.CODEX_IMAGEGEN_BINDING, raising=False)
    service = ModelCatalogService(tmp_path / "model_catalog.json")
    service.update(_config_apply)
    assert service.update(seed.seed_catalog)["services"]["imagegen"]["profiles"] == []


def test_settings_seed_reports_changes_once(tmp_path):
    runtime = RuntimeSettingsService(tmp_path, process_env={})
    assert seed.seed_settings(runtime) is True
    assert seed.seed_settings(runtime) is False
    assert runtime.load_auth(include_process_overrides=False)["enabled"] is True
    assert runtime.load_system(include_process_overrides=False)["sandbox_allow_subprocess"] is False
