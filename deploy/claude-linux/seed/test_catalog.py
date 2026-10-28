"""seed/catalog.py against temp settings files: right shape, and a re-run changes nothing.

Run from the repo root: python -m pytest -q deploy/claude-linux/seed/test_catalog.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deeptutor.services.config.model_catalog import ModelCatalogService
from deeptutor.services.config.provider_runtime import (
    resolve_imagegen_runtime_config,
    resolve_tts_runtime_config,
)
from deeptutor.services.config.runtime_settings import RuntimeSettingsService
from deeptutor.services.llm.provider_core.claude_agent_sdk_provider import CLAUDE_AGENT_SDK_MODELS
from deeptutor_cli.setup_config import SetupConfig, prepare_profiles

_spec = importlib.util.spec_from_file_location(
    "seed_catalog", Path(__file__).with_name("catalog.py")
)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)


def _config_apply(catalog: dict) -> None:
    # The LLM profile `config apply setup.json` writes, built by the CLI's own code.
    setup = SetupConfig.model_validate_json(Path(__file__).with_name("setup.json").read_text())
    catalog["services"]["llm"]["profiles"].append(prepare_profiles(setup)["llm"])


def test_catalog_seed_is_idempotent(tmp_path):
    service = ModelCatalogService(tmp_path / "model_catalog.json")
    service.update(_config_apply)

    first = service.update(seed.seed_catalog)
    assert service.update(seed.seed_catalog) == first

    llm = first["services"]["llm"]
    assert llm["active_model_id"] == "llm-model-claude-sonnet-5-5"
    (profile,) = llm["profiles"]
    assert [m["model"] for m in profile["models"]] == list(CLAUDE_AGENT_SDK_MODELS)
    for model in profile["models"]:
        assert model["context_window"] == 1_000_000
        assert model["context_window_source"] == "manual"
        assert model["capabilities"] == {"tools": True, "vision": True, "json_output": False}
        assert model["codex_supported_reasoning_levels"] == [
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]

    voices = {
        p["id"]: [m["voice"] for m in p["models"]] for p in first["services"]["tts"]["profiles"]
    }
    assert voices == {
        "tts-profile-speaches": ["af_heart"],
        "tts-profile-edge": ["zh-CN-XiaoxiaoNeural", "en-US-AvaMultilingualNeural"],
    }
    tts = resolve_tts_runtime_config(catalog=first)
    assert (tts.base_url, tts.voice) == (seed.EDGE_TTS_URL, "zh-CN-XiaoxiaoNeural")
    assert first["services"]["stt"]["profiles"][0]["base_url"] == seed.SPEACHES_URL
    assert resolve_imagegen_runtime_config(first, service=service).adapter == "openai_codex"
    assert first["services"]["videogen"]["profiles"] == []


def test_settings_seed_reports_changes_once(tmp_path):
    runtime = RuntimeSettingsService(tmp_path, process_env={})
    assert seed.seed_settings(runtime) is True
    assert seed.seed_settings(runtime) is False
    assert runtime.load_auth(include_process_overrides=False)["enabled"] is True
    assert runtime.load_system(include_process_overrides=False)["sandbox_allow_subprocess"] is False
