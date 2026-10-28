"""Second half of seed.sh, run inside the deeptutor container after `config apply`.

`config apply` (setup.json) cannot express several models with metadata, voice,
image generation, login or the exec policy, so this edits them through
DeepTutor's own settings services. Idempotent. Prints "restart required" when
auth.json/system.json changed, because both are only read at startup.
"""

from __future__ import annotations

from deeptutor.services.config import get_model_catalog_service, get_runtime_settings_service

LLM_PROFILE_ID = "llm-profile-agent-setup"  # the profile `config apply` writes
CLAUDE_MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",  # active
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
}
SPEACHES_URL = "http://speaches:8000/v1"
KOKORO = "speaches-ai/Kokoro-82M-v1.0-ONNX"  # same ids as the downloads in seed.sh
WHISPER = "Systran/faster-whisper-small"


def _upsert(service: dict, profile: dict, active_model_id: str) -> None:
    rows = service["profiles"]
    i = next((i for i, row in enumerate(rows) if row.get("id") == profile["id"]), len(rows))
    rows[i : i + 1] = [profile]
    service["active_profile_id"] = profile["id"]
    service["active_model_id"] = active_model_id


def seed_catalog(catalog: dict) -> None:
    services = catalog["services"]
    llm = next((p for p in services["llm"]["profiles"] if p.get("id") == LLM_PROFILE_ID), None)
    if llm is None:
        raise SystemExit("seed: run `config apply` with seed/setup.json first")
    llm["models"] = [
        {
            "id": f"llm-model-{model}",
            "name": name,
            "model": model,
            "context_window": 1_000_000,
            "context_window_source": "manual",
            "capabilities": {"tools": True, "vision": False, "json_output": False},
            "codex_supported_reasoning_levels": ["low", "medium", "high", "xhigh", "max"],
        }
        for model, name in CLAUDE_MODELS.items()
    ]
    services["llm"]["active_profile_id"] = LLM_PROFILE_ID
    services["llm"]["active_model_id"] = "llm-model-claude-sonnet-5-5"

    # speaches picks Kokoro's language from the voice, so zh and en are two
    # models; switch the active one in Settings > Voice. CPU synthesis of a long
    # reply can outrun the 60 s default timeout.
    kokoro = {"model": KOKORO, "response_format": "mp3", "request_timeout": "300"}
    _upsert(
        services["tts"],
        {
            "id": "tts-profile-speaches",
            "name": "speaches (Kokoro)",
            "binding": "custom",  # "OpenAI Compatible": no default URL, base_url is required
            "base_url": SPEACHES_URL,
            "api_key": "",
            "models": [
                {"id": "tts-model-kokoro-zh", "name": "Kokoro zh", "voice": "zf_xiaobei", **kokoro},
                {"id": "tts-model-kokoro-en", "name": "Kokoro en", "voice": "af_heart", **kokoro},
            ],
        },
        "tts-model-kokoro-zh",
    )
    _upsert(
        services["stt"],
        {
            "id": "stt-profile-speaches",
            "name": "speaches (faster-whisper)",
            "binding": "custom",
            "base_url": SPEACHES_URL,
            "api_key": "",
            # Empty language = auto-detect, which suits mixed zh/en dictation.
            "models": [
                {
                    "id": "stt-model-whisper-small",
                    "name": "faster-whisper small",
                    "model": WHISPER,
                    "language": "",
                }
            ],
        },
        "stt-model-whisper-small",
    )
    _upsert(
        services["imagegen"],
        {
            "id": "imagegen-profile-codex",
            "name": "ChatGPT (Codex login)",
            "binding": "openai_codex",  # the Codex image adapter
            "base_url": "",  # empty = the provider's own default
            "api_key": "",  # uses DeepTutor's Codex OAuth login, never a key
            "models": [
                {
                    "id": "imagegen-model-gpt-image-2",
                    "name": "gpt-image-2",
                    "model": "gpt-image-2",
                }
            ],
        },
        "imagegen-model-gpt-image-2",
    )


def seed_settings(runtime) -> bool:
    """Login on and no exec inside the app container; True when a file changed."""
    changed = False
    auth = runtime.load_auth(include_process_overrides=False)
    if not auth["enabled"]:
        # Without a bootstrap password_hash, the first /register becomes the admin.
        runtime.save_auth({**auth, "enabled": True})
        changed = True
    system = runtime.load_system(include_process_overrides=False)
    if system["sandbox_allow_subprocess"]:
        # Exec goes to the runner sidecar; never fall back to the app container.
        runtime.save_system({**system, "sandbox_allow_subprocess": False})
        changed = True
    return changed


def main() -> None:
    get_model_catalog_service().update(seed_catalog)
    print("seed: catalog updated")
    if seed_settings(get_runtime_settings_service()):
        print("seed: restart required")


if __name__ == "__main__":
    main()
