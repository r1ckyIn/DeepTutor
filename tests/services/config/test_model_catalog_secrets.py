from __future__ import annotations

from deeptutor.services.config.model_catalog import (
    CATALOG_SECRET_MASK,
    redact_catalog_secrets,
    restore_catalog_secrets,
)


def _catalog() -> dict:
    return {
        "version": 1,
        "services": {
            "llm": {
                "profiles": [
                    {
                        "id": "profile-a",
                        "name": "A",
                        "api_key": "sk-secret-a",
                        "extra_headers": {
                            "Authorization": "Bearer secret",
                            "X-Tenant": "tenant-secret",
                        },
                    },
                    {
                        "id": "profile-b",
                        "name": "B",
                        "api_key": "sk-secret-b",
                        "extra_headers": '{"Authorization":"Bearer secret"}',
                    },
                ]
            }
        },
    }


def test_redact_catalog_secrets_masks_credentials_without_mutating_source() -> None:
    catalog = _catalog()

    redacted = redact_catalog_secrets(catalog)

    first, second = redacted["services"]["llm"]["profiles"]
    assert first["api_key"] == CATALOG_SECRET_MASK
    assert first["extra_headers"] == {
        "Authorization": CATALOG_SECRET_MASK,
        "X-Tenant": CATALOG_SECRET_MASK,
    }
    assert second["extra_headers"] == CATALOG_SECRET_MASK
    assert catalog["services"]["llm"]["profiles"][0]["api_key"] == "sk-secret-a"


def test_restore_catalog_secrets_matches_profiles_by_id_after_reorder() -> None:
    current = _catalog()
    proposed = redact_catalog_secrets(current)
    proposed["services"]["llm"]["profiles"].reverse()
    proposed["services"]["llm"]["profiles"][0]["name"] = "Renamed B"
    proposed["services"]["llm"]["profiles"][1]["api_key"] = "sk-replaced-a"

    restored = restore_catalog_secrets(proposed, current)

    first, second = restored["services"]["llm"]["profiles"]
    assert first["id"] == "profile-b"
    assert first["api_key"] == "sk-secret-b"
    assert first["extra_headers"] == '{"Authorization":"Bearer secret"}'
    assert first["name"] == "Renamed B"
    assert second["api_key"] == "sk-replaced-a"
    assert second["extra_headers"]["Authorization"] == "Bearer secret"


def test_restore_catalog_secrets_keeps_explicit_clear() -> None:
    current = _catalog()
    proposed = redact_catalog_secrets(current)
    proposed["services"]["llm"]["profiles"][0]["api_key"] = ""

    restored = restore_catalog_secrets(proposed, current)

    assert restored["services"]["llm"]["profiles"][0]["api_key"] == ""


def test_claude_subscription_entries_never_store_a_credential(tmp_path) -> None:
    """The CLI reads its own login from the server environment; a pasted key must
    not reach model_catalog.json, whether typed on the connection or the profile."""
    from deeptutor.services.config.model_catalog import ModelCatalogService

    service = ModelCatalogService(path=tmp_path / "model_catalog.json")
    catalog = service.load()
    catalog["connections"] = [
        {"id": "conn-claude", "provider": "claude_agent_sdk", "api_key": "sk-ant-pasted"}
    ]
    catalog["services"]["llm"]["profiles"] = [
        {
            "id": "llm-linked",
            "binding": "claude_agent_sdk",
            "connection_id": "conn-claude",
            "models": [{"id": "m1", "model": "claude-sonnet-5-5"}],
        },
        {
            "id": "llm-typed",
            "binding": "claude-agent-sdk",
            "api_key": "sk-ant-typed",
            "extra_headers": {"Authorization": "Bearer pasted"},
            "models": [{"id": "m2", "model": "claude-opus-5-5"}],
        },
        {
            "id": "llm-codebuddy",
            "binding": "codebuddy",
            "api_key": "cb-key",
            "models": [{"id": "m3", "model": "codebuddy/hy3"}],
        },
    ]

    service.save(catalog)

    stored = service.path.read_text(encoding="utf-8")
    assert "sk-ant" not in stored and "Bearer pasted" not in stored
    saved = service.load()
    assert saved["connections"][0]["api_key"] == ""
    linked, typed, codebuddy = saved["services"]["llm"]["profiles"]
    assert (linked["api_key"], typed["api_key"], typed["extra_headers"]) == ("", "", {})
    # Matched by binding, not OAuth: CodeBuddy genuinely uses its API key.
    assert codebuddy["api_key"] == "cb-key"
