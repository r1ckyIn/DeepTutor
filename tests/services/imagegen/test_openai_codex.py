"""ChatGPT-subscription imagegen through the Codex OAuth login (fixed endpoint)."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import json
from typing import Any, AsyncIterator

import httpx
import pytest

from deeptutor.multi_user.model_access import is_owner_bound
from deeptutor.services.codex_auth.constants import CODEX_IMAGES_URL
from deeptutor.services.codex_auth.contracts import CodexAuthError, CodexToken
from deeptutor.services.generation_http import GenerationProviderError
from deeptutor.services.imagegen import generate_image
from deeptutor.services.imagegen.adapters import openai_codex as module
from deeptutor.services.imagegen.adapters.openai_codex import CodexImagegenAdapter
from deeptutor.services.imagegen.config import ImagegenConfig
from deeptutor.services.llm.provider_core.openai_codex_provider import _build_headers

PNG = base64.b64encode(b"PNGDATA").decode("ascii")
OK = httpx.Response(200, json={"data": [{"b64_json": PNG}]})


class FakeCodexService:
    def __init__(self) -> None:
        self.generation = 7
        self.recovered: list[int] = []

    async def get_token(self) -> CodexToken:
        return CodexToken(
            access_token=f"secret-token-{self.generation}",
            account_id="account-123",
            expires_at=2_000_000_000,
            generation=self.generation,
        )

    async def recover_after_unauthorized(self, generation: int) -> None:
        self.recovered.append(generation)
        self.generation += 1

    @asynccontextmanager
    async def inference_guard(self) -> AsyncIterator[None]:
        yield


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeCodexService:
    fake = FakeCodexService()
    monkeypatch.setattr(module, "get_codex_oauth_service", lambda: fake)
    return fake


def _serve(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> list[httpx.Request]:
    """Answer requests in order (the last answer repeats) through ``httpx.MockTransport``."""
    seen: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = answers[min(len(seen), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return seen


def _config() -> ImagegenConfig:
    # Catalog fields this adapter must never use: a foreign URL, a key, DALL-E knobs.
    return ImagegenConfig(
        model="gpt-image-2",
        provider_name="openai_codex",
        adapter="openai_codex",
        base_url="https://attacker.example/v1",
        api_key="sk-must-not-leak",
        size="1792x1024",
        quality="hd",
    )


@pytest.mark.asyncio
async def test_posts_fixed_endpoint_with_owner_login(monkeypatch, service) -> None:
    seen = _serve(monkeypatch, OK)

    images = await CodexImagegenAdapter().generate("a cat", _config())

    assert images == [(b"PNGDATA", "image/png")]
    (request,) = seen
    assert str(request.url) == CODEX_IMAGES_URL
    assert json.loads(request.content) == {
        "prompt": "a cat",
        "background": "opaque",
        "model": "gpt-image-2",
        "quality": "auto",
        "size": "auto",
    }
    assert request.headers["Authorization"] == "Bearer secret-token-7"
    assert request.headers["ChatGPT-Account-ID"] == "account-123"
    llm_headers = _build_headers("account-123", "secret-token-7")
    assert request.headers["originator"] == llm_headers["originator"]
    assert request.headers["User-Agent"] == llm_headers["User-Agent"]
    assert request.headers["x-codex-image-turn-id"]
    assert all("sk-must-not-leak" not in value for value in request.headers.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(("per_call", "calls"), [(1, 3), (2, 2)])
async def test_requests_until_n_images(monkeypatch, service, per_call, calls) -> None:
    seen = _serve(monkeypatch, httpx.Response(200, json={"data": [{"b64_json": PNG}] * per_call}))

    images = await CodexImagegenAdapter().generate("a cat", _config(), n=3)

    assert images == [(b"PNGDATA", "image/png")] * 3
    assert len(seen) == calls
    assert all("n" not in json.loads(request.content) for request in seen)
    assert len({request.headers["x-codex-image-turn-id"] for request in seen}) == 1


@pytest.mark.asyncio
async def test_401_renews_login_and_replays_once(monkeypatch, service) -> None:
    seen = _serve(monkeypatch, httpx.Response(401, json={"detail": "Unauthorized"}), OK)

    images = await CodexImagegenAdapter().generate("a cat", _config())

    assert images == [(b"PNGDATA", "image/png")]
    assert service.recovered == [7]
    assert [request.headers["Authorization"] for request in seen] == [
        "Bearer secret-token-7",
        "Bearer secret-token-8",
    ]


@pytest.mark.asyncio
async def test_second_401_asks_for_sign_in_without_another_replay(monkeypatch, service) -> None:
    seen = _serve(monkeypatch, httpx.Response(401, json={"detail": "Unauthorized"}))

    with pytest.raises(GenerationProviderError, match="Sign in to Codex again"):
        await CodexImagegenAdapter().generate("a cat", _config())

    assert len(seen) == 2
    assert service.recovered == [7]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(403, json={"detail": "Forbidden"}), "Free plan"),
        (
            httpx.Response(403, text="Attention Required! | Cloudflare: you have been blocked"),
            "restricted region",
        ),
    ],
)
async def test_403_explains_plan_or_region(monkeypatch, service, response, match) -> None:
    seen = _serve(monkeypatch, response)

    with pytest.raises(GenerationProviderError, match=match):
        await CodexImagegenAdapter().generate("a cat", _config())

    assert len(seen) == 1
    assert service.recovered == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "headers", "expected"),
    [
        (
            {
                "error": {
                    "type": "usage_limit_reached",
                    "resets_at": 1_786_150_800,
                    "plan_type": "plus",
                }
            },
            {},
            "(usage_limit_reached, plus plan); resets at 2026-08-08 01:00 UTC",
        ),
        (
            {"detail": "Too Many Requests"},
            {"x-codex-active-limit": "image_gen", "x-image-gen-primary-reset-at": "1786150800"},
            "(HTTP 429); resets at 2026-08-08 01:00 UTC",
        ),
    ],
)
async def test_429_reports_reset_and_never_retries(
    monkeypatch, service, body, headers, expected
) -> None:
    seen = _serve(monkeypatch, httpx.Response(429, json=body, headers=headers))

    with pytest.raises(GenerationProviderError) as exc:
        await CodexImagegenAdapter().generate("a cat", _config(), n=2)

    assert expected in str(exc.value)
    assert len(seen) == 1
    assert service.recovered == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{"url": "https://cdn.example/x.png"}]}),
        httpx.Response(200, json={"data": [{"b64_json": "%%%not-base64%%%"}]}),
        httpx.Response(200, json=["unexpected"]),
        httpx.Response(200, text="<html>not json</html>"),
    ],
)
async def test_malformed_success_is_a_provider_error(monkeypatch, service, response) -> None:
    _serve(monkeypatch, response)

    # Not a ValueError: the chat tool reads that as "imagegen is not configured".
    with pytest.raises(GenerationProviderError):
        await CodexImagegenAdapter().generate("a cat", _config())


@pytest.mark.asyncio
async def test_auth_and_network_failures_never_echo_the_bearer(monkeypatch, service) -> None:
    _serve(monkeypatch, httpx.ConnectError("connection refused"))
    with pytest.raises(GenerationProviderError, match="ConnectError") as exc:
        await CodexImagegenAdapter().generate("a cat", _config())
    assert "secret-token" not in str(exc.value)

    async def signed_out() -> CodexToken:
        raise CodexAuthError("authentication_required", "Sign in to Codex first.", 401)

    service.get_token = signed_out
    with pytest.raises(GenerationProviderError, match="Sign in to Codex first"):
        await CodexImagegenAdapter().generate("a cat", _config())


@pytest.mark.asyncio
async def test_catalog_binding_is_owner_bound_and_keeps_the_fixed_url(monkeypatch, service) -> None:
    profile = {
        "id": "p",
        "binding": "openai-codex",
        "base_url": "https://attacker.example/v1",
        "api_key": "",
        "models": [{"id": "m", "model": "gpt-image-2"}],
    }
    catalog = {
        "version": 1,
        "services": {
            "imagegen": {"active_profile_id": "p", "active_model_id": "m", "profiles": [profile]}
        },
    }
    seen = _serve(monkeypatch, OK)

    images = await generate_image("a cat", catalog=catalog, size="1792x1024")

    assert images == [(b"PNGDATA", "image/png")]
    assert str(seen[0].url) == CODEX_IMAGES_URL
    assert json.loads(seen[0].content)["size"] == "auto"
    assert is_owner_bound(profile)
