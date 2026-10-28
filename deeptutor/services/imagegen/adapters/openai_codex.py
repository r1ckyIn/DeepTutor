"""ChatGPT-subscription image generation through DeepTutor's own Codex OAuth login.

Mirrors Codex CLI's standalone image tool (openai/codex ``ext/image-generation``,
rust-v0.153.4 through v0.164.0): a plain JSON ``POST`` to the ChatGPT Codex
images endpoint, answered with ``data[].b64_json``. The bearer is the same
owner-scoped sign-in the ``openai_codex`` LLM uses, so a partner turn draws on
its owner's login and every other account must sign in to Codex itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any
import uuid

import httpx

from deeptutor.services.codex_auth import (
    CodexAuthError,
    CodexOAuthService,
    CodexToken,
    get_codex_oauth_service,
)
from deeptutor.services.codex_auth.constants import CODEX_IMAGES_URL
from deeptutor.services.generation_http import (
    GenerationProviderError,
    decode_base64_media,
    raise_for_provider,
)
from deeptutor.services.imagegen.base import BaseImagegenAdapter
from deeptutor.services.imagegen.config import ImagegenConfig

_ACTION = "ChatGPT image generation"


class CodexImagegenAdapter(BaseImagegenAdapter):
    """Generate on the caller's ChatGPT plan; ``base_url`` and ``api_key`` are never read."""

    async def generate(
        self, prompt: str, config: ImagegenConfig, *, n: int = 1
    ) -> list[tuple[bytes, str]]:
        # This path ignores an explicit size/quality (openai/codex#28723) and
        # Codex itself always sends ``auto``, so the catalog knobs stay unused.
        body = {
            "prompt": prompt,
            "background": "opaque",
            "model": config.model,
            "quality": "auto",
            "size": "auto",
        }
        turn_id = str(uuid.uuid4())
        images: list[tuple[bytes, str]] = []
        try:
            service = get_codex_oauth_service()
            # Codex warns a generation can take minutes; 180 s is the chat idle timeout.
            async with (
                service.inference_guard(),
                httpx.AsyncClient(timeout=max(config.request_timeout, 180)) as client,
            ):
                # ponytail: one request at a time until n images (Codex never sends `n`);
                # run them concurrently if n>1 latency starts tripping the chat timeout.
                while len(images) < n:
                    resp = await _post(service, client, body, turn_id)
                    _raise_for_status(resp)
                    images += [
                        (decode_base64_media(b64, _ACTION), "image/png")
                        for b64 in _b64_images(resp)
                    ]
        except CodexAuthError as exc:
            raise GenerationProviderError(exc.public_message) from exc
        except httpx.HTTPError as exc:
            # Type only: the failed request carries the bearer and is never echoed.
            raise GenerationProviderError(
                f"{_ACTION} request failed ({type(exc).__name__})."
            ) from exc
        return images[:n]


async def _post(
    service: CodexOAuthService,
    client: httpx.AsyncClient,
    body: dict[str, Any],
    turn_id: str,
) -> httpx.Response:
    token = await service.get_token()
    resp = await client.post(CODEX_IMAGES_URL, json=body, headers=_headers(token, turn_id))
    if resp.status_code == 401:
        # Rejected before generating, so one replay on a renewed login cannot bill twice.
        await service.recover_after_unauthorized(token.generation)
        token = await service.get_token()
        resp = await client.post(CODEX_IMAGES_URL, json=body, headers=_headers(token, turn_id))
    return resp


def _headers(token: CodexToken, turn_id: str) -> dict[str, str]:
    # Same client identity as the Codex LLM provider (openai_codex_provider._build_headers).
    return {
        "Authorization": f"Bearer {token.access_token}",
        "ChatGPT-Account-ID": token.account_id,
        "originator": "DeepTutor",
        "User-Agent": "DeepTutor (python)",
        "x-codex-image-turn-id": turn_id,
    }


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code == 401:
        raise GenerationProviderError(
            "ChatGPT rejected the Codex sign-in even after renewing it. Sign in to Codex again."
        )
    if resp.status_code == 403:
        if "Cloudflare" in resp.text and "blocked" in resp.text:
            raise GenerationProviderError(
                "Cloudflare blocked the request to ChatGPT, usually because of a restricted "
                "region or network."
            )
        raise GenerationProviderError(
            "This ChatGPT account cannot generate images: the Free plan does not include "
            "image generation, or a workspace admin turned it off."
        )
    if resp.status_code == 429:
        raise GenerationProviderError(_quota_message(resp))
    raise_for_provider(resp, _ACTION)


def _quota_message(resp: httpx.Response) -> str:
    """Name the exhausted limit and its reset, body before headers, as Codex reads them."""
    try:
        error = resp.json()["error"]
    except (ValueError, KeyError, TypeError):
        error = None
    error = error if isinstance(error, dict) else {}
    limit = (resp.headers.get("x-codex-active-limit") or "codex").strip().lower().replace("_", "-")
    now = time.time()
    reset = None
    for value, base in (
        (error.get("resets_at"), 0.0),
        (resp.headers.get(f"x-{limit}-primary-reset-at"), 0.0),
        (resp.headers.get(f"x-{limit}-primary-reset-after-seconds"), now),
        (resp.headers.get("retry-after"), now),
    ):
        try:
            reset = datetime.fromtimestamp(base + float(value), tz=timezone.utc)
            break
        except (TypeError, ValueError, OverflowError, OSError):
            continue
    plan = f", {error['plan_type']} plan" if error.get("plan_type") else ""
    when = f"; resets at {reset:%Y-%m-%d %H:%M} UTC" if reset else ""
    return (
        f"ChatGPT image-generation quota exhausted ({error.get('type') or 'HTTP 429'}{plan})"
        f"{when}. Not retried."
    )


def _b64_images(resp: httpx.Response) -> list[str]:
    try:
        items = resp.json()["data"]
    except (ValueError, KeyError, TypeError):
        items = None
    found = [
        item["b64_json"]
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and isinstance(item.get("b64_json"), str) and item["b64_json"]
    ]
    if not found:
        raise GenerationProviderError(f"{_ACTION} returned no image data.")
    return found


__all__ = ["CodexImagegenAdapter"]
