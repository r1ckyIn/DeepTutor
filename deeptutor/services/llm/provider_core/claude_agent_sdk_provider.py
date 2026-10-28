"""Claude Agent SDK provider: the owner's Claude subscription through the bundled CLI.

Every ``chat()`` / ``chat_stream()`` runs one stateless turn in a fresh,
unmodified Claude Code CLI process (bundled with ``claude-agent-sdk``).
DeepTutor never handles a Claude credential: the CLI authenticates itself from
the ``CLAUDE_CODE_OAUTH_TOKEN`` the deployment puts in the server environment,
which the subprocess inherits unchanged. There is deliberately no login, no
token handling and no HTTP fallback here.

DeepTutor's tools reach the model through an in-process SDK MCP server whose
handlers only acknowledge the call. With ``max_turns=1`` the CLI stops before
its next model request, and the collected ``tool_use`` blocks return as OpenAI
tool calls for DeepTutor's own dispatcher to run.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing, asynccontextmanager, suppress
from itertools import groupby
import logging
import os
from pathlib import Path
import stat
import threading
import time
from types import ModuleType
from typing import Any

from deeptutor.services.llm.provider_core.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from deeptutor.services.llm.usage_frame import usage_breakdown, usage_mapping
from deeptutor.utils.secret_files import ensure_private_directory

CLAUDE_AGENT_SDK_MODELS = (
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
)
DEFAULT_CLAUDE_AGENT_SDK_MODEL = CLAUDE_AGENT_SDK_MODELS[0]

_MCP_SERVER_NAME = "deeptutor"
_MCP_TOOL_PREFIX = f"mcp__{_MCP_SERVER_NAME}__"
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
# The system prompt travels in argv, and Linux caps one argument at 128 KiB.
_MAX_SYSTEM_PROMPT_BYTES = 100_000
_SLOT_WAIT_S = 60.0
_AUTH_MARKERS = ("authentication", "/login", "not logged in", "invalid api key", "oauth token")
_AUTH_HINT = (
    "The DeepTutor server environment must provide CLAUDE_CODE_OAUTH_TOKEN "
    "(create one with `claude setup-token`); restart DeepTutor after setting it."
)
_OWNER_ONLY = (
    "Claude Agent (subscription) is reserved for the DeepTutor owner, the single "
    "administrator account. Choose another model for this request."
)
_PARTNER_REFUSAL = (
    "Claude Agent (subscription) is not available in partner conversations unless "
    "the owner is the one chatting. Give this partner another model."
)

logger = logging.getLogger(__name__)


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name) or default)
    except ValueError:
        return default
    return value if value > 0 else default


# ponytail: one process-wide threading semaphore, polled from the event loop,
# because DeepTutor calls LLMs from several event loops. No fairness or priority;
# add an interactive-first queue if background work ever starves chat.
_CLI_SLOTS = threading.BoundedSemaphore(int(_env_number("DEEPTUTOR_CLAUDE_SDK_MAX_CONCURRENCY", 3)))


@asynccontextmanager
async def _cli_slot() -> AsyncIterator[None]:
    deadline = time.monotonic() + _SLOT_WAIT_S
    while not _CLI_SLOTS.acquire(blocking=False):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"busy: every CLI slot stayed in use for {_SLOT_WAIT_S:g} s (timeout)"
            )
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        _CLI_SLOTS.release()


async def _disconnect(client: Any) -> None:
    """Disconnect, which stops the CLI, and wait for it even through a second cancel.

    The SDK shields its close only from anyio cancellation: another plain
    task.cancel() (a stop pressed twice) would cut off its terminate/kill and
    leave the CLI running outside the slot cap. So the close runs as its own
    task, and a cancel that arrives meanwhile is re-raised once it is done.
    """
    closing = asyncio.ensure_future(client.disconnect())
    cancelled = False
    while not closing.done():
        try:
            await asyncio.wait([closing])
        except asyncio.CancelledError:
            cancelled = True
    with suppress(Exception):
        closing.result()
    if cancelled:
        raise asyncio.CancelledError


class ClaudeAgentSDKProvider(LLMProvider):
    """Owner-only provider that drives the Claude Agent SDK one turn at a time."""

    provider_name = "claude_agent_sdk"

    def __init__(self, default_model: str | None = None):
        # No api_key on purpose: the CLI authenticates itself.
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model or DEFAULT_CLAUDE_AGENT_SDK_MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # max_tokens, temperature, tool_choice and every other kwarg are dropped:
        # the CLI has no such options, and on an output cap it continues by
        # itself (a hit cap still surfaces as finish_reason="length").
        return await self._run(messages, tools, model, reasoning_effort)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self._run(
            messages, tools, model, reasoning_effort, on_content_delta, on_reasoning_delta
        )

    async def _run(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        reasoning_effort: str | None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        total_s = _env_number("DEEPTUTOR_CLAUDE_SDK_TIMEOUT", 900)
        try:
            refusal = _caller_refusal()
            if refusal:
                return LLMResponse(content=refusal, finish_reason="error")
            sdk = _load_sdk()
            system_prompt, blocks = _prompt(messages)
            options = _options(
                sdk, model or self.default_model, system_prompt, tools, _effort(reasoning_effort)
            )
            async with _cli_slot(), asyncio.timeout(total_s):
                client = sdk.ClaudeSDKClient(options=options)
                try:
                    await client.connect()
                    await client.query(_user_turn(blocks))
                    async with aclosing(client.receive_response()) as stream:
                        return await _consume(sdk, stream, on_content_delta, on_reasoning_delta)
                finally:
                    # Terminates the CLI on every path, including the watchdogs.
                    await _disconnect(client)
        except TimeoutError:
            return LLMResponse(
                content=_error_text(f"no result within {total_s:g} s; the CLI was stopped"),
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=_error_text(exc), finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model


def _caller_refusal() -> str | None:
    """Why the current caller may not spend the owner's subscription, or None.

    Checked on every call in the caller's own context: provider instances are
    pooled across users, so a check at construction would prove nothing.
    """
    from deeptutor.multi_user.context import get_current_user_or_none
    from deeptutor.services import auth
    from deeptutor.services.partners.interaction import get_partner_turn_context

    turn = get_partner_turn_context()
    if turn is not None:
        # Judge the human in the chat, not the partner's synthetic or owner scope.
        return None if turn.actor is not None and _is_owner(turn.actor) else _PARTNER_REFUSAL
    if not auth.AUTH_ENABLED:
        return None
    user = get_current_user_or_none()
    return None if user is not None and _is_owner(user) else _OWNER_ONLY


def _is_owner(user: Any) -> bool:
    from deeptutor.multi_user.models import LOCAL_ADMIN_ID
    from deeptutor.services import auth

    if not user.is_admin:
        return False
    if not auth.AUTH_ENABLED:
        return True
    # Exactly one admin account, and the caller is it (scheduled jobs run as the
    # local admin). No admin in the store (PocketBase accounts) fails closed, as
    # does an admin token that outlived its role.
    # ponytail: reads the user store per call; cache it if that ever shows in profiles.
    admins = [u for u in auth.list_users() if u.get("role") == "admin"]
    return len(admins) == 1 and user.id in {admins[0].get("id"), LOCAL_ADMIN_ID}


def _load_sdk() -> ModuleType:
    try:
        import claude_agent_sdk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "claude-agent-sdk is not installed. Install it with "
            '`pip install "deeptutor[claude-agent-sdk]"`.'
        ) from exc
    return claude_agent_sdk


def _effort(reasoning_effort: str | None) -> str | None:
    """DeepTutor reasoning level -> CLI effort; None keeps the CLI default.

    Thinking cannot be turned off on the 5.x models, so the off levels ask for
    the least effort instead.
    """
    effort = (reasoning_effort or "").strip().lower()
    if effort in {"none", "minimal"}:
        return "low"
    return effort if effort in _EFFORTS else None


def _options(
    sdk: ModuleType,
    model: str,
    system_prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    effort: str | None = None,
) -> Any:
    """Options that confine the CLI to one turn without its own tools or settings."""
    home = Path(os.environ.get("DEEPTUTOR_CLAUDE_SDK_HOME") or "/tmp/deeptutor-claude")
    ensure_private_directory(home)
    # /tmp is shared: never hand the CLI a home that another user made first.
    info = home.lstat()
    if os.name != "nt" and (
        not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
    ):
        raise RuntimeError(f"{home} must be a private directory owned by this user")
    for path in (home / ".claude", home / "cwd"):
        ensure_private_directory(path)
    server, allowed = _tool_server(sdk, tools)
    return sdk.ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # no built-in Claude Code tools (Bash, Read, ToolSearch, ...)
        allowed_tools=allowed,
        mcp_servers={_MCP_SERVER_NAME: server} if server else {},
        strict_mcp_config=True,
        setting_sources=[],
        permission_mode="dontAsk",
        max_turns=1,
        include_partial_messages=True,
        thinking={"type": "adaptive", "display": "summarized"},
        effort=effort,
        # Prompt text is delivered as written: no @path file expansion, no slash commands.
        verbatim_prompts=True,
        cwd=str(home / "cwd"),
        # Never the `claude` on PATH. The override exists for hosts where the
        # bundled binary does not start.
        cli_path=os.environ.get("DEEPTUTOR_CLAUDE_SDK_CLI_PATH")
        or str(
            Path(sdk.__file__).parent / "_bundled" / ("claude.exe" if os.name == "nt" else "claude")
        ),
        # Merged over the inherited environment, which still carries the
        # deployment's CLAUDE_CODE_OAUTH_TOKEN. Blanks keep an API key, another
        # endpoint or a cloud backend from replacing the subscription.
        env={
            "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
            "ANTHROPIC_UNIX_SOCKET": "",
            "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR": "",
            # Every backend switch the bundled CLI (2.1.x) knows.
            "CLAUDE_CODE_USE_BEDROCK": "",
            "CLAUDE_CODE_USE_VERTEX": "",
            "CLAUDE_CODE_USE_FOUNDRY": "",
            "CLAUDE_CODE_USE_ANTHROPIC_AWS": "",
            "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD": "",
            "CLAUDE_CODE_USE_MANTLE": "",
            "CLAUDE_CODE_USE_GATEWAY": "",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            # Each call is a fresh session that is never resumed.
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
            "ENABLE_TOOL_SEARCH": "false",
            "DISABLE_AUTO_COMPACT": "1",
        },
    )


async def _acknowledge(_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": "DeepTutor accepted this tool call and will provide its result next.",
            }
        ]
    }


def _tool_server(sdk: ModuleType, tools: list[dict[str, Any]] | None) -> tuple[Any, list[str]]:
    sdk_tools: list[Any] = []
    for schema in tools or []:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = str((function or {}).get("name") or "").strip()
        if not name:
            continue
        parameters = function.get("parameters")
        # The SDK only takes a dict as JSON Schema when it has "type" and "properties".
        parameters = {
            "type": "object",
            "properties": {},
            **(parameters if isinstance(parameters, dict) else {}),
        }
        description = str(function.get("description") or name)
        sdk_tools.append(sdk.tool(name, description, parameters)(_acknowledge))
    if not sdk_tools:
        return None, []
    server = sdk.create_sdk_mcp_server(_MCP_SERVER_NAME, tools=sdk_tools)
    return server, [f"{_MCP_TOOL_PREFIX}{tool.name}" for tool in sdk_tools]


def _prompt(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Fold OpenAI-style messages into a system prompt and one user turn.

    The CLI lives for a single turn, so the dialogue so far becomes a readable
    transcript (the codebuddy_provider pattern, plus the assistant's tool
    calls). Images stay in place as base64 image blocks.
    """
    system: list[str] = []
    segments: list[Any] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        if role == "system":
            system.append("\n".join(filter(None, map(_part_text, parts))))
            continue
        if role == "assistant":
            heading = "Assistant:"
        elif role == "tool":
            name = message.get("name") or "tool"
            heading = f"Tool result {name} (id {message.get('tool_call_id')}):"
        else:
            heading = "User:"
        segments += ["", heading] if segments else [heading]
        for part in parts:
            segment = _image_block(part) or _part_text(part)
            if segment:
                segments.append(segment)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            segments.append(
                f"Tool call {function.get('name')} (id {call.get('id')}): "
                f"{function.get('arguments') or '{}'}"
            )
    system_prompt = "\n\n".join(filter(None, system))
    if len(system_prompt.encode()) > _MAX_SYSTEM_PROMPT_BYTES:
        segments[:0] = ["System instructions:", system_prompt, ""]
        system_prompt = ""
    blocks: list[dict[str, Any]] = []
    for is_text, group in groupby(segments, key=lambda segment: isinstance(segment, str)):
        if is_text:
            blocks.append({"type": "text", "text": "\n".join(group)})
        else:
            blocks.extend(group)
    return system_prompt or None, blocks


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and isinstance(part.get("text"), str):
        return part["text"]
    return ""


def _image_block(part: Any) -> dict[str, Any] | None:
    if not isinstance(part, dict):
        return None
    if part.get("type") == "image_url":
        from deeptutor.services.llm.provider_core.anthropic_provider import AnthropicProvider

        return AnthropicProvider._convert_image_block(part)
    if part.get("type") == "image" and isinstance(part.get("source"), dict):
        return {"type": "image", "source": part["source"]}  # drops DeepTutor's _meta
    return None


async def _user_turn(blocks: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "message": {"role": "user", "content": blocks},
        "parent_tool_use_id": None,
    }


async def _consume(
    sdk: ModuleType,
    stream: AsyncIterator[Any],
    on_content_delta: Callable[[str], Awaitable[None]] | None,
    on_reasoning_delta: Callable[[str], Awaitable[None]] | None,
) -> LLMResponse:
    """Read one turn up to its ResultMessage, streaming text and thinking deltas."""
    idle_s = _env_number("DEEPTUTOR_CLAUDE_SDK_IDLE_TIMEOUT", 120)
    text: list[str] = []
    thinking: list[str] = []
    calls: list[ToolCallRequest] = []
    failure = ""
    while True:
        try:
            # A dying CLI can leave the stream hanging forever (SDK issue #1110).
            async with asyncio.timeout(idle_s):
                message = await anext(stream)
        except TimeoutError:
            raise RuntimeError(f"the Claude CLI stalled with no output for {idle_s:g} s") from None
        except StopAsyncIteration:
            raise RuntimeError("the Claude CLI exited without a result") from None
        if isinstance(message, sdk.StreamEvent):
            event = message.event
            delta = event.get("delta") if event.get("type") == "content_block_delta" else None
            if not isinstance(delta, dict):
                continue
            if delta.get("type") == "text_delta" and on_content_delta:
                await on_content_delta(delta.get("text") or "")
            elif delta.get("type") == "thinking_delta" and on_reasoning_delta:
                await on_reasoning_delta(delta.get("thinking") or "")
        elif isinstance(message, sdk.AssistantMessage):
            failure = message.error or failure
            for block in message.content:
                if isinstance(block, sdk.TextBlock):
                    text.append(block.text)
                elif isinstance(block, sdk.ThinkingBlock):
                    thinking.append(block.thinking)
                elif isinstance(block, sdk.ToolUseBlock) and block.name.startswith(
                    _MCP_TOOL_PREFIX
                ):
                    calls.append(
                        ToolCallRequest(
                            id=block.id,
                            name=block.name.removeprefix(_MCP_TOOL_PREFIX),
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )
        elif isinstance(message, sdk.ResultMessage):
            # Return here: the turn is over, and the SDK may raise after an error result.
            raw = message.usage or {}
            # The CLI counts reasoning as output_tokens_details.thinking_tokens.
            thinking_tokens = usage_mapping(raw.get("output_tokens_details")).get("thinking_tokens")
            usage = usage_breakdown({"reasoning_tokens": thinking_tokens, **raw})
            reasoning = "\n".join(thinking).strip() or None
            if calls:
                # max_turns=1 ends a tool-using turn with error_max_turns; every
                # call of the turn (parallel ones arrive one block per message)
                # is the answer.
                return LLMResponse(
                    content="".join(text),
                    tool_calls=calls,
                    finish_reason="tool_calls",
                    usage=usage,
                    reasoning_content=reasoning,
                )
            if message.is_error:
                detail = "; ".join(message.errors or []) or message.result or message.subtype
                return LLMResponse(
                    content=_error_text(detail, auth=failure == "authentication_failed"),
                    finish_reason="error",
                    usage=usage,
                )
            return LLMResponse(
                content="".join(text) or message.result or "",
                finish_reason="length" if message.stop_reason == "max_tokens" else "stop",
                usage=usage,
                reasoning_content=reasoning,
            )


def _error_text(detail: Any, *, auth: bool = False) -> str:
    text = str(detail or "unknown error").strip()[:500]
    if auth or any(marker in text.lower() for marker in _AUTH_MARKERS):
        text = f"{text.rstrip('.')}. {_AUTH_HINT}"
    return f"Error calling Claude Agent SDK: {text}"


async def inspect_cli(timeout_s: float = 25) -> dict[str, Any]:
    """Settings probe: initialize the CLI and read its account, without a model call.

    Uses the provider's own options and environment, so it sees the login real
    calls will use. The result is persisted as provider discovery, so it never
    carries the account e-mail or plan.
    """
    models = [{"id": model, "services": ["llm"]} for model in CLAUDE_AGENT_SDK_MODELS]
    try:
        sdk = _load_sdk()
        # The clock starts once a slot is free: a busy CLI is not a silent one.
        async with _cli_slot(), asyncio.timeout(timeout_s):
            client = sdk.ClaudeSDKClient(options=_options(sdk, DEFAULT_CLAUDE_AGENT_SDK_MODEL))
            try:
                await client.connect()
                info = await client.get_server_info() or {}
            finally:
                await _disconnect(client)
    except TimeoutError:
        return {"status": "timeout", "models": models}
    except Exception as exc:
        logger.warning("Claude CLI status check failed: %s", exc)
        return {"status": "unreachable", "models": models}
    account = info.get("account") or {}
    signed_in = (
        account.get("apiProvider") in (None, "firstParty")
        and not account.get("apiKeySource")
        and (
            account.get("tokenSource") == "CLAUDE_CODE_OAUTH_TOKEN"
            or bool(account.get("subscriptionType"))
        )
    )
    return {"status": "connected" if signed_in else "auth_error", "models": models}


__all__ = [
    "CLAUDE_AGENT_SDK_MODELS",
    "ClaudeAgentSDKProvider",
    "DEFAULT_CLAUDE_AGENT_SDK_MODEL",
    "inspect_cli",
]
