"""Claude Agent SDK provider against a faked CLI client.

The real SDK types (options, messages, tool decorator) are used so option and
field names are checked against the pinned SDK; only ``ClaudeSDKClient`` — the
part that would spawn the CLI — is replaced.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.runtime.agentic.usage import UsageTracker, record_streamed_usage
from deeptutor.services.llm.config import LLMConfig
from deeptutor.services.llm.provider_core import claude_agent_sdk_provider as provider_module
from deeptutor.services.llm.provider_core.base import LLMProvider
from deeptutor.services.llm.provider_factory import build_isolated_provider

sdk = pytest.importorskip("claude_agent_sdk")
ClaudeAgentSDKProvider = provider_module.ClaudeAgentSDKProvider

HANG = "hang"
TRICKLE = "trickle"


class FakeClient:
    instances: list[FakeClient] = []
    script: list[Any] = []
    connect_error: BaseException | None = None
    server_info: dict[str, Any] | None = None

    def __init__(self, options):
        self.options = options
        self.sent: list[dict[str, Any]] = []
        self.disconnected = False
        FakeClient.instances.append(self)

    async def connect(self):
        if self.connect_error is not None:
            raise self.connect_error

    async def query(self, prompt):
        async for message in prompt:
            self.sent.append(message)

    async def receive_response(self):
        for item in self.script:
            if item == HANG:
                await asyncio.Event().wait()
            if item == TRICKLE:
                while True:
                    await asyncio.sleep(0.01)
                    yield text_delta(".")
            yield item

    async def get_server_info(self):
        return self.server_info

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture(autouse=True)
def fake_cli(monkeypatch, tmp_path):
    FakeClient.instances = []
    FakeClient.script = []
    FakeClient.connect_error = None
    FakeClient.server_info = None
    monkeypatch.setattr(sdk, "ClaudeSDKClient", FakeClient)
    monkeypatch.setenv("DEEPTUTOR_CLAUDE_SDK_HOME", str(tmp_path / "claude-home"))
    monkeypatch.delenv("DEEPTUTOR_CLAUDE_SDK_CLI_PATH", raising=False)
    monkeypatch.setattr("deeptutor.services.auth.AUTH_ENABLED", False)
    monkeypatch.setattr(provider_module, "_CLI_SLOTS", threading.BoundedSemaphore(3))
    return FakeClient


def text_delta(text: str):
    return _delta({"type": "text_delta", "text": text})


def thinking_delta(text: str):
    return _delta({"type": "thinking_delta", "thinking": text})


def _delta(delta: dict[str, Any]):
    event = {"type": "content_block_delta", "index": 0, "delta": delta}
    return sdk.StreamEvent(uuid="u", session_id="s", event=event)


def assistant(*blocks, error=None):
    return sdk.AssistantMessage(content=list(blocks), model="claude-sonnet-5-5", error=error)


def result(**overrides):
    fields = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s",
    }
    return sdk.ResultMessage(**{**fields, **overrides})


def tool(name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name, "description": f"{name} tool"}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def options() -> Any:
    return FakeClient.instances[-1].options


def sent_blocks() -> list[dict[str, Any]]:
    (message,) = FakeClient.instances[-1].sent
    assert message["type"] == "user" and message["parent_tool_use_id"] is None
    return message["message"]["content"]


def test_services_factory_builds_the_provider_without_a_key() -> None:
    provider = build_isolated_provider(
        LLMConfig(
            model="claude-opus-5-5",
            api_key="sk-must-not-be-used",
            binding="claude_agent_sdk",
            provider_name="claude_agent_sdk",
            provider_mode="oauth",
        )
    )

    assert isinstance(provider, ClaudeAgentSDKProvider)
    assert provider.api_key is None
    assert provider.get_default_model() == "claude-opus-5-5"
    assert provider.provider_name == "claude_agent_sdk"


@pytest.mark.asyncio
async def test_streams_text_and_thinking_deltas_and_maps_usage() -> None:
    FakeClient.script = [
        thinking_delta("plan"),
        text_delta("Hel"),
        text_delta("lo"),
        assistant(sdk.ThinkingBlock(thinking="plan", signature="sig")),
        assistant(sdk.TextBlock(text="Hello")),
        result(
            result="Hello",
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "service_tier": "standard",
            },
        ),
    ]
    text: list[str] = []
    reasoning: list[str] = []

    async def on_text(chunk: str) -> None:
        text.append(chunk)

    async def on_reasoning(chunk: str) -> None:
        reasoning.append(chunk)

    response = await ClaudeAgentSDKProvider().chat_stream(
        [{"role": "user", "content": "hi"}],
        on_content_delta=on_text,
        on_reasoning_delta=on_reasoning,
    )

    assert text == ["Hel", "lo"]
    assert reasoning == ["plan"]
    assert response.content == "Hello"
    assert response.reasoning_content == "plan"
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 130,
        "completion_tokens": 5,
        "total_tokens": 135,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 20,
    }
    tracker = UsageTracker()
    record_streamed_usage(tracker, response.usage)
    assert (tracker.prompt_tokens, tracker.completion_tokens) == (130, 5)
    assert FakeClient.instances[-1].disconnected


@pytest.mark.asyncio
async def test_chat_reuses_the_consumer_without_callbacks() -> None:
    FakeClient.script = [
        text_delta("ignored without a callback"),
        assistant(sdk.TextBlock(text="Answer")),
        result(stop_reason="end_turn"),
    ]

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert (response.content, response.finish_reason) == ("Answer", "stop")
    assert options().include_partial_messages is True


@pytest.mark.asyncio
async def test_parallel_tool_calls_are_collected_through_error_max_turns(monkeypatch) -> None:
    declared: list[tuple[str, dict[str, Any]]] = []
    real_tool = sdk.tool

    def spy_tool(name, description, input_schema, annotations=None):
        declared.append((name, input_schema))
        return real_tool(name, description, input_schema, annotations)

    monkeypatch.setattr(sdk, "tool", spy_tool)
    FakeClient.script = [
        assistant(sdk.TextBlock(text="Checking both.")),
        assistant(
            sdk.ToolUseBlock(id="toolu_1", name="mcp__deeptutor__web_search", input={"q": "a"})
        ),
        assistant(sdk.ToolUseBlock(id="toolu_2", name="mcp__deeptutor__rag", input={"q": "b"})),
        assistant(sdk.ToolUseBlock(id="toolu_3", name="Bash", input={"command": "ls"})),
        result(
            subtype="error_max_turns", is_error=True, errors=["Reached maximum number of turns"]
        ),
    ]

    response = await ClaudeAgentSDKProvider().chat(
        [{"role": "user", "content": "search"}],
        tools=[
            tool("web_search", {"type": "object", "properties": {"q": {"type": "string"}}}),
            tool("rag", {"type": "object"}),
        ],
    )

    assert response.finish_reason == "tool_calls"
    assert response.content == "Checking both."
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("toolu_1", "web_search", {"q": "a"}),
        ("toolu_2", "rag", {"q": "b"}),
    ]
    assert options().allowed_tools == ["mcp__deeptutor__web_search", "mcp__deeptutor__rag"]
    assert list(options().mcp_servers) == ["deeptutor"]
    assert options().mcp_servers["deeptutor"]["type"] == "sdk"
    assert options().max_turns == 1
    # The SDK only passes a schema through verbatim with both keys present.
    assert declared[1] == ("rag", {"type": "object", "properties": {}})


@pytest.mark.asyncio
async def test_options_isolate_the_cli(monkeypatch, tmp_path) -> None:
    FakeClient.script = [assistant(sdk.TextBlock(text="ok")), result()]

    await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    opts = options()
    home = tmp_path / "claude-home"
    assert opts.tools == []
    assert opts.allowed_tools == [] and opts.mcp_servers == {}
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert opts.permission_mode == "dontAsk"
    assert opts.user is None
    assert opts.verbatim_prompts is True
    assert opts.thinking == {"type": "adaptive", "display": "summarized"}
    assert opts.effort is None
    assert opts.model == "claude-sonnet-5-5"
    assert opts.cwd == str(home / "cwd") and not any(Path(opts.cwd).iterdir())
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert Path(opts.cli_path) == Path(sdk.__file__).parent / "_bundled" / "claude"
    assert opts.env["HOME"] == str(home)
    assert opts.env["CLAUDE_CONFIG_DIR"] == str(home / ".claude")
    for blanked in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    ):
        assert opts.env[blanked] == ""
    assert opts.env["ENABLE_TOOL_SEARCH"] == "false"
    assert opts.env["DISABLE_AUTO_COMPACT"] == "1"
    assert opts.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    # The token reaches the CLI only by inheritance, never through DeepTutor.
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in opts.env

    monkeypatch.setenv("DEEPTUTOR_CLAUDE_SDK_CLI_PATH", "/opt/claude/bin/claude")
    await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])
    assert options().cli_path == "/opt/claude/bin/claude"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("high", "high"),
        ("XHIGH", "xhigh"),
        ("max", "max"),
        ("none", "low"),
        ("minimal", "low"),
        ("adaptive", None),
        ("", None),
        (None, None),
    ],
)
def test_reasoning_effort_maps_to_cli_effort(requested, expected) -> None:
    assert provider_module._effort(requested) == expected


@pytest.mark.asyncio
async def test_unsupported_parameters_are_dropped() -> None:
    FakeClient.script = [assistant(sdk.TextBlock(text="ok")), result()]

    response = await ClaudeAgentSDKProvider(default_model="claude-haiku-5-5").chat(
        [{"role": "user", "content": "q"}],
        model="claude-opus-5-5",
        max_tokens=60,
        temperature=0.2,
        reasoning_effort="medium",
        tool_choice={"type": "function", "function": {"name": "ask_user"}},
        response_format={"type": "json_object"},
        deeptutor_session_id="session-1",
        max_completion_tokens=10,
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert response.finish_reason == "stop"
    assert options().model == "claude-opus-5-5"
    assert options().effort == "medium"


@pytest.mark.asyncio
async def test_transcript_keeps_tool_calls_and_images_in_place() -> None:
    FakeClient.script = [assistant(sdk.TextBlock(text="ok")), result()]
    png = "data:image/png;base64,iVBORw0KGgo="

    await ClaudeAgentSDKProvider().chat(
        [
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": png}, "_meta": {"path": "a.png"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"q": "x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": "found"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/"},
                        "_meta": {"path": "b.jpg"},
                    }
                ],
            },
        ]
    )

    assert options().system_prompt == "Be brief."
    assert sent_blocks() == [
        {"type": "text", "text": "User:\nWhat is this?"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        },
        {
            "type": "text",
            "text": (
                "\nAssistant:\n"
                'Tool call web_search (id call_1): {"q": "x"}\n'
                "\n"
                "Tool result web_search (id call_1):\n"
                "found\n"
                "\n"
                "User:"
            ),
        },
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/"}},
    ]


@pytest.mark.asyncio
async def test_oversized_system_prompt_moves_into_the_transcript() -> None:
    FakeClient.script = [assistant(sdk.TextBlock(text="ok")), result()]
    system = "x" * 120_000

    await ClaudeAgentSDKProvider().chat(
        [{"role": "system", "content": system}, {"role": "user", "content": "q"}]
    )

    assert options().system_prompt is None
    assert sent_blocks()[0]["text"] == f"System instructions:\n{system}\n\nUser:\nq"


@pytest.mark.asyncio
async def test_output_cap_maps_to_length() -> None:
    FakeClient.script = [assistant(sdk.TextBlock(text="partial")), result(stop_reason="max_tokens")]

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert (response.content, response.finish_reason) == ("partial", "length")


@pytest.mark.asyncio
async def test_auth_failure_points_at_the_deployment_token() -> None:
    FakeClient.script = [
        assistant(sdk.TextBlock(text="Not logged in"), error="authentication_failed"),
        result(is_error=True, result="Not logged in · Please run /login"),
    ]

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert response.finish_reason == "error"
    assert "CLAUDE_CODE_OAUTH_TOKEN" in response.content
    assert not LLMProvider._is_transient_error(response.content)


@pytest.mark.asyncio
async def test_sdk_exceptions_become_error_responses() -> None:
    FakeClient.connect_error = sdk.CLINotFoundError("Claude Code not found", "/x/claude")

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert response.finish_reason == "error"
    assert "Claude Code not found" in response.content
    assert FakeClient.instances[-1].disconnected


@pytest.mark.asyncio
async def test_idle_watchdog_stops_a_silent_cli(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_CLAUDE_SDK_IDLE_TIMEOUT", "0.05")
    FakeClient.script = [HANG]

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert response.finish_reason == "error"
    assert "stalled" in response.content
    assert LLMProvider._is_transient_error(response.content)
    assert FakeClient.instances[-1].disconnected


@pytest.mark.asyncio
async def test_overall_watchdog_stops_a_runaway_cli(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_CLAUDE_SDK_TIMEOUT", "0.1")
    FakeClient.script = [TRICKLE]

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert response.finish_reason == "error"
    assert "no result within" in response.content
    assert not LLMProvider._is_transient_error(response.content)
    assert FakeClient.instances[-1].disconnected


@pytest.mark.asyncio
async def test_cli_slots_cap_concurrency_and_release_on_every_path(monkeypatch) -> None:
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(provider_module, "_CLI_SLOTS", slots)
    monkeypatch.setattr(provider_module, "_SLOT_WAIT_S", 0.1)
    FakeClient.script = [HANG]
    provider = ClaudeAgentSDKProvider()
    first = asyncio.create_task(provider.chat([{"role": "user", "content": "q"}]))
    for _ in range(500):  # until the first call holds the only slot
        if FakeClient.instances or first.done():
            break
        await asyncio.sleep(0.01)
    assert FakeClient.instances, first.result()

    busy = await provider.chat([{"role": "user", "content": "q"}])
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert busy.finish_reason == "error"
    assert "busy" in busy.content and LLMProvider._is_transient_error(busy.content)
    assert len(FakeClient.instances) == 1 and FakeClient.instances[0].disconnected
    assert slots.acquire(blocking=False)


def _user(role: str, user_id: str, tmp_path: Path) -> CurrentUser:
    scope = UserScope(kind=role, user_id=user_id, root=tmp_path / user_id)
    return CurrentUser(id=user_id, username=user_id, role=role, scope=scope)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "admins", "allowed"),
    [(None, 1, False), ("user", 1, False), ("admin", 2, False), ("admin", 1, True)],
)
async def test_identity_guard_with_auth_enabled(monkeypatch, tmp_path, role, admins, allowed):
    monkeypatch.setattr("deeptutor.services.auth.AUTH_ENABLED", True)
    monkeypatch.setattr(
        "deeptutor.services.auth.list_users",
        lambda: [{"role": "admin"}] * admins + [{"role": "user"}],
    )
    FakeClient.script = [assistant(sdk.TextBlock(text="ok")), result()]
    token = set_current_user(_user(role, "owner", tmp_path)) if role else None
    try:
        response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])
    finally:
        if token is not None:
            reset_current_user(token)

    assert (response.finish_reason == "stop") is allowed
    assert bool(FakeClient.instances) is allowed
    if not allowed:
        assert "reserved for the DeepTutor owner" in response.content
        assert not LLMProvider._is_transient_error(response.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actor_role", "allowed"), [(None, False), ("user", False), ("admin", True)]
)
async def test_partner_turns_need_the_owner_as_the_human(
    monkeypatch, tmp_path, actor_role, allowed
):
    actor = _user(actor_role, "someone", tmp_path) if actor_role else None
    monkeypatch.setattr(
        "deeptutor.services.partners.interaction.get_partner_turn_context",
        lambda: SimpleNamespace(actor=actor),
    )
    FakeClient.script = [assistant(sdk.TextBlock(text="ok")), result()]

    response = await ClaudeAgentSDKProvider().chat([{"role": "user", "content": "q"}])

    assert (response.finish_reason == "stop") is allowed
    if not allowed:
        assert "partner conversations" in response.content
        assert not FakeClient.instances
