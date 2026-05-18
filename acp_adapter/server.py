"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Optional

import acp
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    ImageContentBlock,
    AudioContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerHttp,
    McpServerSse,
    McpServerStdio,
    ModelInfo,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionListCapabilities,
    SessionModelState,
    SessionResumeCapabilities,
    SessionInfo,
    TextContentBlock,
    UnstructuredCommandInput,
    Usage,
    UserMessageChunk,
)

# AuthMethodAgent was renamed from AuthMethod in agent-client-protocol 0.9.0
try:
    from acp.schema import AuthMethodAgent
except ImportError:
    from acp.schema import AuthMethod as AuthMethodAgent  # type: ignore[attr-defined]

from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID, build_auth_methods, detect_provider

# Dynamic slash command support — driven by COMMAND_REGISTRY
try:
    from hermes_cli.commands import COMMAND_REGISTRY, resolve_command as _resolve_command
except ImportError:
    COMMAND_REGISTRY = []
    def _resolve_command(name): return None
from acp_adapter.events import (
    _build_plan_update_from_todo_result,
    build_tool_complete,
    build_tool_start,
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.session import SessionManager, SessionState, _expand_acp_enabled_toolsets

logger = logging.getLogger(__name__)

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# Server-side page size for list_sessions. The ACP ListSessionsRequest schema
# does not expose a client-side limit, so this is a fixed cap that clients
# paginate against using `cursor` / `next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 50


def _extract_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Extract plain text from ACP content blocks for display/commands."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(parts)


def _image_block_to_openai_part(block: ImageContentBlock) -> dict[str, Any] | None:
    """Convert an ACP image content block to OpenAI-style multimodal content."""
    data = str(getattr(block, "data", "") or "").strip()
    uri = str(getattr(block, "uri", "") or "").strip()
    mime_type = str(getattr(block, "mime_type", "") or "image/png").strip() or "image/png"

    if data:
        url = data if data.startswith("data:") else f"data:{mime_type};base64,{data}"
    elif uri:
        url = uri
    else:
        return None

    return {"type": "image_url", "image_url": {"url": url}}


def _content_blocks_to_openai_user_content(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str | list[dict[str, Any]]:
    """Convert ACP prompt blocks into a Hermes/OpenAI-compatible user content payload."""
    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in prompt:
        if isinstance(block, TextContentBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            continue
        if isinstance(block, ImageContentBlock):
            image_part = _image_block_to_openai_part(block)
            if image_part is not None:
                parts.append(image_part)
            continue

    if not parts:
        return _extract_text(prompt)

    # Keep pure text prompts as strings so slash-command handling and text-only
    # providers keep the exact legacy path. Switch to structured content only
    # when an actual non-text block is present.
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(text_parts)

    return parts


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    # ACP-specific commands that don't exist in COMMAND_REGISTRY
    _ACP_ONLY_COMMANDS = ("context", "version")

    @classmethod
    def _acp_commands(cls) -> set[str]:
        """Dynamically build the set of commands available in ACP.

        Includes commands from COMMAND_REGISTRY where not cli_only and not gateway_only,
        plus ACP-only commands like 'context' and 'version'.
        Also includes 'tools' which is cli_only but useful in ACP.
        """
        commands: set[str] = set()
        for cmd in COMMAND_REGISTRY:
            if not cmd.cli_only and not cmd.gateway_only:
                commands.add(cmd.name)
                if cmd.aliases:
                    commands.update(cmd.aliases)
        # Include ACP-only commands
        commands.update(cls._ACP_ONLY_COMMANDS)
        # Include 'tools' even though cli_only — useful in ACP
        commands.add("tools")
        return commands

    @classmethod
    def _acp_command_handlers(cls) -> dict[str, callable]:
        """Map command names to handler methods."""
        return {
            # Existing handlers
            "help": cls._cmd_help,
            "model": cls._cmd_model,
            "tools": cls._cmd_tools,
            "context": cls._cmd_context,
            "reset": cls._cmd_reset,
            "new": cls._cmd_reset,
            "compact": cls._cmd_compact,
            "compress": cls._cmd_compact,
            "steer": cls._cmd_steer,
            "queue": cls._cmd_queue,
            "version": cls._cmd_version,
            # New handlers
            "retry": cls._cmd_retry,
            "undo": cls._cmd_undo,
            "title": cls._cmd_title,
            "branch": cls._cmd_branch,
            "background": cls._cmd_background,
            "agents": cls._cmd_agents,
            "goal": cls._cmd_goal,
            "resume": cls._cmd_resume,
            "footer": cls._cmd_footer,
            "yolo": cls._cmd_yolo,
            "reasoning": cls._cmd_reasoning,
            "fast": cls._cmd_fast,
            "curator": cls._cmd_curator,
            "kanban": cls._cmd_kanban,
            "usage": cls._cmd_usage,
            "debug": cls._cmd_debug,
            "stop": cls._cmd_stop,
        }

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    @staticmethod
    def _encode_model_choice(provider: str | None, model: str | None) -> str:
        """Encode a model selection so ACP clients can keep provider context."""
        raw_model = str(model or "").strip()
        if not raw_model:
            return ""
        raw_provider = str(provider or "").strip().lower()
        if not raw_provider:
            return raw_model
        return f"{raw_provider}:{raw_model}"

    def _build_model_state(self, state: SessionState) -> SessionModelState | None:
        """Return the ACP model selector payload for editors like Zed."""
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = getattr(state.agent, "provider", None) or detect_provider() or "openrouter"

        try:
            from hermes_cli.models import curated_models_for_provider, normalize_provider, provider_label

            normalized_provider = normalize_provider(provider)
            provider_name = provider_label(normalized_provider)
            available_models: list[ModelInfo] = []
            seen_ids: set[str] = set()

            for model_id, description in curated_models_for_provider(normalized_provider):
                rendered_model = str(model_id or "").strip()
                if not rendered_model:
                    continue
                choice_id = self._encode_model_choice(normalized_provider, rendered_model)
                if choice_id in seen_ids:
                    continue
                desc_parts = [f"Provider: {provider_name}"]
                if description:
                    desc_parts.append(str(description).strip())
                if rendered_model == model:
                    desc_parts.append("current")
                available_models.append(
                    ModelInfo(
                        model_id=choice_id,
                        name=rendered_model,
                        description=" • ".join(part for part in desc_parts if part),
                    )
                )
                seen_ids.add(choice_id)

            current_model_id = self._encode_model_choice(normalized_provider, model)
            if current_model_id and current_model_id not in seen_ids:
                available_models.insert(
                    0,
                    ModelInfo(
                        model_id=current_model_id,
                        name=model,
                        description=f"Provider: {provider_name} • current",
                    ),
                )

            if available_models:
                return SessionModelState(
                    available_models=available_models,
                    current_model_id=current_model_id or available_models[0].model_id,
                )
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None

        fallback_choice = self._encode_model_choice(provider, model)
        return SessionModelState(
            available_models=[ModelInfo(model_id=fallback_choice, name=model)],
            current_model_id=fallback_choice,
        )

    @staticmethod
    def _resolve_model_selection(raw_model: str, current_provider: str) -> tuple[str, str]:
        """Resolve ``provider:model`` input into the provider and normalized model id.

        The sentinel value ``"default"`` is expanded to the actual default
        model name from config.yaml so that APIs which do not recognise the
        literal string receive a real model identifier.
        """
        target_provider = current_provider
        new_model = raw_model.strip()

        # Expand "default" sentinel to the configured default model.
        if new_model.lower() == "default":
            try:
                from hermes_cli.config import load_config
                _cfg = load_config().get("model") or {}
                _resolved = (_cfg.get("default") or "") if isinstance(_cfg, dict) else str(_cfg)
                if _resolved.strip():
                    new_model = _resolved.strip()
            except Exception:
                logger.debug("Failed to resolve default model from config", exc_info=True)

        try:
            from hermes_cli.models import detect_provider_for_model, parse_model_input

            target_provider, new_model = parse_model_input(new_model, current_provider)
            if target_provider == current_provider:
                detected = detect_provider_for_model(new_model, current_provider)
                if detected:
                    target_provider, new_model = detected
        except Exception:
            logger.debug("Provider detection failed, using model as-is", exc_info=True)

        return target_provider, new_model

    async def _register_session_mcp_servers(
        self,
        state: SessionState,
        mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None,
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return

        try:
            from tools.mcp_tool import register_mcp_servers

            config_map: dict[str, dict] = {}
            for server in mcp_servers:
                name = server.name
                if isinstance(server, McpServerStdio):
                    config = {
                        "command": server.command,
                        "args": list(server.args),
                        "env": {item.name: item.value for item in server.env},
                    }
                else:
                    config = {
                        "url": server.url,
                        "headers": {item.name: item.value for item in server.headers},
                    }
                config_map[name] = config

            await asyncio.to_thread(register_mcp_servers, config_map)
        except Exception:
            logger.warning(
                "Session %s: failed to register ACP MCP servers",
                state.session_id,
                exc_info=True,
            )
            return

        try:
            from model_tools import get_tool_definitions

            enabled_toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"],
                mcp_server_names=[server.name for server in mcp_servers],
            )
            state.agent.enabled_toolsets = enabled_toolsets
            disabled_toolsets = getattr(state.agent, "disabled_toolsets", None)
            state.agent.tools = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
            )
            state.agent.valid_tool_names = {
                tool["function"]["name"] for tool in state.agent.tools or []
            }
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id,
                len(state.agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration",
                state.session_id,
                exc_info=True,
            )

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        resolved_protocol_version = (
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION
        )
        auth_methods = build_auth_methods()

        client_name = client_info.name if client_info else "unknown"
        logger.info(
            "Initialize from %s (protocol v%s)",
            client_name,
            resolved_protocol_version,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only accept authenticate() calls whose method_id matches the
        # provider we advertised in initialize(). Without this check,
        # authenticate() would acknowledge any method_id as long as the
        # server has provider credentials configured — harmless under
        # Hermes' threat model (ACP is stdio-only, local-trust), but poor
        # API hygiene and confusing if ACP ever grows multi-method auth.
        if not isinstance(method_id, str):
            return None
        normalized_method = method_id.strip().lower()
        provider = detect_provider()

        if normalized_method == TERMINAL_SETUP_AUTH_METHOD_ID:
            # Terminal auth launches Hermes setup/model selection out-of-band.
            # Only report success once that flow has produced usable runtime
            # credentials for the normal ACP session.
            return AuthenticateResponse() if provider else None

        if not provider or normalized_method != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    @staticmethod
    def _flatten_history_text(value: Any) -> str:
        """Normalize a persisted text-or-text-parts value into a single string.

        OpenAI-style assistant content (and provider reasoning fields) can arrive
        as either a scalar string or a list of ``{"text": ...}`` /
        ``{"type": "text", "content": ...}`` parts. Whitespace-only inputs
        collapse to an empty string so callers can treat ``""`` as "nothing to
        emit".
        """
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        return ""

    @classmethod
    def _history_message_text(cls, message: dict[str, Any]) -> str:
        """Extract displayable text from a persisted OpenAI-style message."""
        return cls._flatten_history_text(message.get("content"))

    @classmethod
    def _history_reasoning_text(cls, message: dict[str, Any]) -> str:
        """Extract displayable reasoning/thought text from a persisted assistant message.

        Returns the first non-empty value among ``reasoning_content`` (the
        canonical field used by DeepSeek / Moonshot and the post-#16892
        chat-completions normalizer) and ``reasoning`` (used by the codex
        event projector and several other transports). Both keys are
        actively written by live code paths, so neither branch is
        deprecated — they cover different transports rather than old vs.
        new sessions.
        """
        for key in ("reasoning_content", "reasoning"):
            text = cls._flatten_history_text(message.get(key))
            if text:
                return text
        return ""

    @staticmethod
    def _history_message_update(
        *,
        role: str,
        text: str,
    ) -> UserMessageChunk | AgentMessageChunk | None:
        """Build an ACP history replay update for a user/assistant message."""
        block = TextContentBlock(type="text", text=text)
        if role == "user":
            return UserMessageChunk(
                session_update="user_message_chunk",
                content=block,
            )
        if role == "assistant":
            return AgentMessageChunk(
                session_update="agent_message_chunk",
                content=block,
            )
        return None

    @staticmethod
    def _history_thought_update(text: str) -> AgentThoughtChunk:
        """Build an ACP history replay update for an assistant thought."""
        return acp.update_agent_thought_text(text)

    @staticmethod
    def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract function name/arguments from an OpenAI-style tool_call."""
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
        raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception:
                parsed = {"raw": raw_args}
            raw_args = parsed
        if not isinstance(raw_args, dict):
            raw_args = {}
        return name, raw_args

    @staticmethod
    def _history_tool_call_id(tool_call: dict[str, Any]) -> str:
        """Return the stable provider tool call id for ACP history replay."""
        return str(
            tool_call.get("id")
            or tool_call.get("call_id")
            or tool_call.get("tool_call_id")
            or ""
        ).strip()

    async def _replay_session_history(self, state: SessionState) -> None:
        """Replay persisted user/assistant history during session/load or session/resume.

        Invoked inline (``await``) from both ``load_session`` and
        ``resume_session`` so that spec-compliant ACP clients receive the
        full transcript within the request's lifetime — see the comment at
        the call sites for the rationale and prior-art citations.

        Replays the conversation as user/assistant chunks, thinking-mode
        thought chunks, plus reconstructed tool-call start/completion
        notifications. Merely restoring server-side state makes Hermes
        remember context, but leaves the editor looking like a clean thread.
        """
        if not self._conn or not state.history:
            return

        conn = self._conn
        session_id = state.session_id
        active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        async def _send(update) -> bool:
            """Send a single session update; return False on failure."""
            try:
                await conn.session_update(session_id=session_id, update=update)
                return True
            except Exception:
                logger.warning(
                    "Failed to replay ACP history for session %s",
                    session_id,
                    exc_info=True,
                )
                return False

        for message in state.history:
            role = str(message.get("role") or "")

            if role == "user":
                text = self._history_message_text(message)
                if text:
                    update = self._history_message_update(role=role, text=text)
                    if update is not None and not await _send(update):
                        return
                continue

            if role == "assistant":
                thought = self._history_reasoning_text(message)
                if thought and not await _send(self._history_thought_update(thought)):
                    return

                text = self._history_message_text(message)
                if text:
                    update = self._history_message_update(role=role, text=text)
                    if update is not None and not await _send(update):
                        return

                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        tool_call_id = self._history_tool_call_id(tool_call)
                        if not tool_call_id:
                            continue
                        tool_name, args = self._history_tool_call_name_args(tool_call)
                        active_tool_calls[tool_call_id] = (tool_name, args)
                        if not await _send(build_tool_start(tool_call_id, tool_name, args)):
                            return
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                tool_name = str(message.get("tool_name") or "").strip()
                function_args: dict[str, Any] | None = None
                if tool_call_id in active_tool_calls:
                    tool_name, function_args = active_tool_calls.pop(tool_call_id)
                if not tool_call_id or not tool_name:
                    continue
                result = message.get("content")
                result_text = result if isinstance(result, str) else None
                if not await _send(
                    build_tool_complete(
                        tool_call_id,
                        tool_name,
                        result=result_text,
                        function_args=function_args,
                    )
                ):
                    return
                if tool_name == "todo":
                    plan_update = _build_plan_update_from_todo_result(result_text)
                    if plan_update is not None and not await _send(plan_update):
                        return

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        self._schedule_available_commands_update(state.session_id)
        return NewSessionResponse(
            session_id=state.session_id,
            models=self._build_model_state(state),
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Loaded session %s", session_id)
        # Per ACP spec, `session/load` must stream the prior conversation back
        # to the client via `session/update` notifications BEFORE responding,
        # so the client receives the full transcript within the load request's
        # lifetime. Awaiting the replay here matches Codex / Claude Code /
        # OpenCode / Pi and the Zed client (which registers the session-update
        # routing entry before awaiting the loadSession RPC specifically so
        # in-call history replay updates can find the thread). Deferring this
        # via `loop.call_soon` (as we did briefly in May 2026) broke every
        # spec-compliant ACP client that measures notifications synchronously
        # against the load response — see #12285 follow-up.
        try:
            await self._replay_session_history(state)
        except Exception:
            # Replay is best-effort — a corrupted or unexpected message shape
            # must not turn a successful session/load into a JSON-RPC error
            # response. Per-notification failures are already caught inside
            # ``_replay_session_history``; this outer guard covers anything
            # raised by the helpers themselves before reaching ``_send``.
            logger.warning(
                "ACP history replay raised during session/load for %s — "
                "load will still succeed, partial transcript may be missing",
                session_id,
                exc_info=True,
            )
        self._schedule_available_commands_update(session_id)

        # Replay conversation history so the client (Paseo) can rebuild
        # the timeline.  During loadSession the client sets
        # replayingHistory=True, which causes every session_update to be
        # buffered into persistedHistory rather than emitted as events.
        await self._replay_history(state)

        return LoadSessionResponse(models=self._build_model_state(state))

    # ---- History replay ----------------------------------------------------

    async def _replay_history(self, state: "SessionState") -> None:
        """Send conversation history as ACP session updates for timeline rebuild.

        Paseo sets ``replayingHistory=True`` during ``loadSession``, which
        buffers every ``session_update`` into ``persistedHistory``.  Those
        buffered items are later yielded via ``streamHistory()`` to
        populate the UI timeline.  Without this replay the timeline stays
        empty after resuming an archived session.
        """
        if not self._conn or not state.history:
            return

        conn = self._conn
        msg_counter = 0

        for msg in state.history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip non-textual roles that Paseo doesn't render as messages
            if role == "system":
                continue

            # Extract text from content (may be str or list of content blocks)
            text = self._extract_content_text(content)
            if not text:
                continue

            # Skip tool-result messages (role == "tool") — they appear as
            # part of the assistant's tool-call timeline item in Paseo.
            if role == "tool":
                continue

            msg_id = f"replay-{msg_counter}"
            msg_counter += 1

            if role == "user":
                update = acp.update_user_message(acp.text_block(text))
                # Attach message_id so Paseo groups this as one user bubble
                update.message_id = msg_id
            elif role == "assistant":
                update = acp.update_agent_message(acp.text_block(text))
                update.message_id = msg_id
            else:
                continue

            try:
                await conn.session_update(state.session_id, update)
            except Exception:
                logger.debug("Failed to replay history update", exc_info=True)
                break

            # Small yield so the event-loop can flush writes without
            # stalling the loadSession response.
            await asyncio.sleep(0)

        logger.info(
            "Replayed %d history messages for session %s",
            msg_counter,
            state.session_id,
        )

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        """Normalise ``content`` (str | list[dict] | None) to plain text."""
        if not content:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        # Include tool-call names so the user can see what
                        # was invoked, but skip the raw JSON args.
                        parts.append(f"[tool: {block.get('name', '?')}]")
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()
        return str(content).strip()

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Resumed session %s", state.session_id)
        # See `load_session` above for the spec rationale — replay must
        # complete before the response so clients receive the full transcript
        # within the request's lifetime.
        try:
            await self._replay_session_history(state)
        except Exception:
            logger.warning(
                "ACP history replay raised during session/resume for %s — "
                "resume will still succeed, partial transcript may be missing",
                state.session_id,
                exc_info=True,
            )
        self._schedule_available_commands_update(state.session_id)
        return ResumeSessionResponse(models=self._build_model_state(state))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
            with state.runtime_lock:
                if state.is_running and state.current_prompt_text:
                    state.interrupted_prompt_text = state.current_prompt_text
            state.cancel_event.set()
            try:
                if getattr(state, "agent", None) and hasattr(state.agent, "interrupt"):
                    state.agent.interrupt()
            except Exception:
                logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
            logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        new_id = state.session_id if state else ""
        if state is not None:
            await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, new_id)
        if new_id:
            self._schedule_available_commands_update(new_id)
        return ForkSessionResponse(session_id=new_id)

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List ACP sessions with optional ``cwd`` filtering and cursor pagination.

        ``cwd`` is passed through to ``SessionManager.list_sessions`` which already
        normalizes and filters by working directory. ``cursor`` is a ``session_id``
        previously returned as ``next_cursor``; results resume after that entry.
        Server-side page size is capped at ``_LIST_SESSIONS_PAGE_SIZE``; when more
        results remain, ``next_cursor`` is set to the last returned ``session_id``.
        """
        infos = self.session_manager.list_sessions(cwd=cwd)

        if cursor:
            for idx, s in enumerate(infos):
                if s["session_id"] == cursor:
                    infos = infos[idx + 1:]
                    break
            else:
                # Unknown cursor -> empty page (do not fall back to full list).
                infos = []

        has_more = len(infos) > _LIST_SESSIONS_PAGE_SIZE
        infos = infos[:_LIST_SESSIONS_PAGE_SIZE]

        sessions = []
        for s in infos:
            updated_at = s.get("updated_at")
            if updated_at is not None and not isinstance(updated_at, str):
                updated_at = str(updated_at)
            sessions.append(
                SessionInfo(
                    session_id=s["session_id"],
                    cwd=s["cwd"],
                    title=s.get("title"),
                    updated_at=updated_at,
                )
            )

        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        user_content = _content_blocks_to_openai_user_content(prompt)
        has_content = bool(user_text) or (
            isinstance(user_content, list) and bool(user_content)
        )
        if not has_content:
            return PromptResponse(stop_reason="end_turn")

        # /steer on an idle session has no in-flight tool call to inject into.
        # Rewrite it so the payload runs as a normal user prompt, matching the
        # gateway's behavior (gateway/run.py ~L4898). Two sub-cases:
        #   1. Zed-interrupt salvage — a prior prompt was cancelled by the
        #      client right before /steer arrived; replay it with the steer
        #      text attached as explicit correction/guidance so the user's
        #      in-flight work isn't lost.
        #   2. Plain idle — no prior work to salvage; just run the steer
        #      payload as a regular prompt. Without this, _cmd_steer would
        #      silently append to state.queued_prompts and respond with
        #      "No active turn — queued for the next turn", which looks like
        #      /queue even though the user never typed /queue.
        if isinstance(user_content, str) and user_text.startswith("/steer"):
            steer_text = user_text.split(maxsplit=1)[1].strip() if len(user_text.split(maxsplit=1)) > 1 else ""
            interrupted_prompt = ""
            rewrite_idle = False
            with state.runtime_lock:
                if not state.is_running and steer_text:
                    if state.interrupted_prompt_text:
                        interrupted_prompt = state.interrupted_prompt_text
                        state.interrupted_prompt_text = ""
                    else:
                        rewrite_idle = True
            if interrupted_prompt:
                user_text = (
                    f"{interrupted_prompt}\n\n"
                    f"User correction/guidance after interrupt: {steer_text}"
                )
                user_content = user_text
            elif rewrite_idle:
                user_text = steer_text
                user_content = steer_text

        # Intercept slash commands — handle locally without calling the LLM.
        # Slash commands are text-only; if the client included images/resources,
        # send the whole multimodal prompt to the agent instead of treating it as
        # an ACP command.
        if isinstance(user_content, str) and user_text.startswith("/"):
            response_text = self._handle_slash_command(user_text, state)
            if response_text is not None:
                if self._conn:
                    update = acp.update_agent_message_text(response_text)
                    await self._conn.session_update(session_id, update)
                return PromptResponse(stop_reason="end_turn")

        # If Zed sends another regular prompt while the same ACP session is
        # still running, queue it instead of racing two AIAgent loops against
        # the same state.history. /steer and /queue are handled above and can
        # land immediately.
        with state.runtime_lock:
            if state.is_running:
                queued_text = user_text or "[Image attachment]"
                state.queued_prompts.append(queued_text)
                depth = len(state.queued_prompts)
                if self._conn:
                    update = acp.update_agent_message_text(
                        f"Queued for the next turn. ({depth} queued)"
                    )
                    await self._conn.session_update(session_id, update)
                return PromptResponse(stop_reason="end_turn")
            state.is_running = True
            state.current_prompt_text = user_text or "[Image attachment]"

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])

        conn = self._conn
        loop = asyncio.get_running_loop()

        if state.cancel_event:
            state.cancel_event.clear()

        tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
        tool_call_meta: dict[str, dict[str, Any]] = {}
        previous_approval_cb = None

        if conn:
            tool_progress_cb = make_tool_progress_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            thinking_cb = make_thinking_cb(conn, session_id, loop)
            step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            message_cb = make_message_cb(conn, session_id, loop)
            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
        else:
            tool_progress_cb = None
            thinking_cb = None
            step_cb = None
            message_cb = None
            approval_cb = None

        agent = state.agent
        agent.tool_progress_callback = tool_progress_cb
        agent.thinking_callback = thinking_cb
        agent.step_callback = step_cb

        # Track whether stream_delta_callback actually fired during the run.
        # If it did, the final update_agent_message_text is redundant (and
        # causes Paseo to display a duplicate message) because the client
        # already received the complete response via streaming chunks.
        _stream_fired = False

        def _stream_guard(text: str) -> None:
            nonlocal _stream_fired
            if text is not None:
                _stream_fired = True
            if message_cb is not None:
                message_cb(text)

        agent.stream_delta_callback = _stream_guard

        # Approval callback is per-thread (thread-local, GHSA-qg5c-hvr5-hjgr).
        # Set it INSIDE _run_agent so the TLS write happens in the executor
        # thread — setting it here would write to the event-loop thread's TLS,
        # not the executor's. Also set HERMES_INTERACTIVE so approval.py
        # takes the CLI-interactive path (which calls the registered
        # callback via prompt_dangerous_approval) instead of the
        # non-interactive auto-approve branch (GHSA-96vc-wcxf-jjff).
        # ACP's conn.request_permission maps cleanly to the interactive
        # callback shape — not the gateway-queue HERMES_EXEC_ASK path,
        # which requires a notify_cb registered in _gateway_notify_cbs.
        previous_approval_cb = None
        previous_interactive = None

        def _run_agent() -> dict:
            nonlocal previous_approval_cb, previous_interactive
            # Bind HERMES_SESSION_KEY for this session so per-session caches
            # (e.g. the interactive sudo password cache in tools.terminal_tool)
            # scope to the ACP session rather than leaking across sessions
            # that land on the same reused executor thread. This call runs
            # inside a contextvars.copy_context() below, so the ContextVar
            # write is isolated from other concurrent ACP sessions.
            try:
                from gateway.session_context import (
                    clear_session_vars,
                    set_session_vars,
                )
                session_tokens = set_session_vars(session_key=session_id)
            except Exception:
                session_tokens = None
                clear_session_vars = None  # type: ignore[assignment]
                logger.debug("Could not set ACP session context", exc_info=True)
            if approval_cb:
                try:
                    from tools import terminal_tool as _terminal_tool
                    previous_approval_cb = _terminal_tool._get_approval_callback()
                    _terminal_tool.set_approval_callback(approval_cb)
                except Exception:
                    logger.debug("Could not set ACP approval callback", exc_info=True)
            # Signal to tools.approval that we have an interactive callback
            # and the non-interactive auto-approve path must not fire.
            previous_interactive = os.environ.get("HERMES_INTERACTIVE")
            os.environ["HERMES_INTERACTIVE"] = "1"
            try:
                result = agent.run_conversation(
                    user_message=user_content,
                    conversation_history=state.history,
                    task_id=session_id,
                    persist_user_message=user_text or "[Image attachment]",
                )
                return result
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}
            finally:
                # Restore HERMES_INTERACTIVE.
                if previous_interactive is None:
                    os.environ.pop("HERMES_INTERACTIVE", None)
                else:
                    os.environ["HERMES_INTERACTIVE"] = previous_interactive
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)
                if session_tokens is not None and clear_session_vars is not None:
                    try:
                        clear_session_vars(session_tokens)
                    except Exception:
                        logger.debug("Could not clear ACP session context", exc_info=True)

        try:
            # Wrap the executor call in a fresh copy of the current context so
            # concurrent ACP sessions on the shared ThreadPoolExecutor don't
            # stomp on each other's ContextVar writes (HERMES_SESSION_KEY in
            # particular — used by the interactive sudo password cache scope).
            ctx = contextvars.copy_context()
            result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
            return PromptResponse(stop_reason="end_turn")

        if result.get("messages"):
            state.history = result["messages"]
            # Persist updated history so sessions survive process restarts.
            self.session_manager.save_session(session_id)

        final_response = result.get("final_response", "")
        if final_response:
            try:
                from agent.title_generator import maybe_auto_title

                maybe_auto_title(
                    self.session_manager._get_db(),
                    session_id,
                    user_text,
                    final_response,
                    state.history,
                )
            except Exception:
                logger.debug("Failed to auto-title ACP session %s", session_id, exc_info=True)
        if final_response and conn and not _stream_fired:
            # Only send the final update when streaming did NOT fire.
            # When stream_delta_callback was active, the client already
            # received the complete response via streaming chunks;
            # sending it again causes a duplicate message in Paseo.
            update = acp.update_agent_message_text(final_response)
            await conn.session_update(session_id, update)

        # Mark this turn idle before draining queued work so recursive prompt()
        # calls can acquire the session. Queued turns are intentionally run as
        # normal follow-up user prompts, preserving role alternation and history.
        with state.runtime_lock:
            state.is_running = False
            state.current_prompt_text = ""

        while True:
            with state.runtime_lock:
                if not state.queued_prompts:
                    break
                next_prompt = state.queued_prompts.pop(0)
            if conn:
                await conn.session_update(
                    session_id,
                    acp.update_user_message_text(next_prompt),
                )
            await self.prompt(
                prompt=[TextContentBlock(type="text", text=next_prompt)],
                session_id=session_id,
            )

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0),
                output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        stop_reason = "cancelled" if state.cancel_event and state.cancel_event.is_set() else "end_turn"
        return PromptResponse(stop_reason=stop_reason, usage=usage)

    # ---- Slash commands (headless) -------------------------------------------

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        """Advertise commands from COMMAND_REGISTRY (dynamic)."""
        commands: list[AvailableCommand] = []
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only or cmd.gateway_only:
                continue
            input_hint = cmd.args_hint or None
            commands.append(
                AvailableCommand(
                    name=cmd.name,
                    description=cmd.description,
                    input=UnstructuredCommandInput(hint=input_hint)
                    if input_hint
                    else None,
                )
            )
        # Add ACP-only commands
        commands.append(AvailableCommand(name="context", description="Show conversation message counts by role"))
        commands.append(AvailableCommand(name="tools", description="List available tools with descriptions"))
        return commands

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return

        try:
            await self._conn.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._available_commands(),
                ),
            )
        except Exception:
            logger.warning(
                "Failed to advertise ACP slash commands for session %s",
                session_id,
                exc_info=True,
            )

    def _schedule_available_commands_update(self, session_id: str) -> None:
        """Send the command advertisement after the session response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(
            asyncio.create_task, self._send_available_commands_update(session_id)
        )

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command and return the response text.

        Uses ``resolve_command()`` from the central ``COMMAND_REGISTRY``
        so aliases and prefix matching work the same as CLI/Gateway.

        Returns ``None`` for unrecognized commands so they fall through
        to the LLM (the user may have typed ``/something`` as prose).
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        # Resolve aliases via COMMAND_REGISTRY (e.g. "reset" → "new")
        resolved = _resolve_command(cmd)
        canonical = resolved.name if resolved else cmd

        # Check if it's a known ACP command
        if canonical not in self._acp_commands() and cmd not in self._acp_commands():
            return None  # not a known command — let the LLM handle it

        # Use canonical name for handler lookup
        lookup_name = canonical if canonical in self._acp_command_handlers() else cmd
        handler = self._acp_command_handlers().get(lookup_name)

        if handler is None:
            return f"/{canonical} is available but not yet implemented for ACP."

        try:
            return handler(self, args, state)
        except Exception as e:
            logger.error("Slash command /%s error: %s", canonical, e, exc_info=True)
            return f"Error executing /{canonical}: {e}"

    def _cmd_help(self, args: str, state: SessionState) -> str:
        commands = sorted(self._acp_commands())
        lines = ["Available commands:", ""]
        for name in commands:
            # Try to get description from COMMAND_REGISTRY
            resolved = _resolve_command(name)
            if resolved:
                desc = resolved.description
            elif name == "context":
                desc = "Show conversation message counts by role"
            elif name == "version":
                desc = "Show Hermes version"
            else:
                desc = ""
            lines.append(f"  /{name:12s}  {desc}")
        lines.append("")
        lines.append("Unrecognized /commands are sent to the model as normal messages.")
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider = getattr(state.agent, "provider", None) or "openrouter"
        target_provider, new_model = self._resolve_model_selection(args, current_provider)

        state.model = new_model
        state.agent = self.session_manager._make_agent(
            session_id=state.session_id,
            cwd=state.cwd,
            model=new_model,
            requested_provider=target_provider,
        )
        self.session_manager.save_session(state.session_id)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            )
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = t.get("function", {}).get("name", "?")
                desc = t.get("function", {}).get("description", "")
                # Truncate long descriptions
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        n_messages = len(state.history)
        if n_messages == 0:
            return "Conversation is empty (no messages yet)."
        # Count by role
        roles: dict[str, int] = {}
        for msg in state.history:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
        lines = [
            f"Conversation: {n_messages} messages",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        model = state.model or getattr(state.agent, "model", "")
        if model:
            lines.append(f"Model: {model}")
        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compact(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            if not getattr(agent, "compression_enabled", True):
                return "Context compression is disabled for this agent."
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            from agent.model_metadata import estimate_request_tokens_rough

            original_count = len(state.history)
            # Include system prompt + tool schemas so the figure reflects real
            # request pressure, not a transcript-only underestimate (#6217).
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                state.history, system_prompt=_sys_prompt, tools=_tools
            )
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # ACP sessions must keep a stable session id, so avoid the
                # SQLite session-splitting side effect inside _compress_context.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history,
                    getattr(agent, "_cached_system_prompt", "") or "",
                    approx_tokens=approx_tokens,
                    task_id=state.session_id,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_count = len(state.history)
            _sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or _sys_prompt
            _tools_after = getattr(agent, "tools", None) or _tools
            new_tokens = estimate_request_tokens_rough(
                state.history,
                system_prompt=_sys_prompt_after,
                tools=_tools_after,
            )
            return (
                f"Context compressed: {original_count} -> {new_count} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

    def _cmd_steer(self, args: str, state: SessionState) -> str:
        steer_text = args.strip()
        if not steer_text:
            return "Usage: /steer <guidance>"

        if state.is_running and hasattr(state.agent, "steer"):
            try:
                if state.agent.steer(steer_text):
                    preview = steer_text[:80] + ("..." if len(steer_text) > 80 else "")
                    return f"⏩ Steer queued for the active turn: {preview}"
            except Exception as exc:
                logger.warning("ACP steer failed for session %s: %s", state.session_id, exc)
                return f"⚠️ Steer failed: {exc}"

        with state.runtime_lock:
            state.queued_prompts.append(steer_text)
            depth = len(state.queued_prompts)
        return f"No active turn — queued for the next turn. ({depth} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        with state.runtime_lock:
            state.queued_prompts.append(queued_text)
            depth = len(state.queued_prompts)
        return f"Queued for the next turn. ({depth} queued)"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- New slash command handlers ------------------------------------------

    def _cmd_retry(self, args: str, state: SessionState) -> str:
        """Resend last user message to agent."""
        for msg in reversed(state.history):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(p for p in parts if p)
                # Remove the last exchange (user + assistant/tool)
                while state.history and state.history[-1].get("role") != "user":
                    state.history.pop()
                if state.history:
                    state.history.pop()
                with state.runtime_lock:
                    state.queued_prompts.append(content)
                return f"Retrying last message. ({len(state.queued_prompts)} queued)"
        return "No previous message to retry."

    def _cmd_undo(self, args: str, state: SessionState) -> str:
        """Remove last user/assistant exchange."""
        removed = 0
        while state.history and state.history[-1].get("role") in ("assistant", "tool"):
            state.history.pop()
            removed += 1
        if state.history and state.history[-1].get("role") == "user":
            state.history.pop()
            removed += 1
        self.session_manager.save_session(state.session_id)
        return f"Undid {removed} message(s)." if removed else "Nothing to undo."

    def _cmd_title(self, args: str, state: SessionState) -> str:
        """Set session title."""
        if not args:
            return "Usage: /title <name>"
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            db.set_session_title(state.session_id, args)
            return f"Title set to: {args}"
        except Exception as e:
            return f"Could not set title: {e}"

    def _cmd_branch(self, args: str, state: SessionState) -> str:
        """Save current state as a named branch/checkpoint."""
        branch_name = args.strip() or f"branch-{len(state.history)}"
        try:
            import json
            from pathlib import Path
            hermes_home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
            branches_dir = hermes_home / "branches"
            branches_dir.mkdir(parents=True, exist_ok=True)
            safe_branch = branch_name.replace("/", "_").replace(" ", "_")
            branch_file = branches_dir / f"{state.session_id}_{safe_branch}.json"
            branch_file.write_text(json.dumps(state.history, ensure_ascii=False, default=str))
            return f"Branch '{branch_name}' saved ({len(state.history)} messages)."
        except Exception as e:
            return f"Could not create branch: {e}"

    def _cmd_background(self, args: str, state: SessionState) -> str:
        """Run prompt in background (same as queue in ACP)."""
        if not args.strip():
            return "Usage: /background <prompt>"
        with state.runtime_lock:
            state.queued_prompts.append(args.strip())
            depth = len(state.queued_prompts)
        return f"Background task queued. ({depth} queued)"

    def _cmd_agents(self, args: str, state: SessionState) -> str:
        """Show active agent info."""
        model = state.model or getattr(state.agent, "model", "unknown")
        provider = getattr(state.agent, "provider", None) or "auto"
        status = "running" if state.is_running else "idle"
        return (
            f"Session: {state.session_id[:8]}\u2026\n"
            f"Model: {model}\n"
            f"Provider: {provider}\n"
            f"Status: {status}"
        )

    def _cmd_goal(self, args: str, state: SessionState) -> str:
        """GoalManager integration — same as CLI/Gateway."""
        try:
            from hermes_cli.goals import GoalManager
        except ImportError:
            return "Goals module not available."

        mgr = GoalManager(session_id=state.session_id, default_max_turns=20)
        lower = (args or "").strip().lower()

        if not args.strip() or lower == "status":
            return mgr.status_line()

        if lower == "pause":
            s = mgr.pause(reason="user-paused")
            return f"\u23f8 Goal paused: {s.goal}" if s else "No goal set."

        if lower == "resume":
            s = mgr.resume()
            if s is None:
                return "No goal to resume."
            with state.runtime_lock:
                state.queued_prompts.append(s.goal)
            return f"\u25b6 Goal resumed: {s.goal}"

        if lower in ("clear", "stop", "done"):
            had = mgr.has_goal()
            mgr.clear()
            return "\u2713 Goal cleared." if had else "No active goal."

        # Set new goal
        try:
            s = mgr.set(args.strip())
        except ValueError as e:
            return f"Invalid goal: {e}"

        # Queue the goal as first turn
        with state.runtime_lock:
            state.queued_prompts.append(s.goal)
        return f"\u2299 Goal set ({s.max_turns}-turn budget): {s.goal}"

    def _cmd_resume(self, args: str, state: SessionState) -> str:
        """Search for sessions to resume."""
        if not args.strip():
            return "Usage: /resume <session-id-prefix or name>"
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            sessions = db.search_sessions(args.strip(), limit=5)
            if not sessions:
                return f"No sessions matching '{args.strip()}' found."
            lines = ["Matching sessions:"]
            for s in sessions:
                sid = s.get("session_id", "?")
                title = s.get("title", "") or "(untitled)"
                updated = s.get("updated_at", "") or ""
                lines.append(f"  {sid[:12]}\u2026 {title} ({updated})")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not search sessions: {e}"

    def _cmd_footer(self, args: str, state: SessionState) -> str:
        """Toggle footer display."""
        current = getattr(state, "_show_footer", False)
        new_val = (
            not current
            if not args
            else args.strip().lower() in ("on", "true", "1")
        )
        state._show_footer = new_val
        return f"Footer {'enabled' if new_val else 'disabled'}."

    def _cmd_yolo(self, args: str, state: SessionState) -> str:
        """Toggle YOLO mode (skip dangerous command approvals)."""
        current = getattr(state, "_yolo_mode", False)
        new_val = (
            not current
            if not args
            else args.strip().lower() in ("on", "true", "1")
        )
        state._yolo_mode = new_val
        agent = state.agent
        if hasattr(agent, "skip_approvals"):
            agent.skip_approvals = new_val
        action = "auto-approve" if new_val else "require approval"
        if new_val:
            return f"YOLO mode ON \u26a0\ufe0f. Dangerous commands will {action}."
        return f"YOLO mode OFF. Dangerous commands will {action}."

    def _cmd_reasoning(self, args: str, state: SessionState) -> str:
        """Show/manage reasoning effort."""
        agent = state.agent
        rc = getattr(agent, "reasoning_config", None)
        if not args.strip():
            if not rc:
                return "Reasoning: default (medium)"
            effort = rc.get("effort", "medium")
            enabled = rc.get("enabled", True)
            return f"Reasoning: {effort} ({'enabled' if enabled else 'disabled'})"
        lower = args.strip().lower()
        if lower in ("off", "none"):
            if hasattr(agent, "reasoning_config"):
                agent.reasoning_config = {"enabled": False}
            return "Reasoning disabled."
        if lower in ("low", "medium", "high"):
            if hasattr(agent, "reasoning_config"):
                agent.reasoning_config = {"enabled": True, "effort": lower}
            return f"Reasoning effort set to: {lower}"
        return "Usage: /reasoning [low|medium|high|off]"

    def _cmd_fast(self, args: str, state: SessionState) -> str:
        """Toggle fast mode."""
        current = getattr(state, "_fast_mode", False)
        new_val = (
            not current
            if not args
            else args.strip().lower() in ("on", "true", "1")
        )
        state._fast_mode = new_val
        return f"Fast mode {'ON' if new_val else 'OFF'}."

    def _cmd_curator(self, args: str, state: SessionState) -> str:
        """Skill maintenance status."""
        return "Curator is available via the skills tool directly in ACP."

    def _cmd_kanban(self, args: str, state: SessionState) -> str:
        """Kanban board reference."""
        return "Kanban is available via the kanban tools directly in ACP."

    def _cmd_usage(self, args: str, state: SessionState) -> str:
        """Show token usage statistics."""
        agent = state.agent
        total_input = getattr(agent, "total_input_tokens", 0) or 0
        total_output = getattr(agent, "total_output_tokens", 0) or 0
        cost = getattr(agent, "total_cost", 0) or 0
        model = state.model or getattr(agent, "model", "unknown")
        lines = [f"Model: {model}"]
        lines.append(f"Tokens: {total_input:,} in / {total_output:,} out")
        if cost:
            lines.append(f"Est. cost: ${cost:.4f}")
        return "\n".join(lines)

    def _cmd_debug(self, args: str, state: SessionState) -> str:
        """Debug report — not supported in ACP."""
        return "Debug report upload is not supported in ACP mode. Check ~/.hermes/logs/ for logs."

    def _cmd_stop(self, args: str, state: SessionState) -> str:
        """Stop the current agent run."""
        if hasattr(state.agent, "cancel_event") and state.agent.cancel_event:
            state.agent.cancel_event.set()
        if state.cancel_event:
            state.cancel_event.set()
        return "Stop requested."

    # ---- Model switching (ACP protocol method) -------------------------------

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = self.session_manager.get_session(session_id)
        if state:
            current_provider = getattr(state.agent, "provider", None)
            requested_provider, resolved_model = self._resolve_model_selection(
                model_id,
                current_provider or "openrouter",
            )
            state.model = resolved_model
            provider_changed = bool(current_provider and requested_provider != current_provider)
            current_base_url = None if provider_changed else getattr(state.agent, "base_url", None)
            current_api_mode = None if provider_changed else getattr(state.agent, "api_mode", None)
            state.agent = self.session_manager._make_agent(
                session_id=session_id,
                cwd=state.cwd,
                model=resolved_model,
                requested_provider=requested_provider,
                base_url=current_base_url,
                api_mode=current_api_mode,
            )
            self.session_manager.save_session(session_id)
            logger.info(
                "Session %s: model switched to %s via provider %s",
                session_id,
                resolved_model,
                requested_provider,
            )
            return SetSessionModelResponse()
        logger.warning("Session %s: model switch requested for missing session", session_id)
        return None

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        setattr(state, "mode", mode_id)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        options = getattr(state, "config_options", None)
        if not isinstance(options, dict):
            options = {}
        options[str(config_id)] = value
        setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])
