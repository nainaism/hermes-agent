"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import base64
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
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionForkCapabilities,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
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
from acp_adapter.provenance import session_provenance_meta
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

    _EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
    _EDIT_APPROVAL_POLICY_DEFAULT = "ask"
    _MODE_DEFAULT = "default"

    # Thought-level (reasoning) config option
    _THOUGHT_LEVEL_CONFIG_ID = "thought_level"
    _THOUGHT_LEVEL_OPTIONS = ("off", "low", "medium", "high", "max")
    _THOUGHT_LEVEL_DEFAULT = "medium"
    _MODE_ACCEPT_EDITS = "accept_edits"
    _MODE_DONT_ASK = "dont_ask"
    _MODE_TO_EDIT_APPROVAL_POLICY = {
        _MODE_DEFAULT: "ask",
        _MODE_ACCEPT_EDITS: "workspace_session",
        _MODE_DONT_ASK: "session",
    }
    _EDIT_APPROVAL_POLICY_TO_MODE = {
        value: key for key, value in _MODE_TO_EDIT_APPROVAL_POLICY.items()
    }

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

        # Start background watchers early so they run regardless of
        # whether on_connect is called (Paseo ACP connections may not
        # trigger on_connect, but sessions are still created).
        self._watcher_started = False

    def _ensure_watchers_started(self) -> None:
        """Start background watchers if not already running."""
        if self._watcher_started:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop — watchers not started")
            return
        self._watcher_started = True

        # Start the kanban completion watcher so ACP sessions receive
        # notifications when subscribed kanban tasks reach a terminal state.
        try:
            loop.create_task(self._kanban_acp_watcher())
            logger.info("Kanban ACP watcher started")
        except Exception:
            logger.warning("Failed to start kanban ACP watcher", exc_info=True)

        # Start the Codex goal watcher so ACP sessions receive
        # notifications when a dispatched Codex /goal completes.
        try:
            loop.create_task(self._codex_goal_watcher())
            logger.info("Codex goal watcher started")
        except Exception:
            logger.warning("Failed to start Codex goal watcher", exc_info=True)

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection and start watchers."""
        self._conn = conn
        logger.info("ACP client connected")
        self._ensure_watchers_started()

    def _session_modes(self, state: SessionState) -> SessionModeState:
        """Return ACP session modes while preserving Zed's separate model picker.

        Zed renders ``config_options`` in the prominent selector slot where the
        model picker was visible. Claude/Codex expose policy-like controls as ACP
        modes, which coexist with the model picker, so Hermes maps edit approval
        policy onto modes instead of advertising config options.
        """

        current = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        if current not in self._MODE_TO_EDIT_APPROVAL_POLICY:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[
                SessionMode(
                    id=self._MODE_DEFAULT,
                    name="Default",
                    description="Ask before edits.",
                ),
                SessionMode(
                    id=self._MODE_ACCEPT_EDITS,
                    name="Accept Edits",
                    description="Auto-allow workspace and /tmp edits; still asks for sensitive paths.",
                ),
                SessionMode(
                    id=self._MODE_DONT_ASK,
                    name="Don't Ask",
                    description="Auto-allow file edits for this session except sensitive paths.",
                ),
            ],
        )

    def _edit_approval_policy_for_state(self, state: SessionState) -> tuple[str, str | None]:
        mode = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        policy = self._MODE_TO_EDIT_APPROVAL_POLICY.get(mode, self._EDIT_APPROVAL_POLICY_DEFAULT)
        return policy, state.cwd

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

    def _build_thought_level_config(self, state: SessionState) -> list:
        """Return ACP config_options for reasoning/thought-level control.

        Paseo renders a SelectOption selector when config_options is non-empty.
        The current value is derived from the agent's reasoning_config, falling
        back to the configured default.
        """
        try:
            agent = getattr(state, "agent", None)
            rc = getattr(agent, "reasoning_config", None)
            if rc and isinstance(rc, dict):
                if not rc.get("enabled", True):
                    current = "off"
                else:
                    effort = rc.get("effort", self._THOUGHT_LEVEL_DEFAULT)
                    current = effort if effort in self._THOUGHT_LEVEL_OPTIONS else self._THOUGHT_LEVEL_DEFAULT
            else:
                current = self._THOUGHT_LEVEL_DEFAULT

            return [
                SessionConfigOptionSelect(
                    id=self._THOUGHT_LEVEL_CONFIG_ID,
                    name="Thinking",
                    description="Reasoning effort level",
                    category="thought_level",
                    type="select",
                    currentValue=current,
                    options=[
                        SessionConfigSelectOption(name="Off", value="off"),
                        SessionConfigSelectOption(name="Low", value="low"),
                        SessionConfigSelectOption(name="Medium", value="medium"),
                        SessionConfigSelectOption(name="High", value="high"),
                        SessionConfigSelectOption(name="Max", value="max"),
                    ],
                ),
            ]
        except Exception:
            logger.debug("Could not build thought-level config", exc_info=True)
            return []

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

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """Build ACP native context-usage data for clients like Zed.

        Zed's circular context indicator is driven by ACP ``usage_update``
        session updates: ``size`` is the model context window and ``used`` is
        the current request pressure.  Hermes estimates ``used`` from the same
        buckets it sends to providers: system prompt, conversation history, and
        tool schemas.
        """
        agent = state.agent
        compressor = getattr(agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None

        try:
            from agent.model_metadata import estimate_request_tokens_rough

            used = estimate_request_tokens_rough(
                state.history,
                system_prompt=getattr(agent, "_cached_system_prompt", "") or "",
                tools=getattr(agent, "tools", None) or None,
            )
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)

        return UsageUpdate(
            session_update="usage_update",
            size=max(size, 0),
            used=max(used, 0),
        )

    async def _send_usage_update(self, state: SessionState) -> None:
        """Send ACP native context usage to the connected client."""
        if not self._conn:
            return
        update = self._build_usage_update(state)
        if update is None:
            return
        try:
            await self._conn.session_update(
                session_id=state.session_id,
                update=update,
            )
        except Exception:
            logger.warning(
                "Failed to send ACP usage update for session %s",
                state.session_id,
                exc_info=True,
            )

    def _provenance_meta(
        self,
        acp_session_id: str,
        current_hermes_session_id: str,
        previous_hermes_session_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Best-effort ``_meta.hermes.sessionProvenance`` for an ACP session."""
        try:
            return session_provenance_meta(
                self.session_manager._get_db(),
                acp_session_id,
                current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        except Exception:
            logger.debug(
                "Could not build ACP session provenance for %s", acp_session_id, exc_info=True
            )
            return None

    async def _send_session_info_update(
        self,
        session_id: str,
        *,
        current_hermes_session_id: Optional[str] = None,
        previous_hermes_session_id: Optional[str] = None,
    ) -> None:
        """Send ACP native session metadata after Hermes changes it.

        When the internal Hermes head rotated (e.g. compression-driven session
        split during a turn), pass ``previous_hermes_session_id`` so the
        attached ``_meta.hermes.sessionProvenance`` flags the rotation reason.
        """
        if not self._conn:
            return
        try:
            row = self.session_manager._get_db().get_session(session_id)
        except Exception:
            logger.debug("Could not read ACP session info for %s", session_id, exc_info=True)
            return
        if not row:
            return

        title = row.get("title")
        # The `sessions` table does not have an `updated_at` column (see
        # hermes_state.py schema — only started_at/ended_at). Use "now" as
        # the updated_at since we're emitting this notification precisely
        # because the title was just refreshed.
        updated_at = datetime.now(timezone.utc).isoformat()
        meta = self._provenance_meta(
            session_id,
            current_hermes_session_id or session_id,
            previous_hermes_session_id,
        )
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=title if isinstance(title, str) and title.strip() else None,
            updated_at=updated_at,
            field_meta=meta,
        )
        try:
            await self._conn.session_update(
                session_id=session_id,
                update=update,
            )
        except Exception:
            logger.debug("Could not send ACP session info update for %s", session_id, exc_info=True)

    def _schedule_usage_update(self, state: SessionState) -> None:
        """Schedule native context indicator refresh after ACP responses."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, self._send_usage_update(state))

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
            from agent.memory_manager import inject_memory_provider_tools

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
            inject_memory_provider_tools(state.agent)
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
        self._ensure_watchers_started()
        self._schedule_available_commands_update(state.session_id)
        return NewSessionResponse(
            session_id=state.session_id,
            models=self._build_model_state(state),
            modes=self._session_modes(state),
            configOptions=self._build_thought_level_config(state),
            field_meta=self._provenance_meta(
                state.session_id, getattr(state.agent, "session_id", state.session_id)
            ),
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

        self._schedule_usage_update(state)
        return LoadSessionResponse(
            models=self._build_model_state(state),
            modes=self._session_modes(state),
            configOptions=self._build_thought_level_config(state),
            field_meta=self._provenance_meta(
                session_id, getattr(state.agent, "session_id", session_id)
            ),
        )

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
        self._ensure_watchers_started()
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
        self._schedule_usage_update(state)
        return ResumeSessionResponse(
            models=self._build_model_state(state),
            modes=self._session_modes(state),
            field_meta=self._provenance_meta(
                state.session_id, getattr(state.agent, "session_id", state.session_id)
            ),
        )

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
        return ForkSessionResponse(
            session_id=new_id,
            models=self._build_model_state(state) if state is not None else None,
            modes=self._session_modes(state) if state is not None else None,
        )

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
        edit_approval_requester = None

        if conn:
            tool_progress_cb = make_tool_progress_cb(
                conn,
                session_id,
                loop,
                tool_call_ids,
                tool_call_meta,
                edit_approval_policy_getter=lambda: self._edit_approval_policy_for_state(state),
            )
            thinking_cb = make_thinking_cb(conn, session_id, loop)
            step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            message_cb = make_message_cb(conn, session_id, loop)
            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
            try:
                from acp_adapter.edit_approval import make_acp_edit_approval_requester

                edit_approval_requester = make_acp_edit_approval_requester(
                    conn.request_permission,
                    loop,
                    session_id,
                    auto_approve_getter=lambda: self._edit_approval_policy_for_state(state),
                )
            except Exception:
                logger.debug("Could not create ACP edit approval requester", exc_info=True)
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
        edit_approval_token = None
        previous_session_id = None

        def _run_agent() -> dict:
            nonlocal previous_approval_cb, previous_interactive, edit_approval_token, previous_session_id
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
            if edit_approval_requester:
                try:
                    from acp_adapter.edit_approval import set_edit_approval_requester

                    edit_approval_token = set_edit_approval_requester(edit_approval_requester)
                except Exception:
                    logger.debug("Could not set ACP edit approval requester", exc_info=True)
            # Signal to tools.approval that we have an interactive callback
            # and the non-interactive auto-approve path must not fire.
            previous_interactive = os.environ.get("HERMES_INTERACTIVE")
            os.environ["HERMES_INTERACTIVE"] = "1"

            # Expose the ACP session id so kanban_create can auto-subscribe
            # the caller for completion notifications (platform="acp").
            previous_acp_sid = os.environ.get("ACP_SESSION_ID")
            os.environ["ACP_SESSION_ID"] = session_id

            # Propagate the originating ACP session id to tools that want to
            # tag side-effects with it (e.g. ``kanban_create`` stamps it on
            # the new task so clients can render a per-session board). Save
            # and restore around the agent call so a re-used executor thread
            # never leaks one session's id into the next session's tools.
            previous_session_id = os.environ.get("HERMES_SESSION_ID")
            os.environ["HERMES_SESSION_ID"] = session_id
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
                # Restore ACP_SESSION_ID.
                if previous_acp_sid is None:
                    os.environ.pop("ACP_SESSION_ID", None)
                else:
                    os.environ["ACP_SESSION_ID"] = previous_acp_sid
                # Restore HERMES_SESSION_ID symmetrically.
                if previous_session_id is None:
                    os.environ.pop("HERMES_SESSION_ID", None)
                else:
                    os.environ["HERMES_SESSION_ID"] = previous_session_id
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)
                if edit_approval_token is not None:
                    try:
                        from acp_adapter.edit_approval import reset_edit_approval_requester

                        reset_edit_approval_requester(edit_approval_token)
                    except Exception:
                        logger.debug("Could not restore ACP edit approval requester", exc_info=True)
                if session_tokens is not None and clear_session_vars is not None:
                    try:
                        clear_session_vars(session_tokens)
                    except Exception:
                        logger.debug("Could not clear ACP session context", exc_info=True)

        try:
            # Snapshot the internal Hermes DB session id before the turn so we
            # can detect a compression-driven session rotation afterwards. The
            # ACP `session_id` stays the stable client handle; agent.session_id
            # is the live internal head that compression may rotate.
            pre_turn_hermes_id = getattr(state.agent, "session_id", None)
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

        # Detect a compression-driven internal session rotation. If the agent's
        # DB head moved during the turn, emit a session_info_update carrying
        # _meta.hermes.sessionProvenance so ACP clients can render the boundary
        # and keep old/new ids in lineage. The ACP session_id is unchanged.
        post_turn_hermes_id = getattr(state.agent, "session_id", None)
        if (
            conn
            and post_turn_hermes_id
            and pre_turn_hermes_id
            and post_turn_hermes_id != pre_turn_hermes_id
        ):
            try:
                await self._send_session_info_update(
                    session_id,
                    current_hermes_session_id=post_turn_hermes_id,
                    previous_hermes_session_id=pre_turn_hermes_id,
                )
            except Exception:
                logger.debug(
                    "Could not emit ACP provenance update after rotation for %s",
                    session_id,
                    exc_info=True,
                )

        final_response = result.get("final_response", "")
        cancelled = bool(state.cancel_event and state.cancel_event.is_set())
        interrupted = bool(result.get("interrupted")) or cancelled
        # Hermes' local "waiting for model response" interrupt status is metadata,
        # not assistant prose — clients get cancellation from stop_reason instead.
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        suppress_interrupt_response = interrupted and final_response.startswith(
            INTERRUPT_WAITING_FOR_MODEL_PREFIX
        )
        if final_response and not suppress_interrupt_response:
            try:
                from agent.title_generator import maybe_auto_title

                def _notify_title_update(_title: str) -> None:
                    if conn:
                        loop.call_soon_threadsafe(
                            asyncio.create_task,
                            self._send_session_info_update(session_id),
                        )

                maybe_auto_title(
                    self.session_manager._get_db(),
                    session_id,
                    user_text,
                    final_response,
                    state.history,
                    title_callback=_notify_title_update,
                )
            except Exception:
                logger.debug("Failed to auto-title ACP session %s", session_id, exc_info=True)
        if (
            final_response
            and conn
            and not suppress_interrupt_response
            and (not streamed_message or result.get("response_transformed"))
        ):
            # Deliver the final response when streaming did not already send it,
            # or when a plugin hook transformed the response after streaming
            # finished (e.g. transform_llm_output) — otherwise the appended /
            # rewritten text never reaches the client.
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

        await self._send_usage_update(state)

        stop_reason = "cancelled" if cancelled else "end_turn"
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
            from types import SimpleNamespace
            from agent.memory_manager import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            )
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={
                    tool.get("function", {}).get("name")
                    for tool in tools or []
                    if isinstance(tool, dict)
                },
                enabled_toolsets=toolsets,
                _memory_manager=getattr(state.agent, "_memory_manager", None),
            )
            inject_memory_provider_tools(tool_view)
            tools = tool_view.tools
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
        """Dispatch /goal to Codex via Paseo WebSocket.

        Overrides the default GoalManager for ACP sessions — sends the
        goal to a new Codex agent and returns immediately. The codex goal
        watcher polls for completion and wakes the idle session.
        """
        if not args.strip():
            return (
                "Usage: /goal <prompt>\n"
                "Dispatches the goal to Codex via Paseo. "
                "Results are delivered back when Codex finishes."
            )

        try:
            cwd = getattr(state, "cwd", None) or os.getcwd()
            result = self.codex_goal_dispatch(
                goal_prompt=args.strip(),
                session_id=state.session_id,
                cwd=cwd,
            )
        except Exception as exc:
            logger.warning("goal dispatch failed: %s", exc)
            return f"Failed to dispatch goal to Codex: {exc}"

        if result.get("status") == "dispatched":
            agent_id = result.get("paseo_agent_id", "unknown")[:12]
            return (
                f"⤵ Goal dispatched to Codex (agent {agent_id}...)\n"
                f"The session will wake automatically when it completes."
            )
        else:
            return (
                f"Failed to dispatch goal: {result.get('error', 'unknown')}\n"
                "Falling back to local GoalManager."
            )

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
        if lower in ("low", "medium", "high", "max"):
            if hasattr(agent, "reasoning_config"):
                agent.reasoning_config = {"enabled": True, "effort": lower}
            return f"Reasoning effort set to: {lower}"
        return "Usage: /reasoning [low|medium|high|max|off]"

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
        normalized_mode = str(mode_id or "").strip()
        if normalized_mode not in self._MODE_TO_EDIT_APPROVAL_POLICY:
            normalized_mode = self._MODE_DEFAULT
        setattr(state, "mode", normalized_mode)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, normalized_mode)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        if str(config_id) == self._EDIT_APPROVAL_POLICY_CONFIG_ID:
            mode = self._EDIT_APPROVAL_POLICY_TO_MODE.get(str(value), self._MODE_DEFAULT)
            setattr(state, "mode", mode)
        elif str(config_id) == self._THOUGHT_LEVEL_CONFIG_ID:
            # Apply reasoning level to the agent
            agent = getattr(state, "agent", None)
            lower = str(value).strip().lower()
            if lower == "off":
                if hasattr(agent, "reasoning_config"):
                    agent.reasoning_config = {"enabled": False}
                logger.info("Session %s: reasoning disabled", session_id)
            elif lower in self._THOUGHT_LEVEL_OPTIONS:
                if hasattr(agent, "reasoning_config"):
                    agent.reasoning_config = {"enabled": True, "effort": lower}
                logger.info("Session %s: reasoning effort set to %s", session_id, lower)
        else:
            options = getattr(state, "config_options", None)
            if not isinstance(options, dict):
                options = {}
            options[str(config_id)] = value
            setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(
            configOptions=self._build_thought_level_config(state),
        )

    # ---- Kanban completion watcher ------------------------------------------

    async def _kanban_acp_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to ACP sessions.

        Mirrors the Gateway's ``_kanban_notifier_watcher`` but delivers to
        ACP sessions instead of platform adapters.  Subscriptions with
        ``platform="acp"`` are matched to the ACP session identified by
        ``chat_id`` (which holds the ACP session_id).

        On task completion (``completed``), also injects a synthetic
        user message into the ACP session via ``conn.session_update`` so
        the orchestrator (かえで) can review and dispatch the next phase
        without manual intervention.
        """
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban acp watcher: kanban_db not importable; disabled")
            return

        TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out")
        MAX_SEND_FAILURES = 3
        sub_fail_counts: dict[tuple, int] = {}

        # Initial delay so the ACP connection can fully establish.
        await asyncio.sleep(5)

        while True:
            try:
                deliveries = await asyncio.to_thread(self._kanban_acp_collect, _kb, TERMINAL_KINDS)
                if deliveries:
                    for d in deliveries:
                        await self._kanban_acp_deliver(d, sub_fail_counts, MAX_SEND_FAILURES)
            except Exception:
                logger.debug("kanban acp watcher: tick error", exc_info=True)
            await asyncio.sleep(interval)

    # -- collect (runs in thread) ---------------------------------------------

    @staticmethod
    def _kanban_acp_collect(_kb, terminal_kinds: tuple) -> list[dict]:
        """Collect pending kanban events for ACP subscriptions (thread-safe)."""
        deliveries: list[dict] = []
        try:
            boards = _kb.list_boards(include_archived=False)
        except Exception:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        seen_db_paths: set[str] = set()
        for board_meta in boards:
            slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
            db_path = board_meta.get("db_path")
            try:
                resolved = str(
                    __import__("pathlib").Path(db_path).expanduser().resolve()
                ) if db_path else str(_kb.kanban_db_path(slug).resolve())
            except Exception:
                resolved = f"slug:{slug}"
            if resolved in seen_db_paths:
                continue
            seen_db_paths.add(resolved)
            try:
                conn = _kb.connect(board=slug)
            except Exception:
                continue
            try:
                subs = _kb.list_notify_subs(conn)
                for sub in subs:
                    # Only handle ACP-platform subscriptions
                    if (sub.get("platform") or "").lower() != "acp":
                        continue
                    old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                        conn,
                        task_id=sub["task_id"],
                        platform=sub["platform"],
                        chat_id=sub["chat_id"],
                        thread_id=sub.get("thread_id") or "",
                        kinds=terminal_kinds,
                    )
                    if not events:
                        continue
                    task = _kb.get_task(conn, sub["task_id"])
                    deliveries.append({
                        "sub": sub,
                        "old_cursor": old_cursor,
                        "cursor": cursor,
                        "events": events,
                        "task": task,
                        "board": slug,
                    })
            finally:
                conn.close()
        return deliveries

    # -- deliver (runs on event loop) ----------------------------------------

    async def _kanban_acp_deliver(
        self,
        delivery: dict,
        sub_fail_counts: dict[tuple, int],
        max_failures: int,
    ) -> None:
        """Send a kanban notification to the ACP session and inject a prompt on completion."""
        sub = delivery["sub"]
        task = delivery["task"]
        session_id = sub["chat_id"]  # For ACP subs, chat_id = ACP session_id
        events = delivery["events"]
        board_slug = delivery.get("board")
        conn = self._conn

        if not conn:
            logger.debug("kanban acp watcher: no ACP connection; skipping")
            return

        # Check that the target session exists
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.debug(
                "kanban acp watcher: ACP session %s not found; skipping",
                session_id,
            )
            return

        title = (task.title if task else sub["task_id"])[:120]
        tag = f"@{task.assignee} " if task and task.assignee else ""

        for ev in events:
            kind = ev.kind
            if kind == "completed":
                handoff = ""
                payload_summary = None
                if ev.payload and ev.payload.get("summary"):
                    payload_summary = str(ev.payload["summary"])
                if payload_summary:
                    h = payload_summary.strip().splitlines()[0][:200]
                    handoff = f"\n{h}"
                elif task and task.result:
                    r = task.result.strip().splitlines()[0][:160]
                    handoff = f"\n{r}"
                msg = f"✔ {tag}Kanban {sub['task_id']} done — {title}{handoff}"
            elif kind == "blocked":
                reason = ""
                if ev.payload and ev.payload.get("reason"):
                    reason = f": {str(ev.payload['reason'])[:160]}"
                msg = f"⏸ {tag}Kanban {sub['task_id']} blocked{reason}"
            elif kind == "gave_up":
                err = ""
                if ev.payload and ev.payload.get("error"):
                    err = f"\n{str(ev.payload['error'])[:200]}"
                msg = f"✖ {tag}Kanban {sub['task_id']} gave up after repeated spawn failures{err}"
            elif kind == "crashed":
                msg = f"✖ {tag}Kanban {sub['task_id']} worker crashed (pid gone); dispatcher will retry"
            elif kind == "timed_out":
                limit = 0
                if ev.payload and ev.payload.get("limit_seconds"):
                    limit = int(ev.payload["limit_seconds"])
                msg = f"⏱ {tag}Kanban {sub['task_id']} timed out (max_runtime={limit}s); will retry"
            else:
                continue

            # Deliver as an ACP agent message to the session
            sub_key = (
                sub["task_id"], sub["platform"],
                sub["chat_id"], sub.get("thread_id") or "",
            )
            try:
                from acp.schema import AgentMessageChunk, TextContentBlock
                update = AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=msg),
                )
                await conn.session_update(session_id=session_id, update=update)
                logger.info(
                    "kanban acp watcher: delivered %s event for %s to ACP session %s on board %s",
                    kind, sub["task_id"], session_id, board_slug,
                )

                # On completion, wake the idle ACP session so the orchestrator
                # can review and dispatch the next phase automatically.
                if kind == "completed" and state and not state.is_running:
                    wake_text = (
                        f"📋 **Kanban task completed: {sub['task_id']}**\n"
                        f"{msg}\n\n"
                        f"Review the result and proceed with the next phase if ready."
                    )
                    logger.info(
                        "kanban acp watcher: waking ACP session %s for %s",
                        session_id, sub["task_id"],
                    )
                    await self.prompt(
                        prompt=[TextContentBlock(type="text", text=wake_text)],
                        session_id=session_id,
                    )

                sub_fail_counts.pop(sub_key, None)
            except Exception as exc:
                fails = sub_fail_counts.get(sub_key, 0) + 1
                sub_fail_counts[sub_key] = fails
                logger.warning(
                    "kanban acp watcher: send failed for %s (%d/%d): %s",
                    sub["task_id"], fails, max_failures, exc,
                )
                if fails >= max_failures:
                    logger.warning(
                        "kanban acp watcher: dropping sub for %s after %d failures",
                        sub["task_id"], max_failures,
                    )
                    sub_fail_counts.pop(sub_key, None)

    # ── Codex Goal Watcher ──────────────────────────────────────────────

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

    # In-memory tracking of dispatched codex goals
    _codex_goals: dict[str, dict] = {}  # goal_id → {paseo_agent_id, status, result_file, session_id, ...}

    @staticmethod
    def codex_goal_dispatch(
        goal_prompt: str,
        *,
        session_id: str,
        result_file: str = "/tmp/codex-goal-result.md",
        cwd: str | None = None,
        timeout_seconds: int = 0,
    ) -> dict:
        """Dispatch a /goal to Codex via Paseo WebSocket and register for watcher tracking.

        Returns dict with goal_id, paseo_agent_id, and status.
        """
        import json
        import uuid
        import pathlib

        goal_id = f"cg_{uuid.uuid4().hex[:12]}"
        paseo_ws_url = "ws://127.0.0.1:6767/ws"

        # 1. Connect to Paseo WS and send message to codex provider
        try:
            import subprocess
            # Use node for WebSocket (available via Paseo deps)
            # Single-step: create_agent_request with initialPrompt
            # Paseo daemon v0.1.80: initialPrompt triggers execution immediately.
            # Response comes as status(agent_created) — no separate response message.
            js_code = """
const WebSocket = require('ws');
const ws = new WebSocket('WSURL');
const ridCreate = 'create-' + Date.now();
let done = false;
let agentId = null;
let step = 'init';
ws.on('open', () => {
  ws.send(JSON.stringify({type:'hello',clientId:'hermes-codex-goal',clientType:'cli',protocolVersion:1,appVersion:'0.1.75'}));
});
ws.on('message', (data) => {
  const m = JSON.parse(data.toString());
  const mt = m.message?.type || '';
  if (m.type !== 'session') return;
  const payload = m.message?.payload || {};

  if (mt === 'rpc_error' && !done) {
    console.log(JSON.stringify({status:'error', error: m.message.error || m.message.message || 'rpc error'}));
    done = true; ws.close(); return;
  }
  if (mt === 'error' && !done) {
    console.log(JSON.stringify({status:'error', error: m.message.message || 'unknown'}));
    done = true; ws.close(); return;
  }

  if (mt === 'status') {
    if (payload.status === 'server_info' && step === 'init') {
      step = 'creating';
      ws.send(JSON.stringify({type:'session',message:{
        type:'create_agent_request',
        requestId: ridCreate,
        config: { provider: 'codex', cwd: CWD, modeId: 'full-access', thinkingOptionId: 'xhigh', featureValues: { fast_mode: true } },
        initialPrompt: GOALMSG
      }}));
    } else if (payload.status === 'agent_created' && step === 'creating') {
      agentId = payload.agentId;
      console.log(JSON.stringify({status:'sent', agentId: agentId}));
      done = true;
      ws.close();
    }
  }
});
setTimeout(() => { if (!done) { console.log(JSON.stringify({status:'timeout'})); process.exit(1); } }, 30000);
""".replace('WSURL', paseo_ws_url).replace('GOALMSG', json.dumps(goal_prompt)).replace('CWD', json.dumps(cwd or "/tmp"))

            result = subprocess.run(
                ["node", "-e", js_code],
                capture_output=True, text=True, timeout=45,
                env={
                    **os.environ,
                    "NODE_PATH": "/opt/homebrew/lib/node_modules/@getpaseo/cli/node_modules",
                },
            )
            # Parse the JSON output
            output = result.stdout.strip().split('\n')[-1] if result.stdout.strip() else ''
            dispatch_result = json.loads(output) if output else {"status": "unknown"}

        except Exception as exc:
            dispatch_result = {"status": "error", "error": str(exc)}

        if dispatch_result.get("status") != "sent":
            return {
                "goal_id": goal_id,
                "status": "dispatch_failed",
                "error": dispatch_result.get("error", "unknown"),
            }

        agent_id = dispatch_result.get("agentId", "")

        # 2. Register in tracking dict
        HermesACPAgent._codex_goals[goal_id] = {
            "paseo_agent_id": agent_id,
            "status": "dispatched",
            "result_file": result_file,
            "session_id": session_id,
            "cwd": cwd or "",
            "goal_prompt": goal_prompt[:500],
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "goal_id": goal_id,
            "status": "dispatched",
            "paseo_agent_id": agent_id,
            "result_file": result_file,
        }

    async def _codex_goal_watcher(self, interval: float = 5.0) -> None:
        """Poll Paseo codex agents for completion and wake idle ACP sessions.

        Flow:
        1. Dispatch: codex_goal_dispatch() sends /goal to codex via Paseo WS
        2. Watch: this poller checks Paseo agent JSON for status change
        3. Deliver: on completion, read result file and wake idle session via self.prompt()

        Mirrors _kanban_acp_watcher structure but monitors codex agent files instead of kanban DB.
        """
        import pathlib

        PASEO_AGENTS_DIR = pathlib.Path.home() / ".paseo" / "agents"

        await asyncio.sleep(8)  # Let ACP connection establish

        while True:
            try:
                # Re-scan workspace dirs each tick so newly created worktrees
                # (e.g. from paseo worktree create mid-session) are discovered.
                workspace_dirs = list(PASEO_AGENTS_DIR.iterdir()) if PASEO_AGENTS_DIR.exists() else []
                completions = await asyncio.to_thread(
                    self._codex_goal_collect, workspace_dirs
                )
                if completions:
                    for c in completions:
                        await self._codex_goal_deliver(c)
            except Exception:
                logger.debug("codex goal watcher: tick error", exc_info=True)
            await asyncio.sleep(interval)

    @staticmethod
    def _codex_goal_collect(workspace_dirs: list) -> list[dict]:
        """Check dispatched codex goals for completion (thread-safe).

        Reads Paseo agent JSON files to detect status transitions
        from non-closed to closed for tracked codex goal agents.
        """
        import json
        import pathlib

        completions: list[dict] = []
        goals = HermesACPAgent._codex_goals

        for goal_id, info in list(goals.items()):
            if info.get("status") in ("completed", "delivered", "failed"):
                continue

            agent_id = info.get("paseo_agent_id")
            if not agent_id:
                continue

            # Find the agent JSON across all workspace dirs
            agent_json = None
            for wd in workspace_dirs:
                candidate = wd / f"{agent_id}.json"
                if candidate.exists():
                    agent_json = candidate
                    break

            if not agent_json:
                continue

            try:
                data = json.loads(agent_json.read_text())
            except Exception:
                continue

            last_status = data.get("lastStatus", "")
            title = data.get("title", "")

            # Detect completion: status is closed/error, or idle after activity.
            # Codex returns to "idle" on completion (not "closed").
            # attentionReason="finished" is set inconsistently (sometimes None),
            # so we also treat idle with updated timestamp past dispatch as done.
            attention_reason = data.get("attentionReason")
            updated_at = data.get("updatedAt", "")
            dispatched_at = info.get("dispatched_at", "")
            is_done = last_status in ("closed", "error") or (
                last_status == "idle" and attention_reason == "finished"
            ) or (
                last_status == "idle"
                and updated_at
                and dispatched_at
                and updated_at > dispatched_at
            )
            if is_done:
                # Read result file if specified
                result_text = ""
                result_file = info.get("result_file")
                if result_file:
                    try:
                        rp = pathlib.Path(result_file)
                        if rp.exists():
                            result_text = rp.read_text()[:8000]
                    except Exception:
                        pass

                # Check git changes
                cwd = info.get("cwd", "")
                git_summary = ""
                if cwd:
                    try:
                        import subprocess
                        diff = subprocess.run(
                            ["git", "log", "--oneline", "-5"],
                            capture_output=True, text=True, cwd=cwd, timeout=5,
                        )
                        if diff.returncode == 0:
                            git_summary = diff.stdout.strip()[:1000]
                    except Exception:
                        pass

                completions.append({
                    "goal_id": goal_id,
                    "agent_id": agent_id,
                    "last_status": last_status,
                    "title": title,
                    "result_text": result_text,
                    "git_summary": git_summary,
                    "session_id": info.get("session_id"),
                    "cwd": cwd,
                })

                # Mark as completed to avoid re-processing
                goals[goal_id]["status"] = "completed"

        # --- External tracking files (~/.hermes/codex-goals/) ---
        # Supports goals dispatched by Hermes agents directly via Paseo WS
        # (bypassing _cmd_goal). The agent writes a tracking JSON file, and
        # this poller detects completion the same way as _codex_goals dict.
        tracking_dir = pathlib.Path.home() / ".hermes" / "codex-goals"
        if tracking_dir.exists():
            for tf in tracking_dir.glob("*.json"):
                try:
                    td = json.loads(tf.read_text())
                except Exception:
                    continue
                if td.get("status") in ("completed", "delivered", "failed"):
                    continue
                ext_agent_id = td.get("agent_id")
                if not ext_agent_id:
                    continue
                # Find agent JSON in Paseo workspace dirs
                agent_json = None
                for wd in workspace_dirs:
                    candidate = wd / f"{ext_agent_id}.json"
                    if candidate.exists():
                        agent_json = candidate
                        break
                if not agent_json:
                    continue
                try:
                    data = json.loads(agent_json.read_text())
                except Exception:
                    continue
                ext_last_status = data.get("lastStatus", "")
                # Codex returns to "idle" on completion (not "closed").
                # attentionReason="finished" is set inconsistently (sometimes None),
                # so we also treat idle with updated timestamp past dispatch as done.
                attention_reason = data.get("attentionReason")
                ext_updated_at = data.get("updatedAt", "")
                ext_dispatched_at = td.get("dispatched_at", "")
                is_done = ext_last_status in ("closed", "error") or (
                    ext_last_status == "idle" and attention_reason == "finished"
                ) or (
                    ext_last_status == "idle"
                    and ext_updated_at
                    and ext_dispatched_at
                    and ext_updated_at > ext_dispatched_at
                )
                if not is_done:
                    continue
                ext_title = data.get("title", "")
                ext_cwd = td.get("cwd", "")
                ext_git_summary = ""
                if ext_cwd:
                    try:
                        import subprocess
                        diff = subprocess.run(
                            ["git", "log", "--oneline", "-5"],
                            capture_output=True, text=True, cwd=ext_cwd, timeout=5,
                        )
                        if diff.returncode == 0:
                            ext_git_summary = diff.stdout.strip()[:1000]
                    except Exception:
                        pass
                completions.append({
                    "goal_id": tf.stem,
                    "agent_id": ext_agent_id,
                    "last_status": ext_last_status,
                    "title": ext_title,
                    "result_text": "",
                    "git_summary": ext_git_summary,
                    "session_id": td.get("session_id"),
                    "cwd": ext_cwd,
                })
                # Mark as completed in tracking file
                td["status"] = "completed"
                tf.write_text(json.dumps(td))

        return completions

    async def _codex_goal_deliver(self, completion: dict) -> None:
        """Wake idle ACP session with codex goal result."""
        from acp.schema import TextContentBlock

        goal_id = completion["goal_id"]
        session_id = completion.get("session_id")
        if not session_id:
            logger.warning("codex goal watcher: no session_id for goal %s", goal_id)
            return

        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.debug("codex goal watcher: session %s not found", session_id)
            return

        # Build wake message
        title = completion.get("title", "Codex Goal")
        result_text = completion.get("result_text", "")
        git_summary = completion.get("git_summary", "")
        last_status = completion.get("last_status", "")

        status_emoji = "❌" if last_status == "error" else "✅"
        parts = [
            f"{status_emoji} **Codex /goal completed: {title}**\n",
        ]

        if result_text:
            parts.append(f"**Result:**\n```\n{result_text[:3000]}\n```")

        if git_summary:
            parts.append(f"**Recent commits:**\n```\n{git_summary}\n```")

        if not result_text and not git_summary:
            parts.append("(No result file or git changes detected)")

        parts.append(
            "\nReview the results and take any follow-up action needed."
        )

        wake_text = "\n".join(parts)

        # Only wake if session is idle
        if state.is_running:
            logger.info(
                "codex goal watcher: session %s is busy; queuing notification",
                session_id,
            )
            HermesACPAgent._codex_goals.get(goal_id, {})["status"] = "delivered"
            return

        try:
            logger.info(
                "codex goal watcher: waking ACP session %s for goal %s",
                session_id, goal_id,
            )
            await self.prompt(
                prompt=[TextContentBlock(type="text", text=wake_text)],
                session_id=session_id,
            )
            HermesACPAgent._codex_goals.get(goal_id, {})["status"] = "delivered"
        except Exception as exc:
            logger.warning(
                "codex goal watcher: failed to wake session %s: %s",
                session_id, exc,
            )
            HermesACPAgent._codex_goals.get(goal_id, {})["status"] = "failed"
