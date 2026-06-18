"""Tests for Codex Goal Watcher in acp_adapter/server.py.

Tests the dispatch → collect → deliver lifecycle for sending /goal prompts
to Codex via Paseo WebSocket and receiving completion notifications.
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest


# ── Unit tests (no Paseo required) ──────────────────────────────────────


class TestCodexGoalTracking:
    """Test the in-memory _codex_goals dict lifecycle."""

    def setup_method(self):
        from acp_adapter.server import HermesACPAgent
        self._original_goals = HermesACPAgent._codex_goals.copy()
        HermesACPAgent._codex_goals.clear()

    def teardown_method(self):
        from acp_adapter.server import HermesACPAgent
        HermesACPAgent._codex_goals.clear()
        HermesACPAgent._codex_goals.update(self._original_goals)

    def test_register_and_collect(self):
        from acp_adapter.server import HermesACPAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake Paseo agent JSON
            agent_id = "test-agent-001"
            agent_dir = pathlib.Path(tmpdir) / "workspace"
            agent_dir.mkdir()
            agent_file = agent_dir / f"{agent_id}.json"
            agent_file.write_text(json.dumps({
                "id": agent_id,
                "lastStatus": "closed",
                "title": "Test Goal",
                "provider": "codex",
            }))

            # Register the goal
            HermesACPAgent._codex_goals["cg_001"] = {
                "paseo_agent_id": agent_id,
                "status": "dispatched",
                "result_file": "",
                "session_id": "test-session",
                "cwd": "",
            }

            # Collect — workspace_dirs contains the directory where agent files live
            completions = HermesACPAgent._codex_goal_collect([agent_dir])

            assert len(completions) == 1
            assert completions[0]["goal_id"] == "cg_001"
            assert completions[0]["last_status"] == "closed"
            assert completions[0]["session_id"] == "test-session"

            # Status updated to completed
            assert HermesACPAgent._codex_goals["cg_001"]["status"] == "completed"

    def test_collect_skips_already_completed(self):
        from acp_adapter.server import HermesACPAgent

        HermesACPAgent._codex_goals["cg_002"] = {
            "paseo_agent_id": "some-agent",
            "status": "completed",
            "session_id": "s1",
        }

        completions = HermesACPAgent._codex_goal_collect([])
        assert len(completions) == 0

    def test_collect_skips_delivered(self):
        from acp_adapter.server import HermesACPAgent

        HermesACPAgent._codex_goals["cg_003"] = {
            "paseo_agent_id": "some-agent",
            "status": "delivered",
            "session_id": "s1",
        }

        completions = HermesACPAgent._codex_goal_collect([])
        assert len(completions) == 0

    def test_collect_reads_result_file(self):
        from acp_adapter.server import HermesACPAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_id = "test-result-agent"
            agent_dir = pathlib.Path(tmpdir) / "ws"
            agent_dir.mkdir()
            agent_file = agent_dir / f"{agent_id}.json"
            agent_file.write_text(json.dumps({
                "id": agent_id,
                "lastStatus": "closed",
                "title": "Result Test",
                "provider": "codex",
            }))

            result_file = pathlib.Path(tmpdir) / "result.md"
            result_file.write_text("# Result\nAll tests passed!")

            HermesACPAgent._codex_goals["cg_004"] = {
                "paseo_agent_id": agent_id,
                "status": "dispatched",
                "result_file": str(result_file),
                "session_id": "s1",
                "cwd": "",
            }

            completions = HermesACPAgent._codex_goal_collect([agent_dir])

            assert len(completions) == 1
            assert "All tests passed" in completions[0]["result_text"]

    def test_collect_detects_error_status(self):
        from acp_adapter.server import HermesACPAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_id = "test-error-agent"
            agent_dir = pathlib.Path(tmpdir) / "ws"
            agent_dir.mkdir()
            agent_file = agent_dir / f"{agent_id}.json"
            agent_file.write_text(json.dumps({
                "id": agent_id,
                "lastStatus": "error",
                "title": "Failed Goal",
                "provider": "codex",
            }))

            HermesACPAgent._codex_goals["cg_005"] = {
                "paseo_agent_id": agent_id,
                "status": "dispatched",
                "result_file": "",
                "session_id": "s1",
                "cwd": "",
            }

            completions = HermesACPAgent._codex_goal_collect([agent_dir])

            assert len(completions) == 1
            assert completions[0]["last_status"] == "error"

    def test_collect_ignores_running_agent(self):
        from acp_adapter.server import HermesACPAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_id = "test-running-agent"
            agent_dir = pathlib.Path(tmpdir) / "ws"
            agent_dir.mkdir()
            agent_file = agent_dir / f"{agent_id}.json"
            agent_file.write_text(json.dumps({
                "id": agent_id,
                "lastStatus": "running",
                "title": "Still Running",
                "provider": "codex",
            }))

            HermesACPAgent._codex_goals["cg_006"] = {
                "paseo_agent_id": agent_id,
                "status": "dispatched",
                "result_file": "",
                "session_id": "s1",
                "cwd": "",
            }

            completions = HermesACPAgent._codex_goal_collect([agent_dir])

            assert len(completions) == 0
            # Goal status unchanged
            assert HermesACPAgent._codex_goals["cg_006"]["status"] == "dispatched"


class TestCodexGoalDispatch:
    """Test the dispatch method (Paseo WS interaction)."""

    def setup_method(self):
        from acp_adapter.server import HermesACPAgent
        self._original_goals = HermesACPAgent._codex_goals.copy()
        HermesACPAgent._codex_goals.clear()

    def teardown_method(self):
        from acp_adapter.server import HermesACPAgent
        HermesACPAgent._codex_goals.clear()
        HermesACPAgent._codex_goals.update(self._original_goals)

    def test_dispatch_returns_goal_id_on_success(self):
        """Test dispatch with mocked subprocess."""
        from acp_adapter.server import HermesACPAgent

        mock_output = '{"status":"sent","agentId":"fake-agent-123"}'
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0,
            )

            result = HermesACPAgent.codex_goal_dispatch(
                goal_prompt="Test goal",
                session_id="test-session",
                cwd="/tmp",
            )

        assert result["status"] == "dispatched"
        assert result["goal_id"].startswith("cg_")
        assert result["paseo_agent_id"] == "fake-agent-123"

        # Verify tracked
        goal = HermesACPAgent._codex_goals[result["goal_id"]]
        assert goal["status"] == "dispatched"
        assert goal["session_id"] == "test-session"

    def test_dispatch_returns_error_on_failure(self):
        """Test dispatch when subprocess fails."""
        from acp_adapter.server import HermesACPAgent

        mock_output = '{"status":"error","error":"codex provider not found"}'
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0,
            )

            result = HermesACPAgent.codex_goal_dispatch(
                goal_prompt="Test goal",
                session_id="test-session",
            )

        assert result["status"] == "dispatch_failed"
        assert "codex provider not found" in result.get("error", "")

    def test_dispatch_handles_timeout(self):
        """Test dispatch when subprocess times out."""
        from acp_adapter.server import HermesACPAgent
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="node", timeout=45)

            result = HermesACPAgent.codex_goal_dispatch(
                goal_prompt="Test goal",
                session_id="test-session",
            )

        assert result["status"] == "dispatch_failed"


class TestCodexGoalDeliver:
    """Test the deliver (wake) logic."""

    def setup_method(self):
        from acp_adapter.server import HermesACPAgent
        self._original_goals = HermesACPAgent._codex_goals.copy()
        HermesACPAgent._codex_goals.clear()

    def teardown_method(self):
        from acp_adapter.server import HermesACPAgent
        HermesACPAgent._codex_goals.clear()
        HermesACPAgent._codex_goals.update(self._original_goals)

    @pytest.mark.asyncio
    async def test_deliver_skips_missing_session(self):
        """Deliver should handle missing session gracefully."""
        from acp_adapter.server import HermesACPAgent

        HermesACPAgent._codex_goals["cg_deliver"] = {
            "status": "completed",
            "session_id": "nonexistent",
        }

        agent = HermesACPAgent.__new__(HermesACPAgent)
        agent.session_manager = MagicMock()
        agent.session_manager.get_session.return_value = None
        agent._conn = MagicMock()

        completion = {
            "goal_id": "cg_deliver",
            "agent_id": "test",
            "last_status": "closed",
            "title": "Test",
            "result_text": "",
            "git_summary": "",
            "session_id": "nonexistent",
            "cwd": "",
        }

        # Should not raise
        await agent._codex_goal_deliver(completion)

    @pytest.mark.asyncio
    async def test_deliver_wakes_idle_session(self):
        """Deliver should call self.prompt() when session is idle."""
        from acp_adapter.server import HermesACPAgent

        HermesACPAgent._codex_goals["cg_wake"] = {
            "status": "completed",
            "session_id": "active-session",
        }

        agent = HermesACPAgent.__new__(HermesACPAgent)
        agent.session_manager = MagicMock()
        mock_state = MagicMock()
        mock_state.is_running = False
        agent.session_manager.get_session.return_value = mock_state
        agent._conn = MagicMock()
        agent.prompt = MagicMock()

        # Make prompt a coroutine
        async def mock_prompt(**kwargs):
            pass
        agent.prompt = mock_prompt

        completion = {
            "goal_id": "cg_wake",
            "agent_id": "test",
            "last_status": "closed",
            "title": "Wake Test",
            "result_text": "Success!",
            "git_summary": "",
            "session_id": "active-session",
            "cwd": "",
        }

        await agent._codex_goal_deliver(completion)

        assert HermesACPAgent._codex_goals["cg_wake"]["status"] == "delivered"

    @pytest.mark.asyncio
    async def test_deliver_queued_when_session_busy(self):
        """Deliver should not prompt when session is running."""
        from acp_adapter.server import HermesACPAgent

        HermesACPAgent._codex_goals["cg_busy"] = {
            "status": "completed",
            "session_id": "busy-session",
        }

        agent = HermesACPAgent.__new__(HermesACPAgent)
        agent.session_manager = MagicMock()
        mock_state = MagicMock()
        mock_state.is_running = True  # Session is busy
        agent.session_manager.get_session.return_value = mock_state
        agent._conn = MagicMock()

        completion = {
            "goal_id": "cg_busy",
            "agent_id": "test",
            "last_status": "closed",
            "title": "Busy Test",
            "result_text": "",
            "git_summary": "",
            "session_id": "busy-session",
            "cwd": "",
        }

        await agent._codex_goal_deliver(completion)

        # Should be marked delivered without prompting
        assert HermesACPAgent._codex_goals["cg_busy"]["status"] == "delivered"
