"""Tests for src/proactive.py: morning briefing prompt builder."""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.proactive import build_briefing_prompt


class TestBuildBriefingPrompt:
    """Tests for the build_briefing_prompt pure function."""

    def test_includes_current_date(self) -> None:
        """The prompt should contain today's date in human-readable format."""
        result = build_briefing_prompt(
            tasks=[],
            user_context={},
            session_summaries=[],
        )
        today = datetime.now()
        # The function formats as: "Monday, March 07, 2026 at 10:30 AM"
        expected_date_part = today.strftime("%A, %B %d, %Y")
        assert expected_date_part in result

    def test_includes_active_tasks(self) -> None:
        """Active tasks should appear in the prompt with their details."""
        tasks = [
            {
                "title": "Write unit tests",
                "status": "active",
                "priority": "high",
                "due_date": "2026-03-10",
                "context": "Dash v0.3 test coverage",
            },
            {
                "title": "Deploy to production",
                "status": "active",
                "priority": "medium",
                "due_date": None,
                "context": None,
            },
        ]
        result = build_briefing_prompt(
            tasks=tasks,
            user_context={},
            session_summaries=[],
        )
        assert "Write unit tests" in result
        assert "HIGH" in result
        assert "due: 2026-03-10" in result
        assert "Dash v0.3 test coverage" in result
        assert "Deploy to production" in result
        assert "MEDIUM" in result

    def test_includes_user_context(self) -> None:
        """User context with preferences should appear in the prompt."""
        user_context = {
            "preferences": {"communication_style": "direct"},
            "priorities": {"current_focus": "building AI agents"},
            "observations": [
                {"observation": "Works best in the morning", "date": "2026-03-01"}
            ],
        }
        result = build_briefing_prompt(
            tasks=[],
            user_context=user_context,
            session_summaries=[],
        )
        assert "direct" in result
        assert "building AI agents" in result
        assert "Works best in the morning" in result

    def test_includes_session_summaries(self) -> None:
        """Session summaries should appear in the prompt output."""
        summaries = [
            {
                "date": "2026-03-06",
                "summary": "Worked on web search integration.",
                "tasks_changed": ["web_search (added)"],
                "observations_added": [],
                "mood": "focused",
            },
        ]
        result = build_briefing_prompt(
            tasks=[],
            user_context={},
            session_summaries=summaries,
        )
        assert "Worked on web search integration." in result
        assert "2026-03-06" in result

    def test_handles_empty_state(self) -> None:
        """Empty inputs should produce a valid prompt with graceful fallbacks."""
        result = build_briefing_prompt(
            tasks=[],
            user_context={},
            session_summaries=[],
        )
        # Should not raise any errors
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain the "no tasks" fallback
        assert "No tasks yet." in result
        # Should contain the "no context" fallback
        assert "No user context stored yet." in result
        # Should contain the "no sessions" fallback
        assert "No previous sessions yet." in result
        # Should still contain the briefing instructions
        assert "morning briefing" in result
