"""Tests for src/proactive.py: morning briefing prompt builder."""

import datetime as dt
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


class TestBuildBriefingPromptAdaptive:
    """Tests for v0.4 adaptive briefing features."""

    def test_includes_time_context_in_instructions(self):
        result = build_briefing_prompt([], {}, [])
        assert "Time context:" in result

    def test_weekend_detection(self):
        # March 8, 2026 is a Sunday
        mock_now = dt.datetime(2026, 3, 8, 10, 0, 0)
        with patch("src.proactive.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            result = build_briefing_prompt([], {}, [])
            assert "weekend" in result.lower()

    def test_weeknight_evening_detection(self):
        # March 9, 2026 is a Monday, 9 PM
        mock_now = dt.datetime(2026, 3, 9, 21, 0, 0)
        with patch("src.proactive.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            result = build_briefing_prompt([], {}, [])
            assert "weeknight" in result.lower() or "evening" in result.lower()

    def test_weekday_daytime_detection(self):
        # March 9, 2026 is a Monday, 10 AM
        mock_now = dt.datetime(2026, 3, 9, 10, 0, 0)
        with patch("src.proactive.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            result = build_briefing_prompt([], {}, [])
            assert "weekday" in result.lower() or "daytime" in result.lower()

    def test_three_sections_present(self):
        result = build_briefing_prompt([], {}, [])
        assert "Top Priority" in result
        assert "Also On Your Plate" in result
        assert "One Suggestion" in result

    def test_no_quick_checkin_section(self):
        result = build_briefing_prompt([], {}, [])
        assert "Quick Check-in" not in result

    def test_no_pattern_insight_section(self):
        result = build_briefing_prompt([], {}, [])
        assert "Pattern Insight" not in result

    def test_no_mood_label_surfaced(self):
        result = build_briefing_prompt([], {}, [])
        assert "Recent mood:" not in result

    def test_last_session_referenced(self):
        summaries = [
            {"date": "2026-03-07", "summary": "Built the search memory feature", "mood": "focused"},
        ]
        result = build_briefing_prompt([], {}, summaries)
        assert "Built the search memory feature" in result

    def test_emotional_continuity_instructions(self):
        result = build_briefing_prompt([], {}, [])
        assert "pick up the thread" in result.lower() or "last session" in result.lower()

    def test_warm_personality_in_prompt(self):
        result = build_briefing_prompt([], {}, [])
        assert "remember" in result.lower() or "friend" in result.lower()
