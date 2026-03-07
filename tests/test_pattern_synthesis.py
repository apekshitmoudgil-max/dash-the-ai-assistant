"""Tests for pattern synthesis."""

import json
from pathlib import Path

import pytest

from src.pattern_synthesis import (
    MAX_PATTERNS,
    build_synthesis_input,
    synthesize_patterns,
)


# ---------------------------------------------------------------------------
# Fake Anthropic client for mocking
# ---------------------------------------------------------------------------

class FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeContentBlock(text)]


class FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def create(self, **kwargs):
        return FakeResponse(self._response_text)


class FakeClient:
    def __init__(self, response_text: str):
        self.messages = FakeMessages(response_text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create temp data dir and patch config paths."""
    context_file = tmp_path / "user_context.json"
    summaries_file = tmp_path / "session_summaries.json"

    monkeypatch.setattr("src.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.config.USER_CONTEXT_FILE", context_file)
    monkeypatch.setattr("src.config.SESSION_SUMMARIES_FILE", summaries_file)

    return tmp_path


SAMPLE_PATTERNS = [
    {
        "category": "work_habits",
        "pattern": "Prefers to start tasks early in the day.",
        "confidence": "high",
        "first_noticed": "2026-03-01",
        "last_confirmed": "2026-03-07",
    },
    {
        "category": "motivation",
        "pattern": "Learns by building, not reading.",
        "confidence": "medium",
        "first_noticed": "2026-03-01",
        "last_confirmed": "2026-03-07",
    },
]


# ---------------------------------------------------------------------------
# Tests: build_synthesis_input
# ---------------------------------------------------------------------------

class TestBuildSynthesisInput:
    def test_includes_observations(self):
        context = {
            "observations": [
                {"date": "2026-03-07", "observation": "Likes AI"},
            ],
        }
        result = build_synthesis_input(context, [])
        assert "Likes AI" in result
        assert "2026-03-07" in result

    def test_includes_summaries(self):
        summaries = [
            {"date": "2026-03-07", "summary": "Built Dash v0.3", "mood": "focused"},
        ]
        result = build_synthesis_input({}, summaries)
        assert "Built Dash v0.3" in result
        assert "focused" in result

    def test_includes_existing_patterns(self):
        context = {"inferred_patterns": SAMPLE_PATTERNS}
        result = build_synthesis_input(context, [])
        assert "work_habits" in result
        assert "Prefers to start tasks early" in result

    def test_handles_empty_observations(self):
        result = build_synthesis_input({"observations": []}, [])
        assert "No observations yet" in result

    def test_handles_empty_summaries(self):
        result = build_synthesis_input({}, [])
        assert "No session summaries yet" in result

    def test_handles_no_existing_patterns(self):
        result = build_synthesis_input({}, [])
        assert "None yet" in result


# ---------------------------------------------------------------------------
# Tests: synthesize_patterns
# ---------------------------------------------------------------------------

class TestSynthesizePatterns:
    def test_returns_patterns_from_response(self, data_dir):
        response_json = json.dumps(SAMPLE_PATTERNS)
        client = FakeClient(response_json)
        context = {
            "observations": [
                {"date": "2026-03-07", "observation": "Likes AI"},
            ],
        }

        result = synthesize_patterns(client, user_context=context, summaries=[])
        assert len(result) == 2
        assert result[0]["category"] == "work_habits"

    def test_writes_patterns_to_user_context(self, data_dir):
        response_json = json.dumps(SAMPLE_PATTERNS)
        client = FakeClient(response_json)
        context_file = data_dir / "user_context.json"
        context = {
            "preferences": {"name": "Test"},
            "observations": [
                {"date": "2026-03-07", "observation": "test"},
            ],
        }
        context_file.write_text(json.dumps(context))

        synthesize_patterns(client, user_context=context, summaries=[])

        saved = json.loads(context_file.read_text())
        assert "inferred_patterns" in saved
        assert len(saved["inferred_patterns"]) == 2

    def test_preserves_other_context_fields(self, data_dir):
        response_json = json.dumps(SAMPLE_PATTERNS)
        client = FakeClient(response_json)
        context = {
            "preferences": {"name": "Apekshit"},
            "priorities": {"current_focus": "Ship v0.4"},
            "observations": [
                {"date": "2026-03-07", "observation": "test"},
            ],
        }

        synthesize_patterns(client, user_context=context, summaries=[])

        # Context dict should still have preferences and priorities
        assert context["preferences"]["name"] == "Apekshit"
        assert context["priorities"]["current_focus"] == "Ship v0.4"

    def test_caps_at_max_patterns(self, data_dir):
        many_patterns = [
            {"category": f"cat_{i}", "pattern": f"Pattern {i}", "confidence": "low",
             "first_noticed": "2026-03-07", "last_confirmed": "2026-03-07"}
            for i in range(15)
        ]
        client = FakeClient(json.dumps(many_patterns))
        context = {
            "observations": [
                {"date": "2026-03-07", "observation": "test"},
            ],
        }

        result = synthesize_patterns(client, user_context=context, summaries=[])
        assert len(result) == MAX_PATTERNS

    def test_handles_bad_json(self, data_dir):
        client = FakeClient("This is not JSON at all")
        existing = [{"category": "test", "pattern": "Keep me", "confidence": "high",
                      "first_noticed": "2026-03-01", "last_confirmed": "2026-03-07"}]
        context = {
            "observations": [
                {"date": "2026-03-07", "observation": "test"},
            ],
            "inferred_patterns": existing,
        }

        result = synthesize_patterns(client, user_context=context, summaries=[])
        assert len(result) == 1
        assert result[0]["pattern"] == "Keep me"

    def test_strips_markdown_fences(self, data_dir):
        fenced = "```json\n" + json.dumps(SAMPLE_PATTERNS) + "\n```"
        client = FakeClient(fenced)
        context = {
            "observations": [
                {"date": "2026-03-07", "observation": "test"},
            ],
        }

        result = synthesize_patterns(client, user_context=context, summaries=[])
        assert len(result) == 2

    def test_skips_when_no_data(self, data_dir):
        client = FakeClient("should not be called")
        context = {"observations": [], "inferred_patterns": SAMPLE_PATTERNS}

        result = synthesize_patterns(client, user_context=context, summaries=[])
        # Should return existing patterns without calling API
        assert result == SAMPLE_PATTERNS

    def test_handles_non_list_response(self, data_dir):
        client = FakeClient('{"not": "a list"}')
        context = {
            "observations": [
                {"date": "2026-03-07", "observation": "test"},
            ],
        }

        result = synthesize_patterns(client, user_context=context, summaries=[])
        assert result == []
