"""Tests for src/context_manager.py: conversation window management."""

import copy

import pytest

from src.context_manager import should_summarize, summarize_old_messages


# ---------------------------------------------------------------------------
# Fake Anthropic client (same pattern as test_session_memory.py)
# ---------------------------------------------------------------------------


class FakeMessage:
    """Mimics an Anthropic API response with a single text block."""

    def __init__(self, text: str) -> None:
        self.content = [type("Block", (), {"type": "text", "text": text})()]


class FakeMessages:
    @staticmethod
    def create(**kwargs: object) -> FakeMessage:
        return FakeMessage("User discussed tasks and preferences.")


class FakeClient:
    messages = FakeMessages()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(count: int) -> list[dict[str, object]]:
    """Generate a list of alternating user/assistant messages."""
    messages: list[dict[str, object]] = []
    for i in range(count):
        if i % 2 == 0:
            messages.append({"role": "user", "content": f"User message {i}"})
        else:
            messages.append({"role": "assistant", "content": f"Assistant message {i}"})
    return messages


# ---------------------------------------------------------------------------
# TestShouldSummarize
# ---------------------------------------------------------------------------


class TestShouldSummarize:
    """Tests for should_summarize()."""

    def test_false_when_under_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)
        messages = _make_messages(10)
        assert should_summarize(messages) is False

    def test_true_when_over_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)
        messages = _make_messages(30)
        assert should_summarize(messages) is True

    def test_false_when_exactly_at_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)
        messages = _make_messages(24)
        assert should_summarize(messages) is False

    def test_empty_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)
        assert should_summarize([]) is False


# ---------------------------------------------------------------------------
# TestSummarizeOldMessages
# ---------------------------------------------------------------------------


class TestSummarizeOldMessages:
    """Tests for summarize_old_messages()."""

    def test_preserves_recent_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_WINDOW_SIZE", 20)
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)

        messages = _make_messages(30)
        recent_20 = messages[-20:]

        result = summarize_old_messages(messages, FakeClient())

        # The last 20 entries in result should match the last 20 from input
        assert result[2:] == recent_20

    def test_replaces_old_with_summary_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_WINDOW_SIZE", 20)
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)

        messages = _make_messages(30)

        result = summarize_old_messages(messages, FakeClient())

        # Total: 2 (summary pair) + 20 (recent) = 22
        assert len(result) == 22
        # First two are the summary pair
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_summary_user_message_has_context_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_WINDOW_SIZE", 20)
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)

        messages = _make_messages(30)
        result = summarize_old_messages(messages, FakeClient())

        user_msg_content = result[0]["content"]
        assert isinstance(user_msg_content, str)
        assert "[Previous conversation summary:" in user_msg_content

    def test_summary_assistant_acknowledges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_WINDOW_SIZE", 20)
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)

        messages = _make_messages(30)
        result = summarize_old_messages(messages, FakeClient())

        assistant_msg_content = result[1]["content"]
        assert isinstance(assistant_msg_content, str)
        assert "Got it" in assistant_msg_content

    def test_does_not_mutate_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.CONVERSATION_WINDOW_SIZE", 20)
        monkeypatch.setattr("src.config.CONVERSATION_SUMMARY_THRESHOLD", 24)

        messages = _make_messages(30)
        original = copy.deepcopy(messages)

        summarize_old_messages(messages, FakeClient())

        assert messages == original
