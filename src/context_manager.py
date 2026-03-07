"""Dash: Conversation window management.

When conversations grow long, older messages are summarized via Claude Haiku
and replaced with a concise summary pair to keep the context window manageable.
"""

import anthropic

from src import config


# ---------------------------------------------------------------------------
# Compression prompt sent to Haiku for summarization
# ---------------------------------------------------------------------------

_COMPRESSION_PROMPT = (
    "Summarize this earlier conversation between a user and Dash (a task assistant). "
    "Focus on: what tasks were discussed, decisions made, and anything learned about the user. "
    "Keep it under 200 words."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_summarize(messages: list[dict[str, object]]) -> bool:
    """Return True if the message list has grown past the summary threshold.

    Counts ALL messages (user, assistant, tool_result).
    The threshold check is exclusive: exactly at threshold returns False.
    """
    return len(messages) > config.CONVERSATION_SUMMARY_THRESHOLD


def summarize_old_messages(
    messages: list[dict[str, object]],
    client: anthropic.Anthropic,
) -> list[dict[str, object]]:
    """Compress older messages into a summary, keeping recent messages intact.

    Strategy:
    1. Split into old (before the window) and recent (last CONVERSATION_WINDOW_SIZE).
    2. Format old messages into readable text.
    3. Send to Claude Haiku for compression.
    4. Return [summary_user_msg, summary_assistant_msg] + recent.

    Does NOT mutate the input list. Returns a new list.
    """
    window = config.CONVERSATION_WINDOW_SIZE

    # Split: old messages to summarize, recent messages to keep
    old = messages[: len(messages) - window]
    recent = list(messages[len(messages) - window :])

    # Format old messages into a text block for the summarizer
    formatted = _format_messages_for_summary(old)

    # Ask Haiku to compress
    response = client.messages.create(
        model=config.MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"{_COMPRESSION_PROMPT}\n\n---\n\n{formatted}",
            }
        ],
    )
    summary_text = response.content[0].text

    # Build the synthetic summary pair (maintains user/assistant alternation)
    summary_pair: list[dict[str, object]] = [
        {
            "role": "user",
            "content": f"[Previous conversation summary: {summary_text}]",
        },
        {
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        },
    ]

    return summary_pair + recent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_messages_for_summary(messages: list[dict[str, object]]) -> str:
    """Convert a list of message dicts into readable text for the summarizer.

    - User messages: extract text content.
    - Assistant messages: extract text, note tool calls by name.
    - Tool result messages (role=user with tool_result content): skip entirely.
    """
    lines: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        # Skip tool_result messages (they're implementation details)
        if _is_tool_result_message(content):
            continue

        text = _extract_text(content)
        tool_names = _extract_tool_names(content)

        if role == "user" and text:
            lines.append(f"User: {text}")
        elif role == "assistant":
            parts: list[str] = []
            if text:
                parts.append(text)
            if tool_names:
                parts.append(f"[called tools: {', '.join(tool_names)}]")
            if parts:
                lines.append(f"Assistant: {' '.join(parts)}")

    return "\n".join(lines)


def _is_tool_result_message(content: object) -> bool:
    """Check if the message content is a list of tool_result blocks."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _extract_text(content: object) -> str:
    """Extract text from message content (string or list of blocks)."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_val = block.get("text", "")
                if isinstance(text_val, str):
                    texts.append(text_val)
        return " ".join(texts)

    return ""


def _extract_tool_names(content: object) -> list[str]:
    """Extract tool names from tool_use blocks in message content."""
    if not isinstance(content, list):
        return []

    names: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name", "unknown")
            if isinstance(name, str):
                names.append(name)
    return names
