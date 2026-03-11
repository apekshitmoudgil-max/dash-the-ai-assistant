"""Dash: Main agent loop. Handles CLI input, builds context, runs tool-use loop."""

from __future__ import annotations

import json
try:
    import readline  # noqa: F401 -- extends input() buffer and adds line editing
except ImportError:
    pass  # readline is Unix-only; input() still works without it on Windows
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import anthropic

from src.config import (
    ANTHROPIC_API_KEY,
    LOG_DIR,
    MAX_TOOL_ITERATIONS,
    MODEL,
    SYSTEM_PROMPT_DYNAMIC_TEMPLATE,
    SYSTEM_PROMPT_STATIC,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.context_manager import should_summarize, summarize_old_messages
from src.memory import read_tasks, read_user_context
from src.tools import TOOL_DEFINITIONS, TOOL_DEFINITIONS_CACHED, execute_tool

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class StreamingMarkdownRenderer:
    """Converts **bold** markers to ANSI bold during streaming output."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_bold = False

    def feed(self, text: str) -> str:
        """Process a streaming chunk. Returns text ready to print."""
        output = ""
        self._buffer += text

        while "**" in self._buffer:
            idx = self._buffer.index("**")
            # Flush everything before the marker
            output += self._buffer[:idx]
            self._buffer = self._buffer[idx + 2:]
            # Toggle bold
            if self._in_bold:
                output += RESET + MAGENTA
                self._in_bold = False
            else:
                output += BOLD
                self._in_bold = True

        # Flush the buffer, but hold back a trailing "*" in case
        # the next chunk completes a "**" marker
        if self._buffer.endswith("*"):
            output += self._buffer[:-1]
            self._buffer = "*"
        else:
            output += self._buffer
            self._buffer = ""

        return output

    def flush(self) -> str:
        """Flush any remaining buffered text."""
        result = self._buffer
        self._buffer = ""
        if self._in_bold:
            result += RESET
            self._in_bold = False
        return result


# ---------------------------------------------------------------------------
# Session logger: writes JSONL to data/logs/
# ---------------------------------------------------------------------------

class SessionLogger:
    """Logs every event in a session to a JSONL file + readable conversation log."""

    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file: Path = LOG_DIR / f"session_{timestamp}.jsonl"
        self.convo_file: Path = LOG_DIR / f"session_{timestamp}.md"
        self.session_start: str = datetime.now().isoformat()
        self.turn_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self._log("session_start", {"model": MODEL, "timestamp": self.session_start})
        # Write conversation header
        local_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._convo(f"# Dash Session -- {local_time}\n\n")

    def _log(self, event_type: str, data: dict[str, object]) -> None:
        entry = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn": self.turn_count,
            **data,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _convo(self, text: str) -> None:
        """Append to the human-readable conversation log."""
        with open(self.convo_file, "a") as f:
            f.write(text)

    def log_user_input(self, text: str) -> None:
        self.turn_count += 1
        self._log("user_input", {"content": text})
        self._convo(f"**You:** {text}\n\n")

    def log_system_prompt(self, prompt: str) -> None:
        self._log("system_prompt", {"content": prompt})

    def log_llm_call(self, iteration: int) -> None:
        self._log("llm_call_start", {"iteration": iteration})

    def log_llm_response(
        self,
        iteration: int,
        content: list[dict[str, object]],
        stop_reason: str,
        usage: dict[str, int],
        latency_ms: int,
    ) -> None:
        self._log("llm_response", {
            "iteration": iteration,
            "content": content,
            "stop_reason": stop_reason,
            "usage": usage,
            "latency_ms": latency_ms,
        })

    def log_tool_call(self, tool_name: str, tool_input: dict[str, object]) -> None:
        self._log("tool_call", {"tool": tool_name, "input": tool_input})
        input_summary = json.dumps(tool_input, default=str)
        if len(input_summary) > 100:
            input_summary = input_summary[:100] + "..."
        self._convo(f"> *Tool: `{tool_name}` → {input_summary}*\n")

    def log_tool_result(self, tool_name: str, result: str) -> None:
        self._log("tool_result", {"tool": tool_name, "result": result})
        result_preview = result if len(result) <= 100 else result[:100] + "..."
        self._convo(f"> *Result: {result_preview}*\n\n")

    def log_assistant_response(self, text: str) -> None:
        self._log("assistant_response", {"content": text})
        self._convo(f"**Dash:** {text}\n\n---\n\n")

    def log_error(self, error: str) -> None:
        self._log("error", {"message": error})

    def log_turn_summary(
        self,
        user_input: str,
        response: str,
        tool_calls: list[str],
        iterations: int,
        total_input_tokens: int,
        total_output_tokens: int,
        total_latency_ms: int,
    ) -> None:
        cost_input = (total_input_tokens / 1_000_000) * 1.00  # Haiku input: $1/1M
        cost_output = (total_output_tokens / 1_000_000) * 5.00  # Haiku output: $5/1M
        cost = round(cost_input + cost_output, 6)
        self._total_input_tokens += total_input_tokens
        self._total_output_tokens += total_output_tokens
        self._total_cost += cost
        self._log("turn_summary", {
            "user_input": user_input,
            "response_preview": response[:200],
            "tool_calls": tool_calls,
            "iterations": iterations,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_latency_ms": total_latency_ms,
            "cost_estimate": cost,
        })

    def log_session_end(self) -> None:
        self._log("session_end", {"total_turns": self.turn_count})
        self._convo(
            f"---\n\n"
            f"**Session stats:** {self.turn_count} turns | "
            f"{self._total_input_tokens:,} input tokens | "
            f"{self._total_output_tokens:,} output tokens | "
            f"${self._total_cost:.4f} estimated cost\n"
        )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Build the system prompt as a single string (for logging and tests)."""
    _, dynamic = build_system_prompt_parts()
    return SYSTEM_PROMPT_STATIC + dynamic


def _build_dynamic_context() -> str:
    """Build the dynamic portion of the system prompt (changes every turn)."""
    context = read_user_context()
    if context.get("preferences") or context.get("priorities") or context.get("observations"):
        user_context_str = json.dumps(context, indent=2, default=str)
    else:
        user_context_str = "No information about the user yet. Pay attention to what they tell you and store observations."

    # Extract inferred patterns for dedicated prompt section
    patterns = context.get("inferred_patterns", [])
    if patterns:
        pattern_lines = [
            f"- [{p.get('confidence', '?').upper()}] ({p.get('category', '?')}) {p.get('pattern', '')}"
            for p in patterns
        ]
        inferred_patterns_str = "\n".join(pattern_lines)
    else:
        inferred_patterns_str = "No patterns inferred yet. These emerge after a few sessions."

    tasks = read_tasks()
    active_tasks = [t for t in tasks if t.get("status") == "active"]
    if active_tasks:
        task_lines: list[str] = []
        for t in active_tasks:
            line = f"- [{t.get('priority', 'medium').upper()}] {t.get('title')}"
            if t.get("due_date"):
                line += f" (due: {t.get('due_date')})"
            if t.get("context"):
                line += f" -- {t.get('context')}"
            task_lines.append(line)
        active_tasks_str = "\n".join(task_lines)
    else:
        active_tasks_str = "No active tasks yet."

    from src.session_memory import format_summaries_for_prompt, load_recent_summaries

    summaries = load_recent_summaries(n=5)
    recent_sessions_str = format_summaries_for_prompt(summaries)

    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    return SYSTEM_PROMPT_DYNAMIC_TEMPLATE.format(
        current_datetime=current_datetime,
        user_context=user_context_str,
        inferred_patterns=inferred_patterns_str,
        active_tasks=active_tasks_str,
        recent_sessions=recent_sessions_str,
    )


def build_system_prompt_parts() -> tuple[str, str]:
    """Return (static, dynamic) parts of the system prompt."""
    return SYSTEM_PROMPT_STATIC, _build_dynamic_context()


def build_system_prompt_cached() -> list[dict]:
    """Build system prompt as a list of blocks with cache_control for prompt caching.

    Haiku 4.5 requires a minimum of 4,096 tokens for caching to activate.
    Tools (~700 tokens) + static prompt (~600 tokens) alone don't reach this
    threshold. So we cache the FULL system prompt (static + dynamic combined).

    This means the cache is valid within a single turn's tool-use loop
    (where the system prompt is identical across iterations), but invalidates
    between turns (because datetime and possibly tasks/observations change).

    For a typical turn with 2 API calls (tool call + response), this halves
    the input token cost for the system prompt on the second call.
    """
    static, dynamic = build_system_prompt_parts()
    return [
        {"type": "text", "text": static + dynamic, "cache_control": {"type": "ephemeral"}},
    ]


# ---------------------------------------------------------------------------
# Agent turn
# ---------------------------------------------------------------------------

def run_agent_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, object]],
    system_prompt: str | list[dict],
    logger: SessionLogger,
    user_input: str,
) -> str:
    """Run one full agent turn: call Claude, execute tools in a loop, return final text.

    system_prompt can be a plain string or a list of content blocks (for prompt caching).
    """
    final_text = ""
    all_tool_calls: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0
    md_renderer = StreamingMarkdownRenderer()

    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.log_llm_call(iteration)
        call_start = time.monotonic()

        collected_content: list[dict[str, object]] = []
        current_text = ""
        current_tool_use_id = ""
        current_tool_name = ""
        current_tool_input_json = ""

        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=TOOL_DEFINITIONS_CACHED,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "text":
                        current_text = ""
                    elif event.content_block.type == "tool_use":
                        current_tool_use_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        current_tool_input_json = ""

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        rendered = md_renderer.feed(event.delta.text)
                        if rendered:
                            print(f"{MAGENTA}{rendered}{RESET}", end="", flush=True)
                        current_text += event.delta.text
                    elif event.delta.type == "input_json_delta":
                        current_tool_input_json += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_text:
                        remaining = md_renderer.flush()
                        if remaining:
                            print(f"{MAGENTA}{remaining}{RESET}", end="", flush=True)
                        collected_content.append({
                            "type": "text",
                            "text": current_text,
                        })
                        current_text = ""
                    if current_tool_name:
                        tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                        collected_content.append({
                            "type": "tool_use",
                            "id": current_tool_use_id,
                            "name": current_tool_name,
                            "input": tool_input,
                        })
                        current_tool_name = ""
                        current_tool_input_json = ""

            final_message = stream.get_final_message()
            stop_reason = final_message.stop_reason

        call_latency = int((time.monotonic() - call_start) * 1000)
        total_latency_ms += call_latency

        # Extract usage from the final message (including cache stats if available)
        cache_creation = getattr(final_message.usage, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(final_message.usage, 'cache_read_input_tokens', 0) or 0
        usage = {
            "input_tokens": final_message.usage.input_tokens,
            "output_tokens": final_message.usage.output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        }
        total_input_tokens += final_message.usage.input_tokens
        total_output_tokens += final_message.usage.output_tokens

        logger.log_llm_response(iteration, collected_content, stop_reason, usage, call_latency)

        # Check if there are tool calls to execute
        tool_calls = [c for c in collected_content if c.get("type") == "tool_use"]

        if not tool_calls:
            text_parts = [c["text"] for c in collected_content if c.get("type") == "text"]
            final_text = "".join(str(t) for t in text_parts)
            if final_text:
                print()
            messages.append({"role": "assistant", "content": collected_content})
            logger.log_assistant_response(final_text)
            break

        # Execute tool calls
        messages.append({"role": "assistant", "content": collected_content})

        tool_results: list[dict[str, object]] = []
        for tool_call in tool_calls:
            tool_name = str(tool_call["name"])
            tool_input = tool_call.get("input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}

            logger.log_tool_call(tool_name, tool_input)
            result = execute_tool(tool_name, tool_input)
            logger.log_tool_result(tool_name, result)
            all_tool_calls.append(tool_name)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

        if stop_reason == "end_turn":
            text_parts = [c["text"] for c in collected_content if c.get("type") == "text"]
            final_text = "".join(str(t) for t in text_parts)
            if final_text:
                print()
            break
    else:
        print("\n[Dash hit the tool iteration limit. Responding with what I have.]")
        logger.log_error("Hit max tool iterations")

    # Log turn summary with metrics
    logger.log_turn_summary(
        user_input=user_input,
        response=final_text,
        tool_calls=all_tool_calls,
        iterations=iteration + 1,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
    )

    return final_text


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main() -> None:
    """Main REPL loop for Dash."""
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set. Set it in your environment or .env file.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logger = SessionLogger()

    M, B, D, R = MAGENTA, BOLD, DIM, RESET
    W = "\033[97m"   # bright white (head/body)
    GR = "\033[90m"  # dark gray (face panel)
    CY = "\033[96m"  # bright cyan (eyes)
    print()
    print(f"  {W}▄█████▄{R}")
    print(f"  {W}█{GR}█{CY}█{GR}█{CY}█{GR}█{W}█{R}       {B}{M}Dash{R} {D}v0.4{R}")
    print(f"  {W}▀▀███▀▀{R}       {D}Haiku 4.5 · CLI{R}")
    print(f"    {W}█ █{R}")
    print()

    # Morning briefing on startup
    from src.proactive import build_briefing_prompt, StreamingMarkdownRenderer
    from src.memory import read_tasks, read_user_context
    from src.session_memory import load_recent_summaries, format_summaries_for_prompt

    tasks = read_tasks()
    user_context = read_user_context()
    session_summaries = load_recent_summaries(n=5)

    active_tasks = [t for t in tasks if t.get("status") == "active"]
    observations = user_context.get("observations", [])
    patterns = user_context.get("inferred_patterns", [])

    # Log what was loaded at session start
    loaded_info = {
        "active_tasks": len(active_tasks),
        "observations": len(observations),
        "inferred_patterns": len(patterns),
        "recent_sessions": len(session_summaries),
        "task_titles": [t.get("title", "") for t in active_tasks],
    }
    logger._log("session_loaded", loaded_info)
    logger._convo(
        f"*Session loaded: {len(active_tasks)} active tasks, "
        f"{len(observations)} observations, {len(patterns)} patterns, "
        f"{len(session_summaries)} recent sessions*\n\n"
    )

    print()

    briefing_prompt = build_briefing_prompt(tasks, user_context, session_summaries)

    md_renderer = StreamingMarkdownRenderer()
    briefing_text_parts: list[str] = []
    print(f"{M}{B}Dash:{R} ", end="", flush=True)

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": briefing_prompt}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        briefing_text_parts.append(event.delta.text)
                        rendered = md_renderer.feed(event.delta.text)
                        if rendered:
                            print(f"{M}{rendered}{R}", end="", flush=True)
                elif event.type == "content_block_stop":
                    remaining = md_renderer.flush()
                    if remaining:
                        print(f"{M}{remaining}{R}", end="", flush=True)
        print("\n")

        # Log the briefing to both log files
        briefing_full_text = "".join(briefing_text_parts)
        logger._log("morning_briefing", {"content": briefing_full_text})
        logger._convo(f"**Dash (briefing):** {briefing_full_text}\n\n---\n\n")
    except Exception as e:
        print(f"\n  {D}[Briefing unavailable: {e}]{R}\n")

    print(f"  {D}Type 'quit' or 'exit' when done{R}\n")

    messages: list[dict[str, object]] = []

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}You:{RESET}{CYAN} ").strip()
            print(RESET, end="", flush=True)
        except (KeyboardInterrupt, EOFError):
            print(f"{RESET}\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        system_prompt_cached = build_system_prompt_cached()
        system_prompt_flat = build_system_prompt()  # for logging
        logger.log_user_input(user_input)
        logger.log_system_prompt(system_prompt_flat)

        messages.append({"role": "user", "content": user_input})

        # Check if conversation needs trimming
        if should_summarize(messages):
            print("[Compacting conversation history...]")
            try:
                messages = summarize_old_messages(messages, client)
            except Exception as e:
                logger.log_error(f"Summarization failed: {e}")

        print(f"\n{MAGENTA}{BOLD}Dash:{RESET} ", end="", flush=True)

        try:
            run_agent_turn(client, messages, system_prompt_cached, logger, user_input)
        except anthropic.APIError as e:
            logger.log_error(f"API Error: {e}")
            print(f"\n[API Error: {e}]")
            messages.pop()
        except Exception as e:
            logger.log_error(f"Error: {e}")
            print(f"\n[Error: {e}]")
            messages.pop()

        print()

    logger.log_session_end()

    if logger.turn_count > 0:
        print("\nSummarizing session...")
        try:
            from src.session_memory import generate_session_summary, save_session_summary

            summary = generate_session_summary(logger.convo_file, client)
            save_session_summary(summary)

            # Log to JSONL
            logger._log("session_summary", summary)

            # Log to markdown
            logger._convo(f"\n---\n\n**Session summary:**\n")
            logger._convo(f"- Summary: {summary.get('summary', 'N/A')}\n")
            logger._convo(f"- Mood: {summary.get('mood', 'N/A')}\n")
            if summary.get("tasks_changed"):
                logger._convo(f"- Tasks changed: {', '.join(str(t) for t in summary['tasks_changed'])}\n")
            if summary.get("observations_added"):
                logger._convo(f"- Observations added: {', '.join(str(o) for o in summary['observations_added'])}\n")

            print("Session summary saved.")
        except Exception as e:
            print(f"[Could not save summary: {e}]")

        print("Synthesizing patterns...")
        try:
            from src.pattern_synthesis import synthesize_patterns

            patterns = synthesize_patterns(client)

            # Log to JSONL
            logger._log("pattern_synthesis", {"patterns": patterns, "count": len(patterns)})

            # Log to markdown
            logger._convo(f"\n**Patterns updated ({len(patterns)} total):**\n")
            for p in patterns:
                conf = p.get("confidence", "?")
                cat = p.get("category", "?")
                text = p.get("pattern", "")
                logger._convo(f"- [{conf.upper()}] ({cat}) {text}\n")

            print(f"Patterns updated ({len(patterns)} patterns).")
        except Exception as e:
            print(f"[Pattern synthesis skipped: {e}]")

    print(f"\nSession saved:")
    print(f"  Conversation: {logger.convo_file}")
    print(f"  Full log:     {logger.log_file}")


if __name__ == "__main__":
    main()
