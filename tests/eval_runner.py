"""Dash Eval Runner

Runs multi-turn sessions against real Haiku and collects transcripts with code-based scoring.
No mocks. Real API calls. Real tool execution. Isolated data directory.

This is the eval infrastructure. You provide the golden dataset (eval_sessions.json),
and this runner executes each session, scores tool accuracy, and generates a report.

See tests/EVAL_GUIDE.md for how to create your own eval_sessions.json.

Usage:
    python3 -m tests.eval_runner

Output:
    tests/eval_results/run_YYYY-MM-DD_HH-MM-SS.json       (full transcript)
    tests/eval_results/run_YYYY-MM-DD_HH-MM-SS_report.md  (readable report)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

# Patch config BEFORE importing anything that uses it
import src.config as config
from src.agent import build_system_prompt
from src.tools import TOOL_DEFINITIONS, execute_tool


EVAL_SESSIONS_FILE = Path(__file__).parent / "eval_sessions.json"
RESULTS_DIR = Path(__file__).parent / "eval_results"


# ---------------------------------------------------------------------------
# Data isolation: redirect all config paths to eval directory
# ---------------------------------------------------------------------------

def patch_data_dir(data_dir: Path) -> None:
    """Redirect all config paths to an isolated eval data directory."""
    config.DATA_DIR = data_dir
    config.TASKS_FILE = data_dir / "tasks.json"
    config.USER_CONTEXT_FILE = data_dir / "user_context.json"
    config.SESSION_SUMMARIES_FILE = data_dir / "session_summaries.json"
    config.OBSERVATIONS_ARCHIVE_FILE = data_dir / "observations_archive.json"
    config.LOG_DIR = data_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Non-streaming agent turn (for eval, no terminal output)
# ---------------------------------------------------------------------------

def run_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
    user_input: str,
) -> dict[str, Any]:
    """Run one agent turn using client.messages.create() (non-streaming).

    Modifies messages in place (appends user input, assistant responses, tool results).
    Returns a transcript entry with tools called, response text, and token usage.
    """
    # Rebuild system prompt each turn (tasks/context may have changed)
    system_prompt = build_system_prompt()
    messages.append({"role": "user", "content": user_input})

    tools_called: list[str] = []
    tool_details: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    final_text = ""
    start_time = time.monotonic()

    for _iteration in range(config.MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Convert ContentBlock objects to dicts for message history
        content_dicts: list[dict[str, Any]] = []
        tool_blocks: list[Any] = []
        text_parts: list[str] = []

        for block in response.content:
            if block.type == "text":
                content_dicts.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                content_dicts.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_blocks.append(block)

        messages.append({"role": "assistant", "content": content_dicts})

        # No tool calls: we're done
        if not tool_blocks:
            final_text = "".join(text_parts)
            break

        # Execute each tool call
        tool_results: list[dict[str, Any]] = []
        for block in tool_blocks:
            tools_called.append(block.name)
            result = execute_tool(block.name, block.input)
            tool_details.append({
                "tool": block.name,
                "input": block.input,
                "output": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

        # If stop_reason is end_turn after tool execution, capture any text
        if response.stop_reason == "end_turn":
            final_text = "".join(text_parts)
            break
    else:
        final_text = "[Hit max tool iterations]"

    latency_ms = int((time.monotonic() - start_time) * 1000)

    return {
        "user_input": user_input,
        "tools_called": tools_called,
        "tool_details": tool_details,
        "response_text": final_text,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Code-based tool accuracy scoring
# ---------------------------------------------------------------------------

def score_turn(turn_result: dict[str, Any], expectations: dict[str, Any]) -> dict[str, Any]:
    """Compute code-based tool accuracy for a turn.

    Three checks:
    - must_call_all: every tool in this list must appear in tools_called
    - must_call_any: at least one tool from this list must appear
    - must_not_call: none of these tools should appear
    """
    tools_called = set(turn_result["tools_called"])

    must_call_all = set(expectations.get("must_call_all", []))
    must_call_any = set(expectations.get("must_call_any", []))
    must_not_call = set(expectations.get("must_not_call", []))

    all_present = must_call_all.issubset(tools_called)
    any_present = (not must_call_any) or bool(must_call_any & tools_called)
    none_forbidden = not bool(must_not_call & tools_called)

    checks = [all_present, any_present, none_forbidden]
    tool_accuracy = sum(checks) / len(checks) if checks else 1.0

    return {
        "tool_accuracy": round(tool_accuracy, 2),
        "must_call_all_passed": all_present,
        "must_call_any_passed": any_present,
        "must_not_call_passed": none_forbidden,
        "details": {
            "expected_all": sorted(must_call_all),
            "expected_any": sorted(must_call_any),
            "forbidden": sorted(must_not_call),
            "actually_called": sorted(tools_called),
            "missing_required": sorted(must_call_all - tools_called),
            "missing_any_of": sorted(must_call_any - tools_called) if not any_present else [],
            "forbidden_called": sorted(must_not_call & tools_called),
        },
    }


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_session(
    client: anthropic.Anthropic,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Run one full session against real Haiku. Returns session transcript with scores."""
    messages: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []

    for i, turn_spec in enumerate(session["turns"]):
        user_input = turn_spec["user_input"]
        print(f"  Turn {i + 1}/{len(session['turns'])}: {user_input[:60]}...")

        try:
            turn_result = run_turn(client, messages, user_input)
        except Exception as e:
            turn_result = {
                "user_input": user_input,
                "tools_called": [],
                "tool_details": [],
                "response_text": f"[ERROR: {e}]",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
            }

        turn_result["turn"] = i + 1
        turn_result["expectations"] = turn_spec["expectations"]
        turn_result["score"] = score_turn(turn_result, turn_spec["expectations"])
        turns.append(turn_result)

        # Brief console output
        score = turn_result["score"]
        status = "PASS" if score["tool_accuracy"] == 1.0 else "FAIL"
        tools = ", ".join(turn_result["tools_called"]) or "(none)"
        print(f"    [{status}] Tools: {tools} | {turn_result['latency_ms']}ms")

    # Capture data state after session
    from src.memory import read_tasks, read_user_context

    data_state = {
        "tasks": read_tasks(),
        "user_context": read_user_context(),
    }

    return {
        "session_id": session["id"],
        "description": session["description"],
        "turns": turns,
        "data_state_after": data_state,
    }


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def generate_report(results: dict[str, Any]) -> str:
    """Generate a readable markdown report from eval results."""
    lines: list[str] = []
    lines.append("# Dash Eval Report\n")
    lines.append(f"**Run:** {results['run_id']}")
    lines.append(f"**Model:** {results['model']}")
    lines.append(f"**Timestamp:** {results['timestamp']}")
    lines.append("")

    total_turns = 0
    total_passed = 0
    total_cost = 0.0

    for session in results["sessions"]:
        lines.append("---")
        lines.append(f"## {session['session_id']}")
        lines.append(f"*{session['description']}*\n")

        session_cost = 0.0

        for turn in session["turns"]:
            score = turn.get("score", {})
            accuracy = score.get("tool_accuracy", 0)
            status = "PASS" if accuracy == 1.0 else "WARN" if accuracy >= 0.67 else "FAIL"

            tools = ", ".join(turn["tools_called"]) or "(none)"
            cost = (
                turn["input_tokens"] / 1_000_000 * 0.80
                + turn["output_tokens"] / 1_000_000 * 4.00
            )
            session_cost += cost
            total_turns += 1
            if accuracy == 1.0:
                total_passed += 1

            lines.append(f"### Turn {turn['turn']} [{status}]")
            lines.append(f"**User:** {turn['user_input']}")
            lines.append(f"**Tools called:** {tools}")
            lines.append(f"**Tool accuracy:** {accuracy}")

            # Show tool details
            if turn["tool_details"]:
                for td in turn["tool_details"]:
                    input_str = json.dumps(td["input"], default=str)
                    if len(input_str) > 200:
                        input_str = input_str[:200] + "..."
                    lines.append(f"- `{td['tool']}({input_str})`")

            # Show failures
            if accuracy < 1.0:
                details = score.get("details", {})
                if details.get("missing_required"):
                    lines.append(f"  **Missing required:** {details['missing_required']}")
                if details.get("missing_any_of"):
                    lines.append(f"  **Missing any of:** {details['missing_any_of']}")
                if details.get("forbidden_called"):
                    lines.append(f"  **Forbidden called:** {details['forbidden_called']}")

            lines.append(f"\n**Dash:** {turn['response_text']}")
            lines.append(
                f"\n*{turn['input_tokens']} in / {turn['output_tokens']} out / "
                f"{turn['latency_ms']}ms / ${cost:.4f}*\n"
            )

        # Session summary
        session_accuracy = [t["score"]["tool_accuracy"] for t in session["turns"]]
        avg = sum(session_accuracy) / len(session_accuracy) if session_accuracy else 0
        lines.append(f"**Session accuracy:** {avg:.0%} | **Cost:** ${session_cost:.4f}\n")
        total_cost += session_cost

        # Show data state after session
        data_state = session.get("data_state_after", {})
        tasks = data_state.get("tasks", [])
        context = data_state.get("user_context", {})
        active = [t for t in tasks if t.get("status") == "active"]
        completed = [t for t in tasks if t.get("status") == "completed"]
        lines.append(f"**Data after session:** {len(active)} active tasks, {len(completed)} completed, "
                      f"{len(context.get('observations', []))} observations\n")

    # Aggregate summary
    lines.append("---")
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total turns | {total_turns} |")
    lines.append(f"| Turns passed (tool accuracy = 1.0) | {total_passed}/{total_turns} ({total_passed / total_turns:.0%}) |")

    total_input = sum(t["input_tokens"] for s in results["sessions"] for t in s["turns"])
    total_output = sum(t["output_tokens"] for s in results["sessions"] for t in s["turns"])
    total_latency = sum(t["latency_ms"] for s in results["sessions"] for t in s["turns"])

    lines.append(f"| Total input tokens | {total_input:,} |")
    lines.append(f"| Total output tokens | {total_output:,} |")
    lines.append(f"| Total cost | ${total_cost:.4f} |")
    lines.append(f"| Avg latency per turn | {total_latency // total_turns}ms |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Path:
    """Run all eval sessions and generate report."""
    if not config.ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    # Load golden sessions
    if not EVAL_SESSIONS_FILE.exists():
        print("Error: tests/eval_sessions.json not found.")
        print()
        print("This file contains your golden dataset (multi-turn sessions with expectations).")
        print("It's personal to your agent, so it's not committed to Git.")
        print()
        print("See tests/EVAL_GUIDE.md for how to create one.")
        sys.exit(1)

    with open(EVAL_SESSIONS_FILE) as f:
        eval_data = json.load(f)

    total_turns = sum(len(s["turns"]) for s in eval_data["sessions"])

    print("=" * 50)
    print("Dash Eval Runner")
    print(f"Model: {config.MODEL}")
    print(f"Sessions: {len(eval_data['sessions'])}")
    print(f"Total turns: {total_turns}")

    if not config.TAVILY_API_KEY:
        print("\nWARNING: TAVILY_API_KEY not set. Web search turns will fail gracefully.")

    print("=" * 50)
    print()

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up isolated data directory (fresh start, no existing tasks/context)
    eval_data_dir = RESULTS_DIR / f"data_{run_id}"
    patch_data_dir(eval_data_dir)

    results: dict[str, Any] = {
        "run_id": run_id,
        "model": config.MODEL,
        "timestamp": datetime.now().isoformat(),
        "sessions": [],
    }

    for session in eval_data["sessions"]:
        print(f"--- {session['id']} ---")
        session_result = run_session(client, session)
        results["sessions"].append(session_result)
        print()

    # Save JSON results
    results_file = RESULTS_DIR / f"run_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save markdown report
    report = generate_report(results)
    report_file = RESULTS_DIR / f"run_{run_id}_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    # Print summary
    total_passed = sum(
        1
        for s in results["sessions"]
        for t in s["turns"]
        if t["score"]["tool_accuracy"] == 1.0
    )
    total_cost = sum(
        t["input_tokens"] / 1_000_000 * 0.80 + t["output_tokens"] / 1_000_000 * 4.00
        for s in results["sessions"]
        for t in s["turns"]
    )

    print("=" * 50)
    print(f"RESULTS: {total_passed}/{total_turns} turns passed ({total_passed / total_turns:.0%})")
    print(f"COST: ${total_cost:.4f}")
    print(f"JSON:   {results_file}")
    print(f"Report: {report_file}")
    print("=" * 50)

    return results_file


if __name__ == "__main__":
    main()
