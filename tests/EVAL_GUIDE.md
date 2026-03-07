# Evaluating Your Dash Agent

Unit tests verify that your code works. Evals verify that your **agent** works: does Haiku call the right tools, respond with the right tone, and maintain context across a conversation?

This guide covers how to test your Dash agent's actual LLM behavior using multi-turn eval sessions.

## How It Works

```
You create eval_sessions.json (your golden dataset)
  -> eval_runner.py runs each session against real Haiku
  -> Tools execute for real (isolated data directory, not your actual data/)
  -> Each turn is scored: did Haiku call the expected tools?
  -> Results saved as JSON + readable markdown report
```

Cost: roughly $0.01 per turn (Haiku pricing). A 25-turn eval costs about $0.20.

## Quick Start

```bash
# 1. Create your golden dataset (see format below)
#    Save it as tests/eval_sessions.json

# 2. Run the eval
python3 -m tests.eval_runner

# 3. Read the report
cat tests/eval_results/run_*_report.md
```

The runner creates an isolated data directory for each run, so your real `data/` folder is never touched.

## Creating Your Golden Dataset

Create `tests/eval_sessions.json` with this structure:

```json
{
  "metadata": {
    "description": "My Dash eval sessions",
    "version": "1.0"
  },
  "sessions": [
    {
      "id": "session_1_basics",
      "description": "Test basic task creation and listing",
      "turns": [
        {
          "turn": 1,
          "user_input": "I need to finish my report by Friday",
          "expectations": {
            "must_call_all": ["add_task"],
            "must_call_any": [],
            "must_not_call": ["complete_task"],
            "behavioral": [
              "Should create a task with 'report' in the title",
              "Should set a due date for Friday",
              "Should confirm briefly"
            ]
          }
        },
        {
          "turn": 2,
          "user_input": "What's on my plate?",
          "expectations": {
            "must_call_all": ["list_tasks"],
            "must_call_any": [],
            "must_not_call": [],
            "behavioral": [
              "Should list the report task",
              "Should mention the Friday deadline"
            ]
          }
        }
      ]
    }
  ]
}
```

### Expectation Fields

| Field | Type | Meaning |
|-------|------|---------|
| `must_call_all` | `string[]` | Every tool in this list must be called |
| `must_call_any` | `string[]` | At least one tool from this list must be called |
| `must_not_call` | `string[]` | None of these tools should be called |
| `behavioral` | `string[]` | Human-readable expectations (not auto-graded, for your review) |

### Tool Accuracy Scoring

Each turn gets a tool accuracy score from 0.0 to 1.0 based on three checks:

1. **must_call_all passed?** Did the agent call every required tool?
2. **must_call_any passed?** Did it call at least one from the "any" list?
3. **must_not_call passed?** Did it avoid all forbidden tools?

Score = (checks passed) / (total checks). A perfect turn scores 1.0.

## What to Test

### Session Design Tips

**Sessions should be continuous.** Tasks and user_context persist across turns within a session (and across sessions). Design turns that build on previous ones:

```
Turn 1: User adds a task
Turn 2: User asks what's on their plate (should see the task from turn 1)
Turn 3: User completes the task
Turn 4: User asks what's left (completed task should be gone)
```

**Test categories to cover:**

| Category | What to Test | Example |
|----------|-------------|---------|
| Task creation | Does it create tasks from natural language? | "I need to buy groceries" |
| Priority handling | Does it infer priority from cues? | "No rush on that" = low priority |
| Context learning | Does it store observations proactively? | "I'm a morning person" |
| Emotional awareness | Does it respond appropriately to mood? | "Had a rough day" |
| Tool selection | Does it pick the right tool? | URL = web_fetch, question = web_search |
| Edge cases | Does it handle limitations gracefully? | "Delete all tasks" (no delete tool) |
| Minimal input | Does it match energy on short inputs? | "I'm fine." = brief response |
| Multi-task | Can it handle batch requests? | "Add X, Y, and Z" |

### Behavioral Expectations

The `behavioral` field is not auto-graded. It's there so you can review each turn's response against your intentions. When reading the report, check:

- Is the response the right length? (brief for simple inputs, detailed when needed)
- Does it use the user's name (if stored)?
- Does it avoid sycophantic openers ("Certainly!", "Great question!")?
- Does it connect context across turns?

## Reading the Report

The runner generates two files per run:

- **`run_*_.json`**: Full transcript with every tool call, parameter, response, and token count
- **`run_*_report.md`**: Readable markdown with PASS/WARN/FAIL per turn

The report shows:
- Each turn's input, tools called, tool accuracy, and Dash's response
- Per-session accuracy and cost
- Aggregate summary (total turns, pass rate, cost, latency)

### What Good Results Look Like

- **90%+ tool accuracy**: The agent calls the right tools most of the time
- **$0.10-0.25 per full eval**: Haiku is cheap enough to run evals frequently
- **<5s average latency per turn**: Multi-tool turns take longer, simple responses are faster

### Common Failures and Fixes

| Failure | Likely Cause | Fix |
|---------|-------------|-----|
| Doesn't store observations | Prompt not explicit enough about when to call update_user_context | Add concrete examples to the "Learning about the user" section in config.py |
| Answers from memory instead of checking tasks | Prompt doesn't enforce list_tasks-first rule strongly enough | Add "ALWAYS call list_tasks first when asked about priorities" |
| Too verbose on simple inputs | No energy-matching rule | Add "Match the user's energy. One-line input gets a 1-3 line response" |
| Calls web_search when given a URL | Tool descriptions not clear enough | Clarify: web_fetch = specific URL, web_search = general query |

## Extending the System

The eval runner is designed to be simple. Some ways to extend it:

- **Multi-trial runs**: Run the same sessions 3 times to measure consistency (LLM output is non-deterministic)
- **Parameter checking**: Add `param_checks` to expectations and extend `score_turn()` in eval_runner.py
- **LLM-as-judge**: Send transcripts to a stronger model (Sonnet/Opus) for behavioral scoring
- **Regression tracking**: Save results to a CSV and compare across prompt changes
- **CI integration**: Run evals on every prompt change and fail if pass rate drops below a threshold
