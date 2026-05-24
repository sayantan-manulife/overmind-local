"""
GitHub Copilot integration — two modes:

1. GitHub Models API via litellm
   Set GITHUB_TOKEN and use --model github/gpt-4o (or any GitHub-hosted model).
   Full LLM completions, works in all overmind-local commands.

2. gh copilot CLI subprocess
   Uses `gh copilot explain` to summarise traces in plain English.
   Requires `gh` CLI with the copilot extension installed.
"""
import json
import subprocess
from typing import Optional


# ── GitHub Models via litellm ──────────────────────────────────────────────────

# litellm model aliases for GitHub-hosted models (GITHUB_TOKEN required)
COPILOT_MODELS = {
    "copilot":           "github/gpt-4o",
    "copilot-mini":      "github/gpt-4o-mini",
    "copilot-o1":        "github/o1",
    "copilot-o1-mini":   "github/o1-mini",
    "copilot-claude":    "github/claude-3-5-sonnet",
}


def resolve_model(model: str) -> str:
    """Expand shorthand copilot model names to their litellm equivalents."""
    return COPILOT_MODELS.get(model, model)


def check_github_token() -> Optional[str]:
    """Return the GITHUB_TOKEN if set, otherwise try to get it from gh CLI."""
    import os
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # Try gh CLI
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            token = result.stdout.strip()
            os.environ["GITHUB_TOKEN"] = token  # expose to litellm
            return token
    except Exception:
        pass
    return None


# ── gh copilot CLI integration ────────────────────────────────────────────────

def gh_copilot_available() -> bool:
    try:
        r = subprocess.run(
            ["gh", "copilot", "--version"], capture_output=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


def explain_traces(traces: list) -> str:
    """
    Use `gh copilot explain` to describe what's happening in recent traces.

    Formats the traces as a bash-like script description and pipes through
    `gh copilot explain` to get a plain-English summary.
    """
    if not gh_copilot_available():
        return "gh copilot not available — install with: gh extension install github/gh-copilot"

    # Format traces as a pseudo-script for gh copilot explain
    lines = []
    for s in traces[:15]:
        err = s.get("error")
        line = f"# {s['name']} ({s['span_type']}) took {s['duration_ms']:.0f}ms"
        if err:
            line += f"\n# ERROR: {err}"
        lines.append(line)
    script = "\n".join(lines)

    # gh copilot explain reads from stdin or takes a positional arg
    try:
        result = subprocess.run(
            ["gh", "copilot", "explain", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return f"gh copilot explain failed: {exc}"


def suggest_fix(problem_description: str) -> str:
    """
    Use `gh copilot suggest` to get a concrete fix suggestion.
    Returns the raw copilot suggestion text.
    """
    if not gh_copilot_available():
        return "gh copilot not available"
    try:
        result = subprocess.run(
            ["gh", "copilot", "suggest", "-t", "generic", problem_description],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return f"gh copilot suggest failed: {exc}"
