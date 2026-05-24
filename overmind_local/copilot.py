"""
GitHub Copilot CLI integration — subprocess only, no API key needed.

Uses the `gh copilot` extension:
  gh extension install github/gh-copilot

Commands used:
  gh copilot explain  — summarise what code/traces are doing
  gh copilot suggest  — suggest a fix given a problem description
"""
import subprocess
from typing import Optional


def gh_copilot_available() -> bool:
    try:
        r = subprocess.run(
            ["gh", "copilot", "--version"], capture_output=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


def explain_traces(traces: list) -> str:
    """Pipe recent agent traces through `gh copilot explain`."""
    if not gh_copilot_available():
        return (
            "gh copilot not found.\n"
            "Install: gh extension install github/gh-copilot"
        )

    lines = []
    for s in traces[:15]:
        err = s.get("error")
        line = f"# {s['name']} ({s['span_type']}) took {s['duration_ms']:.0f}ms"
        if err:
            line += f"\n# ERROR: {err}"
        lines.append(line)
    script = "\n".join(lines)

    try:
        result = subprocess.run(
            ["gh", "copilot", "explain", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
    except Exception as exc:
        return f"gh copilot explain failed: {exc}"


def suggest_fix(problem_description: str) -> str:
    """Ask `gh copilot suggest` for a concrete fix."""
    if not gh_copilot_available():
        return "gh copilot not found — install: gh extension install github/gh-copilot"
    try:
        result = subprocess.run(
            ["gh", "copilot", "suggest", "-t", "generic", problem_description],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
    except Exception as exc:
        return f"gh copilot suggest failed: {exc}"
