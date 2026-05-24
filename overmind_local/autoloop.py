"""
Karpathy-style autoresearch loop for agent configs.

Cycle:
  1. Read target file + recent traces + policies
  2. LLM proposes a specific change
  3. Apply the change
  4. Run the metric command (must print a float to stdout)
  5. Keep if better, git-revert if not
  6. Log result → repeat

The human steers by editing `program.md` (or equivalent) between runs.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import litellm

from .storage import get_spans, get_policies, get_dataset, write_span


# ── Public entry point ─────────────────────────────────────────────────────────

def run_autoloop(
    agent_name: str,
    target_file: str,
    metric_cmd: str,
    *,
    direction: str = "higher",      # "higher" or "lower" is better
    iterations: int = 20,
    model: str = "gpt-4o",
    program_md: Optional[str] = None,  # optional steering doc (like Karpathy's program.md)
    db_path: Optional[Path] = None,
    log_fn=print,
):
    """
    Run the autoresearch loop.

    Args:
        target_file:  path to the file the LLM is allowed to edit
        metric_cmd:   shell command that prints a single float to stdout
        direction:    "higher" = higher metric is better (e.g. accuracy)
                      "lower"  = lower metric is better (e.g. error rate)
        program_md:   path to a Markdown file with research direction / constraints
        iterations:   max number of propose-test cycles
    """
    target = Path(target_file)
    if not target.exists():
        raise FileNotFoundError(f"Target file not found: {target}")

    program_text = Path(program_md).read_text() if program_md else ""
    best_content = target.read_text()
    best_score: Optional[float] = None
    history: list[dict] = []

    log_fn(f"\n🔬 Autoresearch loop — {iterations} iterations max")
    log_fn(f"   Target file : {target}")
    log_fn(f"   Metric cmd  : {metric_cmd}")
    log_fn(f"   Direction   : {direction} is better")
    log_fn(f"   Model       : {model}\n")

    for i in range(1, iterations + 1):
        log_fn(f"── Iteration {i}/{iterations} ──────────────────────────")

        # 1. Gather context
        traces = get_spans(agent_name=agent_name, limit=40, db_path=db_path)
        policies = get_policies(agent_name=agent_name, db_path=db_path)
        current = target.read_text()

        # 2. Propose a change
        log_fn("  → Asking LLM to propose a change...")
        proposed = _propose(
            current_content=current,
            traces=traces,
            policies=policies,
            program_text=program_text,
            history=history[-10:],  # last 10 outcomes for context
            model=model,
        )

        if proposed.strip() == current.strip():
            log_fn("  ⚠ LLM returned identical content — skipping iteration")
            continue

        # 3. Apply
        target.write_text(proposed)

        # 4. Run metric
        log_fn(f"  → Running metric: {metric_cmd}")
        score = _run_metric(metric_cmd)
        if score is None:
            log_fn("  ✗ Metric command failed or returned non-numeric — reverting")
            target.write_text(best_content)
            continue

        # 5. Keep or revert
        improved = best_score is None or _is_better(score, best_score, direction)
        if improved:
            verb = "KEPT  ✓" if best_score is not None else "BASELINE"
            best_score = score
            best_content = proposed
        else:
            verb = "REVERTED ✗"
            target.write_text(best_content)

        log_fn(f"  Score: {score:.4f}  (best: {best_score:.4f})  → {verb}")

        outcome = {"iteration": i, "score": score, "action": verb.split()[0]}
        history.append(outcome)

        # Record in traces DB so `overmind-local traces` shows the loop
        _record_loop_span(agent_name, i, score, best_score, verb, db_path)

        # Small pause to avoid hammering the LLM API
        time.sleep(1)

    log_fn(f"\n✅ Loop complete. Best score: {best_score:.4f}")
    log_fn(f"   Best content written to: {target}")
    return best_score


# ── Internals ──────────────────────────────────────────────────────────────────

def _propose(current_content, traces, policies, program_text, history, model):
    policy_lines = "\n".join(
        f"  • {p['name']}: {p['description']}" for p in policies
    ) or "  (none)"

    trace_summary = _summarise_traces(traces)
    history_lines = "\n".join(
        f"  iter {h['iteration']}: score={h['score']:.4f} → {h['action']}"
        for h in history
    ) or "  (first iteration)"

    program_section = f"\n## Research Direction\n{program_text}\n" if program_text else ""

    prompt = f"""You are an expert AI agent optimizer running in an autonomous research loop.
Your job: propose ONE specific, targeted improvement to the file below.{program_section}
## Policies (MUST respect)
{policy_lines}

## Recent Experiment History
{history_lines}

## Recent Agent Traces (errors and successes)
{trace_summary}

## Current File Content
```
{current_content}
```

Rules:
- Output ONLY the complete new file content — no explanation, no markdown fences, no preamble.
- Make exactly ONE logical change per iteration (change a prompt phrase, adjust a parameter, add a guard, etc.)
- Do not change the file structure or imports unless strictly necessary.
- If the history shows a change made things worse, try a different direction.
- If you have no improvement to suggest, output the file unchanged.

Output the complete new file content now:"""

    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip()

    # Strip accidental markdown fences if the LLM added them anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw


def _run_metric(cmd: str) -> Optional[float]:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        # Take the last non-empty line that parses as a float
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line:
                try:
                    return float(line)
                except ValueError:
                    continue
        return None
    except Exception:
        return None


def _is_better(score: float, best: float, direction: str) -> bool:
    if direction == "higher":
        return score > best
    return score < best


def _summarise_traces(traces: list) -> str:
    if not traces:
        return "  (no traces yet)"
    lines = []
    for s in traces[:20]:
        err = s.get("error")
        lines.append(
            f"  {s['name']} ({s['span_type']}) "
            f"{s['duration_ms']:.0f}ms"
            + (f"  ✗ {err[:60]}" if err else "  ✓")
        )
    return "\n".join(lines)


def _record_loop_span(agent_name, iteration, score, best_score, verb, db_path):
    import uuid
    write_span({
        "id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "parent_id": None,
        "name": f"autoloop/iter-{iteration}",
        "span_type": "autoloop",
        "agent_name": agent_name,
        "input": {"iteration": iteration},
        "output": {"score": score, "best_score": best_score, "action": verb},
        "error": None,
        "start_time": time.time(),
        "end_time": time.time(),
        "duration_ms": 0,
        "metadata": {"direction": "autoloop"},
    }, db_path=db_path)
