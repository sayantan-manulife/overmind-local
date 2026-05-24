"""Optimization loop: read local traces → LLM analysis → ranked suggestions."""
import json
from pathlib import Path
from typing import Optional

import litellm

from .storage import get_spans, get_policies, get_dataset


def run_optimization(
    agent_name: str,
    model: str = "gpt-4o",
    trace_limit: int = 50,
    db_path: Optional[Path] = None,
) -> str:
    spans = get_spans(agent_name=agent_name, limit=trace_limit, db_path=db_path)
    policies = get_policies(agent_name=agent_name, db_path=db_path)
    dataset = get_dataset(agent_name=agent_name, db_path=db_path)

    if not spans:
        return (
            f"No traces found for agent '{agent_name}'.\n"
            "Instrument your agent with @observe/@tool decorators and run it first."
        )

    # Summarise spans for the prompt (avoid blowing the context window)
    trace_summary = []
    for s in spans:
        entry: dict = {
            "name": s["name"],
            "type": s["span_type"],
            "duration_ms": round(s["duration_ms"] or 0, 1),
        }
        if s.get("error"):
            entry["error"] = s["error"]
        try:
            inp = json.loads(s["input"] or "null")
            if inp and isinstance(inp, dict):
                # Truncate long string values
                entry["input"] = {
                    k: (v[:200] + "…" if isinstance(v, str) and len(v) > 200 else v)
                    for k, v in inp.items()
                }
        except Exception:
            pass
        try:
            out = json.loads(s["output"] or "null")
            if out and isinstance(out, dict) and "content" in out:
                content = out["content"]
                entry["output_preview"] = (
                    content[:200] + "…" if isinstance(content, str) and len(content) > 200
                    else content
                )
        except Exception:
            pass
        trace_summary.append(entry)

    policy_lines = "\n".join(
        f"  • {p['name']}: {p['description']}" for p in policies
    ) or "  (no policies defined)"

    dataset_lines = "\n".join(
        f"  • Input: {p['input']}" + (f"  → Expected: {p['expected_output']}" if p.get("expected_output") else "")
        for p in dataset[:10]
    ) or "  (no test cases defined)"

    # Failure rate
    errors = [s for s in spans if s.get("error")]
    error_rate = f"{len(errors)}/{len(spans)} spans errored"

    prompt = f"""You are an expert AI agent optimizer. Your job is to analyze production traces from an agent and provide concrete, actionable improvement recommendations.

AGENT: {agent_name}
ERROR RATE: {error_rate}

POLICIES (constraints that MUST be respected in any recommendations):
{policy_lines}

TEST DATASET:
{dataset_lines}

RECENT PRODUCTION TRACES ({len(trace_summary)} spans, newest first):
{json.dumps(trace_summary, indent=2)}

Analyze the traces and produce a structured optimization report:

## 1. Failure Analysis
List specific errors or failure patterns you observe, with frequency counts.

## 2. Performance Issues
Identify slow spans (high duration_ms), redundant calls, or inefficient patterns.

## 3. Policy Violations
Flag any behavior that violates the listed policies.

## 4. Recommended Changes (ranked by impact)
For each recommendation:
- **What to change**: specific component (system prompt / tool description / retry logic / model choice)
- **Why**: evidence from the traces
- **How**: concrete code snippet or text change
- **Expected impact**: what metric should improve

## 5. Suggested System Prompt Improvements
If LLM spans are present, propose revised system prompt text that addresses observed failure modes.

Be specific. Reference actual span names and error messages from the traces. Do not make generic suggestions."""

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
