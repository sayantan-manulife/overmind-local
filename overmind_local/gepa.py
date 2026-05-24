"""
GEPA — Generative Evolutionary Prompt Adaptation.

Based on: https://github.com/gepa-ai/gepa

Key innovation over hill-climbing (autoloop):
  - Pareto frontier: maintains candidates that excel on *different* failure modes,
    not just the single top scorer
  - Reflect: after each evaluation, an LLM reads full diagnostic traces (ASI —
    Actionable Side Information) to diagnose *why* a candidate fails
  - Mutate: targeted improvement informed by the candidate's diagnosis and
    accumulated ancestor reflections
  - Merge: system-aware crossover of two Pareto-front candidates with
    complementary error profiles

Cycle (per iteration):
  1. Select — pick a candidate from the Pareto front (round-robin for diversity)
  2. Execute — write to target, run metric, capture new traces as ASI
  3. Reflect — LLM diagnoses failure modes from ASI
  4. Mutate  — LLM proposes a fix targeting diagnosed failures
  5. Merge   — (every merge_every cycles) LLM merges two complementary candidates
  6. Evaluate all new candidates
  7. Accept  — update Pareto front (drop dominated candidates)
"""
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import litellm

from .storage import get_policies, get_spans, write_span


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    content: str
    score: Optional[float] = None
    error_types: Set[str] = field(default_factory=set)  # failure fingerprint from traces
    reflection: str = ""                                  # LLM diagnosis
    ancestor_reflections: List[str] = field(default_factory=list)
    label: str = "seed"


# ── Public entry point ─────────────────────────────────────────────────────────

def run_gepa(
    agent_name: str,
    target_file: str,
    metric_cmd: str,
    *,
    cycles: int = 20,
    pareto_k: int = 5,
    merge_every: int = 5,
    direction: str = "higher",
    model: str = "gpt-4o",
    program_md: Optional[str] = None,
    db_path=None,
    log_fn=print,
) -> float:
    """
    Run GEPA optimization.

    Args:
        cycles:      total Select→Reflect→Mutate iterations
        pareto_k:    maximum Pareto front size (candidates kept)
        merge_every: perform a system-aware merge every N cycles
        direction:   "higher" or "lower" for the metric
    """
    target = Path(target_file)
    if not target.exists():
        raise FileNotFoundError(f"Target file not found: {target}")

    seed_content = target.read_text()
    program_text = Path(program_md).read_text() if program_md else ""
    policies = get_policies(agent_name=agent_name, db_path=db_path)

    log_fn(f"\n🧬 GEPA — {cycles} cycles, Pareto-k={pareto_k}, merge-every={merge_every}")
    log_fn(f"   Target   : {target}")
    log_fn(f"   Metric   : {metric_cmd}")
    log_fn(f"   Direction: {direction} is better\n")

    # Bootstrap: evaluate seed
    seed = Candidate(content=seed_content, label="seed")
    log_fn("── Bootstrap: evaluating seed ──────────────────────────")
    _execute(seed, target, metric_cmd, agent_name, db_path, log_fn)
    if seed.score is not None:
        log_fn(f"  Seed score: {seed.score:.4f}")
        seed.reflection = _reflect(seed, policies, program_text, model)
    else:
        log_fn("  ⚠ Seed metric failed — continuing anyway")

    pareto: List[Candidate] = [seed]
    best_score: Optional[float] = seed.score
    best_content = seed_content
    select_idx = 0

    for cycle in range(1, cycles + 1):
        log_fn(f"\n── Cycle {cycle}/{cycles} ─────────────────────────────────────")

        # 1. Select from Pareto front (round-robin for coverage)
        parent = pareto[select_idx % len(pareto)]
        select_idx += 1
        log_fn(f"  Selected: [{parent.label}]  score={parent.score}")

        # 2. Reflect (reuse existing reflection if available)
        if not parent.reflection:
            parent.reflection = _reflect(parent, policies, program_text, model)

        new_candidates: List[Candidate] = []

        # 3. Mutate
        log_fn("  → Mutating...")
        mutant = _mutate(parent, policies, program_text, model)
        _execute(mutant, target, metric_cmd, agent_name, db_path, log_fn)
        if mutant.score is not None:
            log_fn(f"    Mutant score: {mutant.score:.4f}")
            mutant.reflection = _reflect(mutant, policies, program_text, model)
            new_candidates.append(mutant)

        # 4. Merge (every merge_every cycles, when front has ≥2 candidates)
        if cycle % merge_every == 0 and len(pareto) >= 2:
            log_fn("  → Merging two Pareto candidates...")
            a, b = _select_complementary_pair(pareto, direction)
            merged = _merge(a, b, policies, program_text, model)
            _execute(merged, target, metric_cmd, agent_name, db_path, log_fn)
            if merged.score is not None:
                log_fn(f"    Merged score: {merged.score:.4f}")
                merged.reflection = _reflect(merged, policies, program_text, model)
                new_candidates.append(merged)

        # 5. Accept: add non-dominated candidates, trim front
        for cand in new_candidates:
            if not _is_dominated(cand, pareto, direction):
                pareto.append(cand)
                pareto = _trim_pareto(pareto, pareto_k, direction)

        # Track global best
        for cand in new_candidates:
            if cand.score is not None:
                if best_score is None or _is_better(cand.score, best_score, direction):
                    best_score = cand.score
                    best_content = cand.content
                    log_fn(f"  ✓ New best: {best_score:.4f}")

        scores_str = ", ".join(
            f"{c.score:.4f}" for c in pareto if c.score is not None
        )
        log_fn(f"  Pareto front ({len(pareto)}): [{scores_str}]  best-so-far: "
               f"{best_score:.4f if best_score is not None else 'n/a'}")

        _record_gepa_span(agent_name, cycle, best_score, len(pareto), db_path)

        time.sleep(1)

    target.write_text(best_content)

    if best_score is None:
        log_fn("\n⚠ GEPA complete. No valid score produced.")
        return 0.0

    log_fn(f"\n✅ GEPA complete. Best score: {best_score:.4f}")
    log_fn(f"   Best content written to: {target}")
    return best_score


# ── Core GEPA steps ────────────────────────────────────────────────────────────

def _execute(cand: Candidate, target: Path, metric_cmd: str,
             agent_name: str, db_path, log_fn):
    """Execute: write candidate, run metric, capture traces as ASI."""
    target.write_text(cand.content)
    cand.score = _run_metric(metric_cmd)

    # Capture error fingerprint from fresh traces (ASI)
    try:
        traces = get_spans(agent_name=agent_name, limit=20, db_path=db_path)
        cand.error_types = {
            t["error"].split(":")[0].strip()
            for t in traces
            if t.get("error")
        }
    except Exception:
        cand.error_types = set()


def _reflect(cand: Candidate, policies: list, program_text: str, model: str) -> str:
    """Reflect: LLM reads diagnostic info to diagnose failure modes."""
    policy_lines = "\n".join(f"  • {p['name']}: {p['description']}" for p in policies) or "  (none)"
    program_section = f"\n## Research Direction\n{program_text}\n" if program_text else ""
    score_line = f"{cand.score:.4f}" if cand.score is not None else "metric failed"
    error_list = "\n".join(f"  - {e}" for e in sorted(cand.error_types)) or "  (none observed)"

    prompt = f"""You are an expert AI agent optimizer performing a failure analysis (Reflect step).
Analyze the candidate file below and diagnose specific failure modes.{program_section}
## Policies
{policy_lines}

## Evaluation Result
Score: {score_line}
Error types observed in traces:
{error_list}

## Candidate File
```
{cand.content}
```

Produce a concise failure diagnosis (3-6 bullet points):
- What specific behaviors or patterns are likely causing failures?
- Which parts of the file are most likely responsible?
- What concrete changes would address each failure mode?

Be specific and actionable. Reference exact sections of the file."""

    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"(reflection failed: {exc})"


def _mutate(parent: Candidate, policies: list, program_text: str, model: str) -> Candidate:
    """Mutate: generate improved variant targeting diagnosed failure modes."""
    policy_lines = "\n".join(f"  • {p['name']}: {p['description']}" for p in policies) or "  (none)"
    program_section = f"\n## Research Direction\n{program_text}\n" if program_text else ""

    ancestor_section = ""
    if parent.ancestor_reflections:
        joined = "\n\n".join(f"Ancestor diagnosis:\n{r}" for r in parent.ancestor_reflections[-5:])
        ancestor_section = f"\n## Ancestor Reflections (accumulated lessons)\n{joined}\n"

    prompt = f"""You are an expert AI agent optimizer performing a targeted mutation (Mutate step).
Your job: fix exactly ONE failure mode identified in the diagnosis below.{program_section}{ancestor_section}
## Policies (MUST respect)
{policy_lines}

## Failure Diagnosis for This Candidate
{parent.reflection}

## Current File
```
{parent.content}
```

Rules:
- Output ONLY the complete new file content — no markdown fences, no explanation
- Address the MOST impactful failure mode from the diagnosis
- Make the minimal change needed; do not restructure unrelated parts
- The fix must be concrete and specific (change a specific phrase, add a guard, adjust a value)

Output the improved file now:"""

    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = _strip_fences(resp.choices[0].message.content.strip())
    except Exception:
        content = parent.content  # fallback: no change

    child = Candidate(
        content=content,
        ancestor_reflections=parent.ancestor_reflections + [parent.reflection],
        label=f"mutant(of={parent.label})",
    )
    return child


def _merge(a: Candidate, b: Candidate, policies: list, program_text: str, model: str) -> Candidate:
    """Merge: system-aware crossover of two Pareto-front candidates with complementary strengths."""
    policy_lines = "\n".join(f"  • {p['name']}: {p['description']}" for p in policies) or "  (none)"
    program_section = f"\n## Research Direction\n{program_text}\n" if program_text else ""
    score_a = f"{a.score:.4f}" if a.score is not None else "n/a"
    score_b = f"{b.score:.4f}" if b.score is not None else "n/a"

    prompt = f"""You are an expert AI agent optimizer performing a system-aware merge (Merge step).
Two Pareto-optimal candidates excel on different failure modes. Combine their complementary strengths.{program_section}
## Policies (MUST respect)
{policy_lines}

## Candidate A (score: {score_a})
Failure diagnosis:
{a.reflection}

File:
```
{a.content}
```

## Candidate B (score: {score_b})
Failure diagnosis:
{b.reflection}

File:
```
{b.content}
```

Rules:
- Output ONLY the complete merged file content — no markdown fences, no explanation
- Identify which parts of A address B's failures and vice versa
- Combine these strengths into a single coherent file
- Do not introduce new unrelated changes — pure synthesis only

Output the merged file now:"""

    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = _strip_fences(resp.choices[0].message.content.strip())
    except Exception:
        content = a.content

    return Candidate(
        content=content,
        ancestor_reflections=list(set(a.ancestor_reflections + b.ancestor_reflections +
                                      [a.reflection, b.reflection])),
        label=f"merge({a.label}+{b.label})",
    )


# ── Pareto front helpers ───────────────────────────────────────────────────────

def _is_dominated(cand: Candidate, front: List[Candidate], direction: str) -> bool:
    """A candidate is dominated if another has better/equal score AND a subset of its errors."""
    if cand.score is None:
        return True
    for other in front:
        if other.score is None:
            continue
        score_ok = (other.score >= cand.score) if direction == "higher" else (other.score <= cand.score)
        errors_ok = other.error_types <= cand.error_types  # other has fewer or same error types
        if score_ok and errors_ok and other is not cand:
            return True
    return False


def _trim_pareto(front: List[Candidate], k: int, direction: str) -> List[Candidate]:
    """Remove dominated candidates, then keep top-k by score if still too large."""
    non_dom = [c for c in front if not _is_dominated(c, front, direction)]
    if not non_dom:
        non_dom = front  # safety: keep all if everything looks dominated

    # Sort by score (best first), keep top-k
    scored = [c for c in non_dom if c.score is not None]
    unscored = [c for c in non_dom if c.score is None]
    scored.sort(key=lambda c: c.score, reverse=(direction == "higher"))
    return (scored + unscored)[:k]


def _select_complementary_pair(front: List[Candidate], direction: str):
    """Pick two candidates with the most complementary error profiles."""
    if len(front) < 2:
        return front[0], front[0]
    scored = [c for c in front if c.score is not None]
    if len(scored) < 2:
        return front[0], front[1]

    # Best scorer + candidate with most unique errors
    best = max(scored, key=lambda c: c.score if direction == "higher" else -c.score)
    best_errors = best.error_types
    others = [c for c in scored if c is not best]
    complement = max(others, key=lambda c: len(c.error_types - best_errors))
    return best, complement


# ── Utilities ──────────────────────────────────────────────────────────────────

def _run_metric(cmd: str) -> Optional[float]:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
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
    return score > best if direction == "higher" else score < best


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _record_gepa_span(agent_name, cycle, best_score, front_size, db_path):
    write_span({
        "id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "parent_id": None,
        "name": f"gepa/cycle-{cycle}",
        "span_type": "gepa",
        "agent_name": agent_name,
        "input": {"cycle": cycle},
        "output": {"best_score": best_score, "pareto_front_size": front_size},
        "error": None,
        "start_time": time.time(),
        "end_time": time.time(),
        "duration_ms": 0,
        "metadata": {},
    }, db_path=db_path)
