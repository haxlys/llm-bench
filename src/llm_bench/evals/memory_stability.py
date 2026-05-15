"""Synthetic memory-stability diagnostic inspired by faulty-memory failures.

The runner measures whether repeatedly rewritten textual memory stays useful
for recurring grid-transformation tasks. It is intentionally small and
deterministically scored: models receive an opaque task code plus optional
memory, produce a JSON output grid, and the runner compares it with an oracle
transform.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

DEFAULT_TIMEOUT_S = 10 * 60
DEFAULT_UPDATE_STEPS = 8
DEFAULT_EPISODES_PER_CASE = 2
FORCED_MAX_ITEMS = 3

Grid = list[list[int]]


@dataclass(frozen=True)
class RuleFamily:
    id: str
    description: str
    source_color: int | None = None
    target_color: int | None = None


@dataclass(frozen=True)
class MemoryEpisode:
    id: str
    task_code: str
    input_grid: Grid
    output_grid: Grid


@dataclass(frozen=True)
class EvalCase:
    id: str
    task_code: str
    examples: list[MemoryEpisode]
    input_grid: Grid
    expected_output: Grid


AnswerFn = Callable[[EvalCase, str, str, str], str]
ConsolidateFn = Callable[[str, int], str]


FAMILIES: tuple[RuleFamily, ...] = (
    RuleFamily("f17", "turn every 6 into 0", source_color=6, target_color=0),
    RuleFamily("f23", "turn every 2 into 7", source_color=2, target_color=7),
    RuleFamily("f41", "keep only 4 and erase every other non-zero color", source_color=4),
    RuleFamily("f58", "swap 1 and 9"),
)
FAMILY_BY_ID = {family.id: family for family in FAMILIES}


def run_memory_stability(
    base_url: str,
    model_label: str,
    output_dir: Path,
    limit: int | None = None,
    update_steps: int = DEFAULT_UPDATE_STEPS,
    answer_fn: AnswerFn | None = None,
    consolidate_fn: ConsolidateFn | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Run the memory-stability diagnostic and write a synthetic results JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_call = _model_call_factory(base_url, model_label, timeout_s, api_key)
    if consolidate_fn is None:
        consolidate_fn = model_call
    if answer_fn is None:
        answer_fn = _answer_with_model(model_call)

    families = list(FAMILIES)
    episodes = build_episode_stream(update_steps=update_steps, families=families)
    cases = build_eval_cases(families=families, limit=limit)

    forced_memory = build_forced_memory(episodes, consolidate_fn)
    gated_memories = build_gated_memories(episodes, consolidate_fn)

    memories = {
        "no_memory": "",
        "forced_abstraction": forced_memory,
        "gated_abstraction": "\n\n".join(gated_memories.values()),
        "episodic_only": "",
    }

    samples: list[dict] = []
    policy_correct: dict[str, list[bool]] = {}
    for policy in ("no_memory", "episodic_only", "forced_abstraction", "gated_abstraction"):
        correct_flags: list[bool] = []
        for case in cases:
            memory_context = _memory_for_case(
                policy=policy,
                case=case,
                episodes=episodes,
                forced_memory=forced_memory,
                gated_memories=gated_memories,
            )
            prompt = build_solve_prompt(case, memory_context)
            raw_answer = ""
            parsed_output: Grid | None = None
            error: str | None = None
            try:
                raw_answer = answer_fn(case, prompt, policy, memory_context)
                parsed_output = parse_output_grid(raw_answer)
            except Exception as exc:  # noqa: BLE001 - captured in sample artifact
                error = str(exc)
            is_correct = parsed_output == case.expected_output
            correct_flags.append(is_correct)
            samples.append(
                {
                    "policy": policy,
                    "case": asdict(case),
                    "memory_context": memory_context,
                    "prompt": prompt,
                    "answer": raw_answer,
                    "parsed_output": parsed_output,
                    "correct": is_correct,
                    "error": error,
                }
            )
        policy_correct[policy] = correct_flags

    metrics = aggregate_metrics(
        policy_correct=policy_correct,
        forced_memory=forced_memory,
        gated_memories=gated_memories,
        families=families,
        update_steps=update_steps,
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    results_path = output_dir / f"results_{ts}_memory_stability.json"
    payload = {
        "results": {"memory_stability": metrics},
        "config": {
            "update_steps": update_steps,
            "families": [family.id for family in families],
            "limit": limit,
            "forced_max_items": FORCED_MAX_ITEMS,
        },
        "memories": memories,
        "samples": samples,
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    return {
        "task": "memory_stability",
        "results_file": str(results_path),
        "samples_file": str(results_path),
        "results": {"memory_stability": metrics},
    }


def build_episode_stream(
    update_steps: int = DEFAULT_UPDATE_STEPS,
    families: list[RuleFamily] | None = None,
) -> list[MemoryEpisode]:
    """Build a deterministic mixed-family stream of solved episodes."""
    selected = families or list(FAMILIES)
    if not selected:
        return []
    episodes: list[MemoryEpisode] = []
    for step in range(update_steps):
        family = selected[step % len(selected)]
        grid = _grid_variant(step)
        episodes.append(
            MemoryEpisode(
                id=f"train_{step:03d}_{family.id}",
                task_code=family.id,
                input_grid=grid,
                output_grid=apply_family(family.id, grid),
            )
        )
    return episodes


def build_eval_cases(
    families: list[RuleFamily] | None = None,
    limit: int | None = None,
) -> list[EvalCase]:
    """Build held-out cases with ARC-style input/output examples."""
    selected = families or list(FAMILIES)
    cases = []
    for index, family in enumerate(selected):
        input_grid = _grid_variant(100 + index)
        examples = [
            MemoryEpisode(
                id=f"example_{index:03d}_{slot}_{family.id}",
                task_code=family.id,
                input_grid=_grid_variant(200 + index * 10 + slot),
                output_grid=apply_family(family.id, _grid_variant(200 + index * 10 + slot)),
            )
            for slot in range(2)
        ]
        cases.append(
            EvalCase(
                id=f"heldout_{index:03d}_{family.id}",
                task_code=family.id,
                examples=examples,
                input_grid=input_grid,
                expected_output=apply_family(family.id, input_grid),
            )
        )
    return cases[:limit] if limit is not None else cases


def apply_family(task_code: str, grid: Grid) -> Grid:
    """Apply the oracle transform for a synthetic task code."""
    if task_code == "f17":
        return _replace(grid, 6, 0)
    if task_code == "f23":
        return _replace(grid, 2, 7)
    if task_code == "f41":
        return [[4 if cell == 4 else 0 for cell in row] for row in grid]
    if task_code == "f58":
        return [[9 if cell == 1 else 1 if cell == 9 else cell for cell in row] for row in grid]
    raise KeyError(f"unknown task code: {task_code}")


def build_forced_memory(
    episodes: list[MemoryEpisode],
    consolidate_fn: ConsolidateFn,
) -> str:
    """Rewrite one shared memory after every new episode."""
    memory = ""
    for episode in episodes:
        prompt = "\n\n".join(
            [
                "You maintain a compact long-term memory for future grid tasks.",
                "Rewrite the memory after reading the new solved episode.",
                f"Return at most {FORCED_MAX_ITEMS} numbered bullets.",
                "Generalize aggressively across episodes and remove redundant details.",
                "Keep task codes only when they identify rules that cannot be safely merged.",
                f"Current memory:\n{memory or '(empty)'}",
                f"New solved episode:\n{format_episode(episode)}",
                "Return only the rewritten memory.",
            ]
        )
        memory = consolidate_fn(prompt, 512).strip()
    return memory


def build_gated_memories(
    episodes: list[MemoryEpisode],
    consolidate_fn: ConsolidateFn,
) -> dict[str, str]:
    """Build one family-specific abstract note, avoiding cross-family rewrites."""
    grouped: dict[str, list[MemoryEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.task_code, []).append(episode)

    memories: dict[str, str] = {}
    for task_code, items in sorted(grouped.items()):
        prompt = "\n\n".join(
            [
                "Write one family-specific memory note for future grid tasks.",
                f"Task code: {task_code}",
                "Use only the solved episodes below. Preserve the applicability condition.",
                "\n\n".join(format_episode(item) for item in items),
                "Return exactly one concise sentence that starts with the task code.",
            ]
        )
        memories[task_code] = consolidate_fn(prompt, 256).strip()
    return memories


def build_solve_prompt(case: EvalCase, memory_context: str) -> str:
    return "\n\n".join(
        [
            "You solve small ARC-style grid transformations.",
            "Colors are integers from 0 to 9. The task code is opaque; infer its rule from evidence.",
            "Use the current task examples and any long-term memory from the same benchmark.",
            f"Memory:\n{memory_context or '(none)'}",
            f"Task code: {case.task_code}",
            "Current task examples:",
            "\n\n".join(format_episode(example) for example in case.examples),
            f"Input grid:\n{json.dumps(case.input_grid, separators=(',', ':'))}",
            'Return only JSON in this exact shape: {"output": [[...], ...]}',
        ]
    )


def format_episode(episode: MemoryEpisode) -> str:
    return "\n".join(
        [
            f"Episode: {episode.id}",
            f"Task code: {episode.task_code}",
            f"Input grid: {json.dumps(episode.input_grid, separators=(',', ':'))}",
            f"Correct output grid: {json.dumps(episode.output_grid, separators=(',', ':'))}",
        ]
    )


def parse_output_grid(answer: str) -> Grid:
    """Parse a model answer as either {"output": grid} or a bare grid."""
    candidates = _json_candidates(answer)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("output")
        if _is_grid(parsed):
            return [[int(cell) for cell in row] for row in parsed]
    raise ValueError("answer did not contain a valid output grid")


def aggregate_metrics(
    policy_correct: dict[str, list[bool]],
    forced_memory: str,
    gated_memories: dict[str, str],
    families: list[RuleFamily],
    update_steps: int,
) -> dict[str, float]:
    forced_acc = _accuracy(policy_correct.get("forced_abstraction", []))
    episodic_acc = _accuracy(policy_correct.get("episodic_only", []))
    gated_acc = _accuracy(policy_correct.get("gated_abstraction", []))
    no_memory_acc = _accuracy(policy_correct.get("no_memory", []))
    no_memory = policy_correct.get("no_memory", [])
    forced = policy_correct.get("forced_abstraction", [])
    regressions = sum(1 for base, forced_ok in zip(no_memory, forced) if base and not forced_ok)
    total = len(forced)
    family_ids = [family.id for family in families]
    present_codes = sum(1 for task_code in family_ids if task_code in forced_memory)
    return {
        "acc,none": forced_acc,
        "forced_abstraction_acc,none": forced_acc,
        "episodic_only_acc,none": episodic_acc,
        "gated_abstraction_acc,none": gated_acc,
        "no_memory_acc,none": no_memory_acc,
        "forced_delta_vs_no_memory,none": round(forced_acc - no_memory_acc, 4),
        "forced_delta_vs_episodic,none": round(forced_acc - episodic_acc, 4),
        "episodic_advantage,none": round(episodic_acc - forced_acc, 4),
        "regression_rate,none": round(regressions / total, 4) if total else 0.0,
        "forced_memory_code_coverage,none": round(
            present_codes / len(family_ids), 4
        ) if family_ids else 0.0,
        "forced_memory_collapse_rate,none": round(
            1.0 - (present_codes / len(family_ids)), 4
        ) if family_ids else 0.0,
        "forced_memory_chars,none": float(len(forced_memory)),
        "gated_memory_chars,none": float(sum(len(value) for value in gated_memories.values())),
        "update_steps,none": float(update_steps),
        "n_cases,none": float(total),
    }


def _memory_for_case(
    policy: str,
    case: EvalCase,
    episodes: list[MemoryEpisode],
    forced_memory: str,
    gated_memories: dict[str, str],
) -> str:
    if policy == "no_memory":
        return ""
    if policy == "forced_abstraction":
        return forced_memory
    if policy == "gated_abstraction":
        return gated_memories.get(case.task_code, "")
    if policy == "episodic_only":
        matching = [episode for episode in episodes if episode.task_code == case.task_code]
        return "\n\n".join(format_episode(episode) for episode in matching[:DEFAULT_EPISODES_PER_CASE])
    raise ValueError(f"unknown memory policy: {policy}")


def _answer_with_model(model_call: ConsolidateFn) -> AnswerFn:
    def answer(_case: EvalCase, prompt: str, _policy: str, _memory_context: str) -> str:
        return model_call(prompt, 512)

    return answer


def _model_call_factory(
    base_url: str,
    model_label: str,
    timeout_s: int,
    api_key: str | None,
) -> ConsolidateFn:
    def call(prompt: str, max_tokens: int) -> str:
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            json={
                "model": model_label,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a deterministic benchmark participant. "
                            "Follow the requested output format exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            headers=_auth_headers(api_key),
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    return call


def _replace(grid: Grid, source: int, target: int) -> Grid:
    return [[target if cell == source else cell for cell in row] for row in grid]


def _grid_variant(index: int) -> Grid:
    base = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [2, 4, 6, 1, 9],
        [9, 8, 4, 2, 6],
        [1, 3, 5, 7, 0],
    ]
    shift = index % len(base)
    rotated_rows = base[shift:] + base[:shift]
    if index % 2:
        return [list(reversed(row)) for row in rotated_rows]
    return [row[:] for row in rotated_rows]


def _json_candidates(answer: str) -> list[str]:
    stripped = answer.strip()
    candidates = [stripped]
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    obj_start, obj_end = stripped.find("{"), stripped.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(stripped[obj_start : obj_end + 1])
    arr_start, arr_end = stripped.find("["), stripped.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(stripped[arr_start : arr_end + 1])
    return candidates


def _is_grid(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        if not all(isinstance(cell, int) for cell in row):
            return False
    return True


def _accuracy(flags: list[bool]) -> float:
    return round(sum(1 for flag in flags if flag) / len(flags), 4) if flags else 0.0


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
