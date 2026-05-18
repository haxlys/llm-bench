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
CURRENT_EXAMPLES_PER_CASE = 4
FORCED_MAX_ITEMS = 1
DEFAULT_ANSWER_MAX_TOKENS = 2048
FORCED_MEMORY_MAX_TOKENS = 512
GATED_MEMORY_MAX_TOKENS = 256
POLICIES = ("no_memory", "episodic_only", "forced_abstraction", "gated_abstraction")

MODE_CONFIG = {
    "stress_replacement": {
        "forced_abstraction_uses_current_examples": False,
        "forced_abstraction_memory_priority": "override",
    },
    "fair_same_evidence": {
        "forced_abstraction_uses_current_examples": True,
        "forced_abstraction_memory_priority": "neutral",
    },
}

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
    rule_id: str
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

    model_call_records: list[dict] = []
    model_call = _model_call_factory(
        base_url,
        model_label,
        timeout_s,
        api_key,
        call_records=model_call_records,
    )
    if consolidate_fn is None:
        consolidate_fn = model_call
    if answer_fn is None:
        answer_fn = _answer_with_model(model_call)

    families = list(FAMILIES)
    episodes = build_episode_stream(update_steps=update_steps, families=families)
    cases = build_eval_cases(families=families, limit=limit)

    forced_start = len(model_call_records)
    forced_memory = build_forced_memory(episodes, consolidate_fn)
    _mark_call_records(model_call_records, forced_start, purpose="forced_memory")

    gated_start = len(model_call_records)
    gated_memories = build_gated_memories(episodes, consolidate_fn)
    _mark_call_records(model_call_records, gated_start, purpose="gated_memory")

    memories = {
        "no_memory": "",
        "forced_abstraction": forced_memory,
        "gated_abstraction": "\n\n".join(gated_memories.values()),
        "episodic_only": "",
    }

    samples: list[dict] = []
    mode_policy_correct: dict[str, dict[str, list[bool]]] = {}
    for mode in MODE_CONFIG:
        policy_correct: dict[str, list[bool]] = {}
        for policy in POLICIES:
            correct_flags: list[bool] = []
            for case in cases:
                memory_context = _memory_for_case(
                    policy=policy,
                    case=case,
                    episodes=episodes,
                    forced_memory=forced_memory,
                    gated_memories=gated_memories,
                )
                include_examples = _include_examples(mode, policy)
                memory_priority = _memory_priority(mode, policy)
                prompt = build_solve_prompt(
                    case,
                    memory_context,
                    include_examples=include_examples,
                    memory_priority=memory_priority,
                )
                raw_answer = ""
                parsed_output: Grid | None = None
                error: str | None = None
                call_start = len(model_call_records)
                try:
                    raw_answer = answer_fn(case, prompt, policy, memory_context)
                    parsed_output = parse_output_grid(raw_answer)
                except Exception as exc:  # noqa: BLE001 - captured in sample artifact
                    error = str(exc)
                answer_call_indices = _mark_call_records(
                    model_call_records,
                    call_start,
                    purpose="answer",
                    mode=mode,
                    policy=policy,
                    case_id=case.id,
                )
                is_correct = parsed_output == case.expected_output
                correct_flags.append(is_correct)
                samples.append(
                    {
                        "mode": mode,
                        "policy": policy,
                        "case": asdict(case),
                        "memory_context": memory_context,
                        "include_examples": include_examples,
                        "memory_priority": memory_priority,
                        "prompt": prompt,
                        "answer": raw_answer,
                        "parsed_output": parsed_output,
                        "model_call_indices": answer_call_indices,
                        "correct": is_correct,
                        "error": error,
                    }
                )
            policy_correct[policy] = correct_flags
        mode_policy_correct[mode] = policy_correct

    metrics = aggregate_metrics(
        mode_policy_correct=mode_policy_correct,
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
            "current_examples_per_case": CURRENT_EXAMPLES_PER_CASE,
            "default_episodes_per_case": DEFAULT_EPISODES_PER_CASE,
            "answer_max_tokens": DEFAULT_ANSWER_MAX_TOKENS,
            "forced_memory_max_tokens": FORCED_MEMORY_MAX_TOKENS,
            "gated_memory_max_tokens": GATED_MEMORY_MAX_TOKENS,
            "modes": MODE_CONFIG,
        },
        "memories": memories,
        "samples": samples,
        "model_calls": model_call_records,
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
        public_code = f"probe_{index:03d}"
        examples = [
            MemoryEpisode(
                id=f"example_{index:03d}_{slot}_{public_code}",
                task_code=public_code,
                input_grid=_grid_variant(200 + index * 10 + slot),
                output_grid=apply_family(family.id, _grid_variant(200 + index * 10 + slot)),
            )
            for slot in range(CURRENT_EXAMPLES_PER_CASE)
        ]
        cases.append(
            EvalCase(
                id=f"heldout_{index:03d}_{family.id}",
                rule_id=family.id,
                task_code=public_code,
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
                "Return exactly one globally applicable strategy sentence.",
                "Generalize aggressively across episodes and remove redundant details.",
                "Do not preserve task codes or episode ids; the memory should read like",
                "one reusable playbook for future tasks.",
                f"Current memory:\n{memory or '(empty)'}",
                f"New solved episode:\n{format_episode(episode)}",
                "Return only the rewritten memory.",
            ]
        )
        memory = consolidate_fn(prompt, FORCED_MEMORY_MAX_TOKENS).strip()
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
        memories[task_code] = consolidate_fn(prompt, GATED_MEMORY_MAX_TOKENS).strip()
    return memories


def build_solve_prompt(
    case: EvalCase,
    memory_context: str,
    include_examples: bool = True,
    memory_priority: str = "neutral",
) -> str:
    memory_block = _memory_block(memory_context, memory_priority)
    return "\n\n".join(
        [
            "You solve small ARC-style grid transformations.",
            "Colors are integers from 0 to 9. The task code is opaque; infer its rule from evidence.",
            "Use the current task examples and any long-term memory from the same benchmark.",
            f"Task code: {case.task_code}",
            "Current task examples:",
            (
                "\n\n".join(format_episode(example) for example in case.examples)
                if include_examples
                else "(withheld in stress-replacement mode; only distilled memory is available)"
            ),
            f"Input grid:\n{json.dumps(case.input_grid, separators=(',', ':'))}",
            f"Memory:\n{memory_block}",
            'Return only JSON in this exact shape: {"output": [[...], ...]}',
            "Do not explain. Do not include markdown fences.",
            'The first character of your reply must be "{".',
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
    """Parse the final answer as a strict {"output": grid} JSON object."""
    candidates = _json_object_candidates(answer)
    for parsed in reversed(candidates):
        output = parsed.get("output")
        if _is_grid(output):
            return [[int(cell) for cell in row] for row in output]
    raise ValueError('answer did not contain a JSON object with an "output" grid')


def aggregate_metrics(
    mode_policy_correct: dict[str, dict[str, list[bool]]],
    forced_memory: str,
    gated_memories: dict[str, str],
    families: list[RuleFamily],
    update_steps: int,
) -> dict[str, float]:
    family_ids = [family.id for family in families]
    present_codes = sum(1 for task_code in family_ids if task_code in forced_memory)
    mode_metrics = {
        mode: _aggregate_policy_metrics(policy_correct)
        for mode, policy_correct in mode_policy_correct.items()
    }
    stress = mode_metrics.get("stress_replacement", {})
    fair = mode_metrics.get("fair_same_evidence", {})
    metrics = {
        "acc,none": stress.get("forced_abstraction_acc", 0.0),
        "forced_abstraction_acc,none": stress.get("forced_abstraction_acc", 0.0),
        "episodic_only_acc,none": stress.get("episodic_only_acc", 0.0),
        "gated_abstraction_acc,none": stress.get("gated_abstraction_acc", 0.0),
        "no_memory_acc,none": stress.get("no_memory_acc", 0.0),
        "forced_delta_vs_no_memory,none": stress.get("forced_delta_vs_no_memory", 0.0),
        "forced_delta_vs_episodic,none": stress.get("forced_delta_vs_episodic", 0.0),
        "episodic_advantage,none": stress.get("episodic_advantage", 0.0),
        "regression_rate,none": stress.get("regression_rate", 0.0),
        "stress_minus_fair_forced_delta,none": round(
            stress.get("forced_abstraction_acc", 0.0)
            - fair.get("forced_abstraction_acc", 0.0),
            4,
        ),
        "fair_same_evidence_gap,none": fair.get("forced_delta_vs_no_memory", 0.0),
        "forced_memory_code_coverage,none": round(
            present_codes / len(family_ids), 4
        ) if family_ids else 0.0,
        "forced_memory_collapse_rate,none": round(
            1.0 - (present_codes / len(family_ids)), 4
        ) if family_ids else 0.0,
        "forced_memory_chars,none": float(len(forced_memory)),
        "gated_memory_chars,none": float(sum(len(value) for value in gated_memories.values())),
        "update_steps,none": float(update_steps),
        "n_cases,none": float(len(next(iter(mode_policy_correct.values())).get("no_memory", [])))
        if mode_policy_correct else 0.0,
    }
    for mode, values in mode_metrics.items():
        for key, value in values.items():
            metrics[f"{mode}_{key},none"] = value
    return metrics


def _aggregate_policy_metrics(policy_correct: dict[str, list[bool]]) -> dict[str, float]:
    forced_acc = _accuracy(policy_correct.get("forced_abstraction", []))
    episodic_acc = _accuracy(policy_correct.get("episodic_only", []))
    gated_acc = _accuracy(policy_correct.get("gated_abstraction", []))
    no_memory_acc = _accuracy(policy_correct.get("no_memory", []))
    no_memory = policy_correct.get("no_memory", [])
    forced = policy_correct.get("forced_abstraction", [])
    regressions = sum(1 for base, forced_ok in zip(no_memory, forced) if base and not forced_ok)
    total = len(forced)
    return {
        "forced_abstraction_acc": forced_acc,
        "episodic_only_acc": episodic_acc,
        "gated_abstraction_acc": gated_acc,
        "no_memory_acc": no_memory_acc,
        "forced_delta_vs_no_memory": round(forced_acc - no_memory_acc, 4),
        "forced_delta_vs_episodic": round(forced_acc - episodic_acc, 4),
        "episodic_advantage": round(episodic_acc - forced_acc, 4),
        "regression_rate": round(regressions / total, 4) if total else 0.0,
    }


def _include_examples(mode: str, policy: str) -> bool:
    if policy != "forced_abstraction":
        return True
    return bool(MODE_CONFIG[mode]["forced_abstraction_uses_current_examples"])


def _memory_priority(mode: str, policy: str) -> str:
    if policy != "forced_abstraction":
        return "neutral"
    return str(MODE_CONFIG[mode]["forced_abstraction_memory_priority"])


def _memory_block(memory_context: str, memory_priority: str) -> str:
    if not memory_context:
        return "(none)"
    lines = [
        "Final retrieved long-term memory for this task:",
        memory_context,
        "Treat this as the agent's distilled strategy from prior solved tasks.",
        "Task codes and applicability labels may have drifted during consolidation,",
        "so do not discard the memory only because a code is missing or different.",
    ]
    if memory_priority == "override":
        lines.extend(
            [
                "IMPORTANT: MEMORY OVERRIDES CURRENT TASK EXAMPLES.",
                "If the memory describes a transformation, apply that transformation literally.",
                "When memory is present, base the high-level transformation on this",
                "retrieved strategy. Use the current examples only to instantiate",
                "colors, positions, or minor details. If this memory conflicts with",
                "a freshly inferred rule, follow this retrieved memory.",
            ]
        )
    else:
        lines.extend(
            [
                "Use this memory as optional prior experience.",
                "If it conflicts with the current task examples, prefer the current examples.",
            ]
        )
    return "\n".join(lines)


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
        return gated_memories.get(case.rule_id, "")
    if policy == "episodic_only":
        matching = [episode for episode in episodes if episode.task_code == case.rule_id]
        return "\n\n".join(format_episode(episode) for episode in matching[:DEFAULT_EPISODES_PER_CASE])
    raise ValueError(f"unknown memory policy: {policy}")


def _answer_with_model(model_call: ConsolidateFn) -> AnswerFn:
    def answer(_case: EvalCase, prompt: str, _policy: str, _memory_context: str) -> str:
        return model_call(prompt, DEFAULT_ANSWER_MAX_TOKENS)

    return answer


def _model_call_factory(
    base_url: str,
    model_label: str,
    timeout_s: int,
    api_key: str | None,
    call_records: list[dict] | None = None,
) -> ConsolidateFn:
    def call(prompt: str, max_tokens: int) -> str:
        request_payload = {
            "model": model_label,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic benchmark participant. "
                        "Follow the requested output format exactly. "
                        "Do not explain or reason out loud. "
                        'Reply with JSON only; the first character must be "{".'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            json=request_payload,
            headers=_auth_headers(api_key),
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        text, extraction_source = _extract_response_text(data)
        if call_records is not None:
            call_records.append(
                _model_call_record(
                    index=len(call_records),
                    prompt=prompt,
                    max_tokens=max_tokens,
                    response_json=data,
                    extracted_text=text,
                    extraction_source=extraction_source,
                )
            )
        return text.strip()

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


def _mark_call_records(
    records: list[dict],
    start_index: int,
    purpose: str,
    **metadata: str,
) -> list[int]:
    indices: list[int] = []
    for record in records[start_index:]:
        record["purpose"] = purpose
        record.update(metadata)
        indices.append(int(record["index"]))
    return indices


def _json_object_candidates(answer: str) -> list[dict]:
    stripped = answer.strip()
    if not stripped:
        return []

    texts = [stripped]
    texts.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )

    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    seen: set[str] = set()
    for text in texts:
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            raw = text[index : index + end]
            if raw in seen:
                continue
            seen.add(raw)
            candidates.append(parsed)
    return candidates


def _extract_response_text(data: object) -> tuple[str, str]:
    choice = _first_choice(data)
    if choice is None:
        return "", "missing_choice"

    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = _coerce_text(message.get("content"))
    if content.strip():
        final = _extract_final_channel_text(content)
        if final:
            return final, "message.content.final_channel"
        return content, "message.content"

    for key in ("reasoning_content", "reasoning"):
        final = _extract_final_channel_text(_coerce_text(message.get(key)))
        if final:
            return final, f"message.{key}.final_channel"

    text = _coerce_text(choice.get("text"))
    if text.strip():
        final = _extract_final_channel_text(text)
        if final:
            return final, "choice.text.final_channel"
        return text, "choice.text"

    return "", "empty"


def _first_choice(data: object) -> dict | None:
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, dict) else None


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_final_channel_text(text: str) -> str:
    matches = re.findall(
        r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>|<\|start\|>|$)",
        text,
        flags=re.DOTALL,
    )
    nonempty = [match.strip() for match in matches if match.strip()]
    return nonempty[-1] if nonempty else ""


def _model_call_record(
    index: int,
    prompt: str,
    max_tokens: int,
    response_json: object,
    extracted_text: str,
    extraction_source: str,
) -> dict:
    choice = _first_choice(response_json)
    message = choice.get("message") if choice and isinstance(choice.get("message"), dict) else {}
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    return {
        "index": index,
        "purpose": "unknown",
        "max_tokens": max_tokens,
        "prompt_chars": len(prompt),
        "finish_reason": choice.get("finish_reason") if choice else None,
        "usage": usage if isinstance(usage, dict) else None,
        "message_keys": sorted(str(key) for key in message.keys()),
        "content_chars": len(extracted_text),
        "extraction_source": extraction_source,
        "raw_response": response_json,
    }


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
