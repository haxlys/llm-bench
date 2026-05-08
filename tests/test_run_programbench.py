"""Tests for the ProgramBench run-and-import wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    path = ROOT / "scripts" / "run_programbench.py"
    spec = importlib.util.spec_from_file_location("run_programbench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_programbench_eval_command_includes_core_options(tmp_path):
    mod = _load_script()
    source_dir = tmp_path / "submissions"
    output_dir = tmp_path / "raw"

    cmd = mod._programbench_eval_command(
        programbench_bin="uvx programbench",
        source_dir=source_dir,
        programbench_output_dir=output_dir,
        workers=4,
        branch_workers=2,
        docker_cpus=8,
        branch_retries=0,
        force=True,
        filter_spec="junegunn__fzf.*",
        slice_spec="0:5",
        image_tag="task_cleanroom",
    )

    assert cmd == [
        "uvx",
        "programbench",
        "eval",
        str(source_dir),
        "--workers",
        "4",
        "--branch-workers",
        "2",
        "--docker-cpus",
        "8",
        "--branch-retries",
        "0",
        "--image-tag",
        "task_cleanroom",
        "--output",
        str(output_dir),
        "--force",
        "--filter",
        "junegunn__fzf.*",
        "--slice",
        "0:5",
    ]


def test_evaluated_source_dir_matches_programbench_output_semantics(tmp_path):
    mod = _load_script()

    assert mod._evaluated_source_dir(
        tmp_path / "agent-run",
        tmp_path / "programbench-output",
    ) == tmp_path / "programbench-output" / "agent-run"


def test_programbench_env_sets_blob_dir(tmp_path):
    mod = _load_script()

    env = mod._programbench_env(tmp_path / "blobs")

    assert env["PROGRAMBENCH_BLOB_DIR"] == str(tmp_path / "blobs")
