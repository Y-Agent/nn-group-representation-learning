"""Public CLI safeguards and the time/selection contract of the paper panels."""

import argparse
import importlib
import json
import sys

import pytest

from paper_reproduction.__main__ import render, training_commands
from paper_reproduction.settings import ASSETS


def test_imports_do_not_render_or_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in [
        "paper_reproduction",
        "paper_reproduction.distribution",
        "paper_reproduction.dynamics",
        "paper_reproduction.notebook_adapter",
        "scripts.reproduce",
        "scripts.stage2_from_checkpoint",
    ]:
        importlib.import_module(name)
    assert list(tmp_path.iterdir()) == []


def test_training_plan_uses_refined_stage1_for_both_stage2_runs(tmp_path):
    commands = training_commands(tmp_path, 2)
    assert len(commands) == 10
    assert all(command[0] == sys.executable for command in commands)
    assert all(str(tmp_path) in command for command in commands)
    assert [command[command.index("--step-divisor") + 1] for command in commands[-2:]] == ["1", "2"]
    assert all("stage2_from_checkpoint.py" in command[1] for command in commands[-2:])


def test_render_refuses_to_overwrite_existing_directory(tmp_path, monkeypatch):
    module = importlib.import_module("paper_reproduction.__main__")
    monkeypatch.setattr(module, "verify_inputs", lambda *args: {"inputs": {}})
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep")
    args = argparse.Namespace(runs=tmp_path / "runs", output=out, new_results=False)
    with pytest.raises(ValueError, match="not empty"):
        render(args)
    assert marker.read_text() == "keep"


def test_render_rejects_input_output_overlap(tmp_path, monkeypatch):
    module = importlib.import_module("paper_reproduction.__main__")
    monkeypatch.setattr(module, "verify_inputs", lambda *args: {"inputs": {}})
    runs = tmp_path / "runs"
    runs.mkdir()
    args = argparse.Namespace(runs=runs, output=runs / "nested", new_results=False)
    with pytest.raises(ValueError, match="overlaps inputs"):
        render(args)
    assert list(runs.iterdir()) == []


def test_asset_names_unique():
    assert len(set(ASSETS.values())) == len(ASSETS) == 11


def test_verify_rejects_changed_baseline(tmp_path, monkeypatch):
    module = importlib.import_module("paper_reproduction.__main__")
    monkeypatch.setattr(module.S, "ROOT", tmp_path)
    baseline = tmp_path / "reproduction/accepted"
    baseline.mkdir(parents=True)
    reference = baseline / "reference.txt"
    reference.write_text("accepted")
    (baseline.parent / "paper-contract.json").write_text(
        json.dumps({"baseline_files": {"reference.txt": module.sha(reference)}})
    )
    reference.write_text("changed")
    with pytest.raises(ValueError, match="baseline is missing or changed"):
        module.verify(argparse.Namespace(output=tmp_path / "unused"))
