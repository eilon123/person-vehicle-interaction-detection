import importlib.util
import json
from pathlib import Path


def load_tool(name):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_experiment_number_ignores_unrelated_directories(tmp_path):
    tool = load_tool("run_experiment")
    (tmp_path / "experiment_1").mkdir()
    (tmp_path / "experiment_4").mkdir()
    (tmp_path / "experiment_draft").mkdir()
    assert tool.next_number(tmp_path) == 5


def test_history_contains_experiment_details(tmp_path):
    tool = load_tool("experiment_history")
    run = tmp_path / "experiment_3"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({
        "experiment_name": "experiment_3", "experiment_change": "new sampler", "sample_frames": 20
    }), encoding="utf-8")
    (run / "manual_binary_metrics.json").write_text(json.dumps({
        "event_metrics": [{"tiou_threshold": .5, "type_aware": False, "tp": 2, "fp": 1, "fn": 3,
                           "precision": 2/3, "recall": .4, "f1": .5,
                           "per_clip": {"clip": {"tp": 2, "fp": 1, "fn": 3, "f1": .5}}}],
        "continuous_temporal_overlap": {"temporal_iou": .25}
    }), encoding="utf-8")
    result = tool.build(tmp_path)
    assert "Experiment 3" in result
    assert "new sampler" in result
    assert "sample_frames" in result
