from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dronedet.metrics import choose_f1_threshold
from dronedet.models.cnn import build_cnn, parameter_count
from scripts import evaluate_cnn_v2
from scripts import prepare_cnn_v2


def test_cnn_v2_recipe_uses_the_selected_depth_four_architecture():
    model = build_cnn((1, 64, 28), channels=(16, 32, 64, 64), kernel_size=3,
                      pool=(2, 2), head="global_average", dropout=0.3)
    assert parameter_count(model) == 60_641
    assert tuple(model.channels) == (16, 32, 64, 64)


@pytest.mark.parametrize(("value", "expected"),
                         [(True, True), (False, False), (1, True), (0, False),
                          ("true", True), ("False", False)])
def test_manifest_boolean_parser_is_explicit(value, expected):
    assert prepare_cnn_v2.as_bool(value) is expected


def test_manifest_boolean_parser_rejects_ambiguous_values():
    with pytest.raises(ValueError):
        prepare_cnn_v2.as_bool("yes")


def test_threshold_ties_choose_larger_raw_logit():
    selected = choose_f1_threshold(np.array([1, 1, 0, 0]), np.array([2.0, 1.0, 1.0, 1.0]))
    assert selected["threshold"] == 2.0


def test_v2_preparation_module_has_no_tabular_extractor_dependency():
    source = (ROOT / "scripts/prepare_cnn_v2.py").read_text(encoding="utf-8")
    forbidden = ("cache_split_tabular", "feature_names(", "tabular(", "features_tabular")
    assert not any(name in source for name in forbidden)


def test_group_bootstrap_resamples_groups_and_returns_complete_intervals():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    pred = np.array([0, 1, 1, 1], dtype=np.int8)
    groups = np.array(["a", "b", "c", "d"])
    result = evaluate_cnn_v2.group_bootstrap(y, pred, groups, seed=42, accepted=50)
    assert result["complete"]
    assert result["target_accepted_draws"] == 50
    assert all(metric["accepted_draws"] == 50 for metric in result["metrics"].values())


def test_safe_dataset_verify_does_not_read_test_payload(tmp_path, monkeypatch):
    import json
    import yaml

    config = ROOT / "config/cnn_v2.yaml"
    recipe = yaml.safe_load(config.read_text(encoding="utf-8"))
    output = ROOT / recipe["output_root"] / "data"
    ready = json.loads((output / "DATASET_READY.json").read_text(encoding="utf-8"))
    seen = []
    actual = prepare_cnn_v2.digest_file

    def guarded_digest(path):
        path = Path(path)
        assert "test" not in path.name, f"safe verification opened test artifact: {path}"
        seen.append(path.name)
        return actual(path)

    monkeypatch.setattr(prepare_cnn_v2, "digest_file", guarded_digest)
    prepare_cnn_v2.verify(config)
    assert seen
    assert not any("test" in name for name in seen)
    assert any(Path(r["path"]).name == "mel_power_test.npy" for r in ready["files"])


def test_failed_build_leaves_no_ready_dataset_or_staging_artifact(tmp_path, monkeypatch):
    import yaml

    project = tmp_path / "project"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    source_recipe = yaml.safe_load((ROOT / "config/cnn_v2.yaml").read_text(encoding="utf-8"))
    source_recipe.update(raw_root="absent/raw/source", reference_manifest="absent/reference.csv",
                         reference_splits="absent/splits", output_root="results/cnn_v2")
    config_path = config_dir / "cnn_v2.yaml"
    config_path.write_text(yaml.safe_dump(source_recipe), encoding="utf-8")
    manifest = __import__("pandas").DataFrame([{"clip_id": "unknown/a.wav", "split": "train"}])
    monkeypatch.setattr(prepare_cnn_v2, "_fresh_source_manifest",
                        lambda *args, **kwargs: (manifest, {"fixture": True, "inventory_count": 1}))

    def fail_during_split(*args, **kwargs):
        raise OSError("simulated interrupted cache producer")

    monkeypatch.setattr(prepare_cnn_v2, "_build_split", fail_during_split)
    with pytest.raises(OSError, match="simulated interrupted"):
        prepare_cnn_v2.build(config_path)
    output = project / "results/cnn_v2/data"
    assert not output.exists()
    assert not list(project.glob("cnn_v2_stage_*"))
