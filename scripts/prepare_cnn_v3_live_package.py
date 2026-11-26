"""Export the frozen A1 seed-42 checkpoint as a locally activated V3 package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.bundle import verify_bundle
from dronedet.config import FrontEndConfig
from dronedet.models.cnn import build_cnn, parameter_count


GUARD_FAILURES = [
    "clean_recall_at_least_159_of_185",
    "clean_false_positives_at_most_5_of_1325",
    "each_augmented_far_at_most_1pct",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/cnn_v3/experiments/A1_seed42")
    parser.add_argument("--base-package", type=Path, default=ROOT / "runtime/assets/CNN_V2.0")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/assets/CNN_V3.0")
    parser.add_argument(
        "--experimental-activation", action="store_true",
        help="required to package this A1 candidate despite its failed validation guards",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    record_path = run_dir / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("run_id") != "A1_seed42" or record.get("architecture") != "V2 depth_4 fixed" or record.get("seed") != 42:
        raise RuntimeError("only the registered A1 seed-42 depth_4 run can be exported")
    if record.get("test_access") is not False:
        raise RuntimeError("A1 record does not prove that test access remained false")
    eligible = bool(record.get("eligible"))
    if not eligible and not args.experimental_activation:
        raise RuntimeError("A1 failed validation guards; pass --experimental-activation only for an explicit experimental local activation")
    if not eligible and not record.get("acceptance_guards"):
        raise RuntimeError("A1 validation guard results are missing")
    failures = [key for key, passed in record["acceptance_guards"].items() if not passed]
    if not eligible and len(failures) != len(GUARD_FAILURES):
        raise RuntimeError(f"unexpected A1 guard failure set: {failures}")

    checkpoint_path = run_dir / record["checkpoint"]
    checkpoint_sha = sha(checkpoint_path)
    if checkpoint_sha != record["checkpoint_sha256"]:
        raise RuntimeError("A1 checkpoint differs from its frozen run record")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_cnn(input_shape=(1, 64, 28), channels=(16, 32, 64, 64),
                      kernel_size=3, pool=(2, 2), head="global_average", dropout=0.3)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if parameter_count(model) != 60641:
        raise RuntimeError("A1 checkpoint architecture parameter count changed")

    base_package = args.base_package.resolve()
    base_metadata = json.loads((base_package / "model_metadata.json").read_text(encoding="utf-8"))
    if base_metadata.get("model_sha256") != sha(base_package / base_metadata["model_file"]):
        raise RuntimeError("base V2 frontend package model hash does not verify")
    suite_dir = ROOT / "results/cnn_v3/validation_suite"
    suite_report = json.loads((suite_dir / "TENSOR_CACHE_REPORT.json").read_text(encoding="utf-8"))
    suite_manifest = json.loads((suite_dir / "SUITE_MANIFEST.json").read_text(encoding="utf-8"))
    if suite_manifest.get("status") != "FROZEN" or suite_report.get("status") != "PASS":
        raise RuntimeError("frozen validation tensor suite is unavailable")
    if suite_report.get("tensors_sha256") != sha(suite_dir / "tensors.npz"):
        raise RuntimeError("frozen validation tensor suite checksum mismatch")

    threshold = float(record["threshold"]["threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("A1 threshold is invalid")
    with np.load(suite_dir / "tensors.npz", allow_pickle=False) as archive:
        for condition in ("clean", "mild", "moderate", "severe"):
            x = torch.from_numpy(archive[f"x_{condition}"].copy())
            y = archive[f"y_{condition}"].astype(np.int64)
            if x.shape != (1510, 1, 64, 28):
                raise RuntimeError(f"unexpected frozen {condition} suite shape")
            with torch.inference_mode():
                logits = model(x).cpu().numpy().astype(np.float64)
            predicted = logits >= threshold
            expected = record["condition_metrics"][condition]
            counts = {
                "tp": int(np.sum((y == 1) & predicted)),
                "fp": int(np.sum((y == 0) & predicted)),
                "fn": int(np.sum((y == 1) & ~predicted)),
                "tn": int(np.sum((y == 0) & ~predicted)),
            }
            if any(counts[name] != int(expected[name]) for name in counts):
                raise RuntimeError(f"A1 checkpoint does not reproduce recorded {condition} validation counts: {counts}")

    traced = torch.jit.trace(model, torch.zeros((1, 1, 64, 28), dtype=torch.float32), check_trace=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing runtime package: {output}")
    with tempfile.TemporaryDirectory(prefix="cnn-v3-a1-package-", dir=output.parent) as temporary:
        staging = Path(temporary) / "CNN_V3.0"
        shutil.copytree(base_package, staging)
        old_model = staging / base_metadata["model_file"]
        old_model.unlink()
        model_file = "cnn_v3_a1_torchscript.pt"
        model_path = staging / model_file
        traced.save(str(model_path))

        metadata = dict(base_metadata)
        metadata.update({
            "package": "CNN V3.0",
            "model_file": model_file,
            "model_sha256": sha(model_path),
            "checkpoint_sha256": checkpoint_sha,
            "stage": "A1",
            "training_seed": 42,
            "best_epoch": int(record["best_epoch"]),
            "validation_threshold": threshold,
            "threshold_rule": "weighted pooled validation F1 across clean and three frozen severity conditions; ties choose higher raw logit",
            "validation_eligible": eligible,
            "experimental_activation": not eligible,
            "validation_guard_failures": failures,
            "validation_metrics": {
                "weighted_f1": float(record["threshold"]["f1"]),
                "weighted_pooled_pr_auc": float(record["threshold"]["weighted_pooled_pr_auc"]),
                "condition_metrics": record["condition_metrics"],
                "acceptance_guards": record["acceptance_guards"],
                "eligible": eligible,
            },
            "run_id": "A1_seed42",
            "run_record_sha256": sha(record_path),
            "config_sha256": record["config_sha256"],
            "suite_manifest_sha256": record["suite_manifest_sha256"],
            "test_access": False,
            "operational_contract": (
                "Experimental local activation of the A1 seed-42 depth_4 checkpoint. "
                "Uses the V2 16 kHz/64-band frontend and training normalization; compare raw logits "
                "to the weighted validation-F1 threshold and use the existing three-window policy. "
                "The candidate failed registered validation guards and is not validation-eligible."
            ),
            "evaluation_disclosure": (
                "Experimental local activation despite failed validation eligibility. No V3 test data "
                "was accessed; this package has no independent final-test acceptance claim."
            ),
        })
        write_json(staging / "model_metadata.json", metadata)
        (staging / "MODEL_CARD.md").write_text(
            "# CNN V3.0 A1 · experimental\n\n"
            "This is the single depth_4 CNN trained with the A1 wider-gain recipe (seed 42). "
            "It is activated locally at the user's explicit request despite failing registered validation guards.\n\n"
            "Validation threshold: `" + f"{threshold:.12g}" + "`. Clean recall is 157/185 and clean false positives are 25/1,325. "
            "Moderate and severe false-alarm rates are 2.49% and 5.21%, above the 1% limits. "
            "This package is experimental and validation-ineligible. No test data was opened.\n\n"
            "The verified V2 frontend, normalization statistics, sample-rate goldens and three-window live policy are reused.\n",
            encoding="utf-8",
        )

        frontend_config = json.loads((staging / metadata["frontend_config_file"]).read_text(encoding="utf-8"))
        frontend_config.pop("_identity_schema", None)
        frontend_config["gain_range_db"] = tuple(frontend_config["gain_range_db"])
        frontend_check = verify_bundle(staging, FrontEndConfig(**frontend_config))
        if any(not value.get("verified") for value in frontend_check["normalized_outputs"].values()):
            raise RuntimeError("reused V2 frontend goldens failed in the V3 package")

        sample = torch.zeros((4, 1, 64, 28), dtype=torch.float32)
        with torch.inference_mode():
            eager = model(sample).cpu().numpy()
            scripted = traced(sample).cpu().numpy()
        max_abs = float(np.max(np.abs(eager - scripted)))
        if not np.isfinite(max_abs) or max_abs > 1e-6:
            raise RuntimeError(f"TorchScript differs from A1 eager output by {max_abs}")
        write_json(staging / "software_verification.json", {
            "status": "PASS",
            "frontend_verification": frontend_check,
            "torchscript_eager_max_abs": max_abs,
            "validation_suite_counts_reproduced": True,
            "validation_eligible": eligible,
            "experimental_activation": not eligible,
            "test_access": False,
        })

        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "PACKAGE_MANIFEST.json":
                files.append({"path": path.relative_to(staging).as_posix(), "size": path.stat().st_size, "sha256": sha(path)})
        write_json(staging / "PACKAGE_MANIFEST.json", {
            "schema_version": 1,
            "model_id": "CNN_V3.0",
            "model_version": "3.0",
            "run_id": "A1_seed42",
            "checkpoint_sha256": checkpoint_sha,
            "model_sha256": sha(model_path),
            "validation_eligible": eligible,
            "experimental_activation": not eligible,
            "validation_guard_failures": failures,
            "test_access": False,
            "files": files,
        })
        staging.rename(output)

    print(json.dumps({
        "status": "PASS",
        "package": str(output),
        "model_sha256": sha(output / model_file),
        "checkpoint_sha256": checkpoint_sha,
        "frontend_hash": frontend_check["hash"],
        "validation_eligible": eligible,
        "experimental_activation": not eligible,
        "guard_failures": failures,
        "test_access": False,
        "package_files": len(files),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
