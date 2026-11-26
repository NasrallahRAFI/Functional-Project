"""Export the frozen ONE_CHANNEL_seed42 checkpoint beside existing releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from prepare_cnn_v3_live_package import ROOT, write_json
from dronedet.bundle import verify_bundle
from dronedet.config import FrontEndConfig
from dronedet.models.cnn import build_cnn, parameter_count


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    project = ROOT.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=project / "runs/study-v3/experiments/ONE_CHANNEL_seed42")
    parser.add_argument("--base-package", type=Path, default=ROOT / "runtime/assets/CNN_V2.0")
    parser.add_argument("--output", type=Path, default=project / "releases/study-v3/ONE_CHANNEL_seed42")
    parser.add_argument("--experimental-activation", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    record_path = run_dir / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("run_id") != "ONE_CHANNEL_seed42" or record.get("seed") != 42
            or record.get("architecture") != "V2 depth_4 fixed"):
        raise RuntimeError("export requires the registered ONE_CHANNEL_seed42 depth_4 run")
    if record.get("test_access") is not False:
        raise RuntimeError("run record does not prove held-out test remained locked")
    eligible = bool(record.get("eligible"))
    if not eligible and not args.experimental_activation:
        raise RuntimeError("validation-ineligible export requires the owner's explicit experimental activation")
    guards = record.get("acceptance_guards") or {}
    failures = [key for key, passed in guards.items() if not passed]
    if not guards or len(guards) != 5:
        raise RuntimeError("the five frozen validation guard results are required")

    checkpoint_path = run_dir / record["checkpoint"]
    checkpoint_sha = sha(checkpoint_path)
    if checkpoint_sha != record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash differs from the frozen run record")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_cnn((1, 64, 28), channels=(16, 32, 64, 64), kernel_size=3,
                      pool=(2, 2), head="global_average", dropout=0.3)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if parameter_count(model) != 60641:
        raise RuntimeError("frozen CNN architecture changed")

    suite = ROOT / "results/cnn_v3/validation_suite"
    suite_report = json.loads((suite / "TENSOR_CACHE_REPORT.json").read_text(encoding="utf-8"))
    suite_manifest = json.loads((suite / "SUITE_MANIFEST.json").read_text(encoding="utf-8"))
    if suite_manifest.get("status") != "FROZEN" or suite_report.get("status") != "PASS":
        raise RuntimeError("frozen validation suite is unavailable")
    if suite_report.get("tensors_sha256") != sha(suite / "tensors.npz"):
        raise RuntimeError("validation suite tensor hash mismatch")
    threshold = float(record["threshold"]["threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("selected validation threshold is invalid")
    with np.load(suite / "tensors.npz", allow_pickle=False) as tensors:
        for condition in ("clean", "mild", "moderate", "severe"):
            x = torch.from_numpy(tensors[f"x_{condition}"].copy())
            y = tensors[f"y_{condition}"].astype(np.int64)
            if tuple(x.shape) != (1510, 1, 64, 28):
                raise RuntimeError(f"unexpected {condition} validation tensor shape")
            with torch.inference_mode():
                scores = model(x).cpu().numpy().reshape(-1)
            pred = scores >= threshold
            expected = record["condition_metrics"][condition]
            actual = {"tp": int(np.sum((y == 1) & pred)), "fp": int(np.sum((y == 0) & pred)),
                      "fn": int(np.sum((y == 1) & ~pred)), "tn": int(np.sum((y == 0) & ~pred))}
            if any(actual[k] != int(expected[k]) for k in actual):
                raise RuntimeError(f"{condition} validation counts did not reproduce: {actual}")

    base = args.base_package.resolve()
    base_meta = json.loads((base / "model_metadata.json").read_text(encoding="utf-8"))
    if base_meta.get("model_sha256") != sha(base / base_meta["model_file"]):
        raise RuntimeError("frozen V2 base package hash failed")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing package: {output}")
    traced = torch.jit.trace(model, torch.zeros((1, 1, 64, 28)), check_trace=True)
    with tempfile.TemporaryDirectory(prefix="cnn-v3-one-channel-", dir=output.parent) as tmp:
        staging = Path(tmp) / "ONE_CHANNEL_seed42"
        shutil.copytree(base, staging)
        (staging / base_meta["model_file"]).unlink()
        model_file = "cnn_v3_one_channel_seed42_torchscript.pt"
        traced.save(str(staging / model_file))
        metadata = dict(base_meta)
        metadata.update({
            "package": "CNN V3.0",
            "model_file": model_file,
            "model_sha256": sha(staging / model_file),
            "checkpoint_sha256": checkpoint_sha,
            "stage": "ONE_CHANNEL",
            "training_seed": 42,
            "best_epoch": int(record["best_epoch"]),
            "validation_threshold": threshold,
            "threshold_rule": "maximum weighted pooled validation F1 over clean and three frozen severity conditions; ties choose higher raw logit",
            "validation_eligible": eligible,
            "experimental_activation": bool(args.experimental_activation),
            "validation_guard_failures": failures,
            "validation_metrics": {"weighted_f1": float(record["threshold"]["f1"]),
                                   "weighted_pooled_pr_auc": float(record["threshold"]["weighted_pooled_pr_auc"]),
                                   "condition_metrics": record["condition_metrics"],
                                   "acceptance_guards": guards, "eligible": eligible},
            "run_id": "ONE_CHANNEL_seed42",
            "run_record_sha256": sha(record_path),
            "config_sha256": record["config_sha256"],
            "suite_manifest_sha256": record["suite_manifest_sha256"],
            "test_access": False,
            "final_test_passed": False,
            "operational_contract": "ONE_CHANNEL_seed42 uses the frozen V2 frontend and normalization, a single weighted-F1 raw-logit threshold, 0.496-second cadence, and the existing three-window majority policy.",
            "evaluation_disclosure": "The fixed validation suite was used for selection. No test data has been accessed by this training/export step.",
        })
        write_json(staging / "model_metadata.json", metadata)
        clean = record["condition_metrics"]["clean"]
        (staging / "MODEL_CARD.md").write_text(
            "# CNN V3.0 ONE_CHANNEL_seed42\n\n"
            "Single seed-42 depth_4 CNN trained with the preregistered combined A4 channel recipe.\n\n"
            f"Validation eligibility: **{str(eligible).lower()}**. Experimental activation: **{str(bool(args.experimental_activation)).lower()}**.\n\n"
            f"Selected weighted pooled F1 threshold: `{threshold:.12g}`. Clean validation recall: {clean['tp']}/{clean['tp'] + clean['fn']}; false positives: {clean['fp']}/1325.\n\n"
            f"Failed validation guards: {', '.join(failures) if failures else 'none'}. The V2 frontend, statistics, golden fixtures, and live decision policy are reused. Test data remains locked unless a separate gated final evaluation is authorized by passing validation.\n",
            encoding="utf-8")
        front_doc = json.loads((staging / metadata["frontend_config_file"]).read_text(encoding="utf-8"))
        front_doc.pop("_identity_schema", None)
        front_doc["gain_range_db"] = tuple(front_doc["gain_range_db"])
        front_check = verify_bundle(staging, FrontEndConfig(**front_doc))
        if any(not item.get("verified") for item in front_check["normalized_outputs"].values()):
            raise RuntimeError("bundled V2 frontend golden verification failed")
        sample = torch.zeros((4, 1, 64, 28), dtype=torch.float32)
        with torch.inference_mode():
            parity = float(np.max(np.abs(model(sample).numpy() - traced(sample).numpy())))
        if not np.isfinite(parity) or parity > 1e-6:
            raise RuntimeError(f"TorchScript/eager parity failed: {parity}")
        write_json(staging / "software_verification.json", {
            "status": "PASS", "frontend_verification": front_check,
            "torchscript_eager_max_abs": parity, "validation_suite_counts_reproduced": True,
            "validation_eligible": eligible, "experimental_activation": bool(args.experimental_activation),
            "test_access": False, "final_test_passed": False})
        files = [{"path": p.relative_to(staging).as_posix(), "size": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(staging.rglob("*")) if p.is_file() and p.name != "PACKAGE_MANIFEST.json"]
        write_json(staging / "PACKAGE_MANIFEST.json", {
            "schema_version": 1, "model_id": "CNN_V3.0", "model_version": "3.0",
            "stage": "ONE_CHANNEL", "run_id": "ONE_CHANNEL_seed42", "checkpoint_sha256": checkpoint_sha,
            "model_sha256": sha(staging / model_file), "validation_eligible": eligible,
            "experimental_activation": bool(args.experimental_activation), "validation_guard_failures": failures,
            "test_access": False, "final_test_passed": False, "files": files})
        staging.rename(output)
    print(json.dumps({"status": "PASS", "package": str(output), "model_sha256": sha(output / model_file),
                      "checkpoint_sha256": checkpoint_sha, "frontend_hash": front_check["hash"],
                      "validation_eligible": eligible, "experimental_activation": bool(args.experimental_activation),
                      "guard_failures": failures, "test_access": False, "package_files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
