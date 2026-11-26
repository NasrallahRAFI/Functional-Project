"""Export a self-contained, single-CNN V2.0 inference package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from dronedet.bundle import verify_bundle, write_bundle
from dronedet.config import config_from_mapping
from dronedet.models.cnn import build_cnn


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def export(config_path: Path, freeze_path: Path) -> None:
    cp, fp = config_path.resolve(), freeze_path.resolve()
    doc = yaml.safe_load(cp.read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    root = cp.parent.parent
    data = root / doc["output_root"] / "data"
    training = root / doc["output_root"] / "training"
    out = root / doc["output_root"] / "export" / "CNN_V2.0"
    freeze = json.loads(fp.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_TEST":
        raise RuntimeError("export requires the pre-test immutable freeze record")
    ckpt = fp.parent / freeze["checkpoint"]
    if sha256(ckpt) != freeze["checkpoint_sha256"] or sha256(data / "DATASET_READY.json") != freeze["dataset_ready_sha256"]:
        raise RuntimeError("export checkpoint or data identity differs from freeze")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite CNN V2 package: {out}")
    result_path = root / doc["output_root"] / "evaluation" / "report.json"
    if not result_path.is_file():
        raise FileNotFoundError("the one frozen CNN V2 evaluation report is required before export")
    eval_report = json.loads(result_path.read_text(encoding="utf-8"))
    if eval_report.get("status") != "COMPLETE" or eval_report.get("checkpoint_sha256") != freeze["checkpoint_sha256"]:
        raise RuntimeError("evaluation report does not belong to the frozen CNN checkpoint")

    stage_parent = out.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="cnn_v2_export_", dir=stage_parent))
    try:
        golden = root / "data/interim/frontend_bundle_v2_migration/golden"
        golden_audio = {}
        for name, rate in (("golden16", 16000), ("golden48", 48000)):
            wav = golden / f"{name}_audio.wav"
            info = sf.info(wav)
            if info.samplerate != rate or info.subtype != "FLOAT":
                raise RuntimeError(f"{name} golden fixture encoding/rate changed")
            audio, actual_rate = sf.read(wav, dtype="float32", always_2d=True)
            if actual_rate != rate:
                raise RuntimeError(f"{name} golden fixture sample rate changed")
            golden_audio[name] = audio
        stats_doc = json.loads((data / "normalization.json").read_text(encoding="utf-8"))
        stats = {"mode": stats_doc["mode"], "mean": np.asarray(stats_doc["mean"], dtype=np.float32),
                 "std": np.asarray(stats_doc["std"], dtype=np.float32),
                 "fit_record": stats_doc["fit_record"], "policy": stats_doc["fit_record"]["policy"]}
        write_bundle(stage, fe, stats, golden_audio)
        # Replace generated golden files with the already verified independent
        # expectations and exact native-rate FLOAT fixture files.
        for name in ("golden16", "golden48"):
            for suffix in ("_audio.wav", "_db.npy", "_normalized.npy"):
                shutil.copy2(golden / f"{name}{suffix}", stage / "golden" / f"{name}{suffix}")

        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        model = build_cnn((1, 64, 28), channels=(16, 32, 64, 64), kernel_size=3,
                          pool=(2, 2), head="global_average", dropout=0.3)
        model.load_state_dict(payload["state_dict"]); model.eval()
        scripted = torch.jit.trace(model, torch.zeros((1, 1, 64, 28), dtype=torch.float32), strict=True)
        scripted_path = stage / "cnn_v2_torchscript.pt"
        scripted.save(str(scripted_path))
        threshold = float(freeze["validation_threshold"]["threshold"])
        package = {"package": "CNN V2.0", "representation": "single_convolutional_neural_network",
                   "architecture": freeze["architecture"], "parameter_count": freeze["parameter_count"],
                   "model_file": scripted_path.name, "model_sha256": sha256(scripted_path),
                   "checkpoint_sha256": freeze["checkpoint_sha256"], "score_space": "raw_logit",
                   "validation_threshold": threshold, "threshold_rule": "maximum validation F1; ties choose larger raw logit",
                   "validation_metrics": freeze["validation_metrics"],
                   "input_shape": [1, 64, 28], "label_mapping": {"0": "unknown", "1": "yes_drone"},
                   "frontend_config_file": "config.json", "normalization_file": "stats.npz",
                   "normalization_policy": "train_stats_eligible_per_band_floor_mask_v1",
                   "statistics_sha256": sha256(data / "normalization_stats.npz"),
                   "training_seed": freeze["seed"], "best_epoch": freeze["best_epoch"],
                   "dataset_ready_sha256": freeze["dataset_ready_sha256"],
                   "operational_contract": "Read channel 0, use the bundled 16 kHz/64-band frontend, select centered 28-frame window, normalize with bundled train statistics, feed float32 (B,1,64,28), compare raw logit to the bundled validation threshold.",
                   "evaluation_disclosure": "Test benchmark reused from V1; not an independent unseen test. See project evaluation report.",
                   "external_paths_required": False}
        (stage / "model_metadata.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        (stage / "MODEL_CARD.md").write_text(
            "# CNN V2.0\n\nOne depth_4 CNN (60,641 parameters). Input is a normalized float32 tensor shaped `(B, 1, 64, 28)`. The exported TorchScript model returns raw logits; use the validation-selected threshold in `model_metadata.json`. Frontend and normalization settings are bundled. The V2 test benchmark reuses the previously scored V1 split and is not an independent unseen test.\n",
            encoding="utf-8")
        # The existing frontend verifier checks config/statistics identity and
        # recomputes both native-rate goldens against their preserved expected arrays.
        frontend_check = verify_bundle(stage, fe)
        rng = np.random.default_rng(42)
        sample = torch.from_numpy(rng.normal(size=(5, 1, 64, 28)).astype(np.float32))
        with torch.no_grad():
            eager = model(sample).cpu().numpy()
            bundled = torch.jit.load(str(scripted_path), map_location="cpu").eval()(sample).cpu().numpy()
        parity = float(np.max(np.abs(eager - bundled)))
        if not np.isfinite(bundled).all() or parity > 1e-6:
            raise RuntimeError(f"TorchScript inference parity failed: max_abs={parity}")
        # A separate Python process proves the model artifact loads without any
        # training or V1 checkpoint path.
        code = ("import sys,torch; m=torch.jit.load(sys.argv[1],map_location='cpu').eval(); "
                "x=torch.zeros((1,1,64,28),dtype=torch.float32); "
                "y=m(x); assert y.shape==(1,) and torch.isfinite(y).all(); print(float(y[0]))")
        fresh = subprocess.run([sys.executable, "-c", code, str(scripted_path)], check=True,
                               capture_output=True, text=True)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                model(sample[:1])
        model_ms = (time.perf_counter() - start) * 1000 / 20
        import importlib.metadata as md
        import librosa
        from dronedet.features import file_to_power
        t0 = time.perf_counter()
        for name in ("golden16", "golden48"):
            file_to_power(golden / f"{name}_audio.wav", fe)
        frontend_ms = (time.perf_counter() - t0) * 1000 / 2
        libraries = json.loads((stage / "libraries.json").read_text(encoding="utf-8"))
        libraries.update(model_weights=package["model_sha256"], validation_threshold=threshold)
        (stage / "libraries.json").write_text(json.dumps(libraries, indent=2) + "\n", encoding="utf-8")
        software_report = {"frontend_verification": frontend_check,
                           "torchscript_eager_max_abs": parity,
                           "fresh_process_stdout": fresh.stdout.strip(),
                           "frontend_ms_per_0_9s_golden": frontend_ms,
                           "cnn_forward_ms_per_sample": model_ms,
                           "timing_note": "CNN inference on synthetic normalized tensors; frontend timing on both native-rate golden WAVs.",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "libraries": {name: md.version(name) for name in
                                         ("numpy", "scipy", "librosa", "soxr", "soundfile", "torch")}}
        (stage / "software_verification.json").write_text(json.dumps(software_report, indent=2) + "\n", encoding="utf-8")
        records = []
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                records.append({"path": p.relative_to(stage).as_posix(), "size": p.stat().st_size,
                                "sha256": sha256(p)})
        (stage / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError(f"refusing to overwrite CNN V2 package: {out}")
        os.replace(stage, out)
        print(json.dumps({"status": "EXPORTED_AND_VERIFIED", "package": str(out),
                          "model_sha256": package["model_sha256"], "frontend_parity": frontend_check,
                          "eager_jit_max_abs": parity, "frontend_ms_per_golden": frontend_ms,
                          "cnn_forward_ms": model_ms}, indent=2))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    export(args.config, args.freeze)


if __name__ == "__main__":
    main()
