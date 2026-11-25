"""Verify correction W1/W2 against installed artifacts without replacing them."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dronedet.config import config_hash, load_project_config
from dronedet.data_preparation import measure_resampler_edge
from dronedet.features import fit_cached_normalizer, load_cache_index, load_power_cache
from run_frontend_correction_baseline import sha256


def verify(*, execute_notebook: bool, measure_passband: bool) -> dict:
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    baseline_path = paths["interim"] / "frontend_correction_v1_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    protected_before = {name: sha256(ROOT / name) for name in baseline["artifact_sha256"]}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_sha256": sha256(baseline_path),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "relevant_code_sha256": {str(path.relative_to(ROOT)): sha256(path)
                  for path in [ROOT / "src/dronedet/normalization.py", ROOT / "src/dronedet/features.py",
                               ROOT / "src/dronedet/cache_metadata.py",
                               ROOT / "src/dronedet/data_preparation.py", ROOT / "src/dronedet/resample.py",
                               ROOT / "notebooks/02_feature_extraction.ipynb"]},
              "test_audio_or_array_payloads_opened": False}
    train = pd.read_csv(paths["splits"] / "train.csv", keep_default_na=False)
    index = load_cache_index(paths, "train")
    cache = load_power_cache(paths, "train")
    started = time.perf_counter()
    fitted = fit_cached_normalizer(cache, index, train.to_dict("records"), config)
    elapsed = time.perf_counter() - started
    with np.load(paths["spec"] / "norm_stats_global.npz", allow_pickle=False) as archive:
        original = {"mean": archive["mean"], "std": archive["std"]}
    reproduction = {"fit_record": fitted.fit_record, "seconds": elapsed,
                    "mean_exact": bool(np.array_equal(fitted.mean, original["mean"])),
                    "std_exact": bool(np.array_equal(fitted.std, original["std"])),
                    "max_mean_abs_error": float(np.max(abs(fitted.mean - original["mean"]))),
                    "max_std_abs_error": float(np.max(abs(fitted.std - original["std"]))),
                    "statistics_bearing_hash": config_hash(config, {"mean": fitted.mean, "std": fitted.std})}
    reproduction["pass"] = (reproduction["mean_exact"] and reproduction["std_exact"]
        and reproduction["statistics_bearing_hash"] == baseline["statistics_bearing_hash"]
        and fitted.excluded_count == 88)
    report["masked_fit"] = reproduction
    print(json.dumps({"masked_fit": {key: value for key, value in reproduction.items() if key != "fit_record"}}), flush=True)
    del cache, fitted

    if execute_notebook:
        namespace = {"__name__": "__frontend_reproduction__"}
        notebook = json.loads((ROOT / "notebooks/02_feature_extraction.ipynb").read_text(encoding="utf-8"))
        output = io.StringIO()
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(output):
                for position, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"notebook02:cell{position}", "exec"), namespace)
            errors = {}
            for split in namespace["BUILD_SPLITS"]:
                isolated = namespace["OUTPUT_ROOT"] / "data/processed"
                actual = np.load(isolated / "spectrograms" / f"mel_power_{split}.npy", mmap_mode="r", allow_pickle=False)
                expected = load_power_cache(paths, split)
                power_error = 0.0
                for start in range(0, actual.shape[1], 8192):
                    power_error = max(power_error, float(np.max(abs(actual[:, start:start + 8192] - expected[:, start:start + 8192]))))
                with np.load(isolated / "features_tabular" / f"{split}.npz", allow_pickle=False) as regenerated, \
                        np.load(paths["tab"] / f"{split}.npz", allow_pickle=False) as installed:
                    tabular_error = float(np.max(abs(regenerated["X"] - installed["X"])))
                errors[split] = {"max_power_abs_error": power_error, "max_tabular_abs_error": tabular_error,
                                 "shape": list(actual.shape)}
                if hasattr(actual, "_mmap"):
                    actual._mmap.close()
                if hasattr(expected, "_mmap"):
                    expected._mmap.close()
            with np.load(namespace["OUTPUT_ROOT"] / "data/processed/spectrograms/norm_stats_global.npz",
                         allow_pickle=False) as regenerated_stats:
                statistics_exact = all(np.array_equal(regenerated_stats[key], original[key])
                                       for key in ("mean", "std"))
            report["notebook02"] = {"seconds": time.perf_counter() - started,
                "stdout": output.getvalue(), "splits": errors,
                "saved_statistics_exact": statistics_exact,
                "pass": statistics_exact and all(v["max_power_abs_error"] == 0 and v["max_tabular_abs_error"] == 0 for v in errors.values())}
        finally:
            temporary = namespace.get("_output_directory")
            if temporary is not None:
                temporary.cleanup()
        print("Isolated notebook02 completed", flush=True)

    if measure_passband:
        started = time.perf_counter()
        measured = measure_resampler_edge(out_sr=config.sr, quality=config.resample_quality)
        report["swept_passband"] = {**measured, "seconds": time.perf_counter() - started,
            "pass": abs(measured["f1db"] - config.measured_passband_hz) <= 1e-8 and measured["fmax"] == config.fmax}
        print(json.dumps({"passband_edge_hz": measured["f1db"], "fmax": measured["fmax"]}), flush=True)
    differences = []
    for name, previous in protected_before.items():
        after = sha256(ROOT / name)
        if after != previous or previous != baseline["artifact_sha256"][name]:
            differences.append(name)
    report["protected_artifacts"] = {"count": len(protected_before), "changed_since_baseline": differences,
                                      "pass": not differences}
    report["status"] = "PASS" if all(v.get("pass", True) for v in report.values() if isinstance(v, dict)) else "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true")
    parser.add_argument("--passband", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/frontend_correction_v1_reproduction.json")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {arguments.output}")
    result = verify(execute_notebook=arguments.notebook, measure_passband=arguments.passband)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "record": str(arguments.output),
                      "protected_artifacts": result["protected_artifacts"]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
