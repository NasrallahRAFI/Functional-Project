"""V5.1-only configuration adapter around the unchanged frontend mapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import yaml

from dronedet.config import FrontEndConfig, cache_content_hash, config_from_mapping, transform_hash

ROOT_KEYS = {"frontend", "decision", "inputs", "sampling", "augmentation", "replay", "approval"}
REQUIRED_KEYS = ROOT_KEYS
CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_CONTRACT.json"


@dataclass(frozen=True)
class V51Config:
    frontend: FrontEndConfig
    decision: dict[str, Any]
    inputs: dict[str, Any]
    sampling: dict[str, Any]
    augmentation: dict[str, Any]
    replay: dict[str, Any]
    approval: dict[str, Any]


def load_v51_config(path: str | Path) -> V51Config:
    """Validate V5.1 sections, then map only frontend/decision into V4 helper."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("V5.1 configuration must be a YAML mapping")
    unknown = set(raw) - ROOT_KEYS
    missing = REQUIRED_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(f"invalid V5.1 root keys; missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key in sorted(ROOT_KEYS - {"frontend", "decision"}):
        if not isinstance(raw[key], dict):
            raise ValueError(f"V5.1 {key} must be a mapping")
    if not isinstance(raw["frontend"], dict) or not isinstance(raw["decision"], dict):
        raise ValueError("frontend and decision must be mappings")

    # The shared V4 mapper deliberately sees only the two sections it accepts.
    shared = config_from_mapping({"frontend": raw["frontend"], "decision": raw["decision"]})
    if shared.sr != 16_000 or shared.channel != 0 or shared.resample_quality != "HQ":
        raise ValueError("V5.1 requires 16 kHz, channel 0 and soxr HQ")
    if shared.n_fft != 1024 or shared.hop != 256 or shared.n_mels != 64:
        raise ValueError("V5.1 frontend dimensions differ from the approved contract")
    if shared.win_samples != 7_936 or shared.n_win != 28:
        raise ValueError("V5.1 frontend window must be 7,936 samples and 28 mel frames")
    if shared.fmin != 50.0 or shared.fmax != 7_500.0:
        raise ValueError("V5.1 mel range must be 50–7,500 Hz")
    if shared.gain_range_db != (-12.0, 12.0):
        raise ValueError("historical frontend gain identity must retain V4's range")
    if transform_hash(shared) != "bb7a21a1e5fb317c43e1e214bc04af6e46e04d75053383d1c62ef6849c9466c7":
        raise ValueError("V5.1 transform identity differs from the approved processing identity")
    if cache_content_hash(shared) != "026ff2ecaa8a240383af8666f61bbfcd9ea7ef3d4e9a51c73e803d82259a43b0":
        raise ValueError("V5.1 power-cache processing identity differs from the approved identity")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if raw["sampling"] != contract["sampling"]:
        raise ValueError("sampling settings must match the approved V5.1 contract")
    if raw["augmentation"] != contract["augmentation"]:
        raise ValueError("augmentation settings must match the approved V5.1 contract")
    if raw["replay"] != contract["replay"]:
        raise ValueError("replay settings must match the approved V5.1 contract")
    if set(raw["inputs"]) - set(contract["source_codes"]):
        raise ValueError("inputs contains an unknown source code")
    for source, source_input in raw["inputs"].items():
        if not isinstance(source_input, dict) or set(source_input) - {"source_revision", "status", "path_key"}:
            raise ValueError(f"invalid inputs entry for source {source!r}")
    approval = raw["approval"]
    if set(approval) != {"o0", "plan_sha256", "contract_sha256"}:
        raise ValueError("approval must bind O0 and the exact plan/contract hashes")
    if approval["o0"] is not True:
        raise ValueError("V5.1 execution requires recorded O0")
    if approval["plan_sha256"] != "6a27c99f960e4e0520709216cf204e5890eca6e30ebbb438d7934f300af9f5a1":
        raise ValueError("approval plan hash does not match the approved plan")
    if approval["contract_sha256"] != "6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4":
        raise ValueError("approval contract hash does not match the approved contract")
    return V51Config(
        frontend=shared,
        decision=dict(raw["decision"]),
        inputs=dict(raw["inputs"]),
        sampling=dict(raw["sampling"]),
        augmentation=dict(raw["augmentation"]),
        replay=dict(raw["replay"]),
        approval=dict(raw["approval"]),
    )
