"""Check A3/A4 total added-noise SNR using train-only clips and donors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dronedet.config import config_from_mapping
from dronedet.cnn_v3.augment import AugmentConfig,ChannelAugment,stable_rng
from dronedet.features import fit_length,load_wave
from dronedet.splits import guard_test_access
from scripts.train_cnn_v3_rung import noise_bank_for_training,sha


def calibrate(root:Path)->dict:
    doc=yaml.safe_load((root/"config/cnn_v3.yaml").read_text(encoding="utf-8")); fe=config_from_mapping(doc["frontend"])
    data=root/"results/cnn_v2/data"; manifest=pd.read_csv(data/"manifests/train.csv",keep_default_na=False)
    guard_test_access("train",allow_test=False)
    env=os.environ.get("DRONE_AUDIO_DATASET_ROOT",doc["raw_root"]); raw=Path(env); raw=(root/raw if not raw.is_absolute() else raw).resolve()
    seed=20261126; bank=noise_bank_for_training(root,raw,manifest,fe,seed)
    rng=np.random.default_rng(seed^0x51A); chosen=[]
    for label in (1,0):
        rows=manifest[manifest.label.astype(int)==label].reset_index(drop=True)
        chosen.extend(rows.iloc[rng.choice(len(rows),size=100,replace=False)].to_dict("records"))
    base=doc["augmentation"]; stage=doc["rungs"]["A4"]
    cfg=AugmentConfig(sample_rate=fe.sr,p_clean_recipe=float(base["clean_recipe_probability"]),
        p_band_limit=stage["p_band_limit"],p_eq=stage["p_eq"],p_soft_clip=stage["p_soft_clip"],
        p_reverb=stage["p_reverb"],p_noise=stage["p_noise"],p_floor=stage["p_floor"],
        gain_db=tuple(stage["gain_db"]),clean_gain_db=tuple(base["clean_gain_db"]),
        noise_snr_db=tuple(base["background_snr_db"]),floor_dbfs=tuple(base["microphone_floor_dbfs"]),
        minimum_positive_snr_db=float(base["minimum_added_snr_db"]))
    aug=ChannelAugment(bank,cfg); records=[]
    for i,row in enumerate(chosen):
        path=raw/"Binary_Drone_Audio"/Path(*str(row["clip_id"]).split("/")); clean=fit_length(load_wave(path,fe),fe)
        _,params=aug(clean,stable_rng(seed,0,str(row["clip_id"]),i),label=int(row["label"]))
        snr=params.get("measured_added_noise_snr_db")
        records.append({"clip_id":row["clip_id"],"group":row["group"],"label":int(row["label"]),
            "branch":params["branch"],"measured_added_noise_snr_db":snr,
            "nuisance_limited":params.get("nuisance_limited"),"noise_source":params.get("noise_source"),
            "requested_background_snr_db":params.get("requested_background_snr_db")})
    frame=pd.DataFrame(records)
    channel=frame[(frame.branch=="channel") & frame.measured_added_noise_snr_db.notna()]
    passed=bool(len(channel)>0 and (channel.measured_added_noise_snr_db>=-1e-5).all())
    out=root/"results/cnn_v3/calibration"; out.mkdir(parents=True,exist_ok=True)
    csv=out/"a4_training_added_snr_calibration.csv"; frame.to_csv(csv,index=False,lineterminator="\n")
    report={"status":"PASS" if passed else "FAIL","samples":len(frame),
        "positive_samples":int((frame.label==1).sum()),"negative_samples":int((frame.label==0).sum()),
        "channel_branch_samples":len(channel),"v2_recipe_samples":int((frame.branch=="v2_recipe").sum()),
        "channel_branch_min_added_snr_db":float(channel.measured_added_noise_snr_db.min()) if len(channel) else None,
        "channel_branch_fifth_percentile_added_snr_db":float(channel.measured_added_noise_snr_db.quantile(.05)) if len(channel) else None,
        "nuisance_limited_fraction":float(channel.nuisance_limited.astype(bool).mean()) if len(channel) else None,
        "noise_donors_used":len(bank.donors),"noise_donor_manifest_sha256":sha(data/"manifests/train.csv"),
        "augmentation_config_sha256":cfg.sha256,"augmentation_code_sha256":sha(root/"src/dronedet/cnn_v3/augment.py"),
        "records_sha256":sha(csv),"split":"train","test_access":False}
    (out/"a4_training_added_snr_calibration.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2)); return report


if __name__=="__main__":calibrate(Path(__file__).resolve().parents[1])
