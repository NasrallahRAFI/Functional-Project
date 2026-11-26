"""Training-only severe-channel spectral survival calibration for CNN V3."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dronedet.config import config_from_mapping
from dronedet.cnn_v3.augment import AugmentConfig, ChannelAugment, NoiseBank, stable_rng
from dronedet.features import fit_length, load_wave
from dronedet.splits import guard_test_access


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def band_share(x: np.ndarray, sr: int) -> float:
    a=np.asarray(x,dtype=np.float64)
    # Average overlapping Hann-window spectra avoids a single-crop phase artifact.
    nfft=2048; hop=512; win=np.hanning(nfft)
    if a.size<nfft: a=np.pad(a,(0,nfft-a.size))
    starts=range(0,max(1,a.size-nfft+1),hop)
    total=np.zeros(nfft//2+1,dtype=np.float64)
    count=0
    for start in starts:
        frame=a[start:start+nfft]
        if len(frame)<nfft: frame=np.pad(frame,(0,nfft-len(frame)))
        spec=np.fft.rfft(frame*win)
        total+=np.abs(spec)**2; count+=1
    if count==0 or total.sum()<=0: return 0.0
    f=np.fft.rfftfreq(nfft,d=1/sr)
    mask=(f>=50.0)&(f<=7500.0)
    return float(total[mask].sum()/total.sum())


def calibrate(root: Path) -> dict:
    doc=yaml.safe_load((root/"config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe=config_from_mapping(doc["frontend"])
    data=root/"results/cnn_v2/data"
    manifest=pd.read_csv(data/"manifests/train.csv",keep_default_na=False)
    guard_test_access("train",allow_test=False)
    positives=manifest[manifest.label.astype(int)==1].reset_index(drop=True)
    raw_value=os.environ.get("DRONE_AUDIO_DATASET_ROOT",doc["raw_root"])
    raw_root=Path(raw_value); raw_root=(root/raw_root if not raw_root.is_absolute() else raw_root).resolve()/"Binary_Drone_Audio"
    # Maximum registered severe ranges; p_clean=0 tests every source under stress.
    cfg=AugmentConfig(sample_rate=fe.sr,p_clean_recipe=0.0,
        p_band_limit=1.0,p_eq=1.0,p_soft_clip=1.0,p_reverb=1.0,p_noise=0.6,p_floor=1.0,
        hp_hz=(400.0,600.0),lp_hz=(3500.0,5500.0),gain_db=(-30.0,6.0),
        rt60_s=(0.7,1.0),drr_db=(-5.0,10.0),noise_snr_db=(0.0,10.0),floor_dbfs=(-75.0,-45.0),
        minimum_positive_snr_db=0.0)
    aug=ChannelAugment(NoiseBank(),cfg)
    records=[]
    for i,row in positives.iterrows():
        path=raw_root/Path(*str(row.clip_id).split("/"))
        clean=fit_length(load_wave(path,fe),fe)
        changed,params=aug(clean,stable_rng(20261126,0,str(row.clip_id)),label=1)
        before=band_share(clean,fe.sr); after=band_share(changed,fe.sr)
        retained=after/max(before,1e-12)
        records.append({"clip_id":str(row.clip_id),"group":str(row.group),"family":str(row.family),
                        "clean_model_band_share":before,"severe_model_band_share":after,
                        "normalized_band_energy_retained":retained,
                        "measured_added_noise_snr_db":params.get("measured_added_noise_snr_db")})
        if (i+1)%200==0: print(f"calibrated {i+1}/{len(positives)} training positives",flush=True)
    frame=pd.DataFrame(records)
    passed=(float((frame.normalized_band_energy_retained>=0.01).mean())>=0.90 and
            float(frame.normalized_band_energy_retained.quantile(.05))>=0.01)
    out=root/"results/cnn_v3/calibration"; out.mkdir(parents=True,exist_ok=True)
    csv=out/"severe_positive_spectral_survival.csv"; frame.to_csv(csv,index=False,lineterminator="\n")
    by_family=frame.groupby("family").normalized_band_energy_retained.agg(["count","median","min"]).reset_index().to_dict("records")
    report={"status":"PASS" if passed else "FAIL","split":"train","positive_count":len(frame),
        "positive_groups":int(frame.group.nunique()),"family_summary":by_family,
        "fraction_retaining_at_least_1pct":float((frame.normalized_band_energy_retained>=0.01).mean()),
        "fifth_percentile_retained":float(frame.normalized_band_energy_retained.quantile(.05)),
        "minimum_retained":float(frame.normalized_band_energy_retained.min()),
        "config_sha256":cfg.sha256,"augmentation_code_sha256":sha(root/"src/dronedet/cnn_v3/augment.py"),
        "calibration_script_sha256":sha(Path(__file__).resolve()),"records_sha256":sha(csv),"test_access":False,
        "interpretation":"fraction of each waveform's normalized 0-8 kHz spectral energy retained within the model's 50-7500 Hz band, relative to its clean share"}
    (out/"severe_positive_spectral_survival.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2)); return report


if __name__=="__main__": calibrate(Path(__file__).resolve().parents[1])
