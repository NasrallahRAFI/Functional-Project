"""Create a training-only 20-positive/20-negative severe-channel listening and spectrogram review pack."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from dronedet.config import config_from_mapping
from dronedet.cnn_v3.augment import AugmentConfig, ChannelAugment, NoiseBank, stable_rng
from dronedet.cnn_v3.dataset import frame_aligned_crop
from dronedet.features import fit_length, load_wave
from dronedet.splits import guard_test_access


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def build(root: Path) -> None:
    doc=yaml.safe_load((root/"config/cnn_v3.yaml").read_text(encoding="utf-8"))
    fe=config_from_mapping(doc["frontend"]); data=root/"results/cnn_v2/data"
    manifest=pd.read_csv(data/"manifests/train.csv",keep_default_na=False)
    guard_test_access("train",allow_test=False)
    rng=np.random.default_rng(20261126)
    picked=[]
    for label in (1,0):
        rows=manifest[manifest.label.astype(int)==label]
        # Spread positives across families and negatives across distinct groups.
        key="family" if label else "group"
        unique=rows.groupby(key,sort=True).head(1).reset_index(drop=True)
        chosen=unique.iloc[rng.choice(len(unique),size=20,replace=False)] if len(unique)>=20 else rows.sample(20,random_state=20261126)
        picked.extend(chosen.to_dict("records"))
    rng.shuffle(picked)
    raw_value=os.environ.get("DRONE_AUDIO_DATASET_ROOT",doc["raw_root"])
    raw_root=Path(raw_value); raw_root=(root/raw_root if not raw_root.is_absolute() else raw_root).resolve()/"Binary_Drone_Audio"
    cfg=AugmentConfig(sample_rate=fe.sr,p_clean_recipe=0.0,p_band_limit=1.0,p_eq=1.0,p_soft_clip=1.0,
        p_reverb=1.0,p_noise=0.6,p_floor=1.0,hp_hz=(400.0,600.0),lp_hz=(3500.0,5500.0),
        gain_db=(-30.0,6.0),rt60_s=(0.7,1.0),drr_db=(-5.0,10.0),noise_snr_db=(0.0,10.0),
        floor_dbfs=(-75.0,-45.0),minimum_positive_snr_db=0.0)
    augmenter=ChannelAugment(NoiseBank(),cfg)
    output=root/"results/cnn_v3/review"; output.mkdir(parents=True,exist_ok=True)
    clips=[]; records=[]; gap=np.zeros(int(.35*fe.sr),dtype=np.float32); cursor=0
    fig,axes=plt.subplots(5,8,figsize=(20,12),constrained_layout=True)
    for i,row in enumerate(picked):
        clip_id=str(row["clip_id"]); source=raw_root/Path(*clip_id.split("/")); original=fit_length(load_wave(source,fe),fe)
        aug,r=augmenter(original,stable_rng(20261126,7,clip_id,draw_id=i),label=int(row["label"]))
        crop_rng=stable_rng(20261126,8,clip_id,draw_id=i)
        crop,start_frame=frame_aligned_crop(aug,fe.win_samples,fe.hop,crop_rng,training=True)
        crop=np.asarray(crop,dtype=np.float32)
        clips.append(crop); start_sec=cursor/fe.sr; end_sec=(cursor+len(crop))/fe.sr
        records.append({"sample":i+1,"label":"drone" if int(row["label"]) else "no_drone",
            "clip_id":clip_id,"group":str(row["group"]),"family":str(row["family"]),
            "source_sha256":sha(source),"crop_start_frame":start_frame,
            "montage_start_seconds":start_sec,"montage_end_seconds":end_sec,
            "augmentation_parameters":json.dumps(r,sort_keys=True)})
        ax=axes.flat[i]
        ax.specgram(crop,NFFT=512,Fs=fe.sr,noverlap=384,scale="dB",mode="psd",cmap="magma")
        ax.set_title(f"{i+1:02d} {'DRONE' if int(row['label']) else 'NO DRONE'}",fontsize=8)
        ax.set_ylim(0,8000); ax.set_xticks([]); ax.set_yticks([])
        cursor+=len(crop)+len(gap)
        clips.append(gap)
    if len(clips)>0: clips=clips[:-1]
    montage=np.concatenate(clips)
    montage_peak=float(np.max(np.abs(montage)))
    playback_scale=min(1.0,0.89/max(montage_peak,1e-12))
    montage=np.asarray(montage*playback_scale,dtype=np.float32)
    wav=output/"A4_severe_training_review_montage.wav"
    sf.write(wav,montage,fe.sr,subtype="PCM_16")
    csv=output/"A4_severe_training_review_index.csv"
    pd.DataFrame(records).to_csv(csv,index=False,lineterminator="\n")
    png=output/"A4_severe_training_review_spectrograms.png"
    fig.suptitle("A4 severe channel augmentation — 20 training positives and 20 training negatives",fontsize=15)
    fig.savefig(png,dpi=160); plt.close(fig)
    report={"status":"READY_FOR_HUMAN_REVIEW","count":len(records),"positive_count":sum(r["label"]=="drone" for r in records),
        "negative_count":sum(r["label"]=="no_drone" for r in records),"rung":"A4 severe stress; crop-first review only",
        "config_sha256":cfg.sha256,"montage_sha256":sha(wav),"index_sha256":sha(csv),"spectrogram_sha256":sha(png),
        "train_manifest_sha256":sha(data/"manifests/train.csv"),"shared_playback_scale":playback_scale,
        "peak_after_shared_scale":float(np.max(np.abs(montage))),"test_access":False}
    (output/"REVIEW_REPORT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":build(Path(__file__).resolve().parents[1])
