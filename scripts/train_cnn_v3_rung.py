"""Train one registered CNN V3 augmentation rung on the fixed depth_4 CNN."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from dronedet.config import config_from_mapping
from dronedet.cnn_v3.augment import AugmentConfig, ChannelAugment, NoiseBank, NoiseDonor
from dronedet.cnn_v3.dataset import RawWaveformDataset
from dronedet.features import load_wave
from dronedet.metrics import metrics_at_threshold
from dronedet.models.cnn import build_cnn, parameter_count
from dronedet.splits import guard_test_access


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def weighted_f1_threshold(labels: dict[str,np.ndarray], logits: dict[str,np.ndarray]) -> dict:
    cw={"clean":0.5,"mild":1/6,"moderate":1/6,"severe":1/6}
    y=np.concatenate([labels[k] for k in cw]).astype(np.float64)
    s=np.concatenate([logits[k] for k in cw]).astype(np.float64)
    w=np.concatenate([np.full(len(labels[k]),cw[k]/len(labels[k])) for k in cw])
    best=(-1.0,float("-inf"),None)
    for t in np.unique(s):
        p=(s>=t).astype(np.float64)
        tp=float(np.sum(w*y*p)); fp=float(np.sum(w*(1-y)*p))
        fn=float(np.sum(w*y*(1-p))); tn=float(np.sum(w*(1-y)*(1-p)))
        precision=tp/max(tp+fp,1e-15); recall=tp/max(tp+fn,1e-15)
        f1=2*precision*recall/max(precision+recall,1e-15)
        if f1>best[0] or (f1==best[0] and t>best[1]):
            best=(f1,float(t),{"tp_weight":tp,"fp_weight":fp,"fn_weight":fn,"tn_weight":tn,
                               "precision":precision,"recall":recall,"f1":f1})
    return {"threshold":best[1],**best[2]}


def noise_bank_for_training(root: Path, raw_root: Path, manifest: pd.DataFrame, fe, seed: int, count: int=128) -> NoiseBank:
    guard_test_access("train",allow_test=False)
    negatives=np.flatnonzero(manifest.label.to_numpy(dtype=int)==0)
    rng=np.random.default_rng(seed^0xBADC0FFE)
    selected=np.sort(rng.choice(negatives,size=min(count,len(negatives)),replace=False))
    bank=NoiseBank()
    for i in selected:
        row=manifest.iloc[int(i)]
        path=raw_root/"Binary_Drone_Audio"/Path(*str(row.clip_id).split("/"))
        audio=load_wave(path,fe)
        digest=sha(path)
        try:
            bank.add(NoiseDonor(str(row.clip_id),str(row.group),audio,digest))
        except ValueError as e:
            if "excluded" not in str(e): raise
    return bank


def load_suite(root: Path):
    suite=root/"results/cnn_v3/validation_suite"
    manifest=suite/"SUITE_MANIFEST.json"
    manifest_doc=json.loads(manifest.read_text(encoding="utf-8"))
    manifest_sha=sha(manifest)
    sidecar=suite/"SUITE_MANIFEST.sha256"
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split()[0]!=manifest_sha:
        raise RuntimeError("fixed validation suite manifest checksum sidecar mismatch")
    expected_counts={k:1510 for k in ("mild","moderate","severe")}
    if manifest_doc.get("status")!="FROZEN" or manifest_doc.get("count_per_condition")!=expected_counts:
        raise RuntimeError("fixed validation suite manifest is incomplete or not frozen")
    if len(manifest_doc.get("examples",[]))!=4530:
        raise RuntimeError("fixed validation suite manifest example count is invalid")
    report_path=suite/"TENSOR_CACHE_REPORT.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status")!="PASS" or report.get("suite_manifest_sha256")!=manifest_sha:
        raise RuntimeError("frozen suite tensor cache is missing or stale")
    if report.get("tensors_sha256")!=sha(suite/"tensors.npz"):
        raise RuntimeError("frozen suite tensor cache checksum mismatch")
    archive=np.load(suite/"tensors.npz",allow_pickle=False)
    names=("clean","mild","moderate","severe")
    x={k:torch.from_numpy(archive[f"x_{k}"].copy()) for k in names}
    y={k:archive[f"y_{k}"].astype(np.int64) for k in names}
    clean_labels=y["clean"]
    for k in names:
        if x[k].shape!=(1510,1,64,28) or not torch.isfinite(x[k]).all():
            raise RuntimeError(f"fixed {k} validation tensor geometry or values are invalid")
        if not np.array_equal(y[k],clean_labels):
            raise RuntimeError(f"fixed {k} validation condition row labels differ from clean")
    if int((clean_labels==1).sum())!=185 or int((clean_labels==0).sum())!=1325:
        raise RuntimeError("fixed clean validation class counts changed")
    return x,y,manifest_sha,sha(suite/"tensors.npz")


def train(rung: str, seed: int, config_path: Path) -> dict:
    if rung not in {"A1","A2","A3","A4"}: raise ValueError("rung must be A1, A2, A3, or A4")
    cp=config_path.resolve(); doc=yaml.safe_load(cp.read_text(encoding="utf-8"))
    root=cp.parent.parent; fe=config_from_mapping(doc["frontend"])
    data=root/"results/cnn_v2/data"
    out=root/"results/cnn_v3"/"experiments"/f"{rung}_seed{seed}"
    out.mkdir(parents=True,exist_ok=True)
    train_manifest=pd.read_csv(data/"manifests/train.csv",keep_default_na=False)
    val_manifest=pd.read_csv(data/"manifests/val.csv",keep_default_na=False)
    guard_test_access("train",allow_test=False)
    guard_test_access("val",allow_test=False)
    if len(train_manifest)!=7607 or len(val_manifest)!=1510: raise RuntimeError("frozen split counts changed")
    stats=np.load(data/"normalization_stats.npz",allow_pickle=False)
    mean,std=stats["mean"].astype(np.float32),stats["std"].astype(np.float32)
    raw_value=__import__("os").environ.get("DRONE_AUDIO_DATASET_ROOT",doc["raw_root"])
    raw_root=Path(raw_value); raw_root=(root/raw_root if not raw_root.is_absolute() else raw_root).resolve()
    rp=doc["rungs"][rung]; base=doc["augmentation"]
    augment_config=AugmentConfig(sample_rate=fe.sr,p_clean_recipe=float(base["clean_recipe_probability"]),
        p_band_limit=float(rp["p_band_limit"]),p_eq=float(rp["p_eq"]),p_soft_clip=float(rp["p_soft_clip"]),
        p_reverb=float(rp["p_reverb"]),p_noise=float(rp["p_noise"]),p_floor=float(rp["p_floor"]),
        gain_db=tuple(rp["gain_db"]),clean_gain_db=tuple(base["clean_gain_db"]),
        hp_hz=(150.0,600.0),lp_hz=(3500.0,7500.0),eq_max_gain_db=6.0,
        rt60_s=(0.2,1.0),drr_db=(-5.0,10.0),noise_snr_db=tuple(base["background_snr_db"]),
        floor_dbfs=tuple(base["microphone_floor_dbfs"]),minimum_positive_snr_db=float(base["minimum_added_snr_db"]))
    noise_bank=noise_bank_for_training(root,raw_root,train_manifest,fe,seed) if rung in {"A3","A4"} else NoiseBank()
    augmenter=ChannelAugment(noise_bank,augment_config)
    train_ds=RawWaveformDataset(split="train",raw_root=raw_root,manifest=train_manifest,config=fe,
        mean=mean,std=std,seed=seed,augment=augmenter,channel_enabled=True,v2_gain_db=None)
    val_x,labels,suite_hash,tensor_hash=load_suite(root)
    if not np.array_equal(labels["clean"],val_manifest.label.to_numpy(dtype=np.int64)):
        raise RuntimeError("fixed suite clean labels do not match validation manifest")
    torch.set_num_threads(int(doc["threads"]))
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=build_cnn((1,64,28),channels=(16,32,64,64),kernel_size=3,pool=(2,2),head="global_average",dropout=.3)
    if parameter_count(model)!=60641: raise RuntimeError("fixed CNN depth_4 parameter count changed")
    npos=int(train_manifest.label.sum()); nneg=len(train_manifest)-npos
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos,dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=3)
    history=[]; best_loss=float("inf"); best_epoch=0; stale=0; best_state=None; start_epoch=0
    noise_records=[{"clip_id":d.clip_id,"group_id":d.group_id,"sha256":d.sha256} for d in noise_bank.donors]
    source_files=[Path(__file__).resolve(),Path(__file__).resolve().parents[1]/"src/dronedet/cnn_v3/augment.py",
                  Path(__file__).resolve().parents[1]/"src/dronedet/cnn_v3/dataset.py",
                  Path(__file__).resolve().parents[1]/"src/dronedet/models/cnn.py",
                  Path(__file__).resolve().parents[1]/"results/cnn_v3/records/requirements-v3-environment.txt"]
    source_hashes={p.name:sha(p) for p in source_files}
    identity={"run_id":f"{rung}_seed{seed}","config_sha256":sha(cp),"augmentation_config_sha256":augment_config.sha256,
        "source_hashes":source_hashes,"train_manifest_sha256":sha(data/"manifests/train.csv"),
        "val_manifest_sha256":sha(data/"manifests/val.csv"),"normalization_sha256":sha(data/"normalization_stats.npz"),
        "dependency_lock_sha256":sha(root/"results/cnn_v3/records/requirements-v3-environment.txt"),
        "suite_manifest_sha256":suite_hash,"suite_tensors_sha256":tensor_hash,"noise_donors":noise_records}
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    state_path=out/"checkpoint_epoch.pt"
    if (out/"run_record.json").exists():
        raise FileExistsError(f"run is already complete: {out}")
    if state_path.exists():
        state=torch.load(state_path,map_location="cpu",weights_only=False)
        if state.get("identity_sha256")!=identity_sha:
            raise RuntimeError("checkpoint identity differs; refuse to resume this rung")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        history=state["history"]; best_loss=state["best_loss"]; best_epoch=state["best_epoch"]
        stale=state["stale"]; best_state=state["best_state"]; start_epoch=state["next_epoch"]
    elif any(out.iterdir()):
        raise FileExistsError(f"run directory has artifacts but no resumable checkpoint: {out}")
    ytrain=train_manifest.label.to_numpy(dtype=np.float32)
    started=time.perf_counter()
    stop_reason="weighted_validation_patience_8" if stale>=int(doc["training"]["early_stopping_patience"]) else "maximum_epoch_limit"
    epoch_range=(range(start_epoch,int(doc["training"]["max_epochs"]))
                 if stale<int(doc["training"]["early_stopping_patience"]) else ())
    for epoch in epoch_range:
        train_ds.set_epoch(epoch); model.train()
        order=np.random.default_rng(seed*1_000_003+epoch).permutation(len(train_ds))
        loss_sum=0.0
        for off in range(0,len(order),64):
            ids=order[off:off+64]
            batch=[train_ds[int(i)] for i in ids]
            xb=torch.stack([b["x"] for b in batch]); yb=torch.from_numpy(ytrain[ids])
            optimizer.zero_grad(set_to_none=True); logits=model(xb); loss=criterion(logits,yb)
            loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())*len(ids)
        train_loss=loss_sum/len(train_ds); model.eval(); losses={}
        with torch.inference_mode():
            for name,x in val_x.items():
                out_logits=model(x)
                item_loss=F.binary_cross_entropy_with_logits(out_logits,torch.from_numpy(labels[name].astype(np.float32)),
                    pos_weight=criterion.pos_weight,reduction="none")
                losses[name]=float(item_loss.mean().item())
        val_loss=.5*losses["clean"]+(losses["mild"]+losses["moderate"]+losses["severe"])/6
        scheduler.step(val_loss)
        history.append({"epoch":epoch+1,"train_loss":train_loss,"validation_loss":val_loss,
                        "condition_validation_losses":losses,"learning_rate":float(optimizer.param_groups[0]["lr"])})
        if val_loss<best_loss:
            best_loss,best_epoch,stale=val_loss,epoch+1,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        snapshot={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                  "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                  "history":history,"best_loss":best_loss,"best_epoch":best_epoch,"best_state":best_state,
                  "stale":stale,"next_epoch":epoch+1,"identity_sha256":identity_sha}
        tmp=state_path.with_suffix(".pt.tmp"); torch.save(snapshot,tmp); os.replace(tmp,state_path)
        (out/"history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
        print(f"{rung} seed={seed} epoch={epoch+1} train_loss={train_loss:.6f} weighted_val_loss={val_loss:.6f} best={best_epoch}",flush=True)
        if stale>=int(doc["training"]["early_stopping_patience"]):
            stop_reason="weighted_validation_patience_8"; break
    if best_state is None: raise RuntimeError("no eligible model checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode(): logits={name:model(x).cpu().numpy().astype(np.float64).reshape(-1) for name,x in val_x.items()}
    threshold=weighted_f1_threshold(labels,logits)
    condition_metrics={name:metrics_at_threshold(labels[name],logits[name],threshold["threshold"]) for name in val_x}
    for name in val_x:
        condition_metrics[name]["pr_auc"]=float(average_precision_score(labels[name],logits[name]))
    cw={"clean":0.5,"mild":1/6,"moderate":1/6,"severe":1/6}
    pooled_y=np.concatenate([labels[k] for k in cw])
    pooled_logits=np.concatenate([logits[k] for k in cw])
    pooled_weights=np.concatenate([np.full(len(labels[k]),cw[k]/len(labels[k])) for k in cw])
    threshold["weighted_pooled_pr_auc"]=float(average_precision_score(pooled_y,pooled_logits,sample_weight=pooled_weights))
    v2_reference=json.loads((root/"results/cnn_v3/validation_suite/V2_reference/report.json").read_text(encoding="utf-8"))
    if v2_reference.get("suite_manifest_sha256")!=suite_hash:
        raise RuntimeError("frozen V2 reference does not match the current suite")
    aug_names=("mild","moderate","severe")
    recall_gain=float(np.mean([condition_metrics[k]["recall"]-v2_reference["condition_metrics"][k]["recall"] for k in aug_names]))
    severity_drops={k:float(v2_reference["condition_metrics"][k]["recall"]-condition_metrics[k]["recall"]) for k in aug_names}
    guards={"clean_recall_at_least_159_of_185":condition_metrics["clean"]["tp"]>=159,
        "clean_false_positives_at_most_5_of_1325":condition_metrics["clean"]["fp"]<=5,
        "each_augmented_far_at_most_1pct":all(condition_metrics[k]["false_alarm_rate"]<=0.01 for k in aug_names),
        "mean_augmented_recall_gain_at_least_5pp":recall_gain>=0.05,
        "no_severity_recall_more_than_3pp_below_v2":all(v<=0.03 for v in severity_drops.values())}
    eligible=all(guards.values())
    ckpt=out/f"cnn_v3_{rung.lower()}_seed{seed}.pt"
    torch.save({"state_dict":best_state,"architecture":{"channels":[16,32,64,64],"kernel_size":3,
        "pool":[2,2],"head":"global_average","dropout":.3},"parameter_count":60641,"seed":seed,
        "best_epoch":best_epoch},ckpt)
    parents={"A1":"A0c_weighted_monitor_seed42","A2":"A1_seed42","A3":"A2_seed42","A4":"A3_seed42"}
    record={"status":"FROZEN_BEFORE_TEST","run_id":f"{rung}_seed{seed}","parent":parents[rung],
        "architecture":"V2 depth_4 fixed","seed":seed,"rung":rung,"augmentation_config":augment_config.__dict__,
        "augmentation_config_sha256":augment_config.sha256,"noise_donors":noise_records,
        "best_epoch":best_epoch,"best_weighted_validation_loss":best_loss,"stop_reason":stop_reason,
        "threshold":threshold,"condition_metrics":condition_metrics,"checkpoint":ckpt.name,"checkpoint_sha256":sha(ckpt),
        "acceptance_guards":guards,"eligible":eligible,"mean_augmented_recall_gain":recall_gain,
        "per_severity_recall_drops_vs_v2":severity_drops,"v2_reference_report_sha256":sha(root/"results/cnn_v3/validation_suite/V2_reference/report.json"),
        "identity_sha256":identity_sha,"source_hashes":source_hashes,
        "config_sha256":sha(cp),"train_manifest_sha256":sha(data/"manifests/train.csv"),
        "dataset_ready_sha256":sha(data/"DATASET_READY.json"),
        "val_manifest_sha256":sha(data/"manifests/val.csv"),"normalization_stats_sha256":sha(data/"normalization_stats.npz"),
        "suite_manifest_sha256":suite_hash,"suite_tensors_sha256":tensor_hash,"test_access":False,
        "optimizer":{"name":"AdamW","learning_rate":1e-3,"weight_decay":1e-2},
        "loss":{"name":"BCEWithLogitsLoss","positive_weight":nneg/npos},
        "scheduler":{"name":"ReduceLROnPlateau","factor":0.5,"patience":3},
        "batch_size":64,"max_epochs":int(doc["training"]["max_epochs"]),
        "early_stopping_patience":int(doc["training"]["early_stopping_patience"]),
        "environment":{"python":platform.python_version(),"platform":platform.platform(),
            "torch":torch.__version__,"numpy":np.__version__,
            "scipy":importlib.metadata.version("scipy"),"librosa":importlib.metadata.version("librosa"),
            "soundfile":importlib.metadata.version("soundfile"),"pandas":pd.__version__,"scikit_learn":importlib.metadata.version("scikit-learn"),
            "threads":torch.get_num_threads(),"device":"cpu"},
        "training_seconds":time.perf_counter()-started}
    (out/"run_record.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":record["status"],"run_id":record["run_id"],"best_epoch":best_epoch,
        "weighted_f1":threshold["f1"],"threshold":threshold["threshold"],"checkpoint_sha256":record["checkpoint_sha256"]},indent=2))
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--rung",required=True); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"config/cnn_v3.yaml")
    args=parser.parse_args(); train(args.rung,args.seed,args.config)
