"""Add complete per-condition PR-AUC evidence to the frozen A0c control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from dronedet.models.cnn import build_cnn
from dronedet.metrics import metrics_at_threshold


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/"results/cnn_v3/experiments/A0c_weighted_monitor_seed42"
    freeze_path=run/"freeze.json"; freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    suite=root/"results/cnn_v3/validation_suite"
    tensors=np.load(suite/"tensors.npz",allow_pickle=False)
    model=build_cnn((1,64,28),channels=(16,32,64,64),kernel_size=3,pool=(2,2),head="global_average",dropout=.3)
    ckpt=torch.load(run/freeze["checkpoint"],map_location="cpu",weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    logits={}; labels={}
    with torch.inference_mode():
        for name in ("clean","mild","moderate","severe"):
            x=torch.from_numpy(tensors[f"x_{name}"].copy())
            labels[name]=tensors[f"y_{name}"].astype(int)
            logits[name]=model(x).cpu().numpy().astype(np.float64).reshape(-1)
    threshold=float(freeze["validation_threshold"]["threshold"])
    metrics={k:metrics_at_threshold(labels[k],logits[k],threshold) for k in labels}
    for k in labels:metrics[k]["pr_auc"]=float(average_precision_score(labels[k],logits[k]))
    weights={"clean":.5,"mild":1/6,"moderate":1/6,"severe":1/6}
    y=np.concatenate([labels[k] for k in weights]); s=np.concatenate([logits[k] for k in weights])
    sw=np.concatenate([np.full(len(labels[k]),weights[k]/len(labels[k])) for k in weights])
    freeze["validation_metrics_by_condition"]=metrics
    freeze["weighted_pooled_pr_auc"]=float(average_precision_score(y,s,sample_weight=sw))
    freeze["metrics_report_script_sha256"]=sha(Path(__file__).resolve())
    freeze_path.write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf-8")
    (run/"A0c_record.json").write_text(json.dumps({"status":"COMPLETE","run_id":"A0c_weighted_monitor_seed42",
        "parent":"A0b_rawcache_seed42","change":"validation loss monitor only: clean=1/2, each severity=1/6",
        "freeze_sha256":sha(freeze_path),"checkpoint_sha256":freeze["checkpoint_sha256"],
        "suite_manifest_sha256":freeze["suite_manifest_sha256"],"suite_tensors_sha256":freeze["suite_tensors_sha256"],
        "validation_threshold":freeze["validation_threshold"],"validation_metrics_by_condition":metrics,
        "weighted_pooled_pr_auc":freeze["weighted_pooled_pr_auc"],"test_access":False},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","threshold":threshold,"weighted_pooled_pr_auc":freeze["weighted_pooled_pr_auc"],
                      "condition_metrics":metrics},indent=2))


if __name__=="__main__":main()
