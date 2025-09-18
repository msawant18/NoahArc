"""
Confidence calibration.
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import json

def _nll_temp(T, logits, labels):
    # T is scalar > 0
    T = T[0]
    probs = 1/(1+np.exp(-logits/T))
    # binary NLL
    eps = 1e-12
    return -np.mean(labels * np.log(probs+eps) + (1-labels) * np.log(1-probs+eps))

def fit_temperature(logits, labels):
    # logits and labels flattened
    x0 = np.array([1.0])
    res = minimize(_nll_temp, x0, args=(logits, labels), bounds=[(0.05,10.0)])
    return float(res.x[0]) if res.success else 1.0

def collect_logits_and_labels(proba_dir, labels_dir):
    proba_files = sorted([p for p in Path(proba_dir).glob("*.proba.npy")])
    logits = []
    labels = []
    for pf in proba_files:
        proba = np.load(str(pf)).astype(np.float32)
        # convert prob to "logit" via logit function, clamp
        proba = np.clip(proba, 1e-6, 1-1e-6)
        logit = np.log(proba/(1-proba))
        labf = Path(str(pf).replace(".proba.npy", ".mask.npy"))
        if not labf.exists():
            labf = Path(labels_dir)/ (pf.stem.replace(".proba","") + ".mask.npy")
        if not labf.exists():
            continue
        lab = np.load(str(labf)).astype(np.uint8)
        logits.append(logit.flatten())
        labels.append(lab.flatten())
    if len(logits)==0:
        return None, None
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    return logits, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proba_dir", default="data/outputs")
    parser.add_argument("--labels_dir", default="data/processed")
    parser.add_argument("--out", default="artifacts/thresholds.json")
    args = parser.parse_args()
    lg, lb = collect_logits_and_labels(args.proba_dir, args.labels_dir)
    if lg is None:
        print("[WARN] No logits collected.")
        exit(0)
    T = fit_temperature(lg, lb)
    # compute ECE before and after (rough)
    from src.evaluation.metrics import ece_score
    # compute example
    probs = 1/(1+np.exp(-lg))
    ece_before = ece_score(lb, probs)
    probs_cal = 1/(1+np.exp(-lg/T))
    ece_after = ece_score(lb, probs_cal)
    out = {"temperature": T, "ece_before": float(ece_before), "ece_after": float(ece_after)}
    Path = Path if 'Path' in globals() else None
    import json
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved calibration to {args.out}")
