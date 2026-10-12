"""I130/I144 cached pipeline audit, with both SCut stages fitted on calibration folds.

CPU diagnostic; no model training, submission export, or trusted cache overwrite.
The base models and hyperparameters were historically selected using the same
validation set. This is conditional threshold cross-fitting, NOT fresh model OOF.
Numerical helpers and the two fixed blend recipes come from I144/I130.
Run at repo root: .venv/Scripts/python.exe -u code/tools/audit_i130_calibration.py
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "rel"
N_LABELS = 500


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_helpers():
    path = ROOT / "code/I/I144-cross-attention-towers-eval.py"
    spec = importlib.util.spec_from_file_location("i144_audit_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_prob(filename, key, n_rows, manifest):
    path = REL / filename
    with np.load(path, allow_pickle=False) as d:
        prob = d[key].copy()
    if prob.shape != (n_rows, N_LABELS) or not np.isfinite(prob).all():
        raise ValueError(f"Bad probability cache: {filename}: {prob.shape}")
    manifest[filename] = {"key": key, "shape": list(prob.shape), "sha256": sha256(path)}
    return prob


def build_scores(h, manifest):
    print("Loading fixed seed-42 train/validation split and probability caches...", flush=True)
    data_path = ROOT / "data/train.csv"
    columns = ["protein_id"] + [f"label_{i}" for i in range(N_LABELS)]
    df = pd.read_csv(data_path, usecols=columns)
    y_all = df[[f"label_{i}" for i in range(N_LABELS)]].to_numpy(dtype=np.int8)
    tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_all[tr], y_all[va]
    n = len(va)
    manifest["data/train.csv"] = {"sha256": sha256(data_path), "rows": len(df)}
    manifest["split"] = {"test_size": 0.2, "random_state": 42,
                         "validation_index_sha256": hashlib.sha256(va.tobytes()).hexdigest()}
    r = lambda filename, key: read_prob(filename, key, n, manifest)
    deep4 = (0.12 * r("TCK001_t001a_probs.npz", "t001a_val")
             + 0.18 * r("T002_val_probs.npz", "t002")
             + 0.42 * r("TC003_val_probs.npz", "cnn")
             + 0.28 * r("C005_val_probs.npz", "val"))
    quad = (0.40 * r("multi_plm_mlp_val_probs.npz", "val")
            + 0.20 * r("i106_conv_multiseed_val.npz", "val")
            + 0.20 * r("i122_1664d_swa_val.npz", "val")
            + 0.20 * r("i122_3840d_swa_val.npz", "val"))
    dmt = r("i123_dmt_swa_val.npz", "val")
    cmat = r("i144_cmat_swa_val.npz", "val")
    plms = {"penta_i130": 0.70 * quad + 0.30 * dmt,
            "hexa_i144": 0.70 * quad + 0.15 * dmt + 0.15 * cmat}
    val_ids = df["protein_id"].to_numpy()[va]
    train_map = {p: i for i, p in enumerate(df["protein_id"].to_numpy()[tr])}
    val_map = {p: i for i, p in enumerate(val_ids)}
    aln = REL / "ck005a_tmp/align_val_very-sensitive_e0.001_top100.tsv"
    graph = REL / "ck005a_tmp/align_val_to_val_e0.001.tsv"
    idx, bits = h.i085.parse_topk(str(aln), val_ids, train_map)
    nbr = h.i104.parse_2hop(str(graph), val_map, val_map)
    manifest[str(aln.relative_to(ROOT))] = {"sha256": sha256(aln)}
    manifest[str(graph.relative_to(ROOT))] = {"sha256": sha256(graph)}
    hit = idx[:, 0] >= 0
    parts = []
    for tk in tqdm((20, 3), desc="Fixed homology aggregation"):
        raw = h.i085.agg_topk(idx, bits, y_tr, tk)
        parts.append(h.conditional_diffuse(raw, hit, bits[:, 0], nbr, n,
                                          threshold_b=70, beta_diff=0.20))
    kmer = h.i104.build_kmer(y_tr, "val")
    for k in (4, 5, 6, 7, 8):
        pre = "CK003" if k >= 7 else "CK002"
        path = REL / f"{pre}_neigh_k{k}.npz"
        manifest[path.name] = {"sha256": sha256(path)}
    sim = 0.50 * (0.75 * parts[0] + 0.25 * parts[1]) + 0.50 * kmer
    clip = np.clip(sim, 1e-6, 1.0 - 1e-6)
    ent = -(clip * np.log2(clip) + (1.0 - clip) * np.log2(1.0 - clip)).sum(axis=1)
    ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-7)
    gate = 0.80 * ((1.0 - ent_norm[:, None]) ** 0.10)
    rs = h.ranks(sim)
    scores = {}
    for name, plm in plms.items():
        deep = 0.65 * deep4 + 0.35 * plm
        rrf = h.col_norm(0.6 / (30 + h.ranks(deep) + 1) + 1.4 / (30 + rs + 1))
        scores[name] = deep + gate * sim + 0.50 * rrf
        if not np.isfinite(scores[name]).all():
            raise ValueError(f"Nonfinite score: {name}")
    return scores, y_va, y_tr.mean(0), va


def calibrate(h, score, y_cal, cal_idx, train_prior):
    """Only calibration labels enter either threshold fit; no evaluation labels argument."""
    t_row = h.search_scut(score[cal_idx], y_cal)
    row_m = np.maximum((score >= t_row).sum(axis=1).astype(np.float64), 1.0)
    col_m = np.maximum(train_prior * len(score), 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    kernel = np.exp((score - score.max(axis=1, keepdims=True)) / 0.20)
    transport = h.sinkhorn(kernel, row_m, col_m)
    final = 0.45 * score + 0.55 * transport
    t_final = h.search_scut(final[cal_idx], y_cal)
    return final, t_row, t_final


def binary_metrics(pred, y):
    pred, y = pred.astype(bool), y.astype(bool)
    tp = (pred & y).sum(axis=0)
    fp = (pred & ~y).sum(axis=0)
    fn = (~pred & y).sum(axis=0)
    active = y.sum(axis=0) > 0
    return {
        "macro_f1": float(np.mean((2 * tp / np.maximum(2 * tp + fp + fn, 1))[active])),
        "macro_precision": float(np.mean((tp / np.maximum(tp + fp, 1))[active])),
        "macro_recall": float(np.mean((tp / np.maximum(tp + fn, 1))[active])),
        "mean_cardinality": float(pred.sum(axis=1).mean()),
        "active_labels": int(active.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=REL / "i130_calibration_audit_20261012")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    start = time.monotonic()
    out = args.output.resolve()
    if not out.is_relative_to(REL.resolve()):
        raise ValueError("Diagnostic output must remain under rel/")
    out.mkdir(parents=True, exist_ok=True)
    h = load_helpers()
    manifest = {}
    with threadpool_limits(limits=args.threads):
        scores, y_va, train_prior, va = build_scores(h, manifest)
        n = len(y_va)
        all_idx = np.arange(n)
        reference = {}
        for name, score in scores.items():
            print(f"Reference reproduction: {name}", flush=True)
            final, _, th = calibrate(h, score, y_va, all_idx, train_prior)
            metrics = binary_metrics(final >= th, y_va)
            metrics["macro_auc"] = float(roc_auc_score(y_va, final, average="macro"))
            metrics["micro_auc"] = float(roc_auc_score(y_va, final, average="micro"))
            reference[name] = metrics
        report_i130 = json.loads((REL / "I130_report.json").read_text(encoding="utf-8-sig"))
        report_i144 = json.loads((REL / "I144_report.json").read_text(encoding="utf-8-sig"))
        expected = {"penta_i130": report_i130["val_f1_calibrated"],
                    "hexa_i144": report_i144["full_pipeline_f1"]["hexa_f1"]}
        for name, target in expected.items():
            observed = reference[name]["macro_f1"]
            print(f"  {name}: measured {observed:.8f}, report {target:.5f}", flush=True)
            if abs(observed - target) > 1e-5:
                (out / "baseline_gate_failure.json").write_text(
                    json.dumps({"measured": reference, "expected": expected}, indent=2), encoding="utf-8")
                raise RuntimeError("Original calibrated baseline did not reproduce; stop before comparison")
        pooled_preds = {name: np.zeros_like(y_va, dtype=bool) for name in scores}
        fold_ids = np.full(n, -1, dtype=np.int8)
        folds = []
        thresholds = {}
        split = KFold(n_splits=5, shuffle=True, random_state=20261012)
        for fi, (cal, test) in enumerate(tqdm(split.split(all_idx), total=5, desc="Calibration cross-fit")):
            if time.monotonic() - start > 900:
                raise TimeoutError("15 minute audit budget exhausted")
            if np.intersect1d(cal, test).size:
                raise AssertionError("Calibration/evaluation overlap")
            fold_ids[test] = fi
            row = {"fold": fi, "calibration_rows": len(cal), "evaluation_rows": len(test)}
            for name, score in scores.items():
                final, t_row, th = calibrate(h, score, y_va[cal], cal, train_prior)
                pred = final[test] >= th
                pooled_preds[name][test] = pred
                metrics = binary_metrics(pred, y_va[test])
                metrics["macro_auc"] = float(roc_auc_score(y_va[test], final[test], average="macro"))
                metrics["micro_auc"] = float(roc_auc_score(y_va[test], final[test], average="micro"))
                row[name] = metrics
                thresholds[f"{name}_fold{fi}_row"] = t_row
                thresholds[f"{name}_fold{fi}_final"] = th
            row["hexa_minus_penta_f1"] = row["hexa_i144"]["macro_f1"] - row["penta_i130"]["macro_f1"]
            folds.append(row)
            print(f"  fold {fi}: Penta {row['penta_i130']['macro_f1']:.6f}, "
                  f"Hexa {row['hexa_i144']['macro_f1']:.6f}, "
                  f"difference {row['hexa_minus_penta_f1']:+.6f}", flush=True)
        if (fold_ids < 0).any():
            raise AssertionError("Missing held-out predictions")
        pooled = {name: binary_metrics(pred, y_va) for name, pred in pooled_preds.items()}
        delta = pooled["hexa_i144"]["macro_f1"] - pooled["penta_i130"]["macro_f1"]
        wins = sum(f["hexa_minus_penta_f1"] > 0 for f in folds)
        for name in scores:
            pooled[name]["resubstitution_minus_crossfit_f1"] = (
                reference[name]["macro_f1"] - pooled[name]["macro_f1"])
        report = {
            "analysis": "I130 calibration cross-fit / fixed I144 CMAT inclusion",
            "reference": reference, "folds": folds, "pooled": pooled,
            "hexa_minus_penta_pooled_f1": delta, "hexa_winning_folds": wins,
            "cmat_promotion_gate_passed": bool(delta >= 0.001 and wins >= 4),
            "protocol": {
                "model_split": "80/20 random_state=42, frozen historical caches",
                "calibration_folds": 5, "fold_seed": 20261012,
                "both_scut_stages_calibration_only": True,
                "unlabeled_full_validation_batch_available": True,
                "prior_source": "training 80% labels only",
                "claim_level": "conditional threshold generalization, not fresh-model OOF",
                "limitations": ["Models and hyperparameters previously selected on this validation set",
                                "Cache row alignment follows original scripts; not all caches store protein IDs",
                                "Five fold differences are dependent; no independent-fold significance claim"],
            },
            "input_manifest": manifest,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "helper_source_sha256": {p.name: sha256(p) for p in [
                ROOT / "code/I/I064-diversity-branches.py", ROOT / "code/I/I085-top100-gated-agg.py",
                ROOT / "code/I/I104-top20sim-submit.py", ROOT / "code/I/I128-twilight-markov-diffusion-eval.py",
                ROOT / "code/I/I144-cross-attention-towers-eval.py", Path(__file__).resolve()]},
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        np.savez_compressed(out / "thresholds.npz", **thresholds)
        np.savez_compressed(out / "heldout_decisions.npz", validation_indices=va, fold_ids=fold_ids,
                            **{k: np.packbits(v, axis=1) for k, v in pooled_preds.items()})
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"pooled": pooled, "delta": delta, "winning_folds": wins,
                          "cmat_gate": report["cmat_promotion_gate_passed"]}, ensure_ascii=False, indent=2))
        print(f"Report: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
