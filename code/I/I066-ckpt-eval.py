"""
I066-ckpt-eval — 评估 I066 残基注意力分支已保存的检查点 (只读推理)
==================================================================
用途: 在不继续训练的前提下，判断该分支当前质量与融合价值，供决定是否手动续训。
输出: 控制台指标 + rel/resattn_ckpt_eval.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CKPT = os.path.join(PROJECT_ROOT, "models", "resattn_s7.pt")
N_LABELS = 500

sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "I"))
from importlib.machinery import SourceFileLoader
i066 = SourceFileLoader("i066", os.path.join(PROJECT_ROOT, "code", "I", "I066-residue-attention.py")).load_module()


def search_scut(ens, y, n=60):
    cands = np.linspace(0.01, 0.99, n)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        best[i] = cands[(2*tp/np.maximum(2*tp+fp+fn, 1e-9)).argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp+fp>0, tp/(tp+fp), 0.0); r = np.where(tp+fn>0, tp/(tp+fn), 0.0)
    return float(np.mean(np.where(p+r>0, 2*p*r/(p+r), 0.0)))


def main():
    t0 = time.time()
    print("=" * 70)
    print("  I066 checkpoint evaluation (read-only)")
    print("=" * 70)
    if not os.path.exists(CKPT):
        print(f"  未找到 {CKPT}，I066 尚未产出有效检查点。")
        return

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_va = y_full[idx_va].astype(np.int8)

    va_ds = i066.MemmapDS(i066.TRAIN_H, i066.TRAIN_M, idx_va, y_full)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=192, shuffle=False, num_workers=0)

    model = i066.ResidueAttentionHead().cuda()
    model.load_state_dict(torch.load(CKPT, map_location="cuda"))
    model.eval()

    print("  Predicting validation (mmap read ~7.4GB)...")
    probs = i066.predict(model, va_loader)
    ths = search_scut(probs, y_va)
    f1 = macro_f1(probs, y_va, ths)
    auc = roc_auc_score(y_va, probs, average="macro")
    preds = (probs >= ths).astype(bool); yb = y_va.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    pp = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rr = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    rec, prec = float(np.mean(rr)), float(np.mean(pp))

    print("\n" + "=" * 70)
    print(f"  Residue-attention branch (ckpt: {os.path.basename(CKPT)}):")
    print(f"    Standalone SCut F1 = {f1:.4f} | AUC = {auc:.4f} | R = {rec:.4f} | P = {prec:.4f}")
    print(f"  对照基准:")
    print(f"    1664D 均值池化多种子 (I064): F1 0.2729 | AUC 0.7947")
    print(f"    E004a 旧注意力池化 (历史)  : F1 0.1518")
    print("=" * 70)

    out = {"ckpt": os.path.basename(CKPT), "suct_f1": f1, "auc": auc,
           "recall": rec, "precision": prec,
           "ref_1664d_multiseed": {"f1": 0.2729, "auc": 0.7947}}
    np.savez_compressed(os.path.join(REL_DIR, "resattn_ckpt_val_probs.npz"), val=probs)
    with open(os.path.join(REL_DIR, "resattn_ckpt_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: rel/resattn_ckpt_eval.json + resattn_ckpt_val_probs.npz")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
