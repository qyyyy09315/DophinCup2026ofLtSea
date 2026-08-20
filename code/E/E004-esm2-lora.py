"""
E004-esm2-lora — ESM2-150M + LoRA 参数高效微调多标签分类
================================================================
依据:
  - PEFT-SP (Zeng et al., 2023, doi:10.1101/2023.11.04.565642)
  - InterLabelGO+ (Liu et al., Bioinformatics 2024, doi:10.1093/bioinformatics/btae655)
    无同源蛋白靠可训练 PLM 序列模型兜底（本赛 CK005 miss 子集 F1 仅 0.06）
方案:
  - ESM2-150M + LoRA r=8 / alpha=16 / q,k,v
  - 动态 padding（batch 内按真实长度，不整表 pad 到 256）
  - 默认 15 epoch + 早停 patience=3；默认关闭 gradient checkpointing
  - AMP + TF32；验证间隔 2；--resume 续训
用法:
  python code/E/E004-esm2-lora.py
  python code/E/E004-esm2-lora.py --resume
  python code/E/E004-esm2-lora.py --ckpt --bs=8   # 显存不够时
输出:
  models/e004-lora/best
  rel/E004_report.json
  rel/E004_lora_test_probs.npz
  submit/submission_E004_lora.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import EsmForSequenceClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "e004-lora")
os.makedirs(MODEL_DIR, exist_ok=True)

ESM2_PATH = os.path.join(PROJECT_ROOT, "models", "esm2_150m")
N_LABELS = 500
MAX_LEN = 256
N_EPOCHS = 15
BATCH = 16
EVAL_BATCH = 32
LR = 2e-4
PATIENCE = 3
VALIDATE_EVERY = 2
LORA_R = 8
LORA_ALPHA = 16


def macro_f1(ens, y, t=0.5):
    pred = ens >= t
    f1s = []
    for i in range(N_LABELS):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_rp(ens, y, t):
    pred = ens >= t
    rs, ps = [], []
    for i in range(N_LABELS):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def macro_auc(ens, y):
    aucs = []
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, ens[:, i]))
    return float(np.mean(aucs))


class TokenDS(Dataset):
    def __init__(self, ids, mask, y):
        self.ids = ids
        self.mask = mask
        self.y = y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = int(self.mask[i].sum())
        m = max(m, 1)
        return self.ids[i, :m], self.mask[i, :m], self.y[i]


def collate_dyn(batch):
    ids, mask, y = zip(*batch)
    max_l = max(x.size(0) for x in ids)
    b = len(ids)
    ids_p = torch.zeros(b, max_l, dtype=torch.long)
    mask_p = torch.zeros(b, max_l, dtype=torch.long)
    for i, (a, m) in enumerate(zip(ids, mask)):
        n = a.size(0)
        ids_p[i, :n] = a
        mask_p[i, :n] = m
    return ids_p, mask_p, torch.stack(y)


def load_base():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(ESM2_PATH))
    model = EsmForSequenceClassification.from_pretrained(
        os.path.basename(ESM2_PATH),
        num_labels=N_LABELS,
        problem_type="multi_label_classification",
        torch_dtype=torch.float32,
    )
    os.chdir(cwd)
    return model


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    torch.cuda.empty_cache()
    out = []
    for ids, mask, _ in loader:
        ids, mask = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def parse_args():
    n_epochs, bs, lr = N_EPOCHS, BATCH, LR
    use_ckpt = "--ckpt" in sys.argv
    resume = "--resume" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--epochs="):
            n_epochs = int(a.split("=", 1)[1])
        elif a.startswith("--bs="):
            bs = int(a.split("=", 1)[1])
        elif a.startswith("--lr="):
            lr = float(a.split("=", 1)[1])
    return n_epochs, bs, lr, use_ckpt, resume


def main():
    n_epochs, bs, lr, use_ckpt, resume = parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    t0 = time.time()
    print("=" * 60)
    print(f"E004 LoRA-ESM2: epochs={n_epochs} bs={bs} lr={lr} ckpt={use_ckpt} resume={resume}")
    print(f"  加速: 动态padding / 早停 patience={PATIENCE} / 验证每{VALIDATE_EVERY}ep / TF32"
          f"{' / gradient-ckpt' if use_ckpt else ' / 无ckpt'}")

    print("[1/4] 加载数据...")
    tr_tok = np.load(f"{REL_DIR}/esm2_tokens_train.npz")
    te_tok = np.load(f"{REL_DIR}/esm2_tokens_test.npz")
    ids_tr = tr_tok["input_ids"][:, :MAX_LEN]
    mask_tr = tr_tok["attention_mask"][:, :MAX_LEN]
    ids_te = te_tok["input_ids"][:, :MAX_LEN]
    mask_te = te_tok["attention_mask"][:, :MAX_LEN]
    lens = mask_tr.sum(1)
    print(f"  tokens: train {ids_tr.shape} test {ids_te.shape} | 有效长 mean={lens.mean():.0f} p50={np.median(lens):.0f}")

    train_df = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    tr_ds = TokenDS(torch.from_numpy(ids_tr[idx_train]).long(),
                    torch.from_numpy(mask_tr[idx_train]).long(),
                    torch.from_numpy(y_train.astype(np.float32)))
    va_ds = TokenDS(torch.from_numpy(ids_tr[idx_val]).long(),
                    torch.from_numpy(mask_tr[idx_val]).long(),
                    torch.from_numpy(y_val.astype(np.float32)))
    te_ds = TokenDS(torch.from_numpy(ids_te).long(),
                    torch.from_numpy(mask_te).long(),
                    torch.zeros(len(ids_te), N_LABELS))
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0,
                           collate_fn=collate_dyn, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0,
                           collate_fn=collate_dyn, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0,
                           collate_fn=collate_dyn, pin_memory=True)
    print(f"  步/epoch={len(tr_loader)}  预估: 无ckpt ~8-12min/ep, 有ckpt ~12-18min/ep")

    print("[2/4] 加载 ESM2-150M + LoRA...")
    base = load_base()
    if use_ckpt:
        base.gradient_checkpointing_enable()
        if hasattr(base, "config"):
            base.config.use_cache = False
    print(f"  基模型参数量: {sum(p.numel() for p in base.parameters()) / 1e6:.1f}M")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.1, target_modules=["query", "key", "value"], bias="none")
    model = get_peft_model(base, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA 可训练: {n_trainable / 1e6:.2f}M")
    model = model.cuda()

    print(f"[3/4] 训练 {n_epochs} epoch...")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    n_steps = max(1, len(tr_loader) * n_epochs)
    warmup = max(1, int(n_steps * 0.06))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + np.cos(np.pi * s / n_steps))))
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    start_ep = 0
    bad = 0
    meta_path = os.path.join(MODEL_DIR, "train_meta.json")
    if resume and os.path.isdir(f"{MODEL_DIR}/best"):
        from peft import PeftModel as _PM
        base2 = load_base()
        if use_ckpt:
            base2.gradient_checkpointing_enable()
        model = _PM.from_pretrained(base2, f"{MODEL_DIR}/best").cuda()
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, encoding="utf-8"))
            best_f1 = float(meta.get("best_f1", -1))
            start_ep = int(meta.get("best_epoch", 0))
        print(f"  resume: best_f1={best_f1:.4f} from epoch {start_ep}")

    for ep in range(start_ep, n_epochs):
        model.train()
        total_loss = 0.0
        t_ep = time.time()
        for ids, mask, y in tqdm(tr_loader, desc=f"ep{ep + 1}", leave=False):
            ids = ids.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = bce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item() * ids.size(0)
        dt = time.time() - t_ep
        do_val = ((ep + 1) % VALIDATE_EVERY == 0) or (ep == n_epochs - 1) or (ep < 2)
        msg = f"  ep{ep + 1}: loss={total_loss / len(tr_ds):.4f} {dt:.0f}s ({len(tr_ds) / dt:.1f} seq/s)"
        if do_val:
            f1 = macro_f1(predict_probs(model, va_loader), y_val, 0.5)
            msg += f" F1(0.5)={f1:.4f}"
            print(msg, flush=True)
            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                bad = 0
                model.save_pretrained(f"{MODEL_DIR}/best")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"best_f1": best_f1, "best_epoch": ep + 1}, f)
                print(f"    保存 best F1={best_f1:.4f}", flush=True)
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"    早停 (patience={PATIENCE})", flush=True)
                    break
        else:
            print(msg, flush=True)

    print("[4/4] 评估 + 生成 test probs...")
    if not os.path.isdir(f"{MODEL_DIR}/best"):
        model.save_pretrained(f"{MODEL_DIR}/best")
    base = load_base()
    model = PeftModel.from_pretrained(base, f"{MODEL_DIR}/best").cuda()
    probs_val = predict_probs(model, va_loader)
    probs_test = predict_probs(model, te_loader)
    np.savez_compressed(f"{REL_DIR}/E004_lora_test_probs.npz", val=probs_val, test=probs_test)

    f1_05 = macro_f1(probs_val, y_val, 0.5)
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(N_LABELS, 0.5)
    for i in tqdm(range(N_LABELS), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = probs_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0)
        fp = (p & ~ytm).sum(0)
        fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    f1_100 = np.mean([
        f1_score(y_val[:, i].astype(bool), probs_val[:, i] >= best_t[i], zero_division=0)
        for i in range(N_LABELS) if y_val[:, i].sum() > 0])
    r100, p100 = macro_rp(probs_val, y_val, best_t)
    auc = macro_auc(probs_val, y_val)
    print(f"  E004 LoRA: F1(0.5)={f1_05:.4f} F1(100)={f1_100:.4f} R={r100:.4f} P={p100:.4f} AUC={auc:.4f}")
    print(f"  对照 C002b-ASL F1(0.5)=0.2714: {'提升' if f1_05 > 0.2714 else '未超'}")

    os.makedirs(SUB_DIR, exist_ok=True)
    test_df = pd.read_csv(f"{PROJECT_ROOT}/data/test.csv")
    pred = (probs_test >= best_t[None, :]).astype(np.int8)
    sub = pd.concat([test_df[["protein_id"]], pd.DataFrame(pred, columns=label_cols)], axis=1)
    sub_path = f"{SUB_DIR}/submission_E004_lora.csv"
    sub.to_csv(sub_path, index=False)
    report = {
        "model": "ESM2-150M + LoRA (r=8, target=q/k/v)",
        "f1_05": round(float(f1_05), 4), "f1_100": round(float(f1_100), 4),
        "recall_100": round(r100, 4), "precision_100": round(p100, 4),
        "auc": round(auc, 4), "best_f1_05": round(float(best_f1), 4),
        "c002b_asl_baseline_f1_05": 0.2714,
        "epochs": n_epochs, "bs": bs, "lr": lr, "ckpt": use_ckpt,
        "submission": sub_path, "seconds": round(time.time() - t0),
    }
    with open(f"{REL_DIR}/E004_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REL_DIR}/E004_report.json")
    print(f"  提交: {sub_path}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
