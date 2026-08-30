"""
C018-ss-aux — C002b + 冻结 S4Pred 二级结构辅任务
================================================================
依据:
  - S4Pred: Moffat & Jones, Bioinformatics 2021,
    doi:10.1093/bioinformatics/btab491。单序列二级结构，不要 MSA。
    权重 https://github.com/psipred/s4pred
  - SPOT-1D-Single: Singh et al., Bioinformatics 2021,
    doi:10.1093/bioinformatics/btab316。单序列 SS/RSA，同样不要 profile。
  - NetSurfP-3.0: Høie et al., NAR 2022, doi:10.1093/nar/gkac439。
    要 ESM-1b，8GB 放不下，本实验不用。
  - ECK004 Chou-Fasman 蛋白级均值已关；C006 Atchley 嵌入已关。
    理化/结构不能当输入通道，改成辅任务。
自己的改法:
  骨干/ASL/80ep cosine 同 C002b。输入只有 one-hot 序列。
  离线用冻结 S4Pred 五模型平均，给每条序列打 3 类 SS（C/H/E）。
  卷积特征在全局池化前加 1×1 残基头，L_ss = CE(ss_logit, ss_hat)，
  只在非 PAD 位置。总损失 L = ASL + λ·L_ss，λ=0.20。
  推理只用功能头。接到 C011a，α∈{0.50,1.00}。
门槛:
  相对 C011a 半折 OOF ≥ +0.001。
用法:
  python code/C/C018-ss-aux.py                 # dump SS + 训练
  python code/C/C018-ss-aux.py --dump-only     # 只 dump SS
  python code/C/C018-ss-aux.py --fresh
  python code/C/C018-ss-aux.py --eval-only
输出:
  rel/C018_report.json
  rel/C018_probs.npz
  rel/c018_tmp/ss_labels.npz
  models/c018-ssaux.pt
  models/c018-ssaux.last.pt
"""

import hashlib
import importlib.util
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
C018_TMP = os.path.join(REL_DIR, "c018_tmp")
S4_DIR = os.path.join(MODEL_DIR, "s4pred")
S4_WEIGHT_DIR = os.path.join(S4_DIR, "weights")

C002B_MOD = importlib.util.spec_from_file_location(
    "c002b", os.path.join(SRC, "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(C002B_MOD)
C002B_MOD.loader.exec_module(c002b)

BATCH_SIZE = c002b.BATCH_SIZE
N_EPOCHS = c002b.N_EPOCHS
LR = c002b.LR
WEIGHT_DECAY = c002b.WEIGHT_DECAY
PATIENCE = c002b.PATIENCE
SPLIT_SEED = 42
A_GRID = (0.50, 1.00)
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CNN_W = 0.2
C002B_F05 = 0.2714
C011A_ALPHA = 0.50
LAMBDA_SS = 0.20
SS_PAD = -100
S4_AA = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8,
    "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16,
    "W": 17, "Y": 18, "V": 19,
}
S4_WEIGHT_URLS = (
    "http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz",
    "https://bioinf.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz",
)
S4_WEIGHT_MD5 = "e04ad7d10b61551f7e07a86b65bb88dc"

MODEL_PATH = os.path.join(MODEL_DIR, "c018-ssaux.pt")
LAST_PATH = os.path.join(MODEL_DIR, "c018-ssaux.last.pt")
REPORT_PATH = os.path.join(REL_DIR, "C018_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C018_probs.npz")
TOK_CACHE = os.path.join(REL_DIR, "c004_tokens.npz")
SS_CACHE = os.path.join(C018_TMP, "ss_labels.npz")
SS_PARTIAL = os.path.join(C018_TMP, "ss_labels_partial.npz")
SS_LOCK = os.path.join(C018_TMP, "ss_dump.lock")
DUMP_SAVE_EVERY = 2000
DUMP_PAD_IDX = 21
# 单教师（weights_1）+ 按长度分桶。五模型逐条约 1.3 seq/s，全库不可行。
DUMP_BUCKETS = (
    (128, 64),
    (256, 32),
    (512, 16),
    (1024, 8),
    (1500, 4),
)

EVAL_ONLY = "--eval-only" in sys.argv
GEN_SUB = "--gen-sub" in sys.argv
FRESH = "--fresh" in sys.argv
DUMP_ONLY = "--dump-only" in sys.argv
DEVICE = c002b.DEVICE


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def f1_05(p, y):
    pred = (p >= 0.5)
    f1s = []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        pr = pred[:, i]
        tp = int((pr & yt).sum())
        fp = int((pr & ~yt).sum())
        fn = int((~pr & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


class ResidueEmbedding(nn.Embedding):
    def __init__(self, vocab_size=22, embed_size=128, padding_idx=21):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)


class GRUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ResidueEmbedding()
        self.lstm = nn.GRU(
            128, 1024, num_layers=3, bidirectional=True, batch_first=True)
        self.outlayer = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return F.log_softmax(self.outlayer(x), dim=-1)


class S4PRED(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_1 = GRUnet()
        self.model_2 = GRUnet()
        self.model_3 = GRUnet()
        self.model_4 = GRUnet()
        self.model_5 = GRUnet()

    def forward(self, x):
        y = (self.model_1(x) + self.model_2(x) + self.model_3(x)
             + self.model_4(x) + self.model_5(x)) * 0.2
        return y


class SSAuxCNN(nn.Module):
    def __init__(self, n_labels=500):
        super().__init__()
        self.cnn = c002b.MultiScaleCNNv3(n_labels=n_labels)
        self.ss_head = nn.Conv1d(c002b.FILTERS * len(c002b.KERNELS), 3, 1)

    def encode_map(self, x):
        maps = []
        for branch, shortcut, se in zip(
                self.cnn.branches, self.cnn.shortcuts, self.cnn.ses):
            h = branch(x) + shortcut(x)
            maps.append(se(h))
        return torch.cat(maps, dim=1)

    def forward(self, x, return_ss=False):
        hmap = self.encode_map(x)
        feats = torch.cat(
            [hmap.max(dim=2)[0], hmap.mean(dim=2)], dim=1)
        logits = self.cnn.fc(self.cnn.dropout(feats))
        if return_ss:
            return logits, self.ss_head(hmap)
        return logits


def seq_to_s4(seq):
    return [S4_AA.get(aa, 20) for aa in seq]


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_s4_tar(tar_path):
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    last_err = None
    for url in S4_WEIGHT_URLS:
        print(f"  下载 S4Pred 权重 {url}", flush=True)
        try:
            subprocess.run(
                ["curl", "-kL", "--fail", "--retry", "3",
                 "--retry-delay", "2", "-o", tar_path, url],
                check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last_err = e
            print(f"  curl 失败，改 urllib（跳过过期证书）: {e}", flush=True)
        try:
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx, timeout=120) as src:
                with open(tar_path, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
            return
        except Exception as e:
            last_err = e
            print(f"  urllib 也失败: {e}", flush=True)
    raise RuntimeError(f"S4Pred 权重下载失败: {last_err}")


def ensure_s4_weights():
    needed = [os.path.join(S4_WEIGHT_DIR, f"weights_{i}.pt") for i in range(1, 6)]
    if all(os.path.exists(p) for p in needed):
        return
    os.makedirs(S4_DIR, exist_ok=True)
    tar_path = os.path.join(S4_DIR, "weights.tar.gz")
    if (not os.path.exists(tar_path)
            or file_md5(tar_path) != S4_WEIGHT_MD5):
        download_s4_tar(tar_path)
        got = file_md5(tar_path)
        if got != S4_WEIGHT_MD5:
            raise RuntimeError(f"S4Pred tar MD5={got} 期望 {S4_WEIGHT_MD5}")
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(S4_DIR)
    if not all(os.path.exists(p) for p in needed):
        extracted = os.path.join(S4_DIR, "weights")
        if os.path.isdir(extracted) and extracted != S4_WEIGHT_DIR:
            os.replace(extracted, S4_WEIGHT_DIR)
    if not all(os.path.exists(p) for p in needed):
        raise FileNotFoundError(f"S4Pred 权重未找到: {S4_WEIGHT_DIR}")


def load_s4pred_teacher():
    """只加载 weights_1。五模型平均对辅任务标签过贵，本机不可行。"""
    ensure_s4_weights()
    net = GRUnet()
    path = os.path.join(S4_WEIGHT_DIR, "weights_1.pt")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def _bucket_batch_size(length):
    for cap, bs in DUMP_BUCKETS:
        if length <= cap:
            return cap, bs
    return DUMP_BUCKETS[-1]


def save_npz_atomic(path, **arrays):
    # numpy savez_compressed 会给不以 .npz 结尾的路径自动加后缀
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def acquire_dump_lock():
    os.makedirs(C018_TMP, exist_ok=True)
    if os.path.exists(SS_LOCK):
        try:
            old = open(SS_LOCK, encoding="utf-8").read().strip()
        except OSError:
            old = "?"
        raise RuntimeError(
            f"已有 C018 dump 在跑（lock pid={old}）。先停掉再手动续："
            f" {SS_LOCK}")
    with open(SS_LOCK, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def release_dump_lock():
    try:
        os.remove(SS_LOCK)
    except FileNotFoundError:
        pass


@torch.no_grad()
def dump_ss_labels(seqs, max_len):
    os.makedirs(C018_TMP, exist_ok=True)
    if os.path.exists(SS_CACHE):
        d = np.load(SS_CACHE)
        if d["ss"].shape[0] == len(seqs) and d["ss"].shape[1] == max_len:
            print(f"  复用 {SS_CACHE}", flush=True)
            return d["ss"]
    n_done = 0
    ss = np.full((len(seqs), max_len), SS_PAD, dtype=np.int16)
    if os.path.exists(SS_PARTIAL):
        try:
            d = np.load(SS_PARTIAL)
            if d["ss"].shape == ss.shape:
                ss = d["ss"]
                n_done = int(d["n_done"])
                print(f"  续 dump n_done={n_done}/{len(seqs)}", flush=True)
            d.close()
        except Exception as e:
            bak = SS_PARTIAL + ".bad"
            os.replace(SS_PARTIAL, bak)
            print(f"  partial 损坏已挪到 {bak}: {e}；从头 dump", flush=True)
            n_done = 0
            ss = np.full((len(seqs), max_len), SS_PAD, dtype=np.int16)
    acquire_dump_lock()
    try:
        return _dump_ss_labels_locked(seqs, max_len, ss, n_done)
    finally:
        release_dump_lock()


def _dump_ss_labels_locked(seqs, max_len, ss, n_done):
    dump_dev = DEVICE
    teacher = load_s4pred_teacher().to(dump_dev)
    teacher.eval()
    print(f"  dump 教师=weights_1  device={dump_dev}  已完成={n_done}", flush=True)

    remain = []
    for i in range(n_done, len(seqs)):
        ids = seq_to_s4(seqs[i][:max_len])
        if ids:
            remain.append((len(ids), i, ids))
    remain.sort(key=lambda t: t[0])
    print(f"  待 dump={len(remain)}（按长度排序分桶）", flush=True)

    def flush_bucket(items, pad_to):
        if not items:
            return
        bsz = len(items)
        x = torch.full((bsz, pad_to), DUMP_PAD_IDX, dtype=torch.long, device=dump_dev)
        lengths = []
        for j, (_idx, ids) in enumerate(items):
            n = min(len(ids), pad_to)
            if n:
                x[j, :n] = torch.as_tensor(ids[:n], dtype=torch.long, device=dump_dev)
            lengths.append(n)
        y = teacher(x)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        lab = y.argmax(-1).cpu().numpy().astype(np.int16)
        for j, (idx, _ids) in enumerate(items):
            n = lengths[j]
            if n:
                ss[idx, :n] = lab[j, :n]

    processed = n_done
    last_saved = n_done
    pbar = tqdm(total=len(seqs) - n_done, desc="S4Pred", unit="seq")
    pending, cur_cap, cur_bs = [], None, None
    for length, idx, ids in remain:
        cap, bs = _bucket_batch_size(length)
        if pending and (cap != cur_cap or len(pending) >= cur_bs):
            flush_bucket(pending, cur_cap)
            pbar.update(len(pending))
            processed += len(pending)
            pending = []
            if processed - last_saved >= DUMP_SAVE_EVERY:
                save_npz_atomic(SS_PARTIAL, ss=ss, n_done=processed)
                last_saved = processed
        cur_cap, cur_bs = cap, bs
        pending.append((idx, ids))
    if pending:
        flush_bucket(pending, cur_cap)
        pbar.update(len(pending))
        processed += len(pending)
    pbar.close()
    save_npz_atomic(SS_CACHE, ss=ss)
    save_npz_atomic(SS_PARTIAL, ss=ss, n_done=len(seqs))
    print(f"  SS dump {SS_CACHE}  非PAD={(ss != SS_PAD).mean():.3f}", flush=True)
    return ss


def load_tokens(seqs, test_seqs, idx_train, idx_val):
    if os.path.exists(TOK_CACHE):
        d = np.load(TOK_CACHE)
        return d["X_tr"], d["X_va"], d["X_te"]

    def clean(s):
        return s.replace("U", "X").replace("Z", "X").replace("O", "X")

    X_tr = c002b.encode_sequences([clean(s) for s in seqs[idx_train]], c002b.MAX_LEN)
    X_va = c002b.encode_sequences([clean(s) for s in seqs[idx_val]], c002b.MAX_LEN)
    X_te = c002b.encode_sequences([clean(s) for s in test_seqs], c002b.MAX_LEN)
    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(TOK_CACHE, X_tr=X_tr, X_va=X_va, X_te=X_te)
    return X_tr, X_va, X_te


def collate_ss(batch):
    seqs = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch])
    ss = torch.stack([item[2] for item in batch]).long()
    one_hot = F.one_hot(seqs, num_classes=c002b.N_CHANNELS).permute(0, 2, 1).float()
    return one_hot, labels, ss


def ss_ce(ss_logit, ss_y):
    b, c, l = ss_logit.shape
    return F.cross_entropy(
        ss_logit.permute(0, 2, 1).reshape(-1, c),
        ss_y.reshape(-1),
        ignore_index=SS_PAD)


def train_epoch(model, loader, optimizer, asl, scaler):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Epoch", unit="batch", leave=False)
    for x, y, ss in pbar:
        x, y, ss = x.to(DEVICE), y.to(DEVICE), ss.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with c002b.autocast_ctx():
            logits, ss_logit = model(x, return_ss=True)
            loss = asl(logits, y) + LAMBDA_SS * ss_ce(ss_logit.float(), ss)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate_probs(model, loader):
    model.eval()
    probs = []
    for batch in loader:
        x = batch[0].to(DEVICE)
        with c002b.autocast_ctx():
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def i019_from_cnn(cnn, deep_rest, sim, y_train, y_val):
    deep = deep_rest + CNN_W * cnn
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def save_last(model, opt, sched, scaler, epoch, best_f1, best_epoch, patience_counter):
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": int(epoch),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, LAST_PATH)


def train_model(model, tr_loader, va_loader, y_val):
    asl = c002b.AsymmetricLoss().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS, eta_min=LR / 100)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_f1, best_epoch, patience_counter = -1.0, 0, 0
    start_epoch = 1
    os.makedirs(MODEL_DIR, exist_ok=True)

    auto_resume = (not FRESH) and os.path.exists(LAST_PATH)
    if auto_resume:
        ckpt = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_f1 = float(ckpt.get("best_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        patience_counter = int(ckpt.get("patience_counter", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  [resume] last={LAST_PATH} 下一轮={start_epoch} "
              f"best_ep={best_epoch} F1(0.5)={best_f1:.4f} "
              f"patience={patience_counter}", flush=True)
        if start_epoch > N_EPOCHS:
            print("  last 已训满，跳过训练", flush=True)
            return best_epoch, best_f1

    for ep in range(start_epoch, N_EPOCHS + 1):
        t_ep = time.time()
        loss = train_epoch(model, tr_loader, opt, asl, scaler)
        ev = evaluate_probs(model, va_loader)
        f1 = f1_05(ev, y_val)
        sched.step()
        print(f"  Epoch {ep:2d}/{N_EPOCHS} | loss={loss:.4f} | "
              f"F1(0.5)={f1:.4f} | {time.time()-t_ep:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch, patience_counter = f1, ep, 0
            torch.save({
                "state_dict": model.state_dict(),
                "best_f1": float(best_f1),
                "best_epoch": int(best_epoch),
            }, MODEL_PATH)
        else:
            patience_counter += 1
        save_last(model, opt, sched, scaler, ep, best_f1, best_epoch, patience_counter)
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {ep}", flush=True)
            break
    print(f"  最佳 epoch: {best_epoch}, F1(0.5)={best_f1:.4f}", flush=True)
    return best_epoch, best_f1


def main():
    global i011, i019
    t0 = time.time()
    print("=" * 60)
    print(f"C018: C002b + S4Pred SS aux λ={LAMBDA_SS:g}")
    print(f"  Device: {DEVICE}")

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    n_labels = len(label_cols)
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    train_ids = train_df["protein_id"].values[idx_train]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_ids)}

    print("[ss] dump S4Pred labels (GPU if available) ...")
    ss_full = dump_ss_labels(seqs, c002b.MAX_LEN)
    if DUMP_ONLY:
        print("  --dump-only 结束")
        return
    ss_tr = ss_full[idx_train]
    ss_va = ss_full[idx_val]

    X_tr, X_va, X_te = load_tokens(seqs, test_df["sequence"].values, idx_train, idx_val)
    tr_loader = DataLoader(
        list(zip(torch.from_numpy(X_tr).long(),
                 torch.from_numpy(y_train.astype(np.float32)),
                 torch.from_numpy(ss_tr.astype(np.int64)))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=collate_ss)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(),
                 torch.from_numpy(y_val.astype(np.float32)),
                 torch.from_numpy(ss_va.astype(np.int64)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=collate_ss)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(),
                 torch.zeros(len(X_te), n_labels),
                 torch.full((len(X_te), c002b.MAX_LEN), SS_PAD, dtype=torch.long))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=collate_ss)

    model = SSAuxCNN(n_labels=n_labels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    if EVAL_ONLY:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        best_epoch, best_f1 = ckpt.get("best_epoch"), ckpt.get("best_f1")
    else:
        print("[train] ...")
        best_epoch, best_f1 = train_model(model, tr_loader, va_loader, y_val)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    print("[infer] ...")
    c018_val = evaluate_probs(model, va_loader)
    c018_test = evaluate_probs(model, te_loader)
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    f1_new = f1_05(c018_val, y_val)
    f1_old = f1_05(cnn42, y_val)
    f1_lock = f1_05(cnn_lock, y_val)
    corr = float(np.corrcoef(cnn_lock.ravel(), c018_val.ravel())[0, 1])
    print(f"  C002b F1(0.5)={f1_old:.4f}  C011a-CNN={f1_lock:.4f}  "
          f"C018={f1_new:.4f}  corr(lock)={corr:.3f}")

    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss_va = ~hit
    sim = i011.ALIGN_W * i011.agg_labels(idx_v, bits_v, y_train) + (
        1.0 - i011.ALIGN_W) * i011.build_kmer_val(y_train)
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    s_lock = i019_from_cnn(cnn_lock, deep_rest, sim, y_train, y_val)
    rec0 = i011.eval_ens(s_lock, y_val, hit, miss_va, "c011a")
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    rows = [rec0]
    configs = {"c011a": s_lock}
    mixed_cnn = {"c011a": cnn_lock}
    for a in A_GRID:
        mix_cnn = (1.0 - a) * cnn_lock + a * c018_val
        s = i019_from_cnn(mix_cnn, deep_rest, sim, y_train, y_val)
        name = f"cnn_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss_va, name, {
            "alpha": a, "f1_05": round(f1_05(mix_cnn, y_val), 6)})
        rec["delta_c011a"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = s
        mixed_cnn[name] = mix_cnn
        print(f"  {name:12s} F1(0.5)={rec['f1_05']:.4f} F1={rec['f1_100']:.4f} "
              f"d={rec['delta_c011a']:+.4f} miss={rec['f1_miss']:.4f}")

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_lock = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss_va], y_val[miss_va]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  独立 {'超' if f1_new > C002B_F05 else '未超'} C002b 0.2714")
    print(f"  OOF 最优 {best_oof['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(PROBS_PATH, val=c018_val, test=c018_test, y_true=y_val)
    report = {
        "n_params": n_params,
        "lambda_ss": LAMBDA_SS,
        "best_epoch": best_epoch,
        "f1_05_c002b": round(f1_old, 6),
        "f1_05_c011a_cnn": round(f1_lock, 6),
        "f1_05_c018": round(f1_new, 6),
        "corr_c011a_c018": round(corr, 6),
        "beat_c002b": bool(f1_new > C002B_F05),
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best_oof,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    if useful:
        report["conclusion"] = "C011a OOF≥+0.001，可换 CNN 路"
    elif f1_new <= C002B_F05:
        report["conclusion"] = "独立未超 C002b 0.2714，SS 辅任务关闭"
    else:
        report["conclusion"] = "独立涨、融合未过 +0.001"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        c005_te = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
        cnn_lock_te = (1.0 - C011A_ALPHA) * cnn_te + C011A_ALPHA * c005_te
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 0.0 if best_name == "c011a" else float(best_name.split("a")[1])
        mix_te = (1.0 - alpha) * cnn_lock_te + alpha * c018_test
        deep_te = (w["t001a"] * np.load(
            os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
                   + w["t002"] * t002_te + w["cnn"] * mix_te)
        tsv_te = os.path.join(i011.CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = i011.parse_aln(tsv_te, test_df["protein_id"].values, full_map)
        sim_bit_te = i011.agg_labels(idx_t, bits_t, y_full)
        sim_kmer_te = 0.0
        for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(
                REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
                else f"CK004_neigh_k{kk}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_kmer_te = sim_kmer_te + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_te = i011.ALIGN_W * sim_bit_te + (1.0 - i011.ALIGN_W) * sim_kmer_te
        mix_v = mixed_cnn[best_name]
        deep_v = deep_rest + CNN_W * mix_v
        s011_v = deep_v + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_v, sim, K_RRF))
        t_sub = i011.search_t(configs[best_name], y_val)
        t_i011 = i011.search_t(s011_v, y_val)
        n_te = deep_te.shape[0]
        s011_te = deep_te + i011.SIM_W * sim_te + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_te, sim_te, K_RRF))
        pi = y_train.mean(0).astype(np.float64)
        row_m = np.maximum((s011_te >= t_i011).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(pi * n_te * RHO, 1e-6)
        col_m *= row_m.sum() / col_m.sum()
        ker = np.exp((s011_te - s011_te.max(axis=1, keepdims=True)) / TAU_SK)
        p_te = i019.sinkhorn(ker, row_m, col_m)
        s_te = i019.mix(s011_te, p_te, ALPHA_SK)
        y_pred = (s_te >= t_sub).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_C018_ssaux.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["submit_alpha"] = alpha
        print(f"  提交: {sub_path}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  概率: {PROBS_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
