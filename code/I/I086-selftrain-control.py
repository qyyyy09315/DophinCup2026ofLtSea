"""
I086 — 单轮带护栏自训练的受控对照 (val 模拟测试池)
====================================================
背景: 「纯伪标签迭代 / 熵最小化」在文献与项目层面均已排除, 但
「单轮 + 只加正例伪标签 + 负例侧不监督(masked loss)」这一温和变体从未本地实测。
本实验用 val 模拟测试池给出决定性证据:
  M0 在 train(80%) 上训练 -> 给 val 打伪标签(只取高置信正例) -> 从头重训于 train+伪标签 -> val 真值评估。
若有效, 则真实场景同样操作 (train + 伪标签化的 28,450 条测试序列重训) 即可吃到无标注数据;
若无效, 用本地证据正式关闭该路线 (文献警告 + 本地实测双关闭)。

护栏设计 (文献驱动):
  - 单轮, 不迭代 (Arazo 2020: 迭代放大确认偏差)
  - 只加正例, 负例侧完全无监督 (PU 式; 假负例是多标签自训练的主要伤害源)
  - 伪正例按「训练流行率 x factor」的分位数截断, factor 控制激进程度
变体:
  S0 基线 (仅 train)                  <- 锚, 应复现 ~0.26-0.27 (单种子)
  S1 +伪(0.5x流行率, 硬)  λ=1.0
  S2 +伪(1.0x流行率, 硬)  λ=1.0
  S3 +伪(0.5x, 软目标=概率) λ=0.5
  S4 +伪(0.5x, 硬)        λ=0.3
  S5 上限: val 真值同掩码方式 (伪标签完美时的机制上限)
诊断: 各 factor 下伪正例精度 (对照 val 真值)。
判据: 变体 val Macro F1 (SCut) vs S0。
"""
import os, time, json, io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT = os.path.join(REL_DIR, "I086_report.json")
N_LABELS = 500
EPOCHS = 35
BATCH = 512
SEED = 42

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
MLP, macro_f1, search_scut = i064.MLP, i064.macro_f1, i064.search_scut


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return pos + neg


def train_variant(Xall, T, M, w_map, tag):
    """从头训练。T: 目标矩阵 (train 真值 / val 伪标签), M: 监督掩码, w_map: 每行损失权重。"""
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    tl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xall), torch.from_numpy(T),
                                       torch.from_numpy(M), torch.from_numpy(w_map)),
        batch_size=BATCH, shuffle=True, pin_memory=True)
    m = MLP(Xall.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit = ASL(); sc = torch.amp.GradScaler("cuda")
    for ep in tqdm(range(EPOCHS), desc=f"  {tag}", ncols=90):
        m.train()
        for bx, bt, bm, bw in tl:
            bx, bt, bm, bw = (x.cuda(non_blocking=True) for x in (bx, bt, bm, bw))
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                l = crit(m(bx), bt) * bm * bw[:, None]
                l = l.sum() / bm.sum().clamp(min=1.0)
            sc.scale(l).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt); sc.update()
        sch.step()
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(Xall), 1024):
            bx = torch.from_numpy(Xall[s:s + 1024]).cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out.append(torch.sigmoid(m(bx).float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    t0 = time.time()
    print("=" * 96)
    print("  I086: Single-round guarded self-training — controlled test (val as mock test pool)")
    print("=" * 96)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_tr, n_va = len(idx_tr), len(idx_va)
    prev = y_tr.mean(0)                       # 训练流行率 (伪标签分位数的依据)

    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    Xtr, Xva = X[idx_tr], X[idx_va]
    print(f"  特征 {X.shape} | train {n_tr} | val {n_va}")

    # ---- S0 基线: 仅 train (val 行全部掩码, 只监督 train; 统一在拼接矩阵上预测) ----
    Xall0 = np.concatenate([Xtr, Xva], axis=0)
    T0 = np.concatenate([y_tr.astype(np.float32), np.zeros((n_va, N_LABELS), np.float32)], axis=0)
    M0 = np.concatenate([np.ones((n_tr, N_LABELS), np.float32),
                         np.zeros((n_va, N_LABELS), np.float32)], axis=0)
    w0 = np.ones(n_tr + n_va, np.float32)
    P0 = train_variant(Xall0, T0, M0, w0, "S0 基线")[-n_va:]
    f0 = macro_f1(P0, y_va, search_scut(P0, y_va))
    print(f"  S0 val F1 = {f0:.4f} (应 ~0.26 单种子水平)")
    assert f0 > 0.20, f"S0 基线异常: {f0}"
    res = {"S0_baseline": f0}
    ones_tr = np.ones((n_tr, N_LABELS), np.float32)

    # ---- 伪标签构造 (基于 S0 的 val 预测) ----
    def make_pseudo(factor, hard=True):
        T = np.zeros((n_va, N_LABELS), np.float32)
        prec = []
        for j in range(N_LABELS):
            k = max(int(prev[j] * factor * n_va), 1)
            k = min(k, n_va - 1)
            order = np.argsort(-P0[:, j])[:k]
            T[order, j] = P0[order, j] if not hard else 1.0
            prec.append(float(y_va[order, j].mean()))
        return T, float(np.mean(prec))

    variants = [("S1 factor0.5 硬 λ=1.0", 0.5, True, 1.0),
                ("S2 factor1.0 硬 λ=1.0", 1.0, True, 1.0),
                ("S3 factor0.5 软 λ=0.5", 0.5, False, 0.5),
                ("S4 factor0.5 硬 λ=0.3", 0.5, True, 0.3)]
    diag = {}
    for name, factor, hard, lam in variants:
        Tv, prec = make_pseudo(factor, hard)
        diag[name] = {"factor": factor, "pseudo_precision": prec}
        print(f"  {name}: 伪正例精度 = {prec:.4f}")
        T = np.concatenate([y_tr.astype(np.float32), Tv], axis=0)
        M = np.concatenate([ones_tr, (Tv > 0).astype(np.float32)], axis=0)
        w = np.concatenate([np.ones(n_tr), np.full(n_va, lam)]).astype(np.float32)
        Xall = np.concatenate([Xtr, Xva], axis=0)
        P = train_variant(Xall, T, M, w, name)[-n_va:]
        f = macro_f1(P, y_va, search_scut(P, y_va))
        res[name] = f
        print(f"  {name} val F1 = {f:.4f} ({f - f0:+.4f} vs S0)", flush=True)

    # ---- S5 上限: val 真值同掩码 (伪标签完美时的机制上限) ----
    T5 = np.concatenate([y_tr.astype(np.float32), y_va.astype(np.float32)], axis=0)
    M5 = np.concatenate([ones_tr, (y_va > 0).astype(np.float32)], axis=0)
    w5 = np.concatenate([np.ones(n_tr), np.full(n_va, 1.0)]).astype(np.float32)
    P5 = train_variant(np.concatenate([Xtr, Xva], axis=0), T5, M5, w5, "S5 真值上限")[-n_va:]
    f5 = macro_f1(P5, y_va, search_scut(P5, y_va))
    res["S5_true_label_ceiling"] = f5
    print(f"  S5 (真值同掩码上限) val F1 = {f5:.4f} ({f5 - f0:+.4f} vs S0)")

    best = max((k for k in res if k != "S0_baseline" and not k.startswith("S5")), key=lambda k: res[k])
    d = res[best] - f0
    verdict = ("自训练有效" if d > 0.0005 else
               "无增益, 本地证据正式关闭该路线" if d > -0.0005 else
               "有害, 正式关闭该路线")
    print("\n" + "=" * 96)
    print(f"  最优变体: {best} ({res[best]:.4f}, {d:+.4f} vs S0)")
    print(f"  判定: {verdict}")
    print("=" * 96)

    out = {"results": res, "pseudo_diag": diag, "verdict": verdict, "seed": SEED,
           "note": "S5 为伪标签完美时的机制上限; 各变体均为单轮/只加正例/负例无监督"}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
