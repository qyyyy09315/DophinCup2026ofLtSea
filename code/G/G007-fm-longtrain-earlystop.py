"""
G007 — 流匹配生成端扩轮数 × C2ST 放行率早停 × I096 收敛头下游（G 线复活主刀）
================================================================
背景（计划文档 1.2/3.4）:
  - G001~G006 关闭的原因是三因子联合: 浅生成器短训 × 35ep 弱下游头 × 无物理约束。
    G004 结论原话: "合成样本在 35 epoch 浅层 MLP 下未能转化"。
  - 本刀在 G004 配方上只改两处（消融边界）:
    1. 生成器迭代轮数 20 -> 100 epoch (×5)，且每 10 epoch 用快速 C2ST 放行率评估择优，
       patience=5 次评估（50 epoch）无提升则早停——防止"过训流形外推"与"欠训保真不足"两头失败;
    2. 下游分类头从 35ep 浅头升级为 I096 收敛协议（平坦 MLP / 80ep / 逐轮 SCut 早停 patience=10）。
  - 物理约束不在本刀（留给 G008），保持单变量消融。
中间指标: C2ST 严格区间 [0.35, 0.65] 放行率 > 11%（G004 记录）起、> 30% 良好（仅诊断，不入表）。
判据:
  - Gate 1 独立: SCut F1 > 0.3036（I096 新锚点）
  - Gate 2 融合: 半折 OOF >= 0.4396（基线 0.4386 + 0.001）
用法:
  python code/G/G007-fm-longtrain-earlystop.py            # 正式 (~25-30 min)
  python code/G/G007-fm-longtrain-earlystop.py --smoke    # 沙箱冒烟 (~40s)
输出:
  models/g007_flow_net_best.pt, models/g007_plm_head_80ep.pt
  rel/g007_flow_val_probs.npz, rel/G007_report.json
"""

import os
import json
import copy
import time
import argparse

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
EMB_DIM = 1664
SEED = 42

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
g004 = SourceFileLoader(
    "g004", os.path.join(PROJECT_ROOT, "code", "G", "G004-latent-flow-matching.py")
).load_module()
VectorFieldNet = g004.VectorFieldNet
load_1664d_features = g004.load_1664d_features
build_sim = g004.build_sim
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


@torch.no_grad()
def ode_sample(vnet, y_target, device, ode_steps=20):
    """从标准高斯出发沿学习到的速度场做 Euler 积分（G004 同款）"""
    vnet.eval()
    n = y_target.shape[0]
    y_t = torch.from_numpy(y_target).to(device)
    x = torch.randn(n, EMB_DIM, device=device)
    dt = 1.0 / ode_steps
    for s in range(ode_steps):
        t_curr = torch.full((n, 1), s * dt, device=device)
        v = vnet(x, t_curr, y_t)
        x = x + dt * v
    return x.cpu().numpy()


def c2st_pass_rate(X_real, fake_X, seed=SEED):
    """C2ST 严格区间放行率（G004 同款: LogisticRegression + [0.35, 0.65]）"""
    n_eval = min(fake_X.shape[0], X_real.shape[0])
    rng = np.random.RandomState(seed)
    real_idx = rng.choice(len(X_real), size=n_eval, replace=False)
    X_c2st = np.vstack([X_real[real_idx], fake_X])
    y_c2st = np.hstack([np.ones(n_eval), np.zeros(fake_X.shape[0])])
    clf = LogisticRegression(max_iter=200, C=0.1, random_state=seed)
    clf.fit(X_c2st, y_c2st)
    prob_real = clf.predict_proba(fake_X)[:, 1]
    mask = (prob_real >= 0.35) & (prob_real <= 0.65)
    return float(mask.mean()), prob_real, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="沙箱快速冒烟 (~40s)")
    parser.add_argument("--epochs-fm", type=int, default=100, help="流匹配训练轮数 (默认 100, G004 为 20)")
    parser.add_argument("--eval-every", type=int, default=10, help="C2ST 放行率评估间隔 (默认 10 epoch)")
    parser.add_argument("--patience", type=int, default=5, help="放行率早停耐心 (默认 5 次评估 = 50 epoch)")
    parser.add_argument("--epochs-head", type=int, default=80, help="下游头训练轮数 (默认 80, I096 协议)")
    parser.add_argument("--head-patience", type=int, default=10, help="下游头 SCut 早停耐心 (默认 10)")
    parser.add_argument("--n-synth", type=int, default=3000, help="合成样本量 (默认 3000)")
    parser.add_argument("--ode-steps", type=int, default=20, help="ODE 积分步数 (默认 20)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print(f"  G007: Flow-Matching Long-Train + C2ST-Rate Early-Stop + I096 Converged Head "
          f"({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | FM Epochs: {args.epochs_fm if not args.smoke else 4} | "
          f"Eval Every: {args.eval_every} | Head: {args.epochs_head}ep+SCut早停{args.head_patience}")
    print("=" * 84)

    print("[1/6] 加载 1664D 特征矩阵与标签...")
    X_full = load_1664d_features()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    X_tr, y_tr = X_full[idx_tr], y_full[idx_tr]
    X_va, y_va = X_full[idx_va], y_full[idx_va]

    if args.smoke:
        X_tr, y_tr = X_tr[:2048], y_tr[:2048]
        X_va, y_va = X_va[:512], y_va[:512]
        fm_epochs, eval_every, fm_patience = 4, 2, 2
        head_epochs, head_patience = 2, 2
        n_synth, ode_steps = 200, 5
        c2st_n = 200
    else:
        fm_epochs, eval_every, fm_patience = args.epochs_fm, args.eval_every, args.patience
        head_epochs, head_patience = args.epochs_head, args.head_patience
        n_synth, ode_steps = args.n_synth, args.ode_steps
        c2st_n = 1000  # 快速评估合成子集规模

    print(f"  训练集: {X_tr.shape[0]} 条, 验证集: {X_va.shape[0]} 条, 特征维度: {X_tr.shape[1]}")

    # 2. 流匹配训练 + 放行率监控早停择优
    print(f"\n[2/6] 训练条件速度场网络 ({fm_epochs} epochs, 每 {eval_every} epoch 评估放行率)...")
    loader_fm = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=512, shuffle=True, pin_memory=True,
    )
    vnet = VectorFieldNet().to(device)
    opt_fm = torch.optim.AdamW(vnet.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_fm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fm, T_max=fm_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 靶向切片（G004 同款: 基数 <=10 优先）
    cardinality = y_tr.sum(axis=1)
    target_indices = np.where(cardinality <= 10)[0]
    if len(target_indices) < n_synth:
        target_indices = np.arange(len(y_tr))

    best_rate, best_eval_stall, best_vnet_state = -1.0, 0, None
    rate_history = []
    for ep in range(1, fm_epochs + 1):
        t_ep = time.time()
        vnet.train()
        loss_sum, n_seen = 0.0, 0
        for real_x1, real_y in loader_fm:
            real_x1 = real_x1.to(device, non_blocking=True)
            real_y = real_y.to(device, non_blocking=True)
            b = real_x1.size(0)
            x0 = torch.randn_like(real_x1)
            t = torch.rand(b, 1, device=device)
            xt = (1.0 - t) * x0 + t * real_x1
            target_v = real_x1 - x0
            opt_fm.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred_v = vnet(xt, t, real_y)
                loss_fm = ((pred_v - target_v) ** 2).mean()
            if scaler is not None:
                scaler.scale(loss_fm).backward()
                scaler.step(opt_fm)
                scaler.update()
            else:
                loss_fm.backward()
                opt_fm.step()
            loss_sum += loss_fm.item() * b
            n_seen += b
        sch_fm.step()

        if ep % eval_every == 0 or ep == fm_epochs:
            rng = np.random.RandomState(SEED + ep)
            probe_idx = rng.choice(target_indices, size=min(c2st_n, len(target_indices)), replace=True)
            fake_probe = ode_sample(vnet, y_tr[probe_idx], device, ode_steps)
            rate, _, _ = c2st_pass_rate(X_tr, fake_probe)
            rate_history.append({"epoch": ep, "c2st_pass_rate": round(rate, 4)})
            mark = ""
            if rate > best_rate:
                best_rate, best_vnet_state = rate, copy.deepcopy(vnet.state_dict())
                best_eval_stall = 0
                mark = "(*)"
            else:
                best_eval_stall += 1
            print(f"  flow ep{ep:03d}/{fm_epochs:03d} | fm_mse={loss_sum/n_seen:.4f} | "
                  f"C2ST放行率={rate:.1%} | best={best_rate:.1%} {mark} "
                  f"({time.time()-t_ep:.0f}s)", flush=True)
            if best_eval_stall >= fm_patience:
                print(f"  [生成早停] 连续 {fm_patience} 次评估放行率未提升，早停于 epoch {ep}")
                break
        else:
            print(f"  flow ep{ep:03d}/{fm_epochs:03d} | fm_mse={loss_sum/n_seen:.4f} "
                  f"({time.time()-t_ep:.0f}s)", flush=True)

    if best_vnet_state is not None:
        vnet.load_state_dict(best_vnet_state)
    print(f"  [择优] 最优放行率 {best_rate:.1%}（G004 记录 11.0%）")

    # 3. 用最优检查点合成 n_synth 条
    print(f"\n[3/6] 用择优速度场合成 {n_synth} 条 (ODE {ode_steps} 步)...")
    np.random.seed(SEED)
    chosen_idx = np.random.choice(target_indices, size=n_synth, replace=True)
    y_target = y_tr[chosen_idx]
    fake_X = ode_sample(vnet, y_target, device, ode_steps)
    print(f"  生成向量均值 {fake_X.mean():.4f} / 标准差 {fake_X.std():.4f}")

    # 4. 最终 C2ST 门控（G004 同款，含 50% 拟真保底）
    print("[4/6] 最终 C2ST 门控...")
    rate_final, prob_real, valid_mask = c2st_pass_rate(X_tr, fake_X)
    n_kept = int(valid_mask.sum())
    if n_kept >= 100:
        X_final = np.vstack([X_tr, fake_X[valid_mask]])
        y_final = np.vstack([y_tr, y_target[valid_mask]])
    else:
        print("  [说明] 放行不足，使用最接近中心的 50% 拟真保底（G004 同款）")
        top_half = np.argsort(np.abs(prob_real - 0.5))[: max(n_synth // 2, 100)]
        X_final = np.vstack([X_tr, fake_X[top_half]])
        y_final = np.vstack([y_tr, y_target[top_half]])
        n_kept = len(top_half)
    print(f"  最终放行率 {rate_final:.1%} ({n_kept}/{n_synth})，扩增 {len(X_tr)} -> {len(X_final)}")

    # 5. 下游 I096 收敛头（平坦 MLP / 80ep / SCut 早停 patience=10）
    print(f"\n[5/6] 扩增数据上训练 I096 收敛头 ({head_epochs}ep + SCut 早停 {head_patience})...")
    loader_hd_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_final), torch.from_numpy(y_final)),
        batch_size=512, shuffle=True, pin_memory=True,
    )
    loader_hd_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=512, shuffle=False, pin_memory=True,
    )
    head = MLP(EMB_DIM).to(device)
    crit = ASL()
    opt_h = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=head_epochs, eta_min=1e-5)
    sc2 = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1, best_ep, stall, best_state = 0.0, 0, 0, None
    for ep in range(1, head_epochs + 1):
        t_ep = time.time()
        head.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader_hd_tr:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt_h.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(head(bx), by)
            if sc2 is not None:
                sc2.scale(loss).backward()
                sc2.step(opt_h)
                sc2.update()
            else:
                loss.backward()
                opt_h.step()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch_h.step()

        head.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in loader_hd_va:
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(head(bx.to(device, non_blocking=True)).float()).cpu().numpy()
                preds.append(pr)
        val_probs = np.vstack(preds)
        f1_05 = macro_f1(val_probs, y_va.astype(np.int8), 0.5)
        ths_e = search_scut(val_probs, y_va.astype(np.int8))
        f1_sc = macro_f1(val_probs, y_va.astype(np.int8), ths_e)
        if f1_sc > best_val_f1:
            best_val_f1, best_ep, stall = f1_sc, ep, 0
            best_state = copy.deepcopy(head.state_dict())
            mark = "(*)"
        else:
            stall += 1
            mark = "   "
        print(f"  head ep{ep:02d}/{head_epochs:02d} | loss={loss_sum/n_seen:.4f} | "
              f"val_f1(0.5)={f1_05:.4f} | val_f1(SCut)={f1_sc:.4f} | best={best_val_f1:.4f}@ep{best_ep:02d} {mark} "
              f"({time.time()-t_ep:.0f}s)", flush=True)
        if stall >= head_patience:
            print(f"  [早停] 连续 {head_patience} 轮未提升，早停于 epoch {ep}")
            break

    head.load_state_dict(best_state)
    head.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader_hd_va:
            with torch.amp.autocast("cuda"):
                pr = torch.sigmoid(head(bx.to(device, non_blocking=True)).float()).cpu().numpy()
            preds.append(pr)
    final_val_probs = np.vstack(preds)
    ths_star = search_scut(final_val_probs, y_va.astype(np.int8))
    standalone_f1 = macro_f1(final_val_probs, y_va.astype(np.int8), ths_star)
    standalone_auc = float(roc_auc_score(y_va.astype(np.int8), final_val_probs, average="macro"))
    yb = y_va.astype(bool)
    bp = final_val_probs >= ths_star
    tp = (bp & yb).sum(0); fp = (bp & ~yb).sum(0); fn = (~bp & yb).sum(0)
    p_arr = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r_arr = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    standalone_prec, standalone_rec = float(p_arr.mean()), float(r_arr.mean())

    gate1_pass = standalone_f1 > 0.3036
    print(f"  [Gate 1 独立门槛] 独立 F1={standalone_f1:.4f}, AUC={standalone_auc:.4f}, "
          f"R={standalone_rec:.4f}, P={standalone_prec:.4f} (I096 无扩增收敛头锚点 0.3036) "
          f"-> {'通过' if gate1_pass else '未通过'}")

    # 6. Gate 2 融合流水线
    print("\n[6/6] canonical 融合流水线评估 Gate 2 (RRF + Sinkhorn + 半折 OOF)...")
    if not args.smoke:
        cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
        c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
        t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
        t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
        with open(TCK002_CONFIG, encoding="utf-8") as f:
            wc = json.load(f)["best_w"]
        deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)
        val_pids = train_df["protein_id"].values[idx_va]
        train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
        val_map = {p: i for i, p in enumerate(val_pids)}
        sim = build_sim(y_tr.astype(np.int8), len(idx_va), val_pids, train_map, val_map)
        f1_fused_in, auc_fused_in, mixed_scores, _, _ = full_pipe(
            deep4, sim, y_tr.astype(np.int8), y_va.astype(np.int8), len(idx_va),
            plm=final_val_probs, w=0.35,
        )
        a_idx, b_idx = train_test_split(np.arange(len(idx_va)), test_size=0.5, random_state=SEED)
        preds_oof = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
            t_sub = search_scut(mixed_scores[fi], y_va.astype(np.int8)[fi])
            preds_oof[ei] = mixed_scores[ei] >= t_sub
        yb = y_va.astype(bool)
        tp = (preds_oof & yb).sum(0); fp = (preds_oof & ~yb).sum(0); fn = (~preds_oof & yb).sum(0)
        p_c = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r_c = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1_oof = float(np.mean(np.where(p_c + r_c > 0, 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9), 0.0)))
        delta_oof = f1_oof - 0.4386
        gate2_pass = delta_oof >= 0.001
        print(f"  [Gate 2 融合门槛] 融合样内 F1={f1_fused_in:.4f}, AUC={auc_fused_in:.4f}, "
              f"半折 OOF F1={f1_oof:.4f} (基线 0.4386, ΔOOF={delta_oof:+.4f}) "
              f"-> {'通过' if gate2_pass else '未通过'}")
    else:
        f1_fused_in, auc_fused_in, f1_oof, delta_oof = 0.0, 0.0, 0.0, 0.0
        gate2_pass = False
        print("  [SMOKE] 跳过完整融合管线")

    if gate1_pass and gate2_pass:
        conclusion = "有效，G 线复活，并入 I100 全量换锁流程"
    elif best_rate > 0.11 and not gate2_pass:
        conclusion = "放行率提升但融合未过线，转 G008 物理约束刀"
    else:
        conclusion = "未达门槛，关闭"
    report = {
        "experiment": "G007",
        "method": "Flow-Matching Long-Train + C2ST-Rate Early-Stop + I096 Converged Head",
        "smoke": args.smoke,
        "fm_epochs_trained": fm_epochs,
        "rate_history": rate_history,
        "best_c2st_pass_rate_probe": round(best_rate, 4),
        "final_c2st_pass_rate": round(rate_final, 4),
        "n_synth": n_synth,
        "n_kept": n_kept,
        "ode_steps": ode_steps,
        "standalone_plm": {
            "f1_scut": round(standalone_f1, 4),
            "precision": round(standalone_prec, 4),
            "recall": round(standalone_rec, 4),
            "auc": round(standalone_auc, 4),
            "gate1_pass": gate1_pass,
            "anchor_i096_no_aug": 0.3036,
            "g004_35ep_head": 0.2597,
        },
        "fused_pipeline": {
            "f1_in_sample": round(f1_fused_in, 4),
            "auc_in_sample": round(auc_fused_in, 4),
            "f1_half_fold_oof": round(f1_oof, 4),
            "baseline_oof": 0.4386,
            "delta_oof": round(delta_oof, 4),
            "gate2_pass": gate2_pass,
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with Path(os.path.join(REL_DIR, "G007_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.smoke:
        torch.save(vnet.state_dict(), os.path.join(MODEL_DIR, "g007_flow_net_best.pt"))
        torch.save(head.state_dict(), os.path.join(MODEL_DIR, "g007_plm_head_80ep.pt"))
        np.savez_compressed(os.path.join(REL_DIR, "g007_flow_val_probs.npz"),
                            val=final_val_probs.astype(np.float32))

    print("\n" + "=" * 84)
    print(f"  G007 完成! 放行率 {best_rate:.1%} | 独立 F1={standalone_f1:.4f} | "
          f"融合 OOF={f1_oof:.4f} (Δ={delta_oof:+.4f}) -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'G007_report.json')} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    main()
