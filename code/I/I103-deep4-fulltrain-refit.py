"""
I103 — deep4 全量重训（100% 数据 × 充分收敛，提交侧最后一块全量拼图）
========================================================================
背景:
  - 平台两次证实「全量数据 × 充分收敛」类改动的提交侧增益（本地 val 不可见）:
    I081 PLM 头全量重训 +0.1881、I097 78ep 收敛三种子 +0.0090（现平台锁 45.9636）。
  - deep4（T001a/T002 Transformer + C002b/C005 CNN，占融合 65% 权重）仍用 80% 模型——
    最后一块未全量化拼图。本刀在 100% 全量数据（113,796）上按各原配方重训，
    test 概率替换 deep4、tck002 权重沿用（I081/I097 同逻辑），阈值 val 管线重校。
协议差异声明（相对各原配方，唯一改动）:
  - 数据: 80% -> 100% 全量（无 val，不早停，跑满原 epoch 数）;
  - 调度: ReduceLROnPlateau(需 val 指标) -> CosineAnnealingLR(T_max=N_EPOCHS, eta_min=lr/100)，
    与 I096/I097 收敛协议一致; 其余（模型结构/损失/bs/最大长度/epochs/lr/wd）逐项保持原配方。
用法（每模型独立一段 GPU 会话；中断后重跑同一条命令即自动续跑，勿加 --fresh）:
  python code/I/I103-deep4-fulltrain-refit.py --model c002b
  python code/I/I103-deep4-fulltrain-refit.py --model c005
  python code/I/I103-deep4-fulltrain-refit.py --model t001a
  python code/I/I103-deep4-fulltrain-refit.py --model t002
  python code/I/I103-deep4-fulltrain-refit.py --model c002b --fresh   # 仅需放弃进度重跑时
输出:
  models/i103_{model}_full.pt（最终权重）+ i103_{model}_last.pt（断点）
  rel/i103_{model}_test_probs.npz（键 test）
预计耗时（RTX 4060 单卡）: c002b ~2.5h / c005 ~2.5h / t001a ~4-5h / t002 ~2.5h
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

CONFIGS = {
    "c002b": {"module": "C002b-cnn.py", "series": "C", "epochs": 80, "bs": 256,
              "max_len": 1500, "lr": 1e-3, "wd": 1e-5, "eta_min_frac": 100},
    "c005": {"module": "C005-csra.py", "series": "C", "epochs": 80, "bs": 256,
             "max_len": 1500, "lr": 1e-3, "wd": 1e-5, "eta_min_frac": 100},
    "t001a": {"module": "T001a-rope.py", "series": "T", "epochs": 50, "bs": 128,
              "max_len": 512, "lr": 3e-4, "wd": 1e-5, "eta_min_frac": 100},
    "t002": {"module": "T002-rope.py", "series": "T", "epochs": 30, "bs": 128,
             "max_len": 512, "lr": 1e-3, "wd": 1e-5, "eta_min_frac": 100},
}


def encode_cached(encode_fn, sequences, max_len, cache_path, tag):
    """编码结果 uint8 npz 缓存（断点续跑时免重复编码）；长度指纹校验"""
    import struct
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        if d["arr"].shape == (len(sequences), max_len):
            print(f"  [编码缓存命中] {tag}: {cache_path}")
            return d["arr"]
    arr = encode_fn(sequences, max_len)
    np.savez_compressed(cache_path, arr=arr)
    print(f"  [编码缓存写入] {tag}: {cache_path} ({arr.nbytes/1e6:.0f} MB)")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIGS))
    parser.add_argument("--fresh", action="store_true", help="忽略/清除断点，从头重跑（默认自动续跑）")
    parser.add_argument("--smoke", action="store_true", help="小样本冒烟（512 条 / 2 epoch）")
    args = parser.parse_args()
    cfg = CONFIGS[args.model]
    model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print("=" * 88)
    print(f"  I103: deep4 full-data refit -- {model_name} ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | epochs={cfg['epochs']} bs={cfg['bs']} max_len={cfg['max_len']} "
          f"lr={cfg['lr']} | 调度=Cosine(全量无早停)")
    print("=" * 88)

    # 1. 原脚本构件
    mod = SourceFileLoader(f"i103_{model_name}_orig",
                           os.path.join(PROJECT_ROOT, "code", cfg["series"], cfg["module"])).load_module()
    if model_name == "c005":
        c002b = mod.c002b  # C005 模块级已加载 c002b

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    sequences = train_df["sequence"].values
    test_seqs = test_df["sequence"].values
    n_labels = len(label_cols)
    print(f"  100% 训练集: {len(train_df)}, 测试集: {len(test_df)}, 标签: {n_labels}")

    # 2. 模型与损失（原构造）
    if model_name == "c002b":
        model = mod.MultiScaleCNNv3(n_labels=n_labels).to(device)
        criterion = mod.AsymmetricLoss()
        ds_cls, collate = mod.ProteinDataset, mod.collate_onehot
    elif model_name == "c005":
        model = mod.MultiScaleCNNCSRA(n_labels=n_labels).to(device)
        criterion = c002b.AsymmetricLoss()
        ds_cls, collate = c002b.ProteinDataset, c002b.collate_onehot
    elif model_name in ("t001a", "t002"):
        model = mod.TransformerClassifier(n_labels=n_labels).to(device)
        pos = y_full.sum(axis=0).astype(np.float32)
        neg = len(y_full) - pos
        pos_weight = np.ones(n_labels, dtype=np.float32)
        mask = pos > 0
        pos_weight[mask] = np.clip(neg[mask] / pos[mask], 1.0, mod.POS_WEIGHT_CLIP)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device))
        ds_cls, collate = mod.ProteinDataset, None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 3. 全量 DataLoader（100% 数据，无 val）
    idx_train = np.arange(len(train_df))  # 100% 全量
    if args.smoke:
        idx_train = idx_train[:512]
    suffix = "_smoke" if args.smoke else ""
    impl_mod = mod.c002b if model_name == "c005" else mod  # C005 的 encode/get_probs 均在 c002b
    enc_train = encode_cached(impl_mod.encode_sequences, sequences[idx_train], cfg["max_len"],
                              os.path.join(REL_DIR, f"i103_{model_name}_enc_train{suffix}.npz"),
                              f"{model_name}/train")
    enc_test = encode_cached(impl_mod.encode_sequences, test_seqs, cfg["max_len"],
                             os.path.join(REL_DIR, f"i103_{model_name}_enc_test{suffix}.npz"),
                             f"{model_name}/test")
    if model_name in ("c002b", "c005"):
        train_ds = ds_cls(sequences[idx_train], y_full[idx_train].astype(np.float32), cfg["max_len"])
        train_ds.seqs = enc_train  # 编码缓存直填，免重复 encode
        test_ds = ds_cls(test_seqs, np.zeros((len(test_df), n_labels), dtype=np.int8), cfg["max_len"])
        test_ds.seqs = enc_test
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                                                   collate_fn=collate)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg["bs"], shuffle=False,
                                                  collate_fn=collate)
    else:
        train_ds = ds_cls(sequences[idx_train], y_full[idx_train].astype(np.float32), cfg["max_len"])
        train_ds.seqs = enc_train
        test_ds = ds_cls(test_seqs, np.zeros((len(test_df), n_labels), dtype=np.int8), cfg["max_len"])
        test_ds.seqs = enc_test
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg["bs"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] / cfg["eta_min_frac"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    last_path = os.path.join(MODEL_DIR, f"i103_{model_name}_last{suffix}.pt")
    full_path = os.path.join(MODEL_DIR, f"i103_{model_name}_full{suffix}.pt")
    probs_path = os.path.join(REL_DIR, f"i103_{model_name}_test_probs{suffix}.npz")
    start_epoch = 1
    if args.fresh:
        if os.path.exists(last_path):
            os.remove(last_path)
            print("  [fresh] 已清除断点，从头重跑")
    elif os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if "scaler" in ck and scaler is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"  [自动续跑] 检测到断点，从 epoch {start_epoch} 继续（--fresh 放弃进度重跑）")
    if not args.fresh and os.path.exists(probs_path) and os.path.exists(full_path):
        print(f"  [跳过] {model_name} 已完成（{probs_path} 已存在；--fresh 可强制重跑）")
        return

    # 4. 训练（跑满，无早停；单行日志）
    epochs_eff = 2 if args.smoke else cfg["epochs"]
    n_steps = (len(idx_train) + cfg["bs"] - 1) // cfg["bs"]
    for epoch in range(start_epoch, epochs_eff + 1):
        t_ep = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(bx) if model_name in ("c002b", "c005") else model(bx, (bx != mod.PAD_IDX).long())
                loss = criterion(logits, by)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        scheduler.step()
        print(f"  {model_name} ep{epoch:02d}/{epochs_eff:02d} | loss={loss_sum/n_seen:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {n_steps} 步 ({time.time()-t_ep:.0f}s)", flush=True)
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch}
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, last_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. test 推理
    print(f"\n[推理] test 概率 ({len(test_df)} 条)...")
    test_probs, _ = impl_mod.get_probs(model, test_loader)
    test_probs = test_probs.astype(np.float32)
    np.savez_compressed(probs_path, test=test_probs)
    torch.save(model.state_dict(), full_path)
    print(f"  概率均值 {test_probs.mean():.4f} | 已存 {probs_path}")
    print(f"  最终权重 {full_path}")

    report = {"experiment": "I103", "model": model_name, "smoke": args.smoke,
              "n_train": int(len(idx_train)), "epochs": epochs_eff,
              "test_prob_mean": round(float(test_probs.mean()), 4),
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, f"I103_{model_name}_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  I103/{model_name} 完成! 总用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
