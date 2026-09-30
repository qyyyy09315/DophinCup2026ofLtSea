"""
protbert-finetune-eval — 仅加载已保存的 ProtBERT 微调模型并评估/提交
=====================================================================
用法:
  python protbert-finetune-eval.py
前提:
  rel/protbert_tokens_train.npz, rel/protbert_tokens_test.npz 已存在
  models/protbert-finetune.pt 已保存 (训练脚本每个 epoch 验证后生成)
输出:
  submit/submission_ProtBERT_finetune.csv
  rel/ProtBERT_finetune_val_metrics.json
  models/protbert-finetune-eval.pt
"""

import importlib.util
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score)
from pathlib import Path

spec = importlib.util.spec_from_file_location("protbert_finetune", "B004a-protbert-finetune.py")
pbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbf)

EVAL_MODEL_PATH = "models/protbert-finetune-eval.pt"


def main():
    t_start = time.time()
    print(f"Device: {pbf.DEVICE}")
    if pbf.DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    print("=" * 60)

    print("[1/6] 读取数据与 token 缓存...")
    if not os.path.exists(pbf.TOKEN_CACHE_TRAIN) or not os.path.exists(pbf.TOKEN_CACHE_TEST):
        raise FileNotFoundError("缺少 token 缓存，请先运行 protbert-finetune.py 完成分词")
    train_df = pd.read_csv(pbf.TRAIN_PATH)
    test_df = pd.read_csv(pbf.TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    train_cache = np.load(pbf.TOKEN_CACHE_TRAIN)
    test_cache = np.load(pbf.TOKEN_CACHE_TEST)
    train_ids = train_cache["input_ids"].astype(np.int32)
    train_mask = train_cache["attention_mask"].astype(np.int8)
    test_ids = test_cache["input_ids"].astype(np.int32)
    test_mask = test_cache["attention_mask"].astype(np.int8)
    print(f"  train tokens: {train_ids.shape}, test tokens: {test_ids.shape}, labels: {len(label_cols)}")

    print("[2/6] 划分验证集...")
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=pbf.VAL_SIZE, random_state=pbf.RANDOM_STATE)
    val_ds = pbf.ProteinTokenDataset(train_ids[idx_val], train_mask[idx_val], y_full[idx_val])
    test_ds = pbf.ProteinTokenDataset(
        test_ids, test_mask,
        np.zeros((len(test_df), len(label_cols)), dtype=np.int8))
    val_loader = DataLoader(val_ds, batch_size=pbf.BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=pbf.BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=0)
    print(f"  验证: {len(val_ds)}, 测试: {len(test_ds)}")

    print("[3/6] 构建模型...")
    model = pbf.ProtBertFinetune(pbf.MODEL_DIR, len(label_cols)).to(pbf.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {n_params:,}, 可训练: {n_trainable:,}")

    print("[4/6] 加载 checkpoint...")
    if not os.path.exists(pbf.MODEL_PATH):
        raise FileNotFoundError(f"未找到 {pbf.MODEL_PATH}，请等待训练完成第一个 epoch 验证后再运行")
    ckpt = torch.load(pbf.MODEL_PATH, map_location=pbf.DEVICE, weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"  已加载: {pbf.MODEL_PATH}")

    print("[5/6] 验证集概率 + 阈值搜索...")
    val_prob, val_true = pbf.get_probs(model, val_loader)
    opt_thresholds = pbf.find_optimal_thresholds(val_prob, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    y_val_pred, macro_f1, macro_prec, macro_rec = pbf.evaluate_at_thresholds(
        val_prob, val_true, opt_thresholds)
    macro_auc = np.mean([roc_auc_score(val_true[:, i], val_prob[:, i])
                         for i in range(len(label_cols)) if val_true[:, i].sum() > 0])
    subset_acc = accuracy_score(val_true, y_val_pred)
    hamming_loss = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {macro_auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/6] 测试集预测并保存...")
    y_test_pred = pbf.predict_with_thresholds(model, test_loader, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(pbf.SUB_PATH, index=False)
    print(f"  提交: {pbf.SUB_PATH} ({len(submission)} 行)")

    torch.save({"state_dict": model.state_dict(), "thresholds": opt_thresholds,
                "label_cols": label_cols}, EVAL_MODEL_PATH)
    print(f"  模型: {EVAL_MODEL_PATH}")

    metrics = {
        "model": "protbert-finetune-eval",
        "eval_only": True,
        "checkpoint": pbf.MODEL_PATH,
        "features": "ProtBERT token ids (max_len=512)",
        "classifier": "ProtBERT last 10 layers + Linear head",
        "threshold_stats": {
            "min": round(float(opt_thresholds.min()), 4),
            "max": round(float(opt_thresholds.max()), 4),
            "mean": round(float(opt_thresholds.mean()), 4),
            "n_below_05": int((opt_thresholds < 0.5).sum()),
        },
        "val_samples": len(idx_val), "test_samples": len(test_df),
        "n_labels_total": len(label_cols),
        "n_params": n_params, "n_trainable": n_trainable,
        "total_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(macro_auc, 6),
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming_loss, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
    }
    with Path(pbf.METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标: {pbf.METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

