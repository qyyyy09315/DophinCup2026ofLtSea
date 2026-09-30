
"""
esm2-embed — ESM-2 150M 提取蛋白质序列平均池化嵌入，输出 parquet
=====================================================================
模型: facebook/esm2_t30_150M_UR50D (本地 models/esm2_150m)
输入: data/train.csv, data/test.csv
特征: last_hidden_state 按有效残基做平均池化，维度 640
输出:
  rel/train_esm2_embeddings.parquet  — protein_id + emb_0..639 + label_0..499
  rel/test_esm2_embeddings.parquet   — protein_id + emb_0..639
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_DIR = "models/esm2_150m"
MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
TRAIN_EMB_PATH = f"{REL_DIR}/train_esm2_embeddings.parquet"
TEST_EMB_PATH = f"{REL_DIR}/test_esm2_embeddings.parquet"
INFO_PATH = f"{REL_DIR}/esm2_embed_info.json"

MAX_LEN = 1024          # ESM-2 位置编码上限
BATCH_SIZE = 32
EMB_DIM = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_COLS = [f"emb_{i}" for i in range(EMB_DIM)]


def mean_pool_embeddings(last_hidden_state, input_ids, attention_mask):
    """对非特殊 token 的残基嵌入做平均池化。"""
    valid = (attention_mask.unsqueeze(-1) == 1) & (input_ids.unsqueeze(-1) >= 5)
    emb = last_hidden_state.float()
    denom = valid.float().sum(dim=1).clamp(min=1)
    return (emb * valid.float()).sum(dim=1) / denom


def extract_embeddings(model, tokenizer, sequences, batch_size, desc):
    model.eval()
    all_embs = []
    n = len(sequences)
    n_batches = (n + batch_size - 1) // batch_size
    pbar = tqdm(total=n_batches, desc=desc, unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = sequences[i:i + batch_size]
            enc = tokenizer(list(chunk), padding="max_length", max_length=MAX_LEN,
                            truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool_embeddings(
                out.last_hidden_state, input_ids, attention_mask).cpu().numpy()
            all_embs.append(pooled)
            pbar.update(1)
            pbar.set_postfix(seq=f"{min(i + batch_size, n)}/{n}")
    pbar.close()
    return np.vstack(all_embs)


def main():
    print(f"Device: {DEVICE}, fp16: {DEVICE.type == 'cuda'}")
    t_start = time.time()

    print("[1/4] 加载 ESM-2 150M...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
    model = EsmModel.from_pretrained(MODEL_DIR)
    model.eval().to(DEVICE)
    if DEVICE.type == "cuda":
        model.half()

    print("[2/4] 读取数据...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train.columns if c.startswith("label_")]
    print(f"  train: {len(train)}, test: {len(test)}, labels: {len(label_cols)}")

    print("[3/4] 提取训练集嵌入...")
    train_embs = extract_embeddings(model, tokenizer, train["sequence"].values, BATCH_SIZE, "train")
    print(f"  train embeddings: {train_embs.shape}")

    print("[4/4] 提取测试集嵌入...")
    test_embs = extract_embeddings(model, tokenizer, test["sequence"].values, BATCH_SIZE, "test")
    print(f"  test embeddings: {test_embs.shape}")

    os.makedirs(REL_DIR, exist_ok=True)
    train_df = pd.DataFrame(train_embs.astype(np.float16), columns=EMB_COLS)
    train_df.insert(0, "protein_id", train["protein_id"].values)
    for col in label_cols:
        train_df[col] = train[col].values.astype(np.int8)
    train_df.to_parquet(TRAIN_EMB_PATH, index=False)

    test_df = pd.DataFrame(test_embs.astype(np.float16), columns=EMB_COLS)
    test_df.insert(0, "protein_id", test["protein_id"].values)
    test_df.to_parquet(TEST_EMB_PATH, index=False)

    info = {
        "model": MODEL_NAME,
        "local_dir": MODEL_DIR,
        "pooling": "mean pool over non-special residues",
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "precision": "fp16",
        "embedding_dim": EMB_DIM,
        "train_samples": len(train),
        "test_samples": len(test),
        "total_time_seconds": round(time.time() - t_start, 1),
    }
    with Path(INFO_PATH).open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"  保存: {TRAIN_EMB_PATH}, {TEST_EMB_PATH}")
    print(f"  信息: {INFO_PATH}")
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
