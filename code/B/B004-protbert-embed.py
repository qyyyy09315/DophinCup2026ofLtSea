
"""
protbert-embed — 用 ProtBERT 提取蛋白质序列 CLS 嵌入，输出 parquet
=====================================================================
模型: Rostlab/prot_bert (本地 models/prot_bert)
输入: data/train.csv, data/test.csv
输出:
  rel/train_embeddings.parquet  — protein_id + emb_0..1023 + label_0..499
  rel/test_embeddings.parquet   — protein_id + emb_0..1023
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_DIR = r"models/prot_bert"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
TRAIN_EMB_PATH = f"{REL_DIR}/train_embeddings.parquet"
TEST_EMB_PATH = f"{REL_DIR}/test_embeddings.parquet"
BATCH_SIZE = 64
MAX_LEN = 512          # ProtBERT 长度上限
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_COLS = [f"emb_{i}" for i in range(1024)]


def tokenize(tokenizer, sequences, batch_size):
    """分批 tokenize, 序列字符间加空格 (ProtBERT 要求)"""
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        spaced = [" ".join(s) for s in chunk]
        enc = tokenizer(
            spaced, padding="max_length", max_length=MAX_LEN,
            truncation=True, return_tensors="pt",
        )
        yield {k: v.to(DEVICE) for k, v in enc.items()}


def extract_embeddings(model, tokenizer, sequences, batch_size, desc):
    """提取 CLS 嵌入 (fp16)，带进度条"""
    model.eval()
    all_embs = []
    n = len(sequences)
    n_batches = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        pbar = tqdm(tokenize(tokenizer, sequences, batch_size),
                    total=n_batches, desc=desc, unit="batch",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for enc in pbar:
            out = model(**enc)
            cls = out.last_hidden_state[:, 0].float().cpu().numpy()  # (B, 1024)
            all_embs.append(cls)
            pbar.set_postfix(seq=f"{pbar.n * batch_size}/{n}")
    return np.vstack(all_embs)


def main():
    print(f"Device: {DEVICE}, fp16: {DEVICE.type == 'cuda'}")
    t_start = time.time()

    print("[1/4] 加载 ProtBERT...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = BertModel.from_pretrained(MODEL_DIR)
    model.eval().to(DEVICE)
    if DEVICE.type == "cuda":
        model.half()

    print("[2/4] 读取数据...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train.columns if c.startswith("label_")]
    print(f"  train: {len(train)}, test: {len(test)}")

    print("[3/4] 提取训练集嵌入...")
    train_embs = extract_embeddings(model, tokenizer, train["sequence"].values, BATCH_SIZE, "train")
    print(f"  train embeddings: {train_embs.shape}")

    print("[4/4] 提取测试集嵌入...")
    test_embs = extract_embeddings(model, tokenizer, test["sequence"].values, BATCH_SIZE, "test")
    print(f"  test embeddings: {test_embs.shape}")

    # 保存 parquet
    os.makedirs(REL_DIR, exist_ok=True)
    train_df = pd.DataFrame(train_embs.astype(np.float16), columns=EMB_COLS)
    train_df.insert(0, "protein_id", train["protein_id"].values)
    for col in label_cols:
        train_df[col] = train[col].values.astype(np.int8)
    train_df.to_parquet(TRAIN_EMB_PATH, index=False)

    test_df = pd.DataFrame(test_embs.astype(np.float16), columns=EMB_COLS)
    test_df.insert(0, "protein_id", test["protein_id"].values)
    test_df.to_parquet(TEST_EMB_PATH, index=False)

    train_mb = os.path.getsize(TRAIN_EMB_PATH) / 1024 / 1024
    test_mb = os.path.getsize(TEST_EMB_PATH) / 1024 / 1024
    print(f"  保存: {TRAIN_EMB_PATH} ({train_mb:.0f}MB), {TEST_EMB_PATH} ({test_mb:.0f}MB)")

    info = {
        "model": "Rostlab/prot_bert",
        "pooling": "CLS (last_hidden_state[:, 0])",
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "precision": "fp16",
        "train_samples": len(train),
        "test_samples": len(test),
        "embedding_dim": 1024,
        "total_time_seconds": round(time.time() - t_start, 1),
    }
    with open(f"{REL_DIR}/protbert_embed_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
