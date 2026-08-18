"""
E003-bench — ProstT5 推理吞吐基准（GPU 空闲时测）
==================================================
目的:
  前几轮计时被并发 GPU 任务污染 (QQ/其他进程占卡), 数字不可信。
  本脚本在 GPU 空闲时按长度分层采样 3000 条, 分桶 (每桶总 token 预算
  控制, 防 8GB OOM), 测真实吞吐, 估算全量推理耗时。

用法:
  ./.venv/Scripts/python.exe code/E/E003-bench.py [--n 3000]

输出:
  stdout: 每桶耗时 / 总吞吐 / 全量 91,036 (train) 与 113,796 (全) 预估
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "prostt5")

# transformers 4.46.3 用绝对路径 from_pretrained 返回 bool 的 bug:
# chdir 后用相对路径加载
os.chdir(PROJECT_ROOT)

MAX_LEN = 2000          # 与 E003 主脚本一致
TOKEN_BUDGET = 6000     # 每批总 token 上限 (8GB 显存保守值)
BATCH_CAP = 32
BUCKETS = [(0, 128), (128, 256), (256, 512), (512, 1024),
           (1024, 1536), (1536, 2048)]  # 覆盖截断前全部长度


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {dev} | 采样: {args.n} 条")
    assert dev == "cuda", "基准需要 GPU"

    tok = T5Tokenizer.from_pretrained("models/prostt5", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("models/prostt5", torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    seqs_all = df["sequence"].values
    # 长度分层: 按分位数分层, 保证长序列覆盖
    lens = np.array([len(s) for s in seqs_all])
    n_buckets = min(8, args.n // 100)
    qs = np.quantile(lens, np.linspace(0, 1, n_buckets + 1)[1:-1])
    idxs = np.argsort(lens)
    picked = []
    for lo, hi in zip(np.concatenate([[0], qs]), np.concatenate([qs, [np.inf]])):
        sel = idxs[(lens[idxs] >= lo) & (lens[idxs] < hi)]
        k = max(1, args.n // n_buckets)
        picked.append(np.random.RandomState(42).choice(sel, min(k, len(sel)), replace=False))
    picked = np.concatenate(picked)[:args.n]
    seqs = [s.replace("U", "X").replace("Z", "X").replace("O", "X")
            for s in seqs_all[picked]]

    total_t, total_n = 0.0, 0
    print(f"\n{'长度桶':>12} {'条数':>6} {'批次':>5} {'批均token':>10} {'秒':>8} {'ms/条':>8}")
    for lo, hi in BUCKETS:
        b_seqs = [s for s in seqs if lo <= len(s) < hi]
        if not b_seqs:
            continue
        b_lens = np.array([len(s) for s in b_seqs])
        # 桶内排序, 相邻短序列同批, padding 浪费最小
        order = np.argsort(b_lens)
        b_seqs = [b_seqs[i] for i in order]
        b_lens = b_lens[order]

        t_b = time.time(); n_done = 0; n_b = 0; max_tok = 0
        i = 0
        pbar = tqdm(total=len(b_seqs), desc=f"{lo}-{hi}aa", leave=False)
        while i < len(b_seqs):
            j = i
            # 累积到 token 预算或 batch cap
            while (j < len(b_seqs) and (j - i) < BATCH_CAP
                   and b_lens[j] * (j - i + 1) <= TOKEN_BUDGET):
                j += 1
            batch = b_seqs[i:j]
            texts = ["<AA2fold> " + " ".join(list(seq[:MAX_LEN])) for seq in batch]
            enc = tok(texts, padding="longest", truncation=True,
                      max_length=MAX_LEN + 2, return_tensors="pt").to(dev)
            with torch.no_grad():
                model(**enc)
            n_done += len(batch); n_b += 1
            max_tok = max(max_tok, enc["attention_mask"].sum().item())
            i = j
            pbar.update(len(batch))
        pbar.close()
        dt = time.time() - t_b
        total_t += dt; total_n += n_done
        print(f"{lo:>6}-{hi:<6} {n_done:>6} {n_b:>5} {max_tok:>10.0f} "
              f"{dt:>8.0f} {dt / n_done * 1000:>8.0f}")

    sps = total_t / total_n
    print(f"\n合计 {total_n} 条: {total_t:.0f}s → {sps * 1000:.0f} ms/条")
    for name, n in [("train 91,036", 91036), ("全量 113,796", 113796)]:
        print(f"  {name} 预估: {sps * n / 60:.1f} 分钟 ({sps * n / 3600:.1f} 小时)")


if __name__ == "__main__":
    main()
