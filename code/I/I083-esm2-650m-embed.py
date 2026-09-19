"""
I083 — ESM2-650M 冻结嵌入提取 (手动长任务)
============================================
动机: 当前 PLM 头用 ESM2-150M(640D) + ProtBERT(1024D)。650M 骨干从未以
      冻结均值池化的形式测过 (E006 系列是 LoRA + 残基读出, 不是同一用法)。
      I060 的先例: 加预训练 PLM 特征是唯一成功过的分支扩展 (+0.0017);
      I063 失败的是「同任务训练」特征, 预训练骨干不同属此类。
      EZpred: 用最后 3 层比只用最后 1 层 +2%。

提取: train (113,796) + test (28,450) 全量, fp16, 按长度分桶排序,
      均值池化 (非特殊 token)。同时存:
        *_last   (n, 1280)  最后一层池化
        *_last3  (n, 3840)  最后 3 层池化拼接 (EZpred 依据)
输出: rel/esm2_650m_embeddings.npz
用法:
  python code/I/I083-esm2-650m-embed.py --limit 200      # 冒烟 (分钟级)
  python code/I/I083-esm2-650m-embed.py                  # 全量 (实测约 3-4 小时 GPU)
  python code/I/I083-esm2-650m-embed.py --split test     # 只跑 test
注: 显存不足会自动把 token 预算减半重试, 中断可直接重跑 (只在最后写盘)。
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ESM2_PATH = os.path.join(PROJECT_ROOT, "models", "esm2_650m")
OUT_PATH = os.path.join(REL_DIR, "esm2_650m_embeddings.npz")
MAX_LEN = 1022          # ESM2 上限 (含 BOS/EOS 共 1024)


def load_model_and_tokenizer():
    if not os.path.isdir(ESM2_PATH):
        raise FileNotFoundError(f"缺少 {ESM2_PATH}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import EsmModel, AutoTokenizer
    cwd = os.getcwd()
    os.chdir(os.path.dirname(ESM2_PATH))
    tok = AutoTokenizer.from_pretrained(os.path.basename(ESM2_PATH), local_files_only=True)
    model = EsmModel.from_pretrained(os.path.basename(ESM2_PATH),
                                     torch_dtype=torch.float16,
                                     local_files_only=True).cuda().eval()
    os.chdir(cwd)
    return model, tok


@torch.no_grad()
def encode_split(model, tok, seqs, batch_max_tokens=16384, desc=""):
    """按长度排序分桶, 动态 batch (batch_max_tokens 为每批 token 预算), 均值池化。
    OOM 时把 token 预算减半重试当前批。"""
    n = len(seqs)
    order = np.argsort([-len(s) for s in seqs])
    last_out = np.zeros((n, 1280), np.float32)
    last3_out = np.zeros((n, 3 * 1280), np.float32)
    i = 0
    budget = batch_max_tokens
    pbar = tqdm(total=n, desc=desc, ncols=100)
    while i < n:
        li = order[i]
        L = min(len(seqs[li]) + 2, MAX_LEN + 2)
        bs = max(1, budget // L)
        idx = order[i:i + bs]
        toks = [s[:MAX_LEN] for s in seqs[idx]]
        enc = tok(toks, padding=True, truncation=True, max_length=MAX_LEN + 2,
                  return_tensors="pt", add_special_tokens=True)
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        try:
            with torch.amp.autocast("cuda"):
                out = model(**enc, output_hidden_states=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            budget = max(1024, budget // 2)
            tqdm.write(f"  OOM -> token 预算降到 {budget}")
            continue
        mask = enc["attention_mask"].bool()
        special = torch.zeros_like(mask)
        special[:, 0] = True
        eos_pos = enc["attention_mask"].sum(1) - 1
        special[torch.arange(len(toks)), eos_pos] = True
        keep = mask & ~special
        h_last = out.last_hidden_state.float()
        hs = out.hidden_states                 # tuple(34): emb + 33 层
        h3 = torch.stack([hs[-3], hs[-2], hs[-1]], dim=1).float()
        k = keep.unsqueeze(-1).float()
        pooled_last = (h_last * k).sum(1) / k.sum(1).clamp(min=1)
        pooled_3 = (h3 * k.unsqueeze(1)).sum(2) / k.sum(1).clamp(min=1).unsqueeze(1)
        last_out[idx] = pooled_last.cpu().numpy().astype(np.float32)
        last3_out[idx] = pooled_3.reshape(len(idx), -1).cpu().numpy().astype(np.float32)
        del out, hs, h3
        pbar.update(len(idx))
        i += bs
    pbar.close()
    return last_out, last3_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每 split 只取前 N 条 (冒烟)")
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--batch-max-tokens", type=int, default=16384)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 84)
    print("  I083: ESM2-650M frozen embedding extraction")
    print("=" * 84)

    model, tok = load_model_and_tokenizer()
    print("  模型与分词器就绪 (fp16)")

    store = {}
    for split in (["train", "test"] if args.split == "both" else [args.split]):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
        seqs = df["sequence"].astype(str).values
        if args.limit:
            seqs = seqs[: args.limit]
        print(f"\n  [{split}] {len(seqs)} 条 | 长度 p50={int(np.median([len(s) for s in seqs]))} "
              f"p99={int(np.quantile([len(s) for s in seqs], 0.99))}")
        last, last3 = encode_split(model, tok, seqs, args.batch_max_tokens, desc=f"  {split}")
        store[f"{split}_last"] = last
        store[f"{split}_last3"] = last3
        store[f"{split}_protein_id"] = df["protein_id"].values[: len(seqs)]
        print(f"  [{split}] last {last.shape} | last3 {last3.shape} | "
              f"范数均值 {np.linalg.norm(last, axis=1).mean():.1f}")

    np.savez_compressed(OUT_PATH, **store)
    mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\n  Written: {OUT_PATH} ({mb:.0f} MB)")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
