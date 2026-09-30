"""
I083 — ESM2-650M 冻结嵌入提取（可断点续跑）（手动长任务）
============================================================
动机: 当前 PLM 头用 ESM2-150M(640D) + ProtBERT(1024D)。650M 骨干从未以
      冻结均值池化的形式测过 (E006 系列是 LoRA + 残基读出, 不是同一用法)。
      I060 先例: 加预训练 PLM 特征是唯一成功过的分支扩展; EZpred: 最后 3 层
      比最后 1 层 +2%。评估协议见 code/I/I084-650m-embed-eval.py。

提取: train (113,796) + test (28,450) 全量, fp16, 按长度分桶排序,
      均值池化 (去 BOS/EOS/padding)。同时存:
        *_last   (n, 1280)  末层池化
        *_last3  (n, 3840)  末 3 层池化拼接 (EZpred 依据)

已知限制（2026-09-20 实测）: 数据长度并非赛题文档所写的 100~500 aa —— 实测 train
      min/中位/p95/max = 3/410/1434/35375，>1022 aa 的占 10.70%（train 12,174 条 +
      test 3,097 条），截断会丢掉全部残基的 14.1%。这些长序列的嵌入只覆盖前 1022 个
      残基（MAX_LEN），与 150M/ProtBERT 分支的截断口径是否一致需在 I084 阶段核对。

断点续跑机制:
  - 处理过程中逐批写入磁盘 memmap（行序 = 原表行序，非处理顺序）:
      rel/esm2_650m_last.npy   行序 = [train 原始行序; test 原始行序] 拼接
      rel/esm2_650m_last3.npy
  - 进度文件 rel/esm2_650m_progress.json 记录:
      schema/done/total/n_train/split/max_len/batch_max_tokens/file_rows/
      fingerprint/updated/stage
    默认每 20 秒「先 flush memmap 再原子替换进度文件」，两者顺序不可颠倒。
  - 中断后重跑同一命令：先校验断点与本轮配置（含序列指纹）完全一致，再从 done
    处继续（排序用 stable argsort，与上次逐位一致，已处理行直接跳过）。
    校验不通过**直接报错退出**并提示 --reset，绝不静默改写既有中间结果。
  - 中断（Ctrl+C）、异常、完成后都会立即落盘断点，最多只丢最后一批。
  - 收尾：先把两个 memmap 分块流式写为临时 npz（内存恒定）再原子替换为
    rel/esm2_650m_embeddings.npz，校验形状后写同名 .meta.json，最后才删中间文件。
    故「收尾阶段崩溃」重跑会直接重做收尾（幂等），不会留下半份 npz。
  - 收尾前逐行审计：全零行 / 非有限值行视为断点空洞，直接中止并要求 --reset。

自测沙箱（本次加固核心）:
  --limit / --debug-subset 一律走 rel/_i083_selftest/，与正式产物物理隔离，
  绝不触碰 rel/esm2_650m_*.{npy,json,npz}。配套自测脚本:
      python code/tools/selftest_i083_resume.py
  该脚本用假编码器（不加载模型、不占 GPU）完整走一遍
  中断→续跑→收尾→断点校验拒绝 的全流程，并比对续跑与原跑逐位一致。

用法:
  ./.venv/Scripts/python.exe -u code/I/I083-esm2-650m-embed.py                # 全量（实测约 2.8-3 小时）
  ./.venv/Scripts/python.exe -u code/I/I083-esm2-650m-embed.py --reset        # 清断点后重跑
  ./.venv/Scripts/python.exe -u code/I/I083-esm2-650m-embed.py --limit 64     # 冒烟（分钟级，走沙箱）
  ./.venv/Scripts/python.exe -u code/I/I083-esm2-650m-embed.py --debug-subset 1500   # 断点机制自测（走沙箱）
  ./.venv/Scripts/python.exe -u code/I/I083-esm2-650m-embed.py --save-every 10       # 断点落盘间隔（秒）
  注: 显存不足会自动把 token 预算减半重试当前批；降到 512 仍 OOM 则报错退出（不再死循环）。
      --split train|test 只产 <split> 后缀 npz（I084 不能用），正式产物必须 --split both。

进度显示（2026-09-20 修正）:
  进度条全程单行，形如
    extract:  5%|██▎    | 7136/142246 [18:21, 6.49 行/s], ckpt 7136 · 剩余约 152 min
  其中「剩余」按 token 吞吐计算。tqdm 自带的 {remaining} 已从 bar_format 移除——它用行/s
  线性外推，而本任务按长度降序处理（开头全是 1022 token 长序列），行/s 越跑越快，
  其 ETA 会高估 2 倍以上（实测 356 min vs 真实 152 min；全量真实总时长约 2.84 h）。
  需要留持久日志时用 --log-every N（默认 0=关闭，不换行）。
"""
import os, sys, gc, json, time, shutil, hashlib, argparse, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from numpy.lib import format as npformat
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ESM2_PATH = os.path.join(PROJECT_ROOT, "models", "esm2_650m")
SELFTEST_DIR = os.path.join(REL_DIR, "_i083_selftest")

MAX_LEN = 1022                       # ESM2 token 上限（不含 BOS/EOS）
N_LAST, N_LAST3 = 1280, 3 * 1280
SPLITS = ["train", "test"]
SCHEMA = 2                           # 断点文件结构版本，字段含义变更须 +1
MIN_TOKEN_BUDGET = 512               # OOM 降档下限，到此仍 OOM 即报错
AUDIT_CHUNK = 8192                   # 收尾审计分块行数
NPZ_ROW_CHUNK = 8192                 # 流式写 npz 分块行数
KEYS = ["train_last", "train_last3", "test_last", "test_last3",
        "train_protein_id", "test_protein_id"]
# 进度条单行格式：故意去掉 tqdm 自带的 {remaining}。tqdm 用「行/s」线性外推，
# 而本任务按长度降序处理（开头全是 1022 token 的长序列），行/s 会越跑越快，
# 它的 ETA 会高估 2 倍以上（实测 356min vs 真实 152min）。真实 ETA 按 token
# 吞吐算，放在 postfix 里。
BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]{postfix}"


# ---------------------------------------------------------------- 路径与模式
def resolve_paths(sandbox):
    """沙箱与正式产物目录物理隔离，避免自测污染正式断点。"""
    d = SELFTEST_DIR if sandbox else REL_DIR
    os.makedirs(d, exist_ok=True)
    return {
        "dir": d, "sandbox": bool(sandbox),
        "mm_last": os.path.join(d, "esm2_650m_last.npy"),
        "mm_last3": os.path.join(d, "esm2_650m_last3.npy"),
        "prog": os.path.join(d, "esm2_650m_progress.json"),
    }


def out_path_for(P, split):
    """正式产物只在 --split both 时写规范名；单 split 写后缀名，避免半份产物冒充完整产物。"""
    if P["sandbox"]:
        return os.path.join(P["dir"], "esm2_650m_embeddings.selftest.npz")
    if split == "both":
        return os.path.join(REL_DIR, "esm2_650m_embeddings.npz")
    return os.path.join(REL_DIR, f"esm2_650m_embeddings.{split}.npz")


# ---------------------------------------------------------------- 模型与编码
def load_model_and_tokenizer():
    if not os.path.isdir(ESM2_PATH):
        raise FileNotFoundError(f"缺少 {ESM2_PATH}")
    if not any(os.path.exists(os.path.join(ESM2_PATH, f))
               for f in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"{ESM2_PATH} 下没有 model.safetensors / pytorch_model.bin")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import EsmModel, AutoTokenizer
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(ESM2_PATH))
        tok = AutoTokenizer.from_pretrained(os.path.basename(ESM2_PATH), local_files_only=True)
        model = EsmModel.from_pretrained(os.path.basename(ESM2_PATH),
                                         torch_dtype=torch.float16,
                                         local_files_only=True).cuda().eval()
    finally:
        os.chdir(cwd)
    return model, tok


@torch.no_grad()
def encode_batch(model, tok, toks):
    """单批编码 -> (pooled_last b×1280, pooled_last3 b×3840) float32 numpy"""
    enc = tok(toks, padding=True, truncation=True, max_length=MAX_LEN + 2,
              return_tensors="pt", add_special_tokens=True)
    enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
    with torch.amp.autocast("cuda"):
        out = model(**enc, output_hidden_states=True)
    mask = enc["attention_mask"].bool()
    special = torch.zeros_like(mask)
    special[:, 0] = True
    eos_pos = enc["attention_mask"].sum(1) - 1
    rows = torch.arange(len(toks), device=mask.device)
    special[rows, eos_pos] = True
    keep = mask & ~special
    h_last = out.last_hidden_state.float()
    hs = out.hidden_states                     # tuple(34): emb + 33 层
    h3 = torch.stack([hs[-3], hs[-2], hs[-1]], dim=1).float()
    k = keep.unsqueeze(-1).float()
    pooled_last = (h_last * k).sum(1) / k.sum(1).clamp(min=1)
    pooled_3 = (h3 * k.unsqueeze(1)).sum(2) / k.sum(1).clamp(min=1).unsqueeze(1)
    del out, hs, h3, h_last
    return (pooled_last.cpu().numpy().astype(np.float32),
            pooled_3.reshape(len(toks), -1).cpu().numpy().astype(np.float32))


# ---------------------------------------------------------------- 数据与指纹
def load_sequences(split, limit_per_split=0, debug_subset=0):
    """返回 (seqs, ids, n_train, file_rows)。
    行序 = [train 原始行序; test 原始行序]；n_train = 返回数组中属于 train 的行数。
    limit_per_split / debug_subset 只用于沙箱自测，会同时反映到 n_train。"""
    parts, idparts, file_rows = [], [], {}
    for sp in SPLITS:
        p = os.path.join(DATA_DIR, f"{sp}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 {p}")
        df = pd.read_csv(p, usecols=["protein_id", "sequence"])
        s = df["sequence"].astype(str).values
        i = df["protein_id"].astype(str).values
        file_rows[sp] = int(len(s))
        if limit_per_split:
            s, i = s[:limit_per_split], i[:limit_per_split]
        parts.append(s)
        idparts.append(i)
    seqs = np.concatenate(parts)
    ids = np.concatenate(idparts)
    n_train = len(parts[0])
    if debug_subset:
        k = min(int(debug_subset), len(seqs))
        seqs, ids, n_train = seqs[:k], ids[:k], min(n_train, k)
    if split == "train":
        seqs, ids = seqs[:n_train], ids[:n_train]
    elif split == "test":
        seqs, ids, n_train = seqs[n_train:], ids[n_train:], 0
    return seqs, ids, n_train, file_rows


def fingerprint(lens, ids):
    """序列长度 + 蛋白 ID 联合指纹：长度变动或行序变动都会改变它。"""
    h = hashlib.md5()
    h.update(lens.astype(np.int64).tobytes())
    h.update("".join(map(str, ids)).encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 断点读写
def read_progress(P):
    if not os.path.exists(P["prog"]):
        return None
    with open(P["prog"], encoding="utf-8") as f:
        return json.load(f)


def write_progress(P, state):
    tmp = P["prog"] + ".tmp"
    with Path(tmp).open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, P["prog"])


def make_state(done, total, n_train, args, fp, file_rows, stage="extract"):
    return {
        "schema": SCHEMA, "stage": stage,
        "done": int(done), "total": int(total), "n_train": int(n_train),
        "split": args.split, "max_len": MAX_LEN,
        "batch_max_tokens": int(args.batch_max_tokens),
        "file_rows": {k: int(v) for k, v in file_rows.items()},
        "fingerprint": fp,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def check_state(st, args, total, n_train, fp):
    """返回不一致原因列表；空列表才允许续跑（绝不静默续跑状态未知的断点）。"""
    bad = []
    if st.get("schema") != SCHEMA:
        bad.append(f"schema={st.get('schema')!r} != {SCHEMA}（旧版或非本脚本产物）")
    if st.get("total") != total:
        bad.append(f"total={st.get('total')!r} != {total}")
    if st.get("n_train") != n_train:
        bad.append(f"n_train={st.get('n_train')!r} != {n_train}")
    if st.get("split") not in (None, args.split):
        bad.append(f"split={st.get('split')!r} != {args.split!r}")
    if st.get("max_len") not in (None, MAX_LEN):
        bad.append(f"max_len={st.get('max_len')!r} != {MAX_LEN}")
    if st.get("fingerprint") not in (None, fp):
        bad.append("序列指纹不一致（data/*.csv 变动或行序改变）")
    return bad


def open_mm(P, key, rows, cols, fresh):
    """打开/新建 memmap。既有文件用「声明形状」显式校验，不靠异常猜测。"""
    path = P[key]
    if fresh:
        if os.path.exists(path):
            os.remove(path)
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, cols))
    if not os.path.exists(path):
        raise SystemExit(f"[断点异常] 有进度文件但缺少 {path}\n"
                         f"  处理：确认是否误删；否则加 --reset 重跑。")
    try:
        ex = np.load(path, mmap_mode="r")
        shape, dtype = tuple(ex.shape), ex.dtype
        try:
            ex._mmap.close()
        except Exception:
            pass
        del ex
    except Exception as e:
        raise SystemExit(f"[断点异常] {os.path.basename(path)} 无法读取"
                         f"（{type(e).__name__}: {e}）\n  处理：加 --reset 重跑。")
    if shape != (rows, cols):
        raise SystemExit(f"[断点异常] {os.path.basename(path)} 形状不匹配："
                         f"现有 {shape}，本轮需要 {(rows, cols)}。\n"
                         f"  处理：加 --reset 重跑（会丢弃这份中间结果）。")
    if dtype != np.float32:
        raise SystemExit(f"[断点异常] {os.path.basename(path)} dtype={dtype} != float32，请 --reset")
    return np.lib.format.open_memmap(path, mode="r+", dtype=np.float32, shape=(rows, cols))


def checkpoint(P, mm_last, mm_last3, i, state):
    """先把数据落盘再写进度；顺序不可颠倒。"""
    mm_last.flush()
    mm_last3.flush()
    state = dict(state)
    state["done"] = int(i)
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_progress(P, state)


# ---------------------------------------------------------------- 收尾
def audit_mm(mm, rows, cols, chunk=AUDIT_CHUNK):
    """逐行审计：全零行或含非有限值的行 = 断点空洞 / 未写入，返回问题行号。"""
    bad = []
    for s in range(0, rows, chunk):
        e = min(s + chunk, rows)
        blk = np.asarray(mm[s:e])
        ok = np.isfinite(blk).all(axis=1) & (blk != 0).any(axis=1)
        if not ok.all():
            bad.extend((s + np.where(~ok)[0]).tolist())
    return bad


def _write_npy_member(fh, arr):
    """手写 .npy 头 + 原样字节，避免依赖 write_array 对流的内部判断。
    注意：pandas 3.x 的 protein_id 取出来是 object dtype；object 数组 tobytes() 写的是
    裸指针字节，读回来是垃圾（旧实现靠 savez_compressed 的 allow_pickle 掩盖了这点）。
    这里统一转成 Unicode（'<U…'）写入，既正确又不需要 pickle。"""
    arr = np.asarray(arr)
    if arr.dtype.kind == "O" or arr.dtype.hasobject:
        arr = arr.astype(np.str_)
    arr = np.ascontiguousarray(arr)
    npformat.write_array_header_2_0(fh, {"descr": arr.dtype.str,
                                         "fortran_order": False,
                                         "shape": arr.shape})
    if arr.size:
        fh.write(arr.tobytes(order="C"))


def _write_mm_member(fh, mm, start, rows, cols):
    """memmap 分块流式写入（每块 NPZ_ROW_CHUNK 行），峰值内存恒定。"""
    if rows <= 0:
        _write_npy_member(fh, np.zeros((0, cols), np.float32))
        return
    npformat.write_array_header_2_0(
        fh, {"descr": "<f4", "fortran_order": False, "shape": (rows, cols)})
    for s in range(start, start + rows, NPZ_ROW_CHUNK):
        e = min(s + NPZ_ROW_CHUNK, start + rows)
        fh.write(np.asarray(mm[s:e]).tobytes(order="C"))


def save_npz(path, sources, n_train, total, ids):
    """把 memmap 分块流式写成 .npz（临时文件 + 原子替换）。
    sources 必须是 {"mm_last": mmap, "mm_last3": mmap}。
    断在任何一步都不会留下半份 npz，也不会破坏已有产物。"""
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    n_test = total - n_train
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for key, rows in (("train_last", n_train), ("train_last3", n_train),
                          ("test_last", n_test), ("test_last3", n_test)):
            cols = N_LAST3 if key.endswith("last3") else N_LAST
            start = 0 if key.startswith("train") else n_train
            src = sources["mm_last3" if key.endswith("last3") else "mm_last"]
            with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                _write_mm_member(fh, src, start, rows, cols)
        for key, arr in (("train_protein_id", ids[:n_train]), ("test_protein_id", ids[n_train:])):
            with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                _write_npy_member(fh, arr)
    os.replace(tmp, path)


def release_mm(*mms):
    for mm in mms:
        try:
            mm._mmap.close()
        except Exception:
            pass
    gc.collect()


def remove_quiet(*paths):
    for f in paths:
        for _ in range(3):
            try:
                if os.path.exists(f):
                    os.remove(f)
                break
            except OSError:
                time.sleep(1)


def encode_with_oom_guard(model, tok, seqs, order, tok_ord, i, budget):
    """按当前 token 预算切批并编码。返回 (idx, p1, p3, budget)。
    OOM 时返回 (None, None, None, 降档后的预算) 让调用方重试；到下限仍 OOM 则报错。
    idx 由本函数生成并返回，保证「写入用下标」与「实际编码内容」永不脱节。
    tok_ord[i] = 处理顺序第 i 条实际进入模型的 token 数（已按 MAX_LEN 截断）。"""
    L = int(tok_ord[i])
    bs = max(1, int(budget) // L)
    idx = order[i:i + bs]
    try:
        p1, p3 = encode_batch(model, tok, [s[:MAX_LEN] for s in seqs[idx]])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if budget <= MIN_TOKEN_BUDGET:
            raise RuntimeError(
                f"token 预算已降到下限 {budget} 仍 OOM（行 {int(order[i])}，长度 {L}）。\n"
                f"  处理：断点已保存；降低 --batch-max-tokens（如 4096）后重跑同一命令即可续跑。"
            ) from None
        return None, None, None, max(MIN_TOKEN_BUDGET, int(budget) // 2)
    return idx, p1, p3, budget


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每 split 只取前 N 条（冒烟，走沙箱）")
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--batch-max-tokens", type=int, default=16384)
    ap.add_argument("--reset", action="store_true", help="清空本轮断点与 memmap 后重跑")
    ap.add_argument("--debug-subset", type=int, default=0,
                    help="只取拼接序列前 N 条（走沙箱，用于断点机制自测）")
    ap.add_argument("--save-every", type=float,
                    default=float(os.environ.get("I083_SAVE_EVERY", "20")),
                    help="断点落盘间隔秒数（默认 20，或环境变量 I083_SAVE_EVERY）")
    ap.add_argument("--log-every", type=float, default=0.0,
                    help=">0 时额外每 N 秒用 tqdm.write 打一行持久日志（会换行）；0=关闭（默认，"
                         "进度条保持单行，ETA 在 postfix 里）")
    ap.add_argument("--no-audit", action="store_true", help="跳过收尾逐行审计（不建议）")
    args = ap.parse_args()

    t0 = time.time()
    smoke = args.limit > 0
    sandbox = smoke or args.debug_subset > 0
    P = resolve_paths(sandbox)
    out_npz = out_path_for(P, args.split)
    meta_path = out_npz[:-4] + ".meta.json"

    print("=" * 88)
    print("  I083: ESM2-650M frozen embedding extraction (resumable)")
    print("=" * 88)
    print(f"  模式: {'沙箱自测' if sandbox else '正式全量'}"
          f"{'（--limit 冒烟，内存直算）' if smoke else ''}"
          f"{'（--debug-subset 自测，走完整断点路径）' if args.debug_subset else ''}")
    print(f"  产物目录: {P['dir']}")
    print(f"  最终 npz: {out_npz}")
    if (not sandbox) and args.split != "both":
        print(f"  警告: --split {args.split} 只产出该 split 的嵌入，I084 需要 --split both 的"
              f"规范产物；该文件不会被 I084 使用。")

    if args.reset:
        remove_quiet(P["mm_last"], P["mm_last3"], P["prog"], P["prog"] + ".tmp")
        print("  [--reset] 已清空本轮断点与 memmap")

    seqs, ids, n_train, file_rows = load_sequences(args.split, args.limit, args.debug_subset)
    total = len(seqs)
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=total)
    fp = fingerprint(lens, ids)
    print(f"  待处理行数: {total} ({args.split}) | n_train={n_train} | 指纹 {fp} | "
          f"长度 p50={int(np.median(lens))} p99={int(np.quantile(lens, 0.99))} max={int(lens.max())}")

    if not smoke:
        mm_bytes = total * (N_LAST + N_LAST3) * 4
        need_gb = mm_bytes * 2.2 / 1e9 + 1.0
        free_gb = shutil.disk_usage(P["dir"]).free / 1e9
        print(f"  磁盘: 需要约 {need_gb:.1f} GB（memmap + npz），可用 {free_gb:.1f} GB")
        if free_gb < need_gb:
            raise SystemExit(f"[空间不足] {P['dir']} 需要约 {need_gb:.1f} GB，"
                             f"当前可用 {free_gb:.1f} GB，请先清理。")

    model, tok = load_model_and_tokenizer()
    print("  模型与分词器就绪 (fp16)")

    order = np.argsort(-lens, kind="stable")
    tok_ord = np.minimum(lens + 2, MAX_LEN + 2)[order]     # 每条实际进模型的 token 数
    cum_tok = np.concatenate([[0], np.cumsum(tok_ord)])    # cum_tok[k] = 前 k 条 token 累计
    print(f"  token 总量（截断后）{int(cum_tok[-1]):,}")

    # ---------- 冒烟：内存直算，不碰 memmap / 断点，仍走沙箱 ----------
    if smoke:
        print(f"  [冒烟] 处理前 {total} 行（不写断点）...")
        last_out = np.zeros((total, N_LAST), np.float32)
        last3_out = np.zeros((total, N_LAST3), np.float32)
        budget, i = args.batch_max_tokens, 0
        pbar = tqdm(total=total, desc="  smoke", ncols=100, unit=" 行", bar_format=BAR_FMT)
        while i < total:
            idx, p1, p3, budget = encode_with_oom_guard(model, tok, seqs, order, tok_ord, i, budget)
            if idx is None:
                tqdm.write(f"  OOM -> token 预算降为 {budget}")
                continue
            last_out[idx], last3_out[idx] = p1, p3
            i += len(idx)
            pbar.update(len(idx))
        pbar.close()
        save_npz_arrays(out_npz, last_out, last3_out, n_train, total, ids)
        print(f"  [冒烟] Written: {out_npz} ({os.path.getsize(out_npz) / 1e6:.0f} MB)")
        print(f"  Completed in {(time.time() - t0) / 60:.1f} min")
        print("  注：冒烟产物为截断样本，仅供自测；正式产物请跑 --split both 全量。")
        return

    # ---------- 全量 / 自测：memmap + 断点续跑 ----------
    prog = read_progress(P)
    if prog is not None:
        bad = check_state(prog, args, total, n_train, fp)
        if bad:
            raise SystemExit("[断点校验不通过] 拒绝续跑，原因：\n  - " + "\n  - ".join(bad)
                             + "\n  处理：加 --reset 重跑（丢弃这份中间结果），"
                               "或改回与断点一致的参数。")
        done = int(prog["done"])
        if not (0 <= done <= total):
            raise SystemExit(f"[断点异常] done={done} 超出 [0, {total}]，请加 --reset 重跑")
        mm_last = open_mm(P, "mm_last", total, N_LAST, fresh=False)
        mm_last3 = open_mm(P, "mm_last3", total, N_LAST3, fresh=False)
        print(f"  [断点] 续跑：已完成 {done}/{total}（{done / max(total, 1) * 100:.1f}%）")
    else:
        if os.path.exists(P["mm_last"]) or os.path.exists(P["mm_last3"]):
            raise SystemExit("[断点异常] 存在 memmap 但没有进度文件，状态不可信"
                             "（可能被手动清理或非本脚本写入）。\n"
                             "  处理：确认这些文件无用后加 --reset 重跑。")
        mm_last = open_mm(P, "mm_last", total, N_LAST, fresh=True)
        mm_last3 = open_mm(P, "mm_last3", total, N_LAST3, fresh=True)
        done = 0
        print("  [断点] 无断点，全新开始")

    state = make_state(done, total, n_train, args, fp, file_rows)
    if done >= total:
        print("  [断点] 数据已全部处理完，直接进入收尾阶段")

    budget = args.batch_max_tokens
    i = done
    t_proc = time.time()
    last_save = last_log = time.time()
    interrupted = False

    def postfix(k_processed):
        """实时 ETA 按 token 吞吐算（见 BAR_FMT 注释），单行显示不打断进度条。
        只统计「本次会话」的吞吐：从断点续跑时，done 之前那部分不算进速率。"""
        done_tok = int(cum_tok[k_processed] - cum_tok[done])
        el = max(time.time() - t_proc, 1e-6)
        rate = done_tok / el
        if rate <= 0:
            return f"ckpt {k_processed} · 剩余约 -- min"
        eta = (int(cum_tok[-1]) - int(cum_tok[k_processed])) / rate / 60
        return f"ckpt {k_processed} · 剩余约 {eta:.0f} min"

    pbar = tqdm(total=total, initial=done, desc="  extract", ncols=100,
                unit=" 行", bar_format=BAR_FMT)
    if done:
        pbar.set_postfix_str(f"从断点 {done} 续跑", refresh=False)
    try:
        while i < total:
            idx, p1, p3, budget = encode_with_oom_guard(model, tok, seqs, order, tok_ord, i, budget)
            if idx is None:
                tqdm.write(f"  OOM -> token 预算降为 {budget}")
                continue
            mm_last[idx] = p1
            mm_last3[idx] = p3
            i += len(idx)
            pbar.update(len(idx))
            now = time.time()
            if now - last_save >= args.save_every and i < total:
                checkpoint(P, mm_last, mm_last3, i, state)
                last_save = now
                pbar.set_postfix_str(postfix(i), refresh=False)   # 不换行
                if args.log_every > 0 and now - last_log >= args.log_every:
                    tqdm.write(f"  [断点] {i}/{total} ({i / total * 100:.1f}%) "
                               f"{postfix(i).strip()} -> {os.path.basename(P['prog'])}")
                    last_log = now
    except KeyboardInterrupt:
        interrupted = True
    finally:
        pbar.close()
        if i < total:
            try:
                checkpoint(P, mm_last, mm_last3, i, state)
            except Exception as e:
                print(f"  [警告] 断点落盘失败：{type(e).__name__}: {e}")
    if interrupted:
        print(f"\n  已中断（Ctrl+C）。断点已保存：{i}/{total} 行。")
        print("  续跑：重跑同一命令即可（不要加 --reset）；彻底重来再加 --reset。")
        sys.exit(130)

    print(f"\n  全部 {total} 行处理完毕，收尾中 ...")
    checkpoint(P, mm_last, mm_last3, total, state)
    finalized = False
    try:
        if not args.no_audit:
            print("  审计中间结果（全零行 / 非有限值行）...")
            for mm, cols, name in ((mm_last, N_LAST, "last"), (mm_last3, N_LAST3, "last3")):
                bad = audit_mm(mm, total, cols)
                if bad:
                    raise SystemExit(
                        f"[审计失败] {name} 有 {len(bad)} 行未写入或非法，前若干行号：{bad[:10]}\n"
                        f"  说明中间结果不完整；请加 --reset 重跑（不要直接用于 I084）。")
            print("  审计通过（无空洞行）")
        print("  写入 npz（分块流式 + 原子替换）...")
        save_npz(out_npz, {"mm_last": mm_last, "mm_last3": mm_last3}, n_train, total, ids)
        with np.load(out_npz, mmap_mode="r") as z:
            shapes = {k: tuple(z[k].shape) for k in z.files}
            missing = [k for k in KEYS if k not in shapes]
        if missing:
            raise RuntimeError(f"npz 缺少键：{missing}")
        expect = {"train_last": (n_train, N_LAST), "train_last3": (n_train, N_LAST3),
                  "test_last": (total - n_train, N_LAST), "test_last3": (total - n_train, N_LAST3),
                  "train_protein_id": (n_train,), "test_protein_id": (total - n_train,)}
        wrong = {k: shapes.get(k) for k, v in expect.items() if shapes.get(k) != v}
        if wrong:
            raise RuntimeError(f"npz 形状校验失败：期望 {expect}，实际不符 {wrong}")
        meta = {"schema": SCHEMA, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "code/I/I083-esm2-650m-embed.py", "split": args.split,
                "canonical": args.split == "both", "total_rows": total, "n_train": n_train,
                "n_test": total - n_train, "file_rows": file_rows, "fingerprint": fp,
                "max_len": MAX_LEN, "shapes": {k: list(v) for k, v in shapes.items()},
                "elapsed_min": round((time.time() - t0) / 60, 1)}
        with Path(meta_path).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        release_mm(mm_last, mm_last3)
        remove_quiet(P["mm_last"], P["mm_last3"], P["prog"], P["prog"] + ".tmp",
                     out_npz + ".tmp")
        finalized = True
        print(f"  Written: {out_npz} ({os.path.getsize(out_npz) / 1e6:.0f} MB)"
              f" + {os.path.basename(meta_path)}")
    finally:
        if not finalized:
            release_mm(mm_last, mm_last3)
            print(f"  [收尾未完成] 中间结果保留在 {P['dir']}；重跑同一命令会直接重做收尾"
                  f"（幂等），无需 --reset。")
    print(f"  Completed in {(time.time() - t0) / 60:.1f} min")


def save_npz_arrays(path, last_out, last3_out, n_train, total, ids):
    """冒烟专用：数组已在内存，直接按同一 .npz 结构落盘。"""
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    n_test = total - n_train
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key, arr in (("train_last", last_out[:n_train]), ("train_last3", last3_out[:n_train]),
                         ("test_last", last_out[n_train:]), ("test_last3", last3_out[n_train:])):
            with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                _write_npy_member(fh, arr)
        for key, arr in (("train_protein_id", ids[:n_train]), ("test_protein_id", ids[n_train:])):
            with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                _write_npy_member(fh, arr)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
