"""
I083 断点续跑机制自测（CPU-only：不加载模型、不占 GPU）
=====================================================
为什么需要它: I083 全量约 3-4 小时，断点机制一旦有洞，代价是整轮重跑或被污染的
  中间结果（I084 会直接读到坏数据）。本脚本用「假编码器」把完整机制跑一遍，包括
  中断、续跑、收尾、以及各类断点校验拒绝路径。

覆盖项:
  1  全新开始 → 收尾 → 产物与 meta 齐全、中间文件清除
  2  Ctrl+C 中断（真实 _thread.interrupt_main）→ 断点落盘、进程码 130
  3  断点校验拒绝: 指纹 / total / schema / 缺进度文件 / memmap 形状 / done 越界
  4  续跑 → 与原跑产物**逐位一致**（证明续跑不改变数值）
  5  --limit 冒烟仍写沙箱；隔离性用「泄漏检测」判定（正式路径出现沙箱形状的产物/文件名），
     因此**正式全量正在运行时也能跑**——不能断言「正式路径必须为空」，那只在干净仓库成立
  6  成功后再跑一次（无 --reset）应能正常重做，不因残留状态失败

运行:
  ./.venv/Scripts/python.exe -u code/tools/selftest_i083_resume.py

注: 只在 rel/_i083_selftest/ 下读写；不写腾讯表（分析性自测，规则见 doc/日志/README.md）。
"""
import os, sys, io, time, json, shutil, hashlib, threading, _thread, contextlib, traceback
import numpy as np
from importlib.machinery import SourceFileLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
I083_PATH = os.path.join(PROJECT_ROOT, "code", "I", "I083-esm2-650m-embed.py")
SANDBOX = os.path.join(PROJECT_ROOT, "rel", "_i083_selftest")
PROD = os.path.join(PROJECT_ROOT, "rel")
BACKUP = os.path.join(SANDBOX, "_snapshot")

N_ROWS = 1500            # 自测行数（走 --debug-subset，全在 train 前缀内）
FAKE_SLEEP = 0.25        # 假编码器每批耗时，用于让中断有机会落在循环中
                         # 注：主流程开头读 train.csv 约 1.5s，中断时刻必须晚于它，
                         #     否则中断会落在 CSV 读取段而非提取循环内。
INTERRUPT_AT = [2.8, 3.6, 4.4]   # 中断时刻（秒），逐个尝试直到落在循环中段

SB_FILES = ["esm2_650m_last.npy", "esm2_650m_last3.npy", "esm2_650m_progress.json",
            "esm2_650m_progress.json.tmp", "esm2_650m_embeddings.selftest.npz",
            "esm2_650m_embeddings.selftest.npz.tmp", "esm2_650m_embeddings.selftest.meta.json"]
SNAP_FILES = ["esm2_650m_last.npy", "esm2_650m_last3.npy", "esm2_650m_progress.json"]

i083 = SourceFileLoader("i083_under_test", I083_PATH).load_module()

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""), flush=True)


def _fake_encode_batch(model, tok, toks):
    """确定性假编码器：同一条序列 → 同一向量。故「续跑结果 == 原跑结果」可逐位比对。"""
    time.sleep(FAKE_SLEEP)
    b = len(toks)
    out_l = np.zeros((b, i083.N_LAST), np.float32)
    out_3 = np.zeros((b, i083.N_LAST3), np.float32)
    for j, s in enumerate(toks):
        seed = int.from_bytes(hashlib.md5(str(s).encode("utf-8")).digest()[:8], "little") % (2 ** 32)
        rng = np.random.default_rng(seed)
        out_l[j] = rng.standard_normal(i083.N_LAST).astype(np.float32)
        out_3[j] = rng.standard_normal(i083.N_LAST3).astype(np.float32)
    return out_l, out_3


# 替换掉唯一需要 GPU/模型的环节
i083.encode_batch = _fake_encode_batch
i083.load_model_and_tokenizer = lambda: (None, None)


def run_main(argv, interrupt_after=None):
    """在重定向输出下跑一次 main()，返回 (exit_code, text)。"""
    buf = io.StringIO()
    old_argv = sys.argv[:]
    sys.argv = ["I083-esm2-650m-embed.py"] + argv
    th = None
    try:
        if interrupt_after:
            th = threading.Thread(target=lambda: (time.sleep(interrupt_after),
                                                  _thread.interrupt_main()), daemon=True)
            th.start()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                i083.main()
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (
                    0 if e.code is None else "MSG:" + str(e.code))
    except BaseException:
        buf.write("\n[harness] 未预期异常:\n" + traceback.format_exc())
        code = "TRACE"
    finally:
        sys.argv = old_argv
    return code, buf.getvalue()


def sandbox_clean():
    os.makedirs(SANDBOX, exist_ok=True)
    for f in SB_FILES:
        p = os.path.join(SANDBOX, f)
        if os.path.exists(p):
            os.remove(p)


def sb(f):
    return os.path.join(SANDBOX, f)


def snapshot():
    if os.path.isdir(BACKUP):
        shutil.rmtree(BACKUP)
    os.makedirs(BACKUP)
    for f in SNAP_FILES:
        if os.path.exists(sb(f)):
            shutil.copy2(sb(f), os.path.join(BACKUP, f))


def restore():
    for f in SNAP_FILES:
        src, dst = os.path.join(BACKUP, f), sb(f)
        if os.path.exists(dst):
            os.remove(dst)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def prod_busy():
    """正式路径是否存在断点/中间产物（含「正式全量正在运行」的情形）。"""
    return any(os.path.exists(os.path.join(PROD, f)) for f in SNAP_FILES)


def prod_leak_evidence():
    """检测正式路径里是否有「自测产物泄漏」的痕迹。
    注意：不能用「正式路径必须为空」当断言——正式全量正在跑时它必然非空。
    这里只查「沙箱形状」的痕迹（total/行数 == 自测行数，或出现沙箱专用文件名），
    因此对正在运行的正式任务免疫。"""
    ev = []
    if os.path.exists(os.path.join(PROD, "esm2_650m_embeddings.selftest.npz")):
        ev.append("rel/ 出现沙箱专用 npz 名")
    pg = os.path.join(PROD, "esm2_650m_progress.json")
    if os.path.exists(pg):
        try:
            st = json.load(open(pg, encoding="utf-8"))
        except Exception as e:
            ev.append(f"progress 不可解析: {e}")
            st = None
        if isinstance(st, dict):
            if st.get("schema") is None:
                ev.append("progress 无 schema（异源/旧版写入）")
            elif st.get("total") in (N_ROWS, 64):
                ev.append(f"progress 疑似自测形状 total={st.get('total')}")
    for f in ("esm2_650m_last.npy", "esm2_650m_last3.npy"):
        p = os.path.join(PROD, f)
        if not os.path.exists(p):
            continue
        try:
            a = np.load(p, mmap_mode="r")
            shp = tuple(a.shape)
            if hasattr(a, "_mmap"):
                a._mmap.close()
            if shp[0] in (N_ROWS, 64):
                ev.append(f"{f} shape={shp} 疑似自测形状")
        except Exception as e:
            ev.append(f"{f} 不可读: {e}")
    return ev


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I083 断点续跑机制自测（假编码器，CPU-only）")
    print("=" * 88)
    print(f"  被测脚本: {I083_PATH}")
    print(f"  沙箱目录: {SANDBOX}   自测行数: {N_ROWS}")

    busy0 = prod_busy()
    leak0 = prod_leak_evidence()
    note = "正式全量在跑，故只查泄漏不查「路径为空」" if busy0 else "正式路径为空"
    check("前置：正式路径无自测泄漏痕迹", not leak0,
          note + (f"；泄漏证据 {leak0}" if leak0 else ""))

    # ---------- 1. 全新开始 → 收尾 ----------
    print("\n-- 1) 全新开始 → 收尾 --", flush=True)
    sandbox_clean()
    code, txt = run_main(["--debug-subset", str(N_ROWS), "--reset", "--save-every", "0.001"])
    ok = code == 0 and os.path.exists(sb("esm2_650m_embeddings.selftest.npz"))
    check("全新跑通并产出 npz", ok, f"exit={code}")
    if not ok:
        print(txt[-3000:])
        return finish(t0)
    check("中间文件已清除（memmap/progress）",
          not any(os.path.exists(sb(f)) for f in SNAP_FILES))
    check("meta.json 已写出", os.path.exists(sb("esm2_650m_embeddings.selftest.meta.json")))
    ref = load_npz(sb("esm2_650m_embeddings.selftest.npz"))
    need_keys = {"train_last", "train_last3", "test_last", "test_last3",
                 "train_protein_id", "test_protein_id"}
    check("npz 键完整（I084 兼容）", need_keys <= set(ref), f"缺 {sorted(need_keys - set(ref))}")
    check("train/test 形状正确",
          ref["train_last"].shape == (N_ROWS, i083.N_LAST) and ref["test_last"].shape == (0, i083.N_LAST),
          f"{ref['train_last'].shape} / {ref['test_last'].shape}")
    check("嵌入非全零（审计语义正确）", bool((ref["train_last"] != 0).any()))
    pid = ref["train_protein_id"]
    check("protein_id 以 Unicode 存储（非 object/裸指针）", pid.dtype.kind == "U", f"dtype={pid.dtype}")
    import pandas as pd
    csv_pid = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"),
                          usecols=["protein_id"], nrows=N_ROWS)["protein_id"].astype(str).values
    check("protein_id 内容与 train.csv 逐位一致", np.array_equal(pid.astype(str), csv_pid),
          f"首={pid[0]!r} 末={pid[-1]!r}")

    # ---------- 2. 中断 ----------
    print("\n-- 2) Ctrl+C 中断（真实 interrupt_main） --", flush=True)
    done, code, txt, have_state = None, None, "", False
    for attempt, delay in enumerate(INTERRUPT_AT):
        sandbox_clean()
        code, txt = run_main(["--debug-subset", str(N_ROWS), "--reset", "--save-every", "0.001"],
                             interrupt_after=delay)
        pg = sb("esm2_650m_progress.json")
        if code == 130 and os.path.exists(pg):
            st0 = json.load(open(pg, encoding="utf-8"))
            done = st0.get("done")
            if isinstance(done, int) and 0 < done < N_ROWS:
                break
        print(f"    （第 {attempt + 1} 次中断落点不理想：exit={code}, done={done}，重试）", flush=True)
    check("中断返回码 130（SystemExit）", code == 130, f"exit={code}")
    check("断点文件已落盘且 0 < done < total", isinstance(done, int) and 0 < done < N_ROWS,
          f"done={done}/{N_ROWS}")
    if code != 130 or not isinstance(done, int) or not (0 < done < N_ROWS):
        print("    中断环节未达预期，末尾输出：\n" + txt[-1200:])
    pg = sb("esm2_650m_progress.json")
    have_state = os.path.exists(pg) and os.path.exists(sb("esm2_650m_last.npy")) \
        and os.path.exists(sb("esm2_650m_last3.npy"))
    check("中断后 memmap 仍在（可续跑）", have_state)
    check("中断后未产出 npz", not os.path.exists(sb("esm2_650m_embeddings.selftest.npz")))
    if have_state:
        st = json.load(open(pg, encoding="utf-8"))
        check("断点含全部校验字段",
              {"schema", "done", "total", "n_train", "split", "max_len", "fingerprint"} <= set(st),
              f"字段={sorted(st)}")
        snapshot()
    else:
        check("断点含全部校验字段", False, "断点缺失，跳过")
        check("后续校验/续跑用例", False, "因中断环节未留下可用断点而跳过")

    # ---------- 3. 断点校验拒绝路径 ----------
    print("\n-- 3) 断点校验拒绝路径 --", flush=True)
    if have_state:
        def tamper(name, mutate, keyword):
            restore()
            mutate()
            code, txt = run_main(["--debug-subset", str(N_ROWS), "--save-every", "0.001"])
            msg = code if isinstance(code, str) else f"exit={code}"
            ok = isinstance(code, str) and code.startswith("MSG:") and keyword in code
            check(f"拒绝续跑：{name}", ok, f"命中「{keyword}」" if ok else f"实际 {msg[:160]}")

        def edit_prog(fn):
            st = json.load(open(pg, encoding="utf-8"))
            fn(st)
            json.dump(st, open(pg, "w", encoding="utf-8"), ensure_ascii=False)

        tamper("指纹不一致",
               lambda: edit_prog(lambda s: s.__setitem__("fingerprint", "deadbeefdeadbeef")), "指纹")
        tamper("total 不一致", lambda: edit_prog(lambda s: s.__setitem__("total", 999)), "total")
        tamper("schema 缺失（旧版/异源断点）",
               lambda: edit_prog(lambda s: s.pop("schema", None)), "schema")
        tamper("done 越界", lambda: edit_prog(lambda s: s.__setitem__("done", N_ROWS + 99)), "超出")

        def drop_prog():
            os.remove(pg)
        tamper("有 memmap 无进度文件", drop_prog, "没有进度文件")

        def bad_shape():
            p = sb("esm2_650m_last.npy")
            os.remove(p)
            np.lib.format.open_memmap(p, mode="w+", dtype=np.float32,
                                      shape=(N_ROWS + 5, i083.N_LAST))
        tamper("memmap 形状不匹配", bad_shape, "形状不匹配")

    # ---------- 4. 续跑逐位一致 ----------
    print("\n-- 4) 续跑 → 与原跑逐位一致 --", flush=True)
    if have_state:
        restore()
        st = json.load(open(pg, encoding="utf-8"))
        code, txt = run_main(["--debug-subset", str(N_ROWS), "--save-every", "0.001"])
        check("续跑跑通（未加 --reset）", code == 0, f"exit={code}")
        if code == 0:
            res = load_npz(sb("esm2_650m_embeddings.selftest.npz"))
            same = all(np.array_equal(ref[k], res[k]) for k in need_keys)
            check("续跑与原跑产物逐位一致", same,
                  f"续跑起点 done={st['done']}，比对 {len(need_keys)} 个键")
            for k in need_keys:
                if not np.array_equal(ref[k], res[k]):
                    print(f"    差异键: {k}")
        else:
            print(txt[-2000:])
    else:
        check("续跑跑通（未加 --reset）", False, "无可用断点，跳过")
        check("续跑与原跑产物逐位一致", False, "无可用断点，跳过")

    # ---------- 5. 隔离性 ----------
    print("\n-- 5) 沙箱隔离性 --", flush=True)
    sandbox_clean()
    code, txt = run_main(["--limit", "16", "--save-every", "0.001"])
    check("--limit 冒烟跑通", code == 0, f"exit={code}")
    check("冒烟产物落在沙箱", os.path.exists(sb("esm2_650m_embeddings.selftest.npz")))
    leak1 = prod_leak_evidence()
    check("正式路径未被自测改动（泄漏检查）", not leak1, "; ".join(leak1) if leak1 else "无泄漏")
    if not busy0:
        check("全程未在正式路径产生文件", not prod_busy())
    else:
        check("全程未在正式路径产生文件", True, "跳过（正式全量仍在运行，其产物非自测所写）")
    check("路径解析：自测参数指向沙箱、正式模式指向 rel/",
          i083.SELFTEST_DIR == SANDBOX
          and i083.resolve_paths(True)["dir"] == SANDBOX
          and i083.resolve_paths(False)["dir"] == PROD)
    check("产物命名：正式单 split 不占用规范名",
          i083.out_path_for(i083.resolve_paths(False), "train").endswith(".train.npz")
          and i083.out_path_for(i083.resolve_paths(False), "both").endswith("esm2_650m_embeddings.npz"))

    # ---------- 6. 成功后重跑 ----------
    print("\n-- 6) 成功后再跑一次（无 --reset） --", flush=True)
    sandbox_clean()
    code1, _ = run_main(["--debug-subset", "64", "--reset", "--save-every", "0.001"])
    code2, _ = run_main(["--debug-subset", "64", "--save-every", "0.001"])
    check("二次运行不因残留状态失败", code1 == 0 and code2 == 0, f"exit1={code1} exit2={code2}")

    return finish(t0)


def finish(t0):
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n = len(RESULTS)
    print("\n" + "=" * 88)
    print(f"  自测结果: {n_pass}/{n} 通过   用时 {(time.time() - t0) / 60:.1f} min")
    if n_pass != n:
        print("  失败项:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}  {detail}")
    print("=" * 88)
    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
