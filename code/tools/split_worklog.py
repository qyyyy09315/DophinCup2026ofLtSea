"""One-shot splitter: doc/工作日志.md → doc/日志/*.md (keep archive)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc" / "工作日志.md"
OUT = ROOT / "doc" / "日志"
ARCHIVE = OUT / "_archive" / "工作日志-拆分前.md"

LIT_HEAD = re.compile(r"^### 文献调研")
DATE_HEAD = re.compile(r"^## 2026-")
EXP_BLOCK = re.compile(
    r"^\*\*"
    r"(?:[A-Z]{1,4}\d{3}[a-z]?|[A-Z]{1,4}\d{3}[a-z]? ?/ ?[A-Z]{1,4}\d{3}[a-z]?|"
    r"E004(?:a)?(?: resume| 第四轮| 续训)?|"
    r"C002[a-z]?|CK005[a-z]?)"
)
PROC_KEYS = (
    "环境搭建", "规则建立", "记录规则更新", "规范与环境变更",
    "编号修正", "编号审计", "代码文件统一重命名", "代码按编号系列整理",
    "模型结构命名规则", "C002 参考文献",
)


def is_proc(title: str) -> bool:
    return any(k in title for k in PROC_KEYS)


def split_blocks(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """Return (kind, heading, body_lines) where kind in date/h3/todo/refs/other."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("## 待办"):
            j = i + 1
            while j < n and not (lines[j].startswith("## ") and not lines[j].startswith("### ")):
                j += 1
            blocks.append(("todo", line, lines[i:j]))
            i = j
            continue
        if line.startswith("## 参考文献"):
            blocks.append(("refs", line, lines[i:]))
            break
        if DATE_HEAD.match(line):
            j = i + 1
            while j < n and not lines[j].startswith("## "):
                j += 1
            blocks.append(("date", line, lines[i:j]))
            i = j
            continue
        i += 1
    return blocks


def split_date_body(body: list[str]) -> list[tuple[str, str, list[str]]]:
    """Split a dated section into ### subsections; first chunk is date header + preamble."""
    out = []
    i = 0
    # skip the ## date line
    date_line = body[0]
    i = 1
    while i < len(body) and body[i].strip() == "":
        i += 1
    start = i
    while i < len(body):
        if body[i].startswith("### "):
            if start < i:
                pre = body[start:i]
                if any(x.strip() for x in pre):
                    out.append(("pre", date_line, pre))
            j = i + 1
            while j < len(body) and not body[j].startswith("### "):
                j += 1
            out.append(("h3", body[i], body[i:j]))
            start = j
            i = j
            continue
        i += 1
    if start < len(body) and any(x.strip() for x in body[start:]):
        out.append(("pre", date_line, body[start:]))
    return out


def peel_lit(h3_body: list[str]) -> tuple[list[str], list[str]]:
    """Literature subsection → (survey_lines, experiment_block_lines)."""
    exp_at = None
    for i, line in enumerate(h3_body):
        if i == 0:
            continue
        if line.startswith("**") and ("完成" in line or "脚本就绪" in line or "静默退出" in line
                                      or "无梯度" in line or "resume" in line.lower()):
            # skip literature-internal bold like **可落地** **明确不投入**
            rest = line[2:]
            if rest.startswith(("可落地", "明确不投入", "下一步", "不投入", "跨领域",
                                "已做方法", "新数据分析", "筛过", "路线图",
                                "可迁移", "不并行", "含义")):
                continue
            if re.match(r"^[A-Z0-9/ ]{2,40}(完成|脚本就绪|第四轮|续训|resume)", rest, re.I):
                exp_at = i
                break
    if exp_at is None:
        return h3_body, []
    survey = h3_body[:exp_at]
    while survey and survey[-1].strip() in ("", "---"):
        survey.pop()
    survey.append("\n")
    survey.append("> 本轮落地实验写在 `实验日志.md` 同日条目。\n")
    survey.append("\n")
    return survey, h3_body[exp_at:]


def wrap_date(date_line: str, chunks: list[list[str]]) -> list[str]:
    out = [date_line if date_line.endswith("\n") else date_line + "\n", "\n"]
    for ch in chunks:
        if not ch:
            continue
        out.extend(ch)
        if ch and not ch[-1].endswith("\n"):
            out.append("\n")
        if ch and ch[-1].strip() != "---":
            out.append("\n")
    return out


def main():
    raw = SRC.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(raw, encoding="utf-8")

    exp_by_date: dict[str, list[list[str]]] = {}
    lit_by_date: dict[str, list[list[str]]] = {}
    proc_by_date: dict[str, list[list[str]]] = {}
    todo_lines: list[str] = []
    refs_lines: list[str] = []
    date_order: list[str] = []

    for kind, heading, body in split_blocks(lines):
        if kind == "todo":
            todo_lines = body
            continue
        if kind == "refs":
            refs_lines = body
            continue
        if kind != "date":
            continue
        date_line = heading
        if date_line not in date_order:
            date_order.append(date_line)
        for sk, _h, sub in split_date_body(body):
            title = sub[0] if sub else ""
            if sk == "h3" and LIT_HEAD.match(title):
                survey, exp = peel_lit(sub)
                lit_by_date.setdefault(date_line, []).append(survey)
                if exp:
                    exp_by_date.setdefault(date_line, []).append(exp)
            elif sk == "h3" and is_proc(title):
                proc_by_date.setdefault(date_line, []).append(sub)
            else:
                exp_by_date.setdefault(date_line, []).append(sub)

    def dump_dated(path: Path, title: str, blurb: str, store: dict[str, list[list[str]]]):
        parts = [f"# {title}\n", "\n", f"> {blurb}\n", "\n"]
        for date_line in date_order:
            chunks = store.get(date_line)
            if not chunks:
                continue
            parts.append("---\n")
            parts.extend(wrap_date(date_line, chunks))
        path.write_text("".join(parts), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    dump_dated(
        OUT / "实验日志.md",
        "实验日志",
        "技术实验记录（完成/负结果/分析）。填写规则见 `README.md`。历史原文：`_archive/工作日志-拆分前.md`。",
        exp_by_date,
    )
    dump_dated(
        OUT / "文献调研.md",
        "文献调研日志",
        "文献筛选轮次（可落地 / 不投入 / 下一步）。落地结果写实验日志，引用条目写参考文献。",
        lit_by_date,
    )
    dump_dated(
        OUT / "过程纪要.md",
        "过程纪要",
        "环境、编号、规范变更。不是实验结果，不写腾讯表格。",
        proc_by_date,
    )

    (OUT / "待办.md").write_text(
        "# 待办\n\n"
        "> 只保留未完成项在顶部「未完成」；已完成项追加到底部，不删。规则见 `README.md`。\n\n"
        + "".join(todo_lines[1:] if todo_lines else []),
        encoding="utf-8",
    )
    print(f"wrote 待办.md")

    refs_body = "".join(refs_lines[1:] if refs_lines else [])
    (OUT / "参考文献.md").write_text(
        "# 参考文献\n\n"
        "> 完整引用格式。新增只追加、编号顺延、不改已有条目。三处同步规则见 `README.md` 与 `doc/实验规则.md`。\n\n"
        + refs_body,
        encoding="utf-8",
    )
    print(f"wrote 参考文献.md")


if __name__ == "__main__":
    main()
