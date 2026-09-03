"""Rebuild doc/文献/download-log.json and 清单.md from numbered PDFs."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIT = ROOT / "doc" / "文献"
MAN = json.loads((LIT / "manifest.json").read_text(encoding="utf-8"))
ENTRIES = {e["id"]: e for e in MAN["entries"]}
LOGP = LIT / "download-log.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log = {"items": []}
    if LOGP.exists():
        try:
            log = json.loads(LOGP.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = {"items": []}
    byid = {x["id"]: x for x in log.get("items", [])}
    pdfs = {}
    for p in LIT.glob("*.pdf"):
        m = re.match(r"^(\d{3})-", p.name)
        if m:
            pdfs[int(m.group(1))] = p
    md_status = {}
    idxp = LIT / "清单.md"
    if idxp.exists():
        for ln in idxp.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\| \[(\d+)\] \| (.+?) \| (\S+) \|", ln)
            if m:
                md_status[int(m.group(1))] = m.group(3)
    for i, p in pdfs.items():
        rec = byid.get(i, {})
        rec.update({
            "id": i,
            "name": ENTRIES.get(i, {}).get("name", rec.get("name", "")),
            "doi": ENTRIES.get(i, {}).get("doi"),
            "arxiv": ENTRIES.get(i, {}).get("arxiv"),
            "status": "ok",
            "file": str(p.relative_to(ROOT)).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": rec.get("sha256") if (rec.get("file") or "").endswith(p.name) else sha256(p),
            "source": rec.get("source") or "local",
        })
        byid[i] = rec
    for i in range(1, max(ENTRIES, default=0) + 1):
        if i in byid:
            continue
        e = ENTRIES.get(i, {"name": "?"})
        rec = {
            "id": i,
            "name": e.get("name"),
            "doi": e.get("doi"),
            "arxiv": e.get("arxiv"),
            "status": md_status.get(i, "skip_paywall"),
            "file": None,
            "source": None,
        }
        byid[i] = rec
    items = [byid[i] for i in sorted(byid)]
    LOGP.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in items if r["status"] == "ok")
    fail = sum(1 for r in items if r["status"] == "fail")
    skip = sum(1 for r in items if r["status"] == "skip_paywall")
    lines = [
        "# 参考文献 PDF 清单",
        "",
        f"> 生成自 `code/tools/fetch-refs.py`。ok={ok} fail={fail} skip_paywall={skip} / {len(items)}",
        "",
        "| 编号 | 短名 | 状态 | 文件 | 大小 | 来源 |",
        "|---|---|---|---|---|---|",
    ]
    for r in items:
        sz = r.get("bytes") or ""
        if isinstance(sz, int) and sz:
            sz = f"{sz / 1e6:.1f}M" if sz > 1e6 else f"{sz / 1e3:.0f}K"
        lines.append(
            f"| [{r['id']}] | {r['name']} | {r['status']} | {r.get('file') or ''} | {sz} | {r.get('source') or ''} |"
        )
    idxp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dict(Counter(x["status"] for x in items)), "n", len(items), "pdfs", len(pdfs))


if __name__ == "__main__":
    main()
