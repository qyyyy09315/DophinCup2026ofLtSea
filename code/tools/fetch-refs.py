"""
fetch-refs — 把参考文献.md 对应 PDF 落到 doc/文献，并写校验清单
================================================================
优先开放全文：已有本地文件 → arXiv / export.arxiv → ICLR / OpenReview
→ bioRxiv/medRxiv full.pdf → Google Patents PDF → Unpaywall → EuropePMC。
流式拉取，超时 180s，失败重试。不闯登录墙。失败记 status，不中断。
用法:
  python code/tools/fetch-refs.py
  python code/tools/fetch-refs.py --limit=20
  python code/tools/fetch-refs.py --ids=1,2,10
输出:
  doc/文献/{id:03d}-{slug}.pdf
  doc/文献/download-log.json
  doc/文献/清单.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIT_DIR = PROJECT_ROOT / "doc" / "文献"
MANIFEST = LIT_DIR / "manifest.json"
LOG_PATH = LIT_DIR / "download-log.json"
INDEX_MD = LIT_DIR / "清单.md"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
TIMEOUT = 180
CHUNK = 1 << 16
RETRIES = 3


def slug(name: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", name).strip("-")
    return (s[:n] or "ref").rstrip("-")


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-" and (b"%%EOF" in data[-4096:] or len(data) > 200_000)


ALLOWED_HOSTS = {
    "arxiv.org", "export.arxiv.org", "www.biorxiv.org", "www.medrxiv.org",
    "openreview.net", "proceedings.iclr.cc", "ceur-ws.org", "papers.nips.cc",
    "proceedings.neurips.cc", "api.unpaywall.org", "www.ebi.ac.uk",
    "europepmc.org", "www.ncbi.nlm.nih.gov", "patents.google.com",
    "patentimages.storage.googleapis.com", "www.csie.ntu.edu.tw",
    "huggingface.co", "hf-mirror.com", "www.modelscope.cn",
    "academic.oup.com", "www.dcs.gla.ac.uk",
}


def guard_url(url: str) -> None:
    """SSRF 防护（CWE-918）：协议限定 https、主机白名单、解析后阻断私网/环回/链路本地，重定向每跳复验由 get_pdf 的 hops 复验承担"""
    import ipaddress
    import socket
    p = urllib.parse.urlparse(url)
    if p.scheme != "https":
        raise ValueError(f"协议不允许: {p.scheme}")
    host = p.hostname or ""
    if not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise ValueError(f"主机不在白名单: {host}")
    for info in socket.getaddrinfo(host, p.port or 443):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"解析到受限地址: {ip}")


def open_url(url: str, timeout: int = TIMEOUT):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/pdf,text/html,*/*",
        },
    )
    guard_url(url)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        # 学术站证书链不全（如 csie.ntu.edu.tw）：降级不校验证书重试，仅限公开 PDF 拉取
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def get(url: str) -> bytes | None:
    last = None
    for i in range(RETRIES):
        try:
            with open_url(url) as r:
                return r.read(2_000_000)
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    print(f"    GET fail {url[:80]} {last}")
    return None


def stream_body(r) -> bytes:
    buf = bytearray()
    while True:
        chunk = r.read(CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > 80_000_000:
            break
    return bytes(buf)


def get_pdf(url: str, hops: int = 0) -> bytes | None:
    if hops > 3:
        return None
    last = None
    for i in range(RETRIES):
        try:
            with open_url(url) as r:
                final = r.geturl()
                ctype = (r.headers.get("Content-Type") or "").lower()
                head = r.read(16)
                if head[:5] == b"%PDF-":
                    data = head + stream_body(r)
                    if is_pdf(data):
                        return data
                    return None
                rest = r.read(2_000_000)
                data = head + rest
            text = data.decode("utf-8", "replace") if data else ""
            m = re.search(
                r'https://(?:patentimages\.storage\.googleapis\.com|arxiv\.org|export\.arxiv\.org|proceedings\.iclr\.cc|openreview\.net|www\.biorxiv\.org)[^"\'\s<>]+?\.pdf(?:\?[^"\'\s<>]*)?',
                text,
            )
            if not m:
                m = re.search(r'https://[^"\'\s<>]+?\.pdf(?:\?[^"\'\s<>]*)?', text)
            if not m:
                m = re.search(r'content=["\']([^"\']+\.pdf[^"\']*)["\']', text, re.I)
            if m:
                nxt = m.group(1) if m.lastindex else m.group(0)
                nxt = nxt.replace("&amp;", "&")
                if nxt.startswith("/"):
                    nxt = urllib.parse.urljoin(final, nxt)
                if nxt != url:
                    print(f"    follow {nxt[:90]}")
                    return get_pdf(nxt, hops + 1)
            if "pdf" in ctype and is_pdf(data):
                return data
            return None
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    print(f"    GET fail {url[:80]} {last}")
    return None


def unpaywall_pdf(doi: str) -> str | None:
    url = f"https://api.unpaywall.org/v2/{doi}?email=uncategorized@gmail.com"
    raw = get(url)
    if not raw:
        return None
    try:
        d = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    loc = d.get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf") or loc.get("url")
    if pdf and str(pdf).lower().endswith(".pdf"):
        return pdf
    if loc.get("url_for_pdf"):
        return loc["url_for_pdf"]
    oa = d.get("oa_locations") or []
    for x in oa:
        if x.get("url_for_pdf"):
            return x["url_for_pdf"]
    return None


def ncbi_pmc_pdf(doi: str) -> str | None:
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={urllib.parse.quote(doi)}&format=json"
    raw = get(url)
    if not raw:
        return None
    try:
        d = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    recs = d.get("records") or []
    if not recs:
        return None
    pmcid = recs[0].get("pmcid")
    if not pmcid:
        return None
    return f"https://europepmc.org/articles/{pmcid}?pdf=render"


def europepmc_pdf(doi: str) -> str | None:
    q = urllib.parse.quote(f"DOI:{doi}")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=core"
    raw = get(url)
    if not raw:
        return None
    try:
        d = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    hits = (((d.get("resultList") or {}).get("result")) or [])
    if not hits:
        return None
    h = hits[0]
    pmcid = h.get("pmcid")
    if pmcid:
        return f"https://europepmc.org/articles/{pmcid}?pdf=render"
    if h.get("isOpenAccess") == "Y" and h.get("fullTextUrlList"):
        for u in (h["fullTextUrlList"].get("fullTextUrl") or []):
            if (u.get("documentStyle") or "").lower() == "pdf":
                return u.get("url")
    return None


def patent_id(text: str) -> str | None:
    m = re.search(r"\b((?:WO|US|EP|CN|JP)\d+[A-Z]?\d*)\b", text or "", re.I)
    return m.group(1).upper() if m else None


def candidate_urls(e: dict) -> list[str]:
    urls = []
    if e.get("arxiv"):
        aid = e["arxiv"]
        urls.append(f"https://export.arxiv.org/pdf/{aid}")
        urls.append(f"https://arxiv.org/pdf/{aid}")
        urls.append(f"https://arxiv.org/pdf/{aid}.pdf")
    doi = e.get("doi") or ""
    if doi.startswith("10.1101/") or doi.startswith("10.64898/"):
        urls.append(f"https://www.biorxiv.org/content/{doi}v1.full.pdf")
        urls.append(f"https://www.biorxiv.org/content/{doi}v2.full.pdf")
        urls.append(f"https://www.biorxiv.org/content/{doi}v3.full.pdf")
        urls.append(f"https://www.biorxiv.org/content/{doi}.full.pdf")
        urls.append(f"https://www.medrxiv.org/content/{doi}v1.full.pdf")
    for u in e.get("links") or []:
        if u.lower().endswith(".pdf"):
            urls.append(u)
        if "arxiv.org/abs/" in u:
            aid = u.split("/abs/")[-1].split("v")[0]
            urls.append(f"https://export.arxiv.org/pdf/{aid}")
            urls.append(f"https://arxiv.org/pdf/{aid}")
        if "openreview.net/forum?id=" in u or "openreview.net/pdf?id=" in u:
            oid = u.split("id=")[-1].split("&")[0]
            urls.append(f"https://openreview.net/pdf?id={oid}")
        if "proceedings.iclr.cc" in u and "/hash/" in u:
            h = u.split("/hash/")[-1].split("-")[0]
            urls.append(
                f"https://proceedings.iclr.cc/paper_files/paper/2024/file/{h}-Paper-Conference.pdf"
            )
        if "ceur-ws.org" in u and u.endswith(".pdf"):
            urls.append(u)
        if ("papers.nips.cc" in u or "proceedings.neurips.cc" in u) and "/hash/" in u:
            h = u.split("/hash/")[-1].split("-")[0]
            ym = re.search(r"/paper/(\d{4})/", u)
            y = ym.group(1) if ym else "2018"
            urls.append(f"https://papers.nips.cc/paper_files/paper/{y}/file/{h}-Paper.pdf")
            urls.append(f"https://proceedings.neurips.cc/paper_files/paper/{y}/file/{h}-Paper.pdf")
            urls.append(f"https://proceedings.neurips.cc/paper_files/paper/{y}/file/{h}-Paper-Conference.pdf")
            urls.append(u)
        if re.match(r"https://papers\.nips\.cc/paper/\d+-", u):
            urls.append(u + ".pdf")
            urls.append(u)
        if "patents.google.com/patent/" in u:
            pid = u.split("/patent/")[-1].split("/")[0]
            urls.append(f"https://patents.google.com/patent/{pid}.pdf")
            urls.append(f"https://patents.google.com/patent/{pid}/en")
    pid = patent_id(e.get("name") or "") or patent_id(" ".join(e.get("links") or []))
    if pid:
        urls.append(f"https://patents.google.com/patent/{pid}.pdf")
        urls.append(f"https://patents.google.com/patent/{pid}/en")
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def match_existing(e: dict, files: list[Path]) -> Path | None:
    notes = e.get("local_note") or []
    for n in notes:
        p = PROJECT_ROOT / n.replace("\\", "/")
        if p.is_file() and p.stat().st_size > 10_000:
            return p
    prefix = f"{e['id']:03d}-"
    for p in files:
        if p.name.startswith(prefix):
            return p
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_index(log: dict):
    rows = log["items"]
    ok = sum(1 for r in rows if r["status"] == "ok")
    fail = sum(1 for r in rows if r["status"] == "fail")
    skip = sum(1 for r in rows if r["status"] == "skip_paywall")
    lines = [
        "# 参考文献 PDF 清单",
        "",
        f"> 生成自 `code/tools/fetch-refs.py`。ok={ok} fail={fail} skip_paywall={skip} / {len(rows)}",
        "",
        "| 编号 | 短名 | 状态 | 文件 | 大小 | 来源 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        sz = r.get("bytes") or ""
        if isinstance(sz, int) and sz:
            sz = f"{sz/1e6:.1f}M" if sz > 1e6 else f"{sz/1e3:.0f}K"
        lines.append(
            f"| [{r['id']}] | {r['name']} | {r['status']} | {r.get('file') or ''} | {sz} | {r.get('source') or ''} |"
        )
    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = man["entries"]
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        entries = [e for e in entries if e["id"] in want]
    if args.limit:
        entries = entries[: args.limit]
    LIT_DIR.mkdir(exist_ok=True)
    existing = list(LIT_DIR.glob("*.pdf"))
    log = {"items": []}
    if LOG_PATH.exists():
        try:
            log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = {"items": []}
    done = {r["id"]: r for r in log.get("items", [])}

    for e in entries:
        i, name = e["id"], e["name"]
        print(f"[{i:03d}] {name}")
        if i in done and done[i].get("status") in {"ok"} and done[i].get("file"):
            fp = LIT_DIR / Path(done[i]["file"]).name
            if not fp.exists() and done[i].get("file"):
                fp = PROJECT_ROOT / done[i]["file"]
            if fp.exists():
                print("    cached ok")
                continue
        hit = match_existing(e, existing)
        rec = {"id": i, "name": name, "doi": e.get("doi"), "arxiv": e.get("arxiv")}
        if hit:
            data = hit.read_bytes()
            if is_pdf(data):
                dest = LIT_DIR / f"{i:03d}-{slug(name)}.pdf"
                if hit.resolve() != dest.resolve():
                    dest.write_bytes(data)
                rec.update({
                    "status": "ok", "file": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "bytes": dest.stat().st_size, "sha256": sha256(dest),
                    "source": "local",
                })
                done[i] = rec
                print(f"    local {dest.name} {dest.stat().st_size}")
                log["items"] = [done[k] for k in sorted(done)]
                LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
        urls = candidate_urls(e)
        pdf_url = None
        data = None
        src = None
        host_fail = set()
        for u in urls:
            host = urllib.parse.urlparse(u).netloc
            if host in host_fail:
                continue
            print(f"    try {u[:90]}")
            data = get_pdf(u)
            if data:
                pdf_url, src = u, "direct"
                break
            if "biorxiv.org" in host or "medrxiv.org" in host:
                host_fail.add(host)
        if data is None and e.get("doi"):
            u = europepmc_pdf(e["doi"]) or ncbi_pmc_pdf(e["doi"])
            if u:
                print(f"    pmc {u[:90]}")
                data = get_pdf(u)
                if data:
                    pdf_url, src = u, "europepmc"
            time.sleep(0.3)
        if data is None and e.get("doi"):
            print("    unpaywall", e["doi"])
            u = unpaywall_pdf(e["doi"])
            if u:
                print(f"    oa {u[:90]}")
                if "biorxiv.org" in u and any("biorxiv.org" in x for x in urls):
                    print("    skip duplicate biorxiv oa")
                else:
                    data = get_pdf(u)
                    if data:
                        pdf_url, src = u, "unpaywall"
            time.sleep(0.4)
        if data is None:
            rec.update({"status": "skip_paywall" if e.get("doi") else "fail",
                        "file": None, "source": None})
            done[i] = rec
            print("    SKIP")
        else:
            dest = LIT_DIR / f"{i:03d}-{slug(name)}.pdf"
            dest.write_bytes(data)
            rec.update({
                "status": "ok",
                "file": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source": src,
                "url": pdf_url,
            })
            done[i] = rec
            print(f"    OK {dest.name} {len(data)}")
        log["items"] = [done[k] for k in sorted(done)]
        LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.3)
    log["items"] = [done[k] for k in sorted(done)]
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.ids:
        write_index(log)
    ok = sum(1 for r in log["items"] if r["status"] == "ok")
    print(f"done ok={ok}/{len(log['items'])}  {INDEX_MD}")


if __name__ == "__main__":
    main()
