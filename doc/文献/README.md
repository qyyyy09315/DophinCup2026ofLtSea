# 本地备查文献

`doc/日志/参考文献.md` 共 211 条。PDF 由 `python code/tools/fetch-refs.py` 落到本目录，编号文件 `{id:03d}-{短名}.pdf`。进度见 [清单.md](清单.md) 与 `download-log.json`（含 sha256）。

当前（2026-09-02）：**ok=77 / fail=6 / skip_paywall=128**。新增落地 [204] Ankh、[206] SaProt。

开放全文来源：arXiv、bioRxiv/medRxiv、Unpaywall、EuropePMC `?pdf=render`。NCBI PMC 有 PoW 墙；OUP / IEEE / Science 出版社 PDF 常 403。付费墙不闯登录，记 `skip_paywall`，报告撰写时用 DOI 引用。

重试：`python code/tools/fetch-refs.py`（已 ok 的会跳过）。重建清单：`python code/tools/rebuild-ref-index.py`。

**不上 GitHub**：`*.pdf`、`download-log.json`、`manifest.json` 已进 `.gitignore`。引用以 `doc/日志/参考文献.md` 的 DOI 为准。
