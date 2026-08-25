# 🐬 2026 海豚杯：点燃大海（Light The Sea）团队技术方案

多标签蛋白质功能预测（500 个匿名 GO 类标签，Macro F1）竞赛方案。四路深度概率与序列邻居融合，再用 Reciprocal Rank Fusion 与 Sinkhorn 双边匹配做后处理。

## 当前锁定（2026-08-25）

| 口径 | 编号 | 分数 | 说明 |
| --- | --- | --- | --- |
| **平台最佳** | I011 | **45.6634**（AUC 0.713682） | 实例内 deep vs sim RRF，k=30 β=0.50 |
| **本地融合锁** | I019 | val F1 **0.4449** / OOF 较 I011 **+0.0011** | Sinkhorn τ=0.20 ρ=1.0 α=0.60，叠在 I011 上；提交文件已放桌面，尚未交平台 |
| 平台对照 | I008 | 45.3902（AUC 0.711026） | 门控后再截 hit 过预测 |
| 平台对照 | CK005a | 45.3039（AUC 0.7154） | DIAMOND `--very-sensitive` e=0.001 + 多 k |

ECK001 本地 0.4558 为 val 自匹配泄漏，提交 43.24 作废。缓存后处理 / 冻结 PLM / 异构 CNN 平均均未过 OOF +0.001。下一步是 C004 探针（等上行）与分析报告。

## 融合链（锁）

```
deep（T001a + T002 + CNN） + 0.75·sim
sim = 0.4·DIAMOND bitscore + 0.6·多 k k-mer（k=3~8，k6 主导）
I011: 再 + 0.50·col_norm( 1/(30+rank_deep) + 1/(30+rank_sim) )
I019: 对 I011 分数做 Sinkhorn，(1-0.60)·I011 + 0.60·col_norm(P)
SCut: 逐标签 100 档阈值，目标 Macro F1
```

- val 邻居只从 `idx_train` 建；test 可用全量 113,796
- RRF 公式锁 `1/(k+rank)`；k 锁 30；β 锁 0.50；Sinkhorn 锁在 **I011 融合之后**
- 有效门槛：全局 Macro F1 相对当前锁 OOF ≥ +0.001

## 项目结构

```
├── code/                 # 实验代码（按系列组织）
│   ├── B/                # 基线
│   ├── C/                # CNN（增强轴已关）
│   ├── T/                # Transformer
│   ├── TC/               # Transformer + CNN
│   ├── CK/               # CNN + k-mer / DIAMOND 融合
│   ├── TCK/              # 四路集成权重
│   ├── R/                # RNN（已关闭）
│   ├── E/                # 嵌入 / ECK 融合链（已关闭增量）
│   ├── I/                # 集成与后处理（I011 / I019 有效）
│   └── tools/            # EDA / 分析
├── data/                 # train.csv / test.csv（未上传）
├── doc/                  # 日志入口 doc/日志/README.md；规则与代码索引
├── rel/                  # 中间结果与缓存（gitignore）
├── submit/               # 提交文件（gitignore）
├── tools/                # DIAMOND
└── models/               # 预训练与权重（gitignore）
```

## 快速开始

```bash
# 环境：仓库 .venv，Python 3.11 + PyTorch（不要用系统 Python 3.14）
# 数据：竞赛 train.csv / test.csv 放入 data/
# 需先有 rel/ 中的邻居缓存与单模型概率（见 doc/日志/、doc/代码索引.md）

# 平台锁 I011
python code/I/I011-rrf.py --gen-sub

# 本地锁 I019（叠在 I011 上，尚未交平台）
python code/I/I019-sinkhorn.py --gen-sub
```

## 关键方法

- **kmer 邻居**：k=3~8，权重 `[0, 0.1, 0.2, 0.4, 0.2, 0.1]`，top-5 sim² 加权
- **DIAMOND**：`--very-sensitive` e=0.001，bitscore 加权 top-3，与 kmer 按 0.4 / 0.6 预融合
- **四路 deep**：T001a + T002 + CNN，权重来自 `models/tck002-ensemble.json`
- **I011 RRF**：每条蛋白把 500 标签按 deep、按 sim 各排一次名，`1/(30+rank)` 相加
- **I019 Sinkhorn**：行对齐 SCut 基数、列对齐标签先验，再与 I011 凸组合
- **I008**：半折 Logistic 门控 hit 过预测后按比例截；曾是平台最佳，已被 I011 超过

## 已关闭（不要重开）

CNN 输入增强（Mixup / CutMix / spanmask）、测试时读出（TTA / MC Dropout / AdaBN）、PLM LoRA / 冻结嵌入 kNN、共现图、蛋白图 LGC、二部图 LightGCN、k-mer 变体、比对权改写、PU / F-β / MMR / 段相关 / 融合前匹配、I011 字母派生、异构 CNN 平均、miss 分流。C004 同结构多种子探针未跑。细节见 `doc/日志/待办.md` 已完成节。

## 备注

- 数据、模型权重、`rel/` 缓存体积过大未入库，可依 `doc/日志/README.md` 逐步重建
- 实验编号与运行规则：`doc/实验规则.md`、`doc/代码索引.md`
- 日志按日追加，不要再写 `doc/工作日志.md`

---

## 🙏 团队

**点燃大海（Light The Sea）** —— 2026 海豚杯

- 团队负责人：[**Yue-xin GENG**](https://github.com/qyyyy09315)
- 团队成员：Shu XU、Si-yang LIU、Jiao-nei WU

本项目受 **Hong-xi TAO** 邀请，作为其带领队伍的主力参加。
