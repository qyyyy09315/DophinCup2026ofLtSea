# 🐬 2026 海豚杯：点燃大海（Light The Sea）团队技术方案

多标签蛋白质功能预测（500 个 GO 类标签，Macro F1 评估）竞赛方案。融合序列组成（kmer 多 k 邻居）、DIAMOND 比对邻居与深度模型概率的四路集成框架。

## 当前最佳

提交最高 **CK005 = 44.8375**。ECK001 本地 0.4558 为 val 自匹配泄漏（ECK001a 闭环），提交 43.24 作废。

| 实验 | 方法 | Val F1 | 提交 |
| --- | --- | --- | --- |
| CK005 | DIAMOND bitscore 加权 top-3 邻居 + kmer 多 k 预融合 | **0.4359** | **44.8375** |
| CK004a | kmer 多 k（k6 主导）+ top5 sim² 聚合 | 0.4308 | — |
| CK004 | kmer 多 k 融合 | 0.4295 | — |
| TCK002a | T001a + T002 + CNN + sim 四路集成 | 0.4236 | 43.7276 |

## 项目结构

```
├── code/                 # 实验代码（按系列组织）
│   ├── B/                # 基线
│   ├── C/                # CNN 模型
│   ├── T/                # 单模型特征
│   ├── TC/               # 组合
│   ├── CK/               # kmer 邻居系列（CK002-CK005）
│   ├── TCK/              # 集成框架
│   ├── R/                # 对照实验（已关闭）
│   ├── I/                # 分类器对照（已关闭）
│   └── E/                # 嵌入方向（ESM2/ProstT5 实验）
├── data/                 # 竞赛数据（train.csv / test.csv，未上传）
├── doc/                  # 工作日志 / 实验规则 / 代码索引
├── rel/                  # 中间结果与缓存（npz，未上传）
├── submit/               # 提交文件（未上传）
├── tools/                # 外部工具（DIAMOND）
└── models/               # 预训练模型（未上传）
```

## 快速开始

```bash
# 环境（Python 3.11 + PyTorch + CUDA）
pip install -r requirements.txt   # 依赖见各脚本 import

# 数据：将竞赛 train.csv / test.csv 放入 data/

# 运行最佳集成（需先有 rel/ 中的邻居缓存与单模型概率，见工作日志）
python code/CK/CK005-alignment-neighbor.py --gen-sub
```

## 关键方法

- **kmer 邻居**：k=3~8 多尺度组成特征，k6 主导融合权重 `[0,0.1,0.2,0.4,0.2,0.1]`，top-5 邻居 sim² 加权聚合
- **DIAMOND 比对**：e-value 0.001 过滤，bitscore 加权 top-3 邻居，与 kmer 按 0.4/0.75 权重预融合
- **四路集成**：T001a（LSTM 表示）+ T002（序列统计特征）+ CNN + 邻居相似度，TCK002 框架权重
- **防泄漏**：val 用 train 91,036 空间，test 用全量 113,796 空间
- **逐标签阈值**：100 档阈值网格优化

## 备注

- 数据、模型权重、中间缓存体积过大未入库，可依工作日志（`doc/工作日志.md`）逐步重建
- 实验编号体系与运行规则见 `doc/实验规则.md`、`doc/代码索引.md`

---

## 🙏 团队

**点燃大海（Light The Sea）** —— 2026 海豚杯

- 团队负责人：[**Yue-xin GENG**](https://github.com/qyyyy09315)
- 团队成员：Shu XU、Si-yang LIU、Jiao-nei WU

本项目受 **Hong-xi TAO** 邀请，作为其带领队伍的主力参加。
