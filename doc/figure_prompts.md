# 图 3.1（HDMR-Sinkhorn 三路径架构与推断数据流总览）AI 绘图提示词

> 用途：报告正文图 3.1 目前是占位框（英文原理图待生成）。将下述提示词交给图像生成模型
> （Nano Banana / GPT-Image / 即梦 / 可图 / Midjourney 等），生成后替换占位框并重建报告。
> **生成后必须人工校对图中每一个英文标签的拼写**（AI 生图极易拼错术语），并与正文术语一致：
> P_deep / S_sim / S_plm / S_fused / RRF / Sinkhorn-Knopp / SCut / 1664-d。
> 图内文字一律英文（报告题注已含中英双语，无需画进图里）；画布宽高比约 2:1，300 dpi。

---

## 一、Master prompt（英文，适用于 Nano Banana / GPT-Image / DALL·E 等长文本模型）

```
Create a publication-quality scientific schematic diagram for a bioinformatics methods paper
(Nature / Nature Methods house style, flat vector look, landscape).

Topic: overview of "HDMR-Sinkhorn", a three-path protein function prediction pipeline that fuses
(I) deep sequence perception, (II) homology & k-mer transfer, and (III) frozen pretrained
language-model embeddings, then applies reciprocal rank fusion and a Sinkhorn optimal-transport
decision layer to emit a binary label matrix.

Layout (landscape, left to right, three parallel paths converging rightward):

1. Leftmost: one rounded rectangle "Protein sequence (<=1,022 aa)". Three thin arrows branch from
   it into the three path panels.

2. Three vertically stacked path panels (each a light-tinted rounded box with a saturated header
   strip; each path uses exactly one hue):
   - Panel I, header "Path I · Deep sequence perception" (blue #0072B2): three small stacked blocks
     connected by arrows: "Multi-scale dilated CNN (k = 7/15/31)" -> "CSRA class-specific attention"
     -> "RoFormer encoder"; output arrow labelled "P_deep".
   - Panel II, header "Path II · Homology & k-mer transfer" (green #009E73): two side-by-side
     sub-blocks "DIAMOND alignment (E <= 0.001, top-3 bitscore weighted, 2-hop transfer)" and
     "k-mer spectrum (k = 3-8)", converging into one block "Weighted label prior"; output arrow
     labelled "S_sim".
   - Panel III, header "Path III · Pretrained language models" (orange #E69F00): two side-by-side
     sub-blocks "ESM-2 (640-d, frozen)" and "ProtBERT (1024-d, frozen)" -> "Mean pooling + concat
     (1664-d)" -> "Residual MLP x 3 seeds"; output arrow labelled "S_plm".

3. Centre-right: one tall box "Reciprocal rank fusion" (purple #9467BD) receiving three arrows
   (from P_deep, S_sim, S_plm); inside two stacked sub-blocks: "Linear injection of S_plm (w = 0.35)"
   and "RRF (k = 30, beta = 0.50)"; output arrow labelled "S_fused".

4. Right: one box "Sinkhorn-Knopp optimal transport" (vermillion #D55E00) with two dashed
   constraint arrows pointing to its left edge from two small grey tags "row marginal = predicted
   cardinality r_i" and "column marginal = label frequency c_j"; inside also "tau = 0.20,
   alpha = 0.60". Then -> block "Per-label thresholding (SCut)" -> final rounded box
   "Binary label matrix (N x 500)".

5. Bottom: a thin grey annotation strip, small text: "Training labels (500) · train-only DIAMOND
   database · no external annotations".

Style requirements:
- Flat vector illustration, pure white background, no shadows, no gradients, no 3-D, no textures.
- Sans-serif labels (Helvetica or Arial), all text horizontal, min 7 pt at final print size.
- Colour-blind-safe palette (Okabe-Ito): blue #0072B2 (Path I), green #009E73 (Path II),
  orange #E69F00 (Path III), purple #9467BD (fusion), vermillion #D55E00 (decision);
  grey #666666 for secondary annotations. Panel fills at 12-18% opacity of the panel hue;
  headers and arrowheads at full saturation.
- Arrows: solid 1.2 pt for data flow, dashed 1 pt for the two marginal-constraint annotations;
  small open arrowheads. Even spacing, tidy orthogonal routing, publication quality.
- Spell every label exactly as written above — these are method names and hyper-parameters
  quoted from the paper. No other text anywhere in the figure.

Negative prompt: photorealism, 3-D render, cartoon icons, DNA double-helix decoration,
brain or neuron clichés, network node graphs, watermark, gradient fills, drop shadows,
extra labels, gibberish or misspelled words, Chinese characters in the figure.

Output: 300 dpi, SVG or PDF plus PNG, aspect ratio about 2:1 (e.g. 3600 x 1800 px).
```

---

## 二、中文提示词（适用于即梦 / 可图 / 通义万相 / 豆包生图等中文模型）

```
画一张生物信息学方法论文级别的科学架构示意图（Nature 期刊风格，扁平矢量插画，横向宽幅，白底无阴影无渐变）。

主题：蛋白质功能预测框架 "HDMR-Sinkhorn" 的总览——三条并行路径抽取证据，经互惠秩融合与
Sinkhorn 最优传输决策层，输出二值标签矩阵。

布局（从左到右）：
1. 最左侧：圆角矩形 "Protein sequence (≤1,022 aa)"，三条细箭头分别指向右侧三个路径面板。
2. 三条竖向堆叠的路径面板（每个面板是浅色圆角框 + 深色标题条，各用一种颜色）：
   - 面板 I，标题 "Path I · Deep sequence perception"（蓝色 #0072B2）：内部三个小方块依次为
     "Multi-scale dilated CNN (k = 7/15/31)" → "CSRA class-specific attention" →
     "RoFormer encoder"，输出箭头标注 "P_deep"。
   - 面板 II，标题 "Path II · Homology & k-mer transfer"（绿色 #009E73）：两个并排小方块
     "DIAMOND alignment (E ≤ 0.001, top-3 bitscore weighted, 2-hop transfer)" 与
     "k-mer spectrum (k = 3–8)"，汇入 "Weighted label prior"，输出箭头标注 "S_sim"。
   - 面板 III，标题 "Path III · Pretrained language models"（橙色 #E69F00）：两个并排小方块
     "ESM-2 (640-d, frozen)" 与 "ProtBERT (1024-d, frozen)" → "Mean pooling + concat (1664-d)"
     → "Residual MLP × 3 seeds"，输出箭头标注 "S_plm"。
3. 中右：紫色（#9467BD）高框 "Reciprocal rank fusion"，接收三支箭头；内部两个小块
   "Linear injection of S_plm (w = 0.35)" 与 "RRF (k = 30, β = 0.50)"，输出箭头标注 "S_fused"。
4. 右侧：朱红色（#D55E00）框 "Sinkhorn-Knopp optimal transport (τ = 0.20, α = 0.60)"，
   其左缘有两条虚线约束箭头，来自两个灰色小标签 "row marginal = predicted cardinality r_i"、
   "column marginal = label frequency c_j"；再向右 "Per-label thresholding (SCut)"，
   最后是圆角框 "Binary label matrix (N × 500)"。
5. 底部一条细灰色注释条，小字："Training labels (500) · train-only DIAMOND database ·
   no external annotations"。

风格要求：扁平矢量、纯白背景、无阴影渐变三维；全部标签用无衬线字体（Helvetica/Arial）且水平书写；
配色用色盲安全的 Okabe-Ito 五色（蓝/绿/橙/紫/朱红），面板填充为对应色 12–18% 透明度，标题条与
箭头为全饱和色；实线 1.2pt 数据流箭头，虚线 1pt 约束注释箭头；图内英文标签必须逐字拼写正确，
不得出现中文、乱码或多余文字。

禁止：写实摄影、3D 渲染、卡通图标、DNA 双螺旋装饰、大脑/神经元俗套、网络节点图、水印、
渐变填充、投影、额外标签、乱码错字。

输出：300 dpi，宽高比约 2:1（如 3600×1800）。
```

---

## 三、简化提示词（英文短版，适用于 Midjourney / 对长提示词支持差的模型）

```
Flat vector scientific schematic, landscape, Nature Methods style, white background.
Left: "Protein sequence (<=1,022 aa)" feeding three parallel panels:
  blue "Path I · Deep sequence perception" (dilated CNN -> CSRA attention -> RoFormer -> P_deep),
  green "Path II · Homology & k-mer transfer" (DIAMOND top-3 + k-mer k=3-8 -> S_sim),
  orange "Path III · Pretrained language models" (ESM-2 640-d + ProtBERT 1024-d -> 1664-d residual MLP -> S_plm).
They converge into a purple "Reciprocal rank fusion (RRF k=30, linear injection w=0.35)" box (S_fused),
then a vermillion "Sinkhorn-Knopp optimal transport (tau=0.20, alpha=0.60)" box with dashed constraint tags
"row marginal = cardinality r_i" and "column marginal = label frequency c_j",
then "Per-label thresholding (SCut)" -> "Binary label matrix (N x 500)".
Okabe-Ito palette, Helvetica labels spelled exactly as given, thin solid arrows, no gradients, no 3D,
no shadows, no icons, no DNA helix, 300 dpi, aspect ratio 2:1.
--ar 2:1 --style raw --s 50 --v 6.1
```

---

## 四、负面提示（独立段，给需要 negative prompt 的模型如 Stable Diffusion）

```
photorealistic, 3d render, bevel, drop shadow, gradient, texture, cartoon, clipart,
icon set, dna helix, brain, neuron, network graph, watermark, signature,
misspelled text, gibberish, chinese characters, extra labels, cropped text
```

---

## 五、生成后替换步骤

1. 生成图裁剪到宽高比 ≈ 1.88（报告占位框 540 × 287 px 等比）→ 存 `doc/figures/fig2_architecture_flowchart.png`（或新文件名）。
2. 在 `tmp/build_official_report.py` 中删掉 `fig31-placeholder` 这条 `rep(...)`，恢复原 `imgPara` 调用。
3. 重建链：`python tmp/build_official_report.py` → `node doc/gen_official_report.js` → `python tmp/export_official_pdf.py`（自动更新目录域）。
4. 人工核对图内全部术语拼写与正文一致（P_deep / S_sim / S_plm / S_fused / RRF / Sinkhorn-Knopp / SCut / 1664-d）。
