# 图 3.1（HDMR-Sinkhorn 架构原理图）AI 绘图提示词

> 用途：图 3.1 当前为占位框（见报告正文）。将下述提示词交给图像生成模型（如 Nano Banana / GPT-Image / Midjourney 等）或作为给绘图设计师的规格说明，生成后替换 `doc/gen_paper_hdmr.js` 中 `fig2_architecture_flowchart.png` 的占位调用并重建。
> 生成后必须人工校对图中**每一个**英文标签的拼写与术语（AI 生图极易拼错术语），并确认与正文术语一致：S_deep / S_sim / S_plm / RRF / Sinkhorn / SCut。

---

## Master prompt (English)

Create a publication-quality scientific schematic diagram for a bioinformatics methods paper (Nature/Science standard, vector style, flat design).

**Topic**: Overview of "HDMR-Sinkhorn", a three-path protein function prediction framework that fuses (I) a deep sequence-aware network, (II) sequence-similarity evidence from homology alignment and k-mer statistics, and (III) frozen pretrained language model embeddings, followed by reciprocal rank fusion and an optimal-transport (Sinkhorn) decision layer.

**Layout** (landscape, left → right, three parallel columns converging into a fusion stage):

1. **Left column — input**: a single rounded rectangle labelled `Protein sequence (≤1,022 aa)` at the far left, from which three arrows branch to the three paths.
2. **Three parallel paths** (vertically stacked panels, each a light tinted box with a bold coloured header):
   - Panel I — header `Path I · Deep sequence perception` (blue `#0072B2`): inside, three stacked small blocks `Multi-scale dilated CNN (k = 7/15/31)` → `CSRA class-specific attention` → `RoFormer encoder`; output arrow labelled `P_deep (500 labels)`.
   - Panel II — header `Path II · Homology & k-mer transfer` (green `#009E73`): two sub-blocks `DIAMOND alignment (E ≤ 0.001, top-5 → top-3 weighted, 2-hop transfer)` and `k-mer spectrum (k = 3–8)` converging into `Weighted label prior S_sim`.
   - Panel III — header `Path III · Pretrained language models` (orange `#E69F00`): two sub-blocks `ESM-2 (640-d, frozen)` and `ProtBERT (1024-d, frozen)` → `Mean pooling + concat (1664-d)` → `Residual MLP (3 seeds)`; output arrow labelled `S_plm`.
3. **Fusion stage** (centre-right, a single tall box, purple `#9467BD`): three incoming arrows merge; inside, two stacked sub-blocks: `Reciprocal Rank Fusion (k = 30)` and `Linear injection of S_plm (w = 0.35)`; output labelled `S_fused`.
4. **Decision stage** (right, a box, vermillion `#D55E00`): `Sinkhorn-Knopp optimal transport (τ = 0.20, α = 0.60)` with two small annotation tags on its left edge: `row marginal = card(r_i)` and `column marginal = label frequency(c_j)`; then → `Per-label thresholding (SCut)` → final rounded box `Binary label matrix (N × 500)`.
5. Under the three path panels, add a thin grey annotation strip: `Training labels (500) · train-only database · no external annotations`.

**Style requirements**
- Flat vector look, white background, no shadows, no gradients, no 3-D bevels, no textures, no clip-art.
- Sans-serif typeface (Helvetica/Arial) for every label; all text horizontal; minimum font size equivalent to 7 pt at final print size.
- Colour-blind-safe palette (Okabe–Ito): one hue per path as specified above; boxes filled with the hue at 12–18% opacity, headers and arrowheads at full saturation; greys `#666666` for secondary annotations.
- Arrows: solid 1.2 pt for the main data flow, dashed 1 pt for the marginal-constraint annotations; arrowheads small and open.
- Every box label must be spelled exactly as given (these are method names and hyper-parameters quoted from the paper).
- Canvas ~180 mm × 90 mm (double-column width), 300 dpi, export as SVG/PDF + PNG.

**Negative prompt**: no photorealism, no 3D render, no cartoon icons, no DNA double-helix decoration, no brain/neuron clichés, no watermark, no gradients, no drop shadows, no extra text or gibberish labels, no misspelled words.

---

## 备用简化提示词（若生成模型对长提示词支持差）

Flat vector scientific schematic, landscape: three parallel processing paths (blue "Deep sequence perception" CNN+attention; green "Homology & k-mer transfer" DIAMOND+k-mer; orange "Pretrained language models" ESM-2+ProtBERT embeddings) converging into a purple fusion box "Reciprocal Rank Fusion + linear injection", then a vermillion decision box "Sinkhorn optimal transport → per-label thresholding", output "N × 500 binary labels". White background, Okabe-Ito colour-blind-safe palette, Helvetica labels, thin solid arrows, no gradients, no 3D, no shadows, publication quality, Nature journal style.

---

## 生成后的替换步骤

1. 生成图 → 裁剪为 540×287 px（或保持宽高比 ≈ 1.88）→ 存为 `doc/figures/fig2_architecture_flowchart.png`（或新文件名）。
2. 在 `tmp/build_official_report.py` 中删除 `fig31-placeholder` 这条 `rep(...)`（恢复原 `imgPara` 调用）。
3. 重建：`python tmp/build_official_report.py` → `node doc/gen_official_report.js` → `python tmp/finalize_official_report.py`。
4. 人工核对图中术语拼写与正文一致性后，更新目录域并交付。
