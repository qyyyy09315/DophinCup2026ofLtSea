# -*- coding: utf-8 -*-
"""
gen_paper_figures_v2 — 论文新增图 (Okabe-Ito 色盲安全配色, Arial, Nature 规范)
==============================================================================
fig2  架构总览重绘 (三路: 深度/同源/PLM, 含 2-hop 与线性注入)
fig7  排序头部诊断 (a: 分支平均 ROC 曲线 FPR<=0.1; b: R@FPR0.069 条形)
fig8  「独立增益不传导」设计法则散点 (I063/I074/I087)
fig9  融合算子与同源算子消融 (a/b 两面板水平条形)
fig10 频带 x 分支 AP 热图 (I075)
fig11 自训练变体与真值上限 (I086)
数据: doc/figures/fig_data_v2.json + rel/I075/I086/I088/I089_report.json
"""
import os, io, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- Nature 规范 + Okabe-Ito (Wong 2011) ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.linewidth": 0.9, "xtick.direction": "out",
    "ytick.direction": "out", "lines.linewidth": 1.6, "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
BLUE, GREEN, VERM, ORANGE, SKY, PINK, GRAY = (
    "#0072B2", "#009E73", "#D55E00", "#E69F00", "#56B4E9", "#CC79A7", "#6E6E6E")
DARK = "#262626"
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "doc", "figures")
ROOT = os.path.dirname(os.path.dirname(FIG_DIR))


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  saved {name} (.svg/.png)")


def panel_label(ax, s, dx=-0.02, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom", color=DARK)


data = json.load(io.open(os.path.join(FIG_DIR, "fig_data_v2.json"), encoding="utf-8"))

# ================= FIG 2: 架构总览 (三路) =================
print("Figure 2: architecture (3-path, updated)...")
fig = plt.figure(figsize=(12.8, 6.8), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
LIGHT = {"b": "#EAF2F9", "g": "#E7F4EF", "p": "#F4EAF4", "y": "#FBF3E2", "r": "#FBEDE4"}

rect_in = patches.FancyBboxPatch((1.5, 36), 11, 28, boxstyle="round,pad=0.8",
                                 ec=GRAY, fc="#FFFFFF", lw=1.6)
ax.add_patch(rect_in)
ax.text(7, 54, "Input\nProtein\nSequence", ha="center", va="center", fontsize=10,
        fontweight="bold", color=DARK)
ax.text(7, 43, "Primary Residues\n$L\\in[20,1024]$", ha="center", va="center",
        fontsize=8, color=GRAY)

# 三条泳道
lanes = [
    ("b", 66, "Branch I: Multi-Scale Deep Sequence Network",
     [("RoFormer", "Rotary Position\nSelf-Attention"), ("Dual 1D-CNN", "Dilated Residual\n+ CSRA Attention")],
     "Deep Prior $S_{deep}$\n($w_1P_{T1}+w_2P_{T2}+w_cP_{cnn}$)"),
    ("g", 36, "Branch II: Homology & Manifold Transfer",
     [("DIAMOND", "Very-sensitive Align\ntop-3 Bitscore + 2-hop"), ("Multi-k-mer", "$k\\in\\{4..8\\}$ Profile\nsim$^2$ Weighting")],
     "Homology Prior $S_{sim}$\n($0.4S_{bit}+0.6S_{kmer}$)"),
    ("p", 6, "Branch III: Pretrained Language Model",
     [("ESM2 + ProtBERT", "Frozen Encoder\n640d + 1024d"), ("Residual MLP", "1664d $\\rightarrow$ 500\n$\\times$ 3 Seeds")],
     "PLM Prior $S_{plm}$\n(seed-averaged)"),
]
for color_key, y0, title, mods, agg_text in lanes:
    c = {"b": BLUE, "g": GREEN, "p": PINK}[color_key]
    rect = patches.FancyBboxPatch((17, y0), 36, 26, boxstyle="round,pad=1.0",
                                  ec=c, fc=LIGHT[color_key], lw=1.6, ls="--")
    ax.add_patch(rect)
    ax.text(35, y0 + 23, title, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=c)
    for k, (name, sub) in enumerate(mods):
        x0 = 19 + k * 17
        box = patches.FancyBboxPatch((x0, y0 + 6), 15, 13, boxstyle="round,pad=0.4",
                                     ec=c, fc="#FFFFFF", lw=1.2)
        ax.add_patch(box)
        ax.text(x0 + 7.5, y0 + 15, name, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=DARK)
        ax.text(x0 + 7.5, y0 + 10, sub, ha="center", va="center", fontsize=7, color=GRAY)
    agg_box = patches.FancyBboxPatch((21, y0 + 1), 28, 4.6, boxstyle="round,pad=0.3",
                                     ec=c, fc="#FFFFFF", lw=1.2)
    ax.add_patch(agg_box)
    ax.text(35, y0 + 3.3, agg_text, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=DARK)
    ax.annotate("", xy=(17, y0 + 13), xytext=(12.8, 50),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=c))

# PLM 线性注入节点
inj = patches.FancyBboxPatch((56, 9), 15, 9, boxstyle="round,pad=0.5",
                             ec=PINK, fc=LIGHT["p"], lw=1.5)
ax.add_patch(inj)
ax.text(63.5, 15, "Linear Injection", ha="center", va="center", fontsize=8.5,
        fontweight="bold", color=DARK)
ax.text(63.5, 11.5, "$S_{deep'}=(1-w)S_{deep}+wS_{plm}$\n$w=0.35$", ha="center",
        va="center", fontsize=7, color=GRAY)
ax.annotate("", xy=(56, 13.5), xytext=(53, 13.5),
            arrowprops=dict(arrowstyle="->", lw=1.6, color=PINK))
ax.annotate("", xy=(63.5, 55), xytext=(63.5, 18.5),
            arrowprops=dict(arrowstyle="->", lw=1.6, color=PINK))

# Stage 1 RRF
rect_rrf = patches.FancyBboxPatch((57, 32), 17, 22, boxstyle="round,pad=0.8",
                                  ec=ORANGE, fc=LIGHT["y"], lw=1.8)
ax.add_patch(rect_rrf)
ax.text(65.5, 50, "Stage 1: Fusion", ha="center", va="center", fontsize=9,
        fontweight="bold", color="#B37700")
ax.text(65.5, 42, "Reciprocal Rank\nFusion (RRF)\n$k=30,\\ \\beta=0.50$", ha="center",
        va="center", fontsize=8.5, fontweight="bold", color=DARK)
ax.text(65.5, 35.5, "$S_{rrf}=S_{deep'}+0.75S_{sim}$\n$+\\beta\\cdot\\mathrm{ColNorm}(\\mathrm{RRF})$",
        ha="center", va="center", fontsize=7, color=GRAY)
ax.annotate("", xy=(57, 52), xytext=(52, 79), arrowprops=dict(arrowstyle="->", lw=1.8, color=BLUE))
ax.annotate("", xy=(57, 46), xytext=(53, 47), arrowprops=dict(arrowstyle="->", lw=1.8, color=GREEN))

# Stage 2 Sinkhorn
rect_sk = patches.FancyBboxPatch((78, 28), 20, 30, boxstyle="round,pad=0.8",
                                 ec=VERM, fc=LIGHT["r"], lw=1.8)
ax.add_patch(rect_sk)
ax.text(88, 54, "Stage 2: Regularization", ha="center", va="center", fontsize=9,
        fontweight="bold", color=VERM)
ax.text(88, 47, "Dual-Marginal\nOptimal Transport", ha="center", va="center",
        fontsize=9, fontweight="bold", color=DARK)
ax.text(88, 38.5, "Sinkhorn-Knopp\nRow: Instance Cardinality\nCol: Label Prior Frequencies",
        ha="center", va="center", fontsize=7, color=GRAY)
ax.text(88, 31.5, "$S_{final}=0.4S_{rrf}+0.6P_{sk}$\nDecision: SCut Thresholds",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color=VERM)
ax.annotate("", xy=(78, 44), xytext=(74, 44), arrowprops=dict(arrowstyle="->", lw=1.8, color=VERM))

plt.tight_layout()
save(fig, "fig2_architecture_flowchart")

# ================= FIG 7: 排序头部诊断 =================
print("Figure 7: head-of-ranking diagnosis...")
roc = data["roc"]
labels = {"plm1664": "PLM-MLP (1664D)", "deep5": "Deep-5 path", "sim": "Homology (Sim)", "fused": "HDMR-Sinkhorn (final)"}
colors = {"plm1664": ORANGE, "deep5": BLUE, "sim": GREEN, "fused": VERM}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
for key in ["sim", "deep5", "plm1664", "fused"]:
    r = roc[key]
    fpr = np.array(r["fpr"]); tpr = np.array(r["tpr_mean"])
    m = fpr <= 0.10
    lw = 2.2 if key == "fused" else 1.5
    ax.plot(fpr[m] * 100, tpr[m], color=colors[key], lw=lw,
            label=f"{labels[key]} (pAUC={r['pauc01']:.3f})")
ax.plot([0, 10], [0, 10], color=GRAY, lw=0.8, ls=":")
ax.set_xlabel("False positive rate (%)")
ax.set_ylabel("Mean true positive rate")
ax.set_xlim(0, 10); ax.set_ylim(0, 0.85)
ax.legend(loc="upper left", handlelength=1.6)
panel_label(ax, "a")
ax = axes[1]
keys = ["plm1664", "deep5", "sim", "fused"]
vals = [roc[k]["r@0.069"] for k in keys]
bars = ax.bar(np.arange(4), vals, width=0.62,
              color=[colors[k] for k in keys], edgecolor="white", linewidth=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5, color=DARK)
ax.axhline(0.82, color=GRAY, lw=1.0, ls="--")
ax.text(3.45, 0.828, "leaderboard top-1\n(estimated)", fontsize=6.5, color=GRAY, ha="right", va="bottom")
ax.set_xticks(np.arange(4))
ax.set_xticklabels(["PLM\n1664D", "Deep-5", "Homology\n(Sim)", "Fused"], fontsize=8)
ax.set_ylabel("Recall at FPR = 6.9%")
ax.set_ylim(0, 0.92)
panel_label(ax, "b")
fig.tight_layout(w_pad=2.0)
save(fig, "fig7_head_diagnosis")

# ================= FIG 8: 设计法则散点 =================
print("Figure 8: standalone-vs-fused design law...")
law = data["law"]
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.axhspan(-0.003, 0.0008, xmin=0, xmax=1, color="#F2F2F2", zorder=0)
ax.axhline(0, color=GRAY, lw=0.8)
ax.axvline(0, color=GRAY, lw=0.8)
ax.text(0.0125, 0.00045, "standalone up,\nfused down", fontsize=7.5, color=VERM,
        ha="center", style="italic")
pts = {
    "I063 feature expansion": ("I063_2176D特征扩容", SKY, "o"),
    "I074 propensity sweep": ("I074_a0.5", ORANGE, "s"),
    "I087 self-training": ("I087_自训练", GREEN, "^"),
}
for name, (key, c, mk) in pts.items():
    d = law[key]
    ax.scatter(d["d_standalone"], d["d_fused"], s=42, color=c, marker=mk,
               zorder=3, edgecolor="white", linewidth=0.6, label=name)
path = np.array([[law["I074_a0.5"]["d_standalone"], law["I074_a0.5"]["d_fused"]],
                 [law["I074_a1.0_w3.33"]["d_standalone"], law["I074_a1.0_w3.33"]["d_fused"]],
                 [law["I074_a1.0_w6.0"]["d_standalone"], law["I074_a1.0_w6.0"]["d_fused"]]])
ax.plot(path[:, 0], path[:, 1], color=ORANGE, lw=1.0, ls="--", zorder=2)
ax.annotate("stronger\nstandalone", xy=(0.0125, 0.0145), fontsize=7, color=GRAY, ha="center")
ax.annotate("worse\nfused", xy=(0.0012, -0.0021), fontsize=7, color=GRAY, ha="center")
ax.set_xlabel("Standalone Macro-F1 gain (pp)")
ax.set_ylabel("Fused Macro-F1 gain (pp)")
ax.set_xlim(-0.002, 0.032); ax.set_ylim(-0.0026, 0.0016)
ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.1f}")
ax.yaxis.set_major_formatter(lambda y, _: f"{y * 100:.2f}")
ax.legend(loc="upper right", fontsize=6.2, handlelength=1.2, borderpad=0.4)
fig.tight_layout()
save(fig, "fig8_design_law")

# ================= FIG 9: 算子消融 =================
print("Figure 9: operator ablation (fusion & homology)...")
i088 = json.load(io.open(os.path.join(ROOT, "rel", "I088_report.json"), encoding="utf-8"))
i089 = json.load(io.open(os.path.join(ROOT, "rel", "I089_report.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
items = [("A0 current (RRF+Sinkhorn)", "A0 现状(RRF+Sinkhorn)"),
         ("A1 w/o RRF", "A1 去RRF(+Sinkhorn)"),
         ("A2 w/o Sinkhorn", "A2 去Sinkhorn(RRF)"),
         ("A3 equal-weight mean", "A3 等权算术平均"),
         ("A4 equal-weight + Sinkhorn", "A4 等权+Sinkhorn"),
         ("A5 max combination", "A5 max组合(col_norm)")]
ypos = np.arange(len(items))[::-1]
for yp, (en, cn) in zip(ypos, items):
    v = i088[cn]["f1"]
    c = VERM if cn.startswith("A0") else (GRAY if v < 0.4460 else BLUE)
    ax.barh(yp, v, height=0.62, color=c, edgecolor="white", linewidth=0.5)
    ax.text(v + 0.0004, yp, f"{v:.4f}", va="center", fontsize=7, color=DARK)
ax.axvline(0.4475, color=DARK, lw=0.8, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels([e for e, _ in items], fontsize=7.5)
ax.set_xlabel("Validation Macro-F1"); ax.set_xlim(0.400, 0.452)
panel_label(ax, "a")
ax = axes[1]
items_b = [("B0 current (bitscore top-3)", "B0_bitscore_top3(现状)"),
           ("B1 max sequence identity", "B1_max_pident"),
           ("B2 max bitscore", "B2_max_bitscore"),
           ("B3 dedup voting", "B3_dedup_vote")]
ypos = np.arange(len(items_b))[::-1]
for yp, (en, cn) in zip(ypos, items_b):
    v = i089["results"][cn]["fused_f1"]
    c = GREEN if cn.startswith("B0") else GRAY
    ax.barh(yp, v, height=0.55, color=c, edgecolor="white", linewidth=0.5)
    ax.text(v + 0.0004, yp, f"{v:.4f}", va="center", fontsize=7, color=DARK)
ax.axvline(0.4475, color=DARK, lw=0.8, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels([e for e, _ in items_b], fontsize=7.5)
ax.set_xlabel("Fused Macro-F1"); ax.set_xlim(0.415, 0.452)
panel_label(ax, "b")
fig.tight_layout(w_pad=2.4)
save(fig, "fig9_operator_ablation")

# ================= FIG 10: 频带 x 分支 AP 热图 =================
print("Figure 10: frequency-band AP heatmap...")
i075 = json.load(io.open(os.path.join(ROOT, "rel", "I075_report.json"), encoding="utf-8"))
band = i075["section2_band_ap"]
rows = [("cnn(C002b)", "CNN (C002b)"), ("c005(CSRA)", "CSRA (C005)"),
        ("t001a", "RoFormer (T001a)"), ("plm_asl", "PLM-MLP"),
        ("deep4", "Deep-4 ensemble"), ("sim", "Homology (Sim)")]
cols = ["极稀有 <Q1", "稀有 Q1-Q2", "中等 Q2-Q3", "高频 >Q3"]
col_en = ["Tail <Q1", "Q1-Q2", "Q2-Q3", "Head >Q3"]
M = np.array([[band[rk][c] for c in cols] for rk, _ in rows])
fig, ax = plt.subplots(figsize=(3.9, 3.0))
im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0.09, vmax=0.50)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(col_en, fontsize=8)
ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels([r for _, r in rows], fontsize=8)
for i in range(len(rows)):
    for j in range(4):
        best = j == int(np.argmax(M[i]))
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                color="white" if M[i, j] > 0.36 else DARK,
                fontweight="bold" if best else "normal")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
cbar.set_label("Average precision", fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.set_xlabel("Training-frequency quartile of labels (125 labels per band)")
fig.tight_layout()
save(fig, "fig10_band_ap_heatmap")

# ================= FIG 11: 自训练变体 =================
print("Figure 11: guarded self-training variants...")
i086 = json.load(io.open(os.path.join(ROOT, "rel", "I086_report.json"), encoding="utf-8"))
res = i086["results"]
items = [("S0 baseline\n(train only)", res["S0_baseline"], GRAY),
         ("S1 hard\n$\\lambda$=1.0", res["S1 factor0.5 硬 λ=1.0"], GREEN),
         ("S2 hard\nfactor=1.0", res["S2 factor1.0 硬 λ=1.0"], GREEN),
         ("S3 soft\n$\\lambda$=0.5", res["S3 factor0.5 软 λ=0.5"], GREEN),
         ("S4 hard\n$\\lambda$=0.3", res["S4 factor0.5 硬 λ=0.3"], GREEN)]
fig, ax = plt.subplots(figsize=(3.8, 2.9))
x = np.arange(len(items))
vals = [v for _, v, _ in items]
bars = ax.bar(x, vals, width=0.62, color=[c for _, _, c in items],
              edgecolor="white", linewidth=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
            ha="center", va="bottom", fontsize=7, color=DARK)
ax.axhline(res["S5_true_label_ceiling"], color=VERM, lw=1.1, ls="--")
ax.text(4.35, res["S5_true_label_ceiling"] + 0.002,
        f"true-label ceiling {res['S5_true_label_ceiling']:.4f}",
        fontsize=6.8, color=VERM, ha="right")
ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in items], fontsize=7)
ax.set_ylabel("Standalone Macro-F1"); ax.set_ylim(0.24, 0.385)
fig.tight_layout()
save(fig, "fig11_selftrain")

print("\nAll v2 figures done.")
