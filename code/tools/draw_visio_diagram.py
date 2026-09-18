import os, sys, time
import win32com.client

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
OUT_DIR = os.path.join(PROJECT_ROOT, "doc", "visio")
os.makedirs(OUT_DIR, exist_ok=True)
VSDX_PATH = os.path.join(OUT_DIR, "HDMR_Sinkhorn_Architecture.vsdx")
PDF_PATH = os.path.join(OUT_DIR, "HDMR_Sinkhorn_Architecture.pdf")

# 颜色函数
def rgb(r, g, b):
    return f"RGB({r},{g},{b})"

C_NAVY_TITLE = rgb(27, 54, 93)     # #1B365D 深海军蓝
C_GOLD_TITLE = rgb(190, 130, 20)   # #D99B26 琥珀金
C_CORAL_TITLE = rgb(195, 74, 62)   # #C35A48 珊瑚赭红
C_DEEP_BLUE = rgb(15, 45, 90)      # 深蓝

C_STAGE1_BG = rgb(244, 248, 252)
C_STAGE1_BORDER = rgb(50, 100, 170)
C_STAGE2_BG = rgb(254, 252, 243)
C_STAGE2_BORDER = rgb(217, 155, 38)
C_STAGE3_BG = rgb(253, 246, 245)
C_STAGE3_BORDER = rgb(205, 90, 75)
C_STAGE4_BG = rgb(246, 248, 252)
C_STAGE4_BORDER = rgb(30, 60, 110)

C_WHITE = rgb(255, 255, 255)
C_DARK_TEXT = rgb(26, 32, 44)
C_SLATE = rgb(100, 110, 125)

print("Starting clean Visio generation...")
visio = win32com.client.Dispatch("Visio.Application")
visio.Visible = False

try:
    # 巧妙利用已存在的空文档，杜绝多余文档弹窗
    if visio.Documents.Count > 0:
        doc = visio.Documents.Item(1)
    else:
        doc = visio.Documents.AddEx("", 0, 0)
    
    page = doc.Pages.Item(1)
    page.Name = "HDMR-Sinkhorn Architecture"

    # 设置画布大小：宽 420mm (16.5英寸), 高 236mm (9.3英寸)
    ps = page.PageSheet
    ps.CellsU("PageWidth").FormulaU = "420 mm"
    ps.CellsU("PageHeight").FormulaU = "236 mm"
    ps.CellsU("DrawingScale").FormulaU = "1 mm"
    ps.CellsU("PageScale").FormulaU = "1 mm"

    # 坐标转换：Visio COM 图形绘制方法 (DrawRectangle / DrawLine) 接受以英寸为单位的浮点数
    def mm_to_in(val):
        return float(val) / 25.4

    def draw_box(x, y, w, h, fill, line, lw="1.5 pt", dash="0", text="", fsize="9 pt", fcolor=C_DARK_TEXT, bold=False):
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w/2, y + h/2
        s = page.DrawRectangle(mm_to_in(x1), mm_to_in(y1), mm_to_in(x2), mm_to_in(y2))
        s.CellsU("FillForegnd").FormulaU = fill
        s.CellsU("FillBkgnd").FormulaU = fill
        s.CellsU("FillPattern").FormulaU = "1"
        s.CellsU("LineColor").FormulaU = line
        s.CellsU("LineWeight").FormulaU = lw
        s.CellsU("LinePattern").FormulaU = dash
        s.CellsU("Rounding").FormulaU = "3.5 mm"
        if text:
            s.Text = text
            s.CellsU("Char.Size").FormulaU = fsize
            s.CellsU("Char.Color").FormulaU = fcolor
            if bold:
                s.CellsU("Char.Style").FormulaU = "1"
        return s

    def add_line(x1, y1, x2, y2, color=C_NAVY_TITLE, lw="1.5 pt"):
        l = page.DrawLine(mm_to_in(x1), mm_to_in(y1), mm_to_in(x2), mm_to_in(y2))
        l.CellsU("LineColor").FormulaU = color
        l.CellsU("LineWeight").FormulaU = lw
        l.CellsU("EndArrow").FormulaU = "13"
        l.CellsU("EndArrowSize").FormulaU = "3"
        return l

    top_y = 115
    # 4 个主阶段容器 (Stage 1..4)
    draw_box(68, top_y, 115, 175, C_STAGE1_BG, C_STAGE1_BORDER, "1.8 pt", "2")
    draw_box(179, top_y, 94, 175, C_STAGE2_BG, C_STAGE2_BORDER, "1.8 pt", "2")
    draw_box(287, top_y, 110, 175, C_STAGE3_BG, C_STAGE3_BORDER, "1.8 pt", "2")
    draw_box(378, top_y, 60, 175, C_STAGE4_BG, C_STAGE4_BORDER, "1.8 pt", "2")

    # 主阶段标题
    def add_title(x, y, text, col):
        t = page.DrawRectangle(mm_to_in(x-50), mm_to_in(y-6), mm_to_in(x+50), mm_to_in(y+6))
        t.CellsU("LinePattern").FormulaU = "0"
        t.CellsU("FillPattern").FormulaU = "0"
        t.Text = text
        t.CellsU("Char.Size").FormulaU = "12 pt"
        t.CellsU("Char.Style").FormulaU = "1"
        t.CellsU("Char.Color").FormulaU = col
        t.CellsU("Para.HorzAlign").FormulaU = "1"

    add_title(68, 194, "Stage 1: Input & Decoupled Representation", C_NAVY_TITLE)
    add_title(179, 194, "Stage 2: Ordinal Alignment via Rank Fusion", C_GOLD_TITLE)
    add_title(287, 194, "Stage 3: Dual-Marginal Optimal Transport Regularization", C_CORAL_TITLE)
    add_title(378, 194, "Stage 4: Calibrated Multi-Label Decision", C_DEEP_BLUE)

    # --- Stage 1 内部细节 ---
    draw_box(68, 172, 105, 18, C_WHITE, C_NAVY_TITLE, "1 pt", "0",
             "Primary Protein Sequence (L = 20-1024 aa)\nN-terminus  ●-●-●-●-●-●-●-●-●-●-●-●  C-terminus", "8.5 pt", C_NAVY_TITLE, True)
    add_line(68, 163, 68, 151, C_NAVY_TITLE)

    # 深度模型支路
    draw_box(68, 118, 105, 60, C_WHITE, C_NAVY_TITLE, "1.2 pt", "0")
    t1 = page.DrawRectangle(mm_to_in(18), mm_to_in(143), mm_to_in(118), mm_to_in(149))
    t1.CellsU("LinePattern").FormulaU = "0"
    t1.CellsU("FillPattern").FormulaU = "0"
    t1.Text = "Deep Sequence Network (Context & Motif Modeling)"
    t1.CellsU("Char.Size").FormulaU = "9 pt"
    t1.CellsU("Char.Style").FormulaU = "1"
    t1.CellsU("Char.Color").FormulaU = C_NAVY_TITLE

    draw_box(36, 122, 30, 30, C_STAGE1_BG, C_NAVY_TITLE, "0.8 pt", "0",
             "1D Dilated Residual\nConvolutions\nd=1, k=7\nd=2, k=15\nd=4, k=31", "7 pt")
    draw_box(68, 122, 30, 30, C_STAGE1_BG, C_NAVY_TITLE, "0.8 pt", "0",
             "Class-Specific Residual\nAttention (CSRA)\na_ij = exp(T w_j f_i) / Σ\n(Catalytic Residue Focus)", "7 pt")
    draw_box(100, 122, 30, 30, C_STAGE1_BG, C_NAVY_TITLE, "0.8 pt", "0",
             "RoFormer Position\nEmbedding\n+θ_i, +θ_j\nRelative Encoding", "7 pt")

    draw_box(68, 93, 72, 8, C_NAVY_TITLE, C_NAVY_TITLE, "1 pt", "0",
             "S_deep ∈ R^D (Context & Motif Output)", "8 pt", C_WHITE, True)

    # 进化同源支路
    draw_box(68, 56, 105, 58, C_WHITE, rgb(0, 128, 128), "1.2 pt", "0")
    t2 = page.DrawRectangle(mm_to_in(18), mm_to_in(80), mm_to_in(118), mm_to_in(86))
    t2.CellsU("LinePattern").FormulaU = "0"
    t2.CellsU("FillPattern").FormulaU = "0"
    t2.Text = "Evolutionary Manifold Transfer (Homology & Composition)"
    t2.CellsU("Char.Size").FormulaU = "9 pt"
    t2.CellsU("Char.Style").FormulaU = "1"
    t2.CellsU("Char.Color").FormulaU = rgb(0, 128, 128)

    draw_box(44, 63, 44, 24, C_STAGE2_BG, rgb(0, 128, 128), "0.8 pt", "0",
             "DIAMOND Alignment\n(very-sensitive, E<=0.001)\nBitscore-Weighted Neighbors\n[b_1, b_2, ..., b_M]", "7 pt")
    draw_box(92, 63, 44, 24, C_STAGE2_BG, rgb(0, 128, 128), "0.8 pt", "0",
             "Multi-scale k-mer Manifold\nk ∈ [4..8] Frequency Profiles\nCosine Distance Topological\nNearest Neighbors", "7 pt")

    draw_box(68, 33, 72, 8, rgb(0, 128, 128), rgb(0, 128, 128), "1 pt", "0",
             "S_sim = 0.4 S_bit + 0.6 S_kmer ∈ R^D", "8 pt", C_WHITE, True)

    # --- Stage 2 内部细节 (RRF) ---
    add_line(126, 93, 138, 115, C_NAVY_TITLE)
    add_line(126, 33, 138, 105, rgb(0, 128, 128))

    draw_box(179, 172, 84, 16, C_WHITE, C_GOLD_TITLE, "1 pt", "0",
             "Stage 1: Reciprocal Rank Fusion (RRF)", "9 pt", C_GOLD_TITLE, True)

    draw_box(157, 134, 34, 45, C_WHITE, C_GOLD_TITLE, "0.8 pt", "0",
             "Rank by S_deep\n(Deep Model)\n\n█  Rank 1\n▆  Rank 250\n▃  Rank 500", "7.5 pt", C_NAVY_TITLE)
    draw_box(201, 134, 34, 45, C_WHITE, C_GOLD_TITLE, "0.8 pt", "0",
             "Rank by S_sim\n(Homology Model)\n\n█  Rank 1\n▆  Rank 250\n▃  Rank 500", "7.5 pt", C_GOLD_TITLE)

    add_line(174, 140, 184, 132, C_GOLD_TITLE, "1 pt")
    add_line(174, 128, 184, 144, C_NAVY_TITLE, "1 pt")

    draw_box(179, 88, 84, 26, C_WHITE, C_GOLD_TITLE, "1 pt", "0",
             "RRF(S_dp, S_sm; k) = \n1 / [ k + rank(S_dp) + 1 ] + \n1 / [ k + rank(S_sm) + 1 ],  k=30", "7.5 pt", C_DARK_TEXT, True)

    draw_box(179, 58, 84, 16, C_WHITE, C_GOLD_TITLE, "0.8 pt", "0",
             "Column Min-Max Normalization\nS_rrf = S_deep + w S_sim + β RRF", "7.5 pt")

    draw_box(179, 36, 60, 10, C_GOLD_TITLE, C_GOLD_TITLE, "1 pt", "0",
             "S_rrf ∈ R^500", "8.5 pt", C_WHITE, True)

    # --- Stage 3 内部细节 (Sinkhorn) ---
    add_line(226, 36, 238, 105, C_CORAL_TITLE)

    draw_box(287, 172, 98, 16, C_WHITE, C_CORAL_TITLE, "1 pt", "0",
             "Stage 2: Sinkhorn-Knopp Regularization", "9 pt", C_CORAL_TITLE, True)

    draw_box(287, 130, 96, 52, C_WHITE, C_CORAL_TITLE, "0.8 pt", "0",
             "Optimal Transport Plan  P ∈ R^(N × 500)\n\n"
             "Row Marginal Prior r_i (Instance Cardinality) → [ P ] ← Column Marginal Prior c_j (Global Frequency)\n"
             "Σ_j P_ij = r_i                                          Σ_i P_ij = c_j\n"
             "Heatmap Mass: [Low ░░▒▒▓▓██ High]", "7 pt")

    draw_box(287, 85, 96, 25, C_WHITE, C_CORAL_TITLE, "0.8 pt", "0",
             "Gibbs Kernel: K = exp( S_rrf / τ )\n"
             "u^(t+1) = r / (K v^(t))   ⇄   v^(t+1) = c / (K^T u^(t+1))\n"
             "Iterative Sinkhorn Projections (until convergence)", "7.5 pt", C_CORAL_TITLE, True)

    draw_box(287, 50, 96, 22, C_WHITE, C_CORAL_TITLE, "1 pt", "0",
             "min_P  ⟨ P, -S_rrf ⟩ + τ Σ_ij P_ij log(P_ij)\n"
             "s.t.   P 1 = r,   P^T 1 = c", "8 pt", C_DARK_TEXT, True)

    # --- Stage 4 内部细节 (Decision) ---
    add_line(342, 50, 350, 105, C_DEEP_BLUE)

    draw_box(378, 172, 52, 16, C_WHITE, C_DEEP_BLUE, "1 pt", "0",
             "Per-Label Adaptive\nSCut Thresholds t_j*", "8 pt", C_DEEP_BLUE, True)

    add_line(378, 162, 378, 150, C_DEEP_BLUE)

    draw_box(378, 108, 52, 65, C_WHITE, C_DEEP_BLUE, "0.8 pt", "0",
             "Final Prediction Matrix\ny ∈ {0, 1}^500\n\n"
             "[ ■ 1 (Active) ]\n"
             "[ □ 0 (Inactive)]\n"
             "[ ■ 1 (Active) ]\n"
             "[ ■ 1 (Active) ]\n"
             "[ □ 0 (Inactive)]\n"
             "500 Functional Categories", "7.5 pt")

    draw_box(378, 48, 52, 28, C_DEEP_BLUE, C_DEEP_BLUE, "1 pt", "0",
             "S_final = (1 - α) S_rrf + α P_sk\n\n"
             "y_j = I[ S_final,j ≥ t_j* ]\n(Calibrated Output)", "7.5 pt", C_WHITE, True)

    # 底部流水线总括
    draw_box(210, 14, 380, 12, C_WHITE, C_SLATE, "0.8 pt", "0",
             "Decoupled Feature Extraction → Reciprocal Rank Fusion (Ordinal Alignment) → Optimal Transport (Marginal Regularization) → Adaptive Decision",
             "8 pt", C_SLATE, True)

    SVG_PATH = os.path.join(OUT_DIR, "HDMR_Sinkhorn_Architecture.svg")
    PNG_PATH = os.path.join(OUT_DIR, "HDMR_Sinkhorn_Architecture.png")

    # 保存文件
    print(f"Saving Visio drawing to {VSDX_PATH}...")
    doc.SaveAs(VSDX_PATH)
    
    print(f"Exporting Vector SVG to {SVG_PATH}...")
    page.Export(SVG_PATH)

    print(f"Exporting PNG to {PNG_PATH}...")
    page.Export(PNG_PATH)
    
    # 彻底杜绝弹窗的关键两行
    doc.Saved = True
    doc.Close()
    print("Document successfully saved and closed.")

finally:
    visio.Quit()
    print("Visio exited cleanly without hanging.")