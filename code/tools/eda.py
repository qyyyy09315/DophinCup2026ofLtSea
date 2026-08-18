"""蛋白质功能预测 — 数据探索性分析 (EDA)"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── 全局设置 ──────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
OUT_DIR = 'rel'

print("=" * 60)
print("1. 加载数据")
print("=" * 60)

t0 = time.time()
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"train: {train.shape}  |  耗时 {time.time() - t0:.1f}s")
print(f"test:  {test.shape}")

label_cols = [c for c in train.columns if c.startswith('label_')]
n_labels = len(label_cols)
print(f"标签数: {n_labels}")


# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. 序列长度分析")
print("=" * 60)

train_len = train['sequence'].str.len()
test_len = test['sequence'].str.len()

def seq_len_stats(s, name):
    stats = {
        'count': len(s), 'min': s.min(), 'max': s.max(),
        'mean': s.mean(), 'median': s.median(), 'std': s.std(),
        'p5': s.quantile(0.05), 'p95': s.quantile(0.95),
    }
    print(f"\n--- {name} ---")
    for k, v in stats.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    return stats

train_len_stats = seq_len_stats(train_len, '训练集序列长度')
test_len_stats = seq_len_stats(test_len, '测试集序列长度')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 直方图 (限制 x 轴到 p99 以内，避免长尾拉伸)
xlim_max = int(np.percentile(train_len, 99))
axes[0].hist(train_len.clip(upper=xlim_max), bins=80, alpha=0.6, label='Train', color='#2c7bb6', edgecolor='white')
axes[0].hist(test_len.clip(upper=xlim_max), bins=80, alpha=0.6, label='Test', color='#d7191c', edgecolor='white')
axes[0].axvline(train_len.median(), color='#2c7bb6', ls='--', lw=1.5)
axes[0].axvline(test_len.median(), color='#d7191c', ls='--', lw=1.5)
axes[0].set_xlabel('Sequence Length (aa)')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Sequence Length Distribution (≤p99={xlim_max})')
axes[0].legend()

# Boxplot
bp = axes[1].boxplot([train_len, test_len], tick_labels=['Train', 'Test'], patch_artist=True,
                       boxprops=dict(facecolor='#abd9e9'), medianprops=dict(color='#d7191c'))
axes[1].set_ylabel('Sequence Length (aa)')
axes[1].set_title('Length Boxplot')

# 分桶 (根据实际数据分布调整)
bins_edges = [0, 200, 400, 800, 1600, 40000]
bins_labels = ['≤200', '200-400', '400-800', '800-1600', '>1600']
train_binned = pd.cut(train_len, bins=bins_edges, labels=bins_labels).value_counts().sort_index()
test_binned = pd.cut(test_len, bins=bins_edges, labels=bins_labels).value_counts().sort_index()
x = np.arange(len(bins_labels))
w = 0.35
axes[2].bar(x - w/2, train_binned.values, w, label='Train', color='#2c7bb6')
axes[2].bar(x + w/2, test_binned.values, w, label='Test', color='#d7191c')
axes[2].set_xticks(x)
axes[2].set_xticklabels(bins_labels)
axes[2].set_ylabel('Count')
axes[2].set_title('Length Bins')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_seq_length.png')
plt.close()
print(f"\n→ 已保存 {OUT_DIR}/01_seq_length.png")


# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. 氨基酸组成分析")
print("=" * 60)

# 训练集氨基酸频率
all_train_seq = ''.join(train['sequence'])
train_aa_counts = Counter(all_train_seq)
train_aa_freq = {aa: train_aa_counts.get(aa, 0) / len(all_train_seq) for aa in AMINO_ACIDS}
train_aa_df = pd.DataFrame({'AA': AMINO_ACIDS, 'Frequency': [train_aa_freq[aa] for aa in AMINO_ACIDS]})

# 测试集氨基酸频率
all_test_seq = ''.join(test['sequence'])
test_aa_counts = Counter(all_test_seq)
test_aa_freq = {aa: test_aa_counts.get(aa, 0) / len(all_test_seq) for aa in AMINO_ACIDS}
test_aa_df = pd.DataFrame({'AA': AMINO_ACIDS, 'Frequency': [test_aa_freq[aa] for aa in AMINO_ACIDS]})

print("\n训练集氨基酸频率 (Top 5):")
for aa in sorted(AMINO_ACIDS, key=lambda a: train_aa_freq[a], reverse=True)[:5]:
    print(f"  {aa}: {train_aa_freq[aa]:.3f}")
print("训练集氨基酸频率 (Bottom 5):")
for aa in sorted(AMINO_ACIDS, key=lambda a: train_aa_freq[a])[:5]:
    print(f"  {aa}: {train_aa_freq[aa]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(AMINO_ACIDS))
w = 0.35
axes[0].bar(x - w/2, train_aa_df['Frequency'], w, label='Train', color='#2c7bb6')
axes[0].bar(x + w/2, test_aa_df['Frequency'], w, label='Test', color='#d7191c')
axes[0].set_xticks(x)
axes[0].set_xticklabels(AMINO_ACIDS)
axes[0].set_ylabel('Frequency')
axes[0].set_title('Amino Acid Composition: Train vs Test')
axes[0].legend()

# 差异图
diff = (test_aa_df['Frequency'].values - train_aa_df['Frequency'].values) * 100
colors = ['#d7191c' if d > 0 else '#2c7bb6' for d in diff]
axes[1].bar(AMINO_ACIDS, diff, color=colors, edgecolor='white')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_ylabel('Diff (Test - Train) × 100')
axes[1].set_title('AA Frequency Difference')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_aa_composition.png')
plt.close()
print(f"→ 已保存 {OUT_DIR}/02_aa_composition.png")


# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. 标签分布分析")
print("=" * 60)

y = train[label_cols].values.astype(int)

# 每个蛋白质的标签数
labels_per_protein = y.sum(axis=1)
lp_max = int(labels_per_protein.max())
print(f"\n每蛋白标签数:")
print(f"  mean={labels_per_protein.mean():.1f}, median={np.median(labels_per_protein):.0f}")
print(f"  std={labels_per_protein.std():.1f}, min={labels_per_protein.min()}, max={labels_per_protein.max()}")

# 每个标签的正样本数
pos_per_label = y.sum(axis=0)
print(f"\n每标签正样本数:")
print(f"  mean={pos_per_label.mean():.1f}, median={np.median(pos_per_label):.0f}")
print(f"  min={pos_per_label.min()}, max={pos_per_label.max()}")
print(f"  density (avg): {pos_per_label.mean() / len(train):.2%}")

# 四分位数
q25, q50, q75 = np.percentile(pos_per_label, [25, 50, 75])
print(f"  25%: {q25:.0f}, 50%: {q50:.0f}, 75%: {q75:.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 每蛋白标签数分布
axes[0, 0].hist(labels_per_protein, bins=min(60, lp_max + 1), color='#2c7bb6', edgecolor='white', alpha=0.8)
axes[0, 0].axvline(labels_per_protein.mean(), color='#d7191c', ls='--', lw=2, label=f'Mean={labels_per_protein.mean():.1f}')
axes[0, 0].set_xlabel('Number of Labels per Protein')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Labels per Protein Distribution')
axes[0, 0].legend()

# 每标签正样本数分布
axes[0, 1].hist(pos_per_label, bins=50, color='#fc8d59', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(pos_per_label.mean(), color='#d7191c', ls='--', lw=2, label=f'Mean={pos_per_label.mean():.0f}')
axes[0, 1].set_xlabel('Positive Samples per Label')
axes[0, 1].set_ylabel('Number of Labels')
axes[0, 1].set_title('Label Frequency Distribution')
axes[0, 1].legend()

# Top 20 高频标签
top20_idx = np.argsort(pos_per_label)[::-1][:20]
top20_labels = [label_cols[i] for i in top20_idx]
top20_values = pos_per_label[top20_idx]
axes[1, 0].barh(range(20), top20_values[::-1], color='#2c7bb6', edgecolor='white')
axes[1, 0].set_yticks(range(20))
axes[1, 0].set_yticklabels([top20_labels[i] for i in range(19, -1, -1)], fontsize=8)
axes[1, 0].set_xlabel('Positive Count')
axes[1, 0].set_title('Top 20 Most Frequent Labels')

# Bottom 20 低频标签
bottom20_idx = np.argsort(pos_per_label)[:20]
bottom20_labels = [label_cols[i] for i in bottom20_idx]
bottom20_values = pos_per_label[bottom20_idx]
axes[1, 1].barh(range(20), bottom20_values[::-1], color='#d7191c', edgecolor='white')
axes[1, 1].set_yticks(range(20))
axes[1, 1].set_yticklabels([bottom20_labels[i] for i in range(19, -1, -1)], fontsize=8)
axes[1, 1].set_xlabel('Positive Count')
axes[1, 1].set_title('Bottom 20 Least Frequent Labels')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_label_distribution.png')
plt.close()
print(f"→ 已保存 {OUT_DIR}/03_label_distribution.png")


# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. 标签共现分析")
print("=" * 60)

# 标签共现矩阵 (Top 30 高频标签)
top30_idx = np.argsort(pos_per_label)[::-1][:30]
y_top30 = y[:, top30_idx]
co_occur = y_top30.T @ y_top30
# 转为 Jaccard
n = len(train)
jaccard = np.zeros_like(co_occur, dtype=float)
for i in range(30):
    for j in range(30):
        union = pos_per_label[top30_idx[i]] + pos_per_label[top30_idx[j]] - co_occur[i, j]
        jaccard[i, j] = co_occur[i, j] / union if union > 0 else 0

top30_names = [label_cols[i].replace('label_', 'L') for i in top30_idx]

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(jaccard, dtype=bool), k=1)
sns.heatmap(jaccard, mask=mask, annot=False, cmap='YlOrRd', square=True,
            xticklabels=top30_names, yticklabels=top30_names,
            cbar_kws={'label': 'Jaccard Similarity'}, ax=ax)
ax.set_title('Label Co-occurrence (Jaccard) — Top 30 Labels')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_label_cooccurrence.png')
plt.close()
print(f"→ 已保存 {OUT_DIR}/04_label_cooccurrence.png")


# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. 序列长度 vs 标签数 关系")
print("=" * 60)

corr_len_label = np.corrcoef(train_len, labels_per_protein)[0, 1]
print(f"序列长度与标签数 Pearson 相关系数: {corr_len_label:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散点图 (采样)
sample_idx = np.random.RandomState(42).choice(len(train), size=5000, replace=False)
xlim = int(np.percentile(train_len, 99))
sample_x = np.clip(train_len.values[sample_idx], a_min=None, a_max=xlim)
axes[0].scatter(sample_x, labels_per_protein[sample_idx],
                alpha=0.3, s=8, c='#2c7bb6', edgecolors='none')
axes[0].set_xlabel('Sequence Length (aa)')
axes[0].set_ylabel('Number of Labels')
axes[0].set_title(f'Length vs Labels (r={corr_len_label:.3f})')

# 分桶
bins_edges2 = [0, 200, 400, 800, 1600, 40000]
bins_labels2 = ['≤200', '200-400', '400-800', '800-1600', '>1600']
binned = pd.cut(train_len, bins=bins_edges2, labels=bins_labels2)
avg_labels = [labels_per_protein[binned == b].mean() for b in bins_labels2]
axes[1].bar(bins_labels2, avg_labels, color='#2c7bb6', edgecolor='white')
axes[1].set_xlabel('Sequence Length Range')
axes[1].set_ylabel('Average Number of Labels')
axes[1].set_title('Avg Labels by Length Bin')
for i, v in enumerate(avg_labels):
    axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_length_vs_labels.png')
plt.close()
print(f"→ 已保存 {OUT_DIR}/05_length_vs_labels.png")


# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. 汇总统计表")
print("=" * 60)

summary = {
    '训练集样本数': len(train),
    '测试集样本数': len(test),
    '标签总数': n_labels,
    '标签密度': f"{pos_per_label.mean() / len(train):.2%}",
    '训练集序列平均长度': f"{train_len.mean():.1f} ± {train_len.std():.1f}",
    '测试集序列平均长度': f"{test_len.mean():.1f} ± {test_len.std():.1f}",
    '每蛋白平均标签数': f"{labels_per_protein.mean():.1f} ± {labels_per_protein.std():.1f}",
    '每标签平均正样本数': f"{pos_per_label.mean():.0f}",
    '序列长度与标签数相关系数': f"{corr_len_label:.3f}",
}
print()
for k, v in summary.items():
    print(f"  {k}: {v}")

summary_df = pd.DataFrame(summary.items(), columns=['指标', '值'])
summary_df.to_csv(f'{OUT_DIR}/summary.csv', index=False, encoding='utf-8-sig')
print(f"\n→ 汇总表已保存 {OUT_DIR}/summary.csv")

print(f"\n{'='*60}")
print(f"EDA 完成，所有图表已保存到 {OUT_DIR}/ 目录")
print(f"总耗时: {time.time() - t0:.1f}s")
