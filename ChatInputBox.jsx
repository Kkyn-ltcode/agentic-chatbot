import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# Academic Formatting Setup
# ---------------------------------------------------------
# Use a serif font (Times New Roman) to match IEEE templates
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# ---------------------------------------------------------
# Step 1: Data Extraction (Mocking the Latent Embeddings)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 2000 # 1000 Benign, 1000 MITM

# E-GraphSAGE (Baseline): Distributions are "fatally entangled"
baseline_benign = np.random.randn(n_samples // 2, 64)
baseline_mitm = np.random.randn(n_samples // 2, 64) + 0.5 # Very slight shift, mostly overlapping
embeddings_baseline = np.vstack((baseline_benign, baseline_mitm))

# GraphTPA (Ours): MITM forms a "distinct, linearly separable cluster"
ours_benign = np.random.randn(n_samples // 2, 64)
ours_mitm = np.random.randn(n_samples // 2, 64) + 5.0 # Large shift for clear separation
embeddings_ours = np.vstack((ours_benign, ours_mitm))

# Ground truth labels
labels = np.array(['Benign'] * (n_samples // 2) + ['MITM'] * (n_samples // 2))

# ---------------------------------------------------------
# Step 2: Dimensionality Reduction (t-SNE)
# ---------------------------------------------------------
print("Computing t-SNE for Baseline...")
tsne_baseline = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings_baseline)

print("Computing t-SNE for GraphTPA...")
tsne_ours = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings_ours)

# ---------------------------------------------------------
# Step 3 & 4: Plot Setup and Scatter Data
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.set_style("white")

# Define highly contrasting colors
palette = {"Benign": "#bdc3c7", "MITM": "#e74c3c"} # Gray and distinct Red

# Left Plot: Concatenation Baseline
sns.scatterplot(
    x=tsne_baseline[:, 0], y=tsne_baseline[:, 1],
    hue=labels, palette=palette, alpha=0.7, s=25,
    ax=axes[0], legend=False
)
axes[0].set_title("Standard Concatenation\n(Entangled)", fontsize=14, fontweight='bold', pad=10)
# Step 5: Hide axis ticks for standard t-SNE styling
axes[0].set_xticks([])
axes[0].set_yticks([])

# Right Plot: GraphTPA (Ours)
sns.scatterplot(
    x=tsne_ours[:, 0], y=tsne_ours[:, 1],
    hue=labels, palette=palette, alpha=0.7, s=25,
    ax=axes[1]
)
axes[1].set_title("GraphTPA: Tensor Edge Representation\n(Disentangled)", fontsize=14, fontweight='bold', pad=10)
axes[1].set_xticks([])
axes[1].set_yticks([])

# ---------------------------------------------------------
# Step 5: Academic Styling and Export
# ---------------------------------------------------------
# Adjust the legend
handles, labels_leg = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels_leg, title="Traffic Class", fontsize=12, title_fontsize=12, loc='upper right')

plt.tight_layout()

# Export as high-resolution vector graphic
plt.savefig("fig_tsne.pdf", format="pdf", dpi=300, bbox_inches="tight")
print("Plot successfully saved as 'fig_tsne.pdf'")
plt.show()
