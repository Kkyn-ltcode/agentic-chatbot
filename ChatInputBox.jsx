import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# 1. Mock Data Setup (Replace with your actual extracted data)
# ---------------------------------------------------------
# Assuming embeddings are shape (N, d) where d is hidden dim (e.g., 256)
N = 4000 # 2000 Benign, 2000 MITM
np.random.seed(42)

# Mocking the baseline (entangled)
embeddings_baseline = np.random.randn(N, 256) 

# Mocking GraphTPA (disentangled)
embeddings_graphtpa_benign = np.random.randn(N//2, 256) + np.array([2]*256)
embeddings_graphtpa_mitm = np.random.randn(N//2, 256) - np.array([2]*256)
embeddings_graphtpa = np.vstack([embeddings_graphtpa_benign, embeddings_graphtpa_mitm])

# Labels: 0 for Benign, 1 for MITM
labels = np.array(['Benign'] * (N//2) + ['MITM'] * (N//2))

# ---------------------------------------------------------
# 2. Apply t-SNE Dimensionality Reduction
# ---------------------------------------------------------
print("Computing t-SNE for Baseline...")
tsne_baseline = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results_baseline = tsne_baseline.fit_transform(embeddings_baseline)

print("Computing t-SNE for GraphTPA...")
tsne_graphtpa = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results_graphtpa = tsne_graphtpa.fit_transform(embeddings_graphtpa)

# ---------------------------------------------------------
# 3. Plotting the Figure
# ---------------------------------------------------------
# Set aesthetic style
sns.set_style("white")
palette = {"Benign": "#3498db", "MITM": "#e74c3c"} # Blue and Red

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Baseline
sns.scatterplot(
    x=tsne_results_baseline[:, 0], 
    y=tsne_results_baseline[:, 1],
    hue=labels,
    palette=palette,
    alpha=0.7,
    s=30,
    ax=axes[0],
    legend=False # Hide legend here to avoid duplicates
)
axes[0].set_title("Standard Concatenation\n(Entangled)", fontsize=16, fontweight='bold')
axes[0].set_xticks([])
axes[0].set_yticks([])

# Plot GraphTPA
sns.scatterplot(
    x=tsne_results_graphtpa[:, 0], 
    y=tsne_results_graphtpa[:, 1],
    hue=labels,
    palette=palette,
    alpha=0.7,
    s=30,
    ax=axes[1]
)
axes[1].set_title("GraphTPA: Tensor Edge Representation\n(Disentangled)", fontsize=16, fontweight='bold')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Adjust legend
handles, labels_leg = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels_leg, title="Traffic Type", fontsize=12, title_fontsize=14, loc='upper right')

# Add the caption/title text at the bottom if desired, or let LaTeX handle it
plt.tight_layout()
plt.savefig("tsne_edge_representations.pdf", dpi=300, bbox_inches='tight')
plt.show()
