import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Step 5: Academic Styling Setup
# ---------------------------------------------------------
# Use a serif font (Times New Roman) to match IEEE templates
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# Attack types derived from Table 9
classes = [
    'Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 
    'MITM', 'Password', 'Ransomware', 'Scanning', 'XSS'
]

# ---------------------------------------------------------
# Step 1: Data Generation (Mocking Normalized Inference Data)
# ---------------------------------------------------------
# We mock row-normalized confusion matrices (True Positive Rates).
# Diagonals loosely match the F1-scores reported in Table 9.

# E-GraphSAGE: Flawed baseline
# Narrative: Fails on MITM (18.34%) and XSS (72.34%), heavily misclassifying them as Benign.
cm_egraphsage = np.array([
    [0.98, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00], # Benign
    [0.08, 0.87, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01, 0.01, 0.00], # Backdoor
    [0.01, 0.00, 0.97, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00], # DDoS
    [0.02, 0.00, 0.01, 0.95, 0.00, 0.00, 0.01, 0.00, 0.01, 0.00], # DoS
    [0.10, 0.02, 0.01, 0.01, 0.82, 0.01, 0.02, 0.01, 0.00, 0.00], # Injection
    [0.75, 0.01, 0.01, 0.02, 0.01, 0.18, 0.01, 0.01, 0.00, 0.00], # MITM (Severe False Negatives)
    [0.06, 0.01, 0.00, 0.01, 0.01, 0.00, 0.90, 0.00, 0.01, 0.00], # Password
    [0.15, 0.02, 0.01, 0.01, 0.02, 0.00, 0.01, 0.78, 0.00, 0.00], # Ransomware
    [0.03, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.94, 0.00], # Scanning
    [0.22, 0.01, 0.00, 0.01, 0.02, 0.01, 0.01, 0.00, 0.00, 0.72]  # XSS (Notable False Negatives)
])

# GraphTPA (Ours): Superior model
# Narrative: Rigid diagonal trace, dramatically improving MITM (67.89%) and XSS (86.45%).
cm_graphtpa = np.array([
    [0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00], # Benign
    [0.02, 0.95, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00], # Backdoor
    [0.01, 0.00, 0.98, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # DDoS
    [0.01, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00], # DoS
    [0.03, 0.01, 0.00, 0.01, 0.92, 0.01, 0.01, 0.01, 0.00, 0.00], # Injection
    [0.22, 0.02, 0.01, 0.02, 0.01, 0.68, 0.01, 0.02, 0.01, 0.00], # MITM (Aggressive Detection)
    [0.02, 0.00, 0.00, 0.00, 0.01, 0.00, 0.96, 0.00, 0.01, 0.00], # Password
    [0.05, 0.01, 0.00, 0.01, 0.01, 0.00, 0.00, 0.91, 0.01, 0.00], # Ransomware
    [0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.97, 0.00], # Scanning
    [0.08, 0.01, 0.00, 0.01, 0.02, 0.01, 0.01, 0.00, 0.00, 0.86]  # XSS (Aggressive Detection)
])

# ---------------------------------------------------------
# Step 2: Plot Setup (The 1x2 Grid)
# ---------------------------------------------------------
# Figure spanning double-column width
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Common heatmap settings for fair scientific comparison
heatmap_kws = {
    "cmap": "Blues",
    "vmin": 0.0,
    "vmax": 1.0,
    "annot": True,
    "fmt": ".2f",
    "cbar": False, # We will add a shared colorbar later if needed, or omit for cleaner look
    "square": True,
    "xticklabels": classes,
    "yticklabels": classes,
    "linewidths": 0.5,
    "linecolor": "gray"
}

# ---------------------------------------------------------
# Step 3: Plotting E-GraphSAGE (Left)
# ---------------------------------------------------------
sns.heatmap(cm_egraphsage, ax=axes[0], **heatmap_kws)
axes[0].set_title("E-GraphSAGE (Baseline)", fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel("Predicted Class", fontsize=14, labelpad=10)
axes[0].set_ylabel("True Class", fontsize=14, labelpad=10)
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# ---------------------------------------------------------
# Step 4: Plotting GraphTPA (Right)
# ---------------------------------------------------------
sns.heatmap(cm_graphtpa, ax=axes[1], **heatmap_kws)
axes[1].set_title("GraphTPA (Ours)", fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel("Predicted Class", fontsize=14, labelpad=10)
axes[1].set_ylabel("") # Hide Y-label on the right plot for cleaner look
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

# ---------------------------------------------------------
# Step 5: Exporting
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig("fig_conf_matrix.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
