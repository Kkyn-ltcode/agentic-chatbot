import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Mock Data Setup (Replace with your actual training logs)
# ---------------------------------------------------------
epochs = np.arange(1, 101) # 100 epochs

# Mocking GraphTPA: Rapid, smooth convergence to ~95.8%
# Using an exponential decay function to simulate smooth learning
graphtpa_f1 = 95.77 - 40 * np.exp(-0.1 * epochs) 
# Adding just a tiny bit of realistic noise
graphtpa_f1 += np.random.normal(0, 0.3, len(epochs))

# Mocking TransformerConv: Volatile, spiky, slower convergence to ~90.1%
transformer_f1 = 90.12 - 40 * np.exp(-0.05 * epochs)
# Adding severe gradient variance and validation spikes
noise = np.random.normal(0, 1.5, len(epochs))
# Introduce a few large random spikes to simulate the volatility mentioned
spikes = np.random.choice([0, -5, 4, -8], size=len(epochs), p=[0.85, 0.05, 0.05, 0.05])
transformer_f1 = transformer_f1 + noise + spikes

# ---------------------------------------------------------
# 2. Plotting the Convergence Dynamics
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(9, 6))

# Plot the volatile baseline
plt.plot(epochs, transformer_f1, label='TransformerConv (Full-Rank)', 
         color='#e74c3c', linewidth=1.5, alpha=0.8)

# Plot the stable GraphTPA
plt.plot(epochs, graphtpa_f1, label='GraphTPA (Low-Rank / Ours)', 
         color='#2ecc71', linewidth=2.5)

# ---------------------------------------------------------
# 3. Formatting for Academic Publication
# ---------------------------------------------------------
plt.title("Training Convergence Dynamics on NF-ToN-IoT", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Training Epoch", fontsize=14)
plt.ylabel("Validation Macro F1-Score (%)", fontsize=14)

# Set logical limits for the Y-axis
plt.ylim(50, 100)
plt.xlim(0, 100)

plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

# Save as a vector graphic for LaTeX
plt.tight_layout()
plt.savefig("fig_convergence.pdf", dpi=300, bbox_inches='tight')
plt.show()
