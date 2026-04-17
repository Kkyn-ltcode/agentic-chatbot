Excellent. I am now your research supervisor for this project. Let's begin.

You have the full roadmap. Now we execute it one concrete task at a time. I will assign tasks, explain why each matters, show you the code you need to write, and tell you what success looks like.

---

Session 1: Phase 0 — Environment & First Contact with the Stack

Goal for this session: By the end, you will have a working development environment, understand the core data structures of lambeq, and have run a complete "Hello World" of the hybrid pipeline in a single Jupyter notebook.

Time estimate: 4-6 hours (one focused evening or weekend morning).

---

Task 0.1: Create the Project Environment

Why this matters: Quantum libraries have finicky dependencies. Isolating them in a conda environment prevents "works on my machine" disasters later. Using conda specifically because lambeq's dependency discopy sometimes has C-extensions that pip alone struggles with.

Instructions:

Open your terminal and run:

```bash
# Create environment with Python 3.10 (most stable for lambeq)
conda create -n qnlp python=3.10 -y

# Activate it
conda activate qnlp

# Install core quantum and NLP libraries
pip install pennylane pennylane-lightning lambeq discopy
pip install sentence-transformers torch torchvision
pip install scikit-learn matplotlib seaborn pandas jupyter tqdm

# Install analysis and tracking tools
pip install cca-zoo wandb pytest hydra-core

# Verify the key import works
python -c "import lambeq; print('lambeq version:', lambeq.__version__)"
python -c "import pennylane as qml; print('PennyLane version:', qml.__version__)"
python -c "from sentence_transformers import SentenceTransformer; print('SBERT OK')"
```

Expected output: You should see version numbers printed without errors. lambeq version should be ≥0.3.0.

Troubleshooting: If lambeq fails, try pip install lambeq --no-deps then manually install discopy and pylatexenc. If you're on Windows and discopy fails, use WSL2 (highly recommended for quantum dev).

Success criterion: All three test imports run without error.

---

Task 0.2: Set Up the Project Skeleton

Why this matters: A clean, modular structure prevents you from drowning in spaghetti code 8 weeks from now. We separate data loading, model definition, and analysis from the start.

Instructions:

Create the following directory structure. You can do this manually or run this script:

```bash
mkdir -p Quantum/{src/{data,models,probing,syntax,utils},notebooks,configs,scripts,data/{raw,processed},results/{figures,tables,raw},paper,tests}

touch Quantum/src/__init__.py
touch Quantum/src/data/__init__.py
touch Quantum/src/models/__init__.py
touch Quantum/src/probing/__init__.py
touch Quantum/src/syntax/__init__.py
touch Quantum/src/utils/__init__.py
touch Quantum/tests/__init__.py
touch Quantum/requirements.txt
touch Quantum/README.md
```

Now create Quantum/requirements.txt with the following content (copy exactly):

```
pennylane>=0.38.0
pennylane-lightning>=0.38.0
lambeq>=0.4.0
discopy>=1.1.0
sentence-transformers>=2.2.0
torch>=2.0.0
scikit-learn>=1.3.0
cca-zoo>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
jupyter>=1.0.0
tqdm>=4.65.0
wandb>=0.15.0
pytest>=7.0.0
hydra-core>=1.3.0
```

Success criterion: Directory structure exists. requirements.txt is populated.

---

Task 0.3: The "Hello World" of QNLP — Parse a Sentence and Draw Its Diagram

Why this matters: You need to see the DisCoCat string diagram before you can build a structural fingerprint of it. This task demystifies what lambeq actually does.

Instructions:

Open a Jupyter notebook at Quantum/notebooks/01_hello_discocat.ipynb. Run the following cells:

Cell 1: Imports

```python
from lambeq import BobcatParser
from discopy import Word, Cup, Id, Ty

# Define atomic types
n = Ty('n')  # noun
s = Ty('s')  # sentence

print("Atomic types defined: n =", n, ", s =", s)
```

Cell 2: Parse a simple sentence

```python
parser = BobcatParser(verbose=False)

sentence = "Alice loves Bob"
diagram = parser.sentence2diagram(sentence)

print(f"Diagram for '{sentence}':")
diagram.draw(figsize=(8, 4), fontsize=12)
```

Cell 3: Inspect the diagram's structure

```python
# What are the boxes (words)?
print("Boxes (words) in diagram:")
for box in diagram.boxes:
    print(f"  - {box.name} : {box.dom} -> {box.cod}")

# What are the cups (grammatical reductions)?
print("\nCups in diagram:")
for cup in diagram.cups:
    print(f"  - {cup}")

# Overall type signature
print(f"\nDiagram type: {diagram.dom} -> {diagram.cod}")
```

Expected output:

· A visual diagram appears with wires and boxes labeled "Alice", "loves", "Bob".
· The box for "loves" should have type n.r @ s @ n.l (read as: takes a noun on the right, produces a sentence, takes a noun on the left).
· The diagram should have Cup operations connecting wires.

Troubleshooting: If BobcatParser fails to download the model, check your internet connection. If the diagram doesn't render, ensure matplotlib is installed and you're in a notebook environment (not plain Python script).

Success criterion: You see a diagram and understand that loves has a complex adjoint type while Alice and Bob are simple nouns.

---

Task 0.4: Extract SBERT Embeddings and Run a Toy PQC

Why this matters: This is the minimal viable hybrid pipeline — the thing you'll scale up over the next 28 weeks. Getting it running now confirms your environment and mental model.

Instructions:

Continue in the same notebook or create notebooks/02_hybrid_toy.ipynb.

Cell 1: Load SBERT and encode sentences

```python
from sentence_transformers import SentenceTransformer

# Load the lightweight model
sbert = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Alice loves Bob",
    "Bob loves Alice",
    "The cat chased the mouse",
    "The mouse was chased by the cat"
]

embeddings = sbert.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")  # Should be (4, 384)

# Check cosine similarity between first two (should be high, same words different order)
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity between 'Alice loves Bob' and 'Bob loves Alice': {sim:.3f}")
```

Cell 2: Build a 4-qubit PennyLane circuit

```python
import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    """
    inputs: array of shape (n_qubits,) - rotation angles
    weights: array of shape (n_layers, n_qubits, 3) - trainable parameters
    """
    # Data encoding: apply RY rotations based on input
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Variational layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure expectation of PauliZ on first qubit
    return qml.expval(qml.PauliZ(0))

# Test with random inputs
dummy_input = np.random.uniform(0, np.pi, size=(n_qubits,))
dummy_weights = np.random.uniform(0, 2*np.pi, size=(2, n_qubits, 3))

result = quantum_circuit(dummy_input, dummy_weights)
print(f"Circuit output (expectation value in [-1, 1]): {result:.4f}")
```

Cell 3: Connect to PyTorch for training

```python
import torch
import torch.nn as nn
import torch.optim as optim

class HybridModel(nn.Module):
    def __init__(self, input_dim=384, compressed_dim=4, n_qubits=4, n_layers=2):
        super().__init__()
        self.compress = nn.Linear(input_dim, compressed_dim)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # PennyLane device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Trainable quantum weights
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float64) * 0.1
        )
        
    def quantum_circuit(self, inputs):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit(x, w):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            qml.StronglyEntanglingLayers(w, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))
        return circuit(inputs, self.q_weights)
    
    def forward(self, x):
        # x shape: (batch, 384)
        compressed = self.compress(x)  # (batch, compressed_dim)
        # Map to [0, pi] range for rotation angles
        angles = torch.sigmoid(compressed) * np.pi
        # Process each sample in batch (PennyLane expects one sample at a time)
        outputs = torch.stack([self.quantum_circuit(angles[i]) for i in range(angles.shape[0])])
        return outputs.squeeze()

# Quick test
model = HybridModel(compressed_dim=4, n_qubits=4)
x = torch.randn(2, 384, dtype=torch.float64)  # batch of 2
out = model(x)
print(f"Model output shape: {out.shape}, values: {out.detach().numpy()}")
```

Success criterion: The code runs without error. You see an output tensor of shape (2,) with values between -1 and 1.

---

Task 0.5: Understand the Math (Assigned Reading)

Why this matters: You can't debug what you don't understand. These three papers are your theoretical foundation. Read them alongside coding over the next 2 weeks.

Assigned Reading for This Week:

1. Coecke, Sadrzadeh & Clark (2010) — "Mathematical Foundations for a Compositional Distributional Model of Meaning"
   · Focus on: Sections 2-3 (pregroups, string diagrams, the functor F).
   · Key takeaway: Why a verb is n^r \cdot s \cdot n^l and how "cups" represent grammatical reduction.
2. Kornblith et al. (2019) — "Similarity of Neural Network Representations Revisited"
   · Focus on: Section 3 (Centered Kernel Alignment).
   · Key takeaway: Why CKA works better than CCA for high-dimensional representations.
3. PennyLane Documentation on qml.StronglyEntanglingLayers
   · Read: The exact gate sequence and parameter count.

Action: After reading the first 5 pages of Coecke et al., come back and explain to me (in your own words) why "Alice loves Bob" is grammatically valid but "Alice Bob loves" is not, using the type system. This is a mental check — I don't need a written answer, just verify you can articulate it.

---

Task 0.6: Initialize Git and Weights & Biases

Why this matters: Version control from day zero. W&B will track every experiment automatically.

Instructions:

```bash
cd Quantum
git init
echo "data/raw/" > .gitignore
echo "results/raw/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".wandb/" >> .gitignore
git add .
git commit -m "Initial project structure"
```

W&B setup:

```python
# In a notebook or script:
import wandb
wandb.login()  # You'll need a free account at wandb.ai
```

Success criterion: git log shows one commit. wandb.login() succeeds.

---

Session 1 Wrap-Up

By the end of this session, you should have:

· A conda environment qnlp with all libraries working.
· A project directory Quantum/ with the proper structure and git initialized.
· A Jupyter notebook where you can parse a sentence, draw its diagram, extract SBERT embeddings, and run a tiny hybrid PQC model.
· W&B connected.
· Started reading Coecke et al.

---

Your Next Steps

1. Complete Tasks 0.1 through 0.6 above.
2. Once done, report back with:
   · Any errors encountered and their resolutions.
   · A screenshot of the diagram for "Alice loves Bob".
   · Confirmation that the toy hybrid model runs.

Then I will assign Session 2: Week 1 Tasks — Data Loading and Baseline DisCoCat.

Take your time with Task 0.3 and 0.4. This is the "aha" moment where abstract DisCoCat becomes concrete code. Let me know when you're ready to proceed.
