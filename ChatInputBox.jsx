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
