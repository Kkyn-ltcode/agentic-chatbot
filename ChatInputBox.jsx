# 1. Install PyTorch compiled for CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. Install the exact matching CUDA toolkit into this specific environment
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

# 3. Now try the installation again
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
