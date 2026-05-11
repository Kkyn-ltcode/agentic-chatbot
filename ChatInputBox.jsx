# Check your system CUDA toolkit
nvcc --version

# Check what CUDA PyTorch was built with
python -c "import torch; print('PyTorch:', torch.__version__); print('Built with CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
