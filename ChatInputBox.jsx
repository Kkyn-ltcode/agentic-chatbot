# 1. Control experiment (uses existing broad checkpoints)
python -m src.pipeline.control_experiment \
    --checkpoint checkpoints/thyn_v0/best.pt \
    --config configs/thyn_v0.yaml

python -m src.pipeline.control_experiment \
    --checkpoint checkpoints/baseline_a/best.pt \
    --config configs/baseline_a.yaml

# 2. Cross-process training (the C1 test!)
torchrun --nproc_per_node=4 -m src.pipeline.train \
    --config configs/thyn_crossprocess.yaml

torchrun --nproc_per_node=4 -m src.pipeline.train \
    --config configs/baseline_a_crossprocess.yaml

# 3. TRACE pipeline (in parallel)
python -m src.pipeline.feature_extractor --dataset trace
python -m src.pipeline.build_graph --dataset trace
python -m src.pipeline.build_sequences --dataset trace
