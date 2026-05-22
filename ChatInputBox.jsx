# 1. Re-extract features (will pick up object-aware features)
#    Delete old global_stats.npz and thyne_shard*.npz first
rm data/processed/darpa_tc_e3/theia/features/global_stats.npz
rm data/processed/darpa_tc_e3/theia/features/thyne_shard*.npz
python -m src.pipeline.batch_features --dataset theia

# 2. Re-normalize
python -m src.pipeline.normalize --dataset theia

# 3. Re-run L1 relabel (if not already done)
python -m src.pipeline.novel_binary_relabel --dataset theia

# 4. Train with new improvements
torchrun --nproc_per_node=4 -m src.pipeline.train --config configs/theia_l1_thyn.yaml --dataset theia
torchrun --nproc_per_node=4 -m src.pipeline.train --config configs/theia_l1_baseline_a.yaml --dataset theia
