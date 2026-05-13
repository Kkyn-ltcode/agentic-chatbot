# ─── Stage 1: Extract JSON shards from tar.gz ───
python -m src.data.download_darpa_tc --extract --dataset theia

# ─── Stage 2: Parse JSON → Parquet (events, subjects, objects per shard) ───
python -m src.data.darpa_tc_parser --shards all --dataset theia

# ─── Stage 3: Label events (broad + crossprocess) ───
python -m src.pipeline.batch_ingest --dataset theia

# ─── Stage 4: Relabel (adds crossprocess labels) ───
python -m src.pipeline.relabel --dataset theia

# ─── Stage 5: Extract features ───
python -m src.pipeline.batch_features --dataset theia

# ─── Stage 6: Normalize features (train shards 0-6) ───
python -m src.pipeline.normalize --dataset theia --train-shards 0-6

# ─── Stage 7: Build graph (entity vocab + incidence matrix) ───
python -m src.pipeline.build_graph --dataset theia

# ─── Stage 8: Build per-subject sequences ───
python -m src.pipeline.build_sequences --dataset theia

# ─── Stage 9: Train ───
# Single GPU:
python -m src.pipeline.train --config configs/thyn_v0.yaml
# Multi-GPU:
torchrun --nproc_per_node=4 -m src.pipeline.train --config configs/thyn_v0.yaml
