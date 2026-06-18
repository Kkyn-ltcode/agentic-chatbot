cd /path/to/HyperMamba-NIDS

python -c "
import json, numpy as np, pandas as pd
from pathlib import Path

root = Path('data/processed/darpa_tc_e3/theia')

# 1. Summary
print('=== SUMMARY ===')
s = json.load(open(root / 'summary.json'))
print(json.dumps(s, indent=2))

# 2. Shard count & sizes
print('\n=== SHARDS ===')
shards = sorted((root / 'features_norm').glob('thyne_shard*.npz'))
print(f'Total shards: {len(shards)}')
for f in shards:
    d = np.load(f)
    print(f'  {f.name}: {d[\"X\"].shape[0]:>10,} events, {d[\"X\"].shape[1]} features')

# 3. Timestamps per shard (to figure out the date boundaries)
print('\n=== TIMESTAMP RANGES ===')
labeled = sorted((root / 'labeled').glob('labeled_shard*.parquet'))
for f in labeled:
    df = pd.read_parquet(f, columns=['timestamp_nanos'])
    ts_min = pd.to_datetime(df['timestamp_nanos'].min(), unit='ns')
    ts_max = pd.to_datetime(df['timestamp_nanos'].max(), unit='ns')
    print(f'  {f.name}: {ts_min} to {ts_max}')

# 4. Label distribution per shard
print('\n=== LABELS (crossprocess) ===')
for f in labeled:
    df = pd.read_parquet(f, columns=['label_crossprocess'])
    n = len(df)
    n_pos = int((df['label_crossprocess'] == 1).sum())
    print(f'  {f.name}: {n:>10,} events, {n_pos:>6,} attack ({100*n_pos/n:.3f}%)')

# 5. Entity vocab size
vocab = np.load(root / 'graph' / 'entity_vocab.npz', allow_pickle=True)
print(f'\n=== ENTITIES ===')
print(f'  num_entities: {vocab[\"num_entities\"]}')
"
