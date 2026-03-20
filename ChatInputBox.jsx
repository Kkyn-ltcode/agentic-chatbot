import numpy as np

# 1. Identify only the numeric columns (ignores categories/strings)
numeric_cols = X.select_dtypes(include=[np.number]).columns

# 2. Get the safe maximum and minimum for float32
# This is roughly 3.4 x 10^38 and -3.4 x 10^38
safe_max = np.finfo(np.float32).max * 0.99  # 99% of max just to be safe
safe_min = np.finfo(np.float32).min * 0.99

print(f"Clipping {len(numeric_cols)} numeric columns to safe float32 limits...")

# 3. Apply the clip to all numeric columns at once
X[numeric_cols] = X[numeric_cols].clip(lower=safe_min, upper=safe_max)

# Now it is 100% safe to downcast to float32
X[numeric_cols] = X[numeric_cols].astype(np.float32)
