import numpy as np
import pandas as pd

# Assuming X is your features DataFrame (16M rows, 44 columns)

print("1. Converting infinite values to NaN...")
# We use inplace=True to modify the existing DataFrame without copying it
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# (Optional but recommended) Verify that no infs remain
# assert not np.isinf(X.select_dtypes(include=np.number)).any().any(), "Still have infs!"

print("2. Proceeding to XGBoost...")
# You don't need to impute or fill the NaNs! 
# XGBoost's DMatrix natively flags NaNs as missing data.

# Create your DMatrix (using the splits we made in the previous step)
# The 'missing=np.nan' parameter explicitly tells XGBoost what your missing values look like
dtrain = xgb.QuantileDMatrix(X_train, label=y_train, missing=np.nan)
dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)
dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

# ... Proceed with training exactly as before!
