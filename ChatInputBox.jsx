import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import gc

# Assuming your data is already loaded into X (features) and y (labels 0-9)
# TIP: Ensure X is float32 and y is int8 to save ~50% of your RAM!

# --- 1. DATA SPLITTING (70 / 20 / 10) ---

# First split: 70% Train, 30% Temp (which will become Val and Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y # Ensures all 10 classes are balanced across splits
)

# Second split: The remaining 30% is split into 2/3 (20% overall) and 1/3 (10% overall)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=(1/3), 
    random_state=42, 
    stratify=y_temp
)

# Free up memory immediately
del X_temp, y_temp
gc.collect()

# --- 2. XGBOOST DATA STRUCTURES ---

# For 16M rows, standard DMatrix is okay, but QuantileDMatrix uses less memory 
# and initializes faster when using the 'hist' tree method.
dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# --- 3. MODEL PARAMETERS ---

params = {
    'objective': 'multi:softprob', # Outputs probabilities for each of the 10 classes
    'num_class': 10,               # Since your labels are 0-9
    'tree_method': 'hist',         # CRITICAL for 16M rows. Do not use 'exact'.
    'device': 'cuda',              # Change to 'cpu' if you don't have a GPU
    'eval_metric': 'mlogloss',     # Multiclass logloss for evaluation
    'learning_rate': 0.1,
    'max_depth': 6
}

# --- 4. TRAINING WITH EARLY STOPPING ---

evals = [(dtrain, 'train'), (dval, 'validation')]

print("Starting training...")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,          # High number, rely on early stopping
    evals=evals,
    early_stopping_rounds=50,      # Stop if validation score doesn't improve for 50 rounds
    verbose_eval=25                # Print progress every 25 trees
)

# --- 5. EVALUATION ---

# Predict on the unseen 10% test set
preds_proba = model.predict(dtest)

# Convert probabilities to definitive class labels (0-9)
preds_labels = np.argmax(preds_proba, axis=1)

print("Finished!")
