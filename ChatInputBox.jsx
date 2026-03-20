from sklearn.metrics import f1_score, classification_report

# Assuming you already ran this from step 4:
# preds_proba = model.predict(dtest)
# preds_labels = np.argmax(preds_proba, axis=1)

# --- 1. MACRO F1 SCORE ---
# Calculates F1 for each of the 10 classes separately, then takes the unweighted mean.
# Use this if you care about your model performing well on ALL classes equally, 
# even the rare ones.
f1_macro = f1_score(y_test, preds_labels, average='macro')
print(f"Macro F1 Score: {f1_macro:.4f}")

# --- 2. WEIGHTED F1 SCORE ---
# Calculates F1 for each class, but weights the average by the number of true instances in each class.
# Use this if your classes are heavily imbalanced and you want the metric to reflect that.
f1_weighted = f1_score(y_test, preds_labels, average='weighted')
print(f"Weighted F1 Score: {f1_weighted:.4f}")

# --- 3. THE FULL BREAKDOWN (Highly Recommended) ---
# Since you have 10 classes, seeing a single number doesn't tell the whole story.
# This report shows you the Precision, Recall, and F1 score for EVERY single class (0 through 9).
print("\n--- Detailed Classification Report ---")
report = classification_report(y_test, preds_labels)
print(report)
