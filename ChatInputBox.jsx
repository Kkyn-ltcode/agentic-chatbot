import wandb
import pandas as pd
from sklearn.metrics import classification_report

# 1. Initialize your W&B run (if you haven't already at the top of your script)
run = wandb.init(project="xgboost-16M", name="xgb-multiclass-eval")

# 2. Get the report as a DICTIONARY instead of a string
# output_dict=True is the magic parameter here
report_dict = classification_report(y_test, preds_labels, output_dict=True)

# 3. Convert the dictionary into a Pandas DataFrame for clean formatting
# We use .transpose() so the 10 classes are the rows, and precision/recall/f1 are the columns
df_report = pd.DataFrame(report_dict).transpose()

# 4. Save it locally as a CSV file
report_filename = "classification_report.csv"
df_report.to_csv(report_filename, index=True)

# 5. Create the W&B Artifact
# 'name' is what it will be called in the UI
# 'type' helps you organize your artifacts (e.g., 'model', 'dataset', 'evaluation')
eval_artifact = wandb.Artifact(
    name="xgb_classification_report", 
    type="evaluation",
    description="Full classification report for the 10-class XGBoost model"
)

# 6. Attach the CSV file to the artifact
eval_artifact.add_file(report_filename)

# 7. Log the artifact to the W&B servers
run.log_artifact(eval_artifact)

print("Classification report successfully saved to W&B Artifacts!")

# Optional: Close the run when your script is totally finished
# wandb.finish()
