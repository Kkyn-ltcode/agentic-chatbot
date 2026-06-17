Viewed train_full.py:219-250

Here is the step-by-step guide to run Experiment 1 from start to finish.

### Step 1: Train & Evaluate HyperMamba
First, you need to run HyperMamba's training pipeline. Because of the code updates I just made, the evaluation step at the very end of this script will now correctly collect and save the integer `entity_ids` into a `preds.pt` file.

From your `HyperMamba-NIDS` directory, run:
```bash
python -m src.pipeline.train_full --dataset theia --label_type crossprocess
```
*Note: This will output a `results/hypermamba/.../preds.pt` file. Keep track of this path.*

### Step 2: Run PIDSMaker Baselines
Next, you need to run the three baseline models (KAIROS, ThreaTrace, MAGIC) using the PIDSMaker framework. Since PIDSMaker is a separate codebase, you'll run this according to your standard PIDSMaker workflow (e.g., on your Kubeflow cluster).

Ensure that for each baseline, you save the `edge_losses` directory from the test window. You should end up with three directories containing CSVs:
- `/path/to/kairos/edge_losses/test/model_epoch_X`
- `/path/to/threatrace/edge_losses/test/model_epoch_X`
- `/path/to/magic/edge_losses/test/model_epoch_X`

### Step 3: Run the Dual-Evaluation Script
Finally, we use the updated comparison script to match the timestamps, aggregate the nodes, and generate the final metrics tables for your paper.

Run the following command (replace the paths with your actual output paths from Steps 1 and 2):
```bash
python -m src.analysis.experiment1_comparison \
    --hypermamba_preds results/hypermamba/preds.pt \
    --kairos_dir /path/to/PIDSMaker/kairos/edge_losses/test/model_epoch_0 \
    --threatrace_dir /path/to/PIDSMaker/threatrace/edge_losses/test/model_epoch_0 \
    --magic_dir /path/to/PIDSMaker/magic/edge_losses/test/model_epoch_0 \
    --pidsmaker_gt /Users/nguyen/Documents/Work/NIDS/PIDSMaker/Ground_Truth/orthrus/E3-THEIA \
    --dataset theia \
    --label_type crossprocess \
    --out_dir results/experiment1
```

### What You Will Get
Check the `results/experiment1/` directory. You will find:
1. `event_level_table.tex`: The perfectly fair apples-to-apples comparison on events.
2. `node_level_table.tex`: The apples-to-oranges comparison evaluating each model on its own ground truth (which you can discuss in the paper).
3. `pr_curves_event.png` & `pr_curves_node.png`: The overlaid Precision-Recall plots.

Let me know when you're ready to run it or if you hit any snags during the training!
