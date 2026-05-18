import wandb

# Initialize your wandb run
run = wandb.init(project="my-awesome-project")

# 1. Create a new artifact
# type can be 'model', 'dataset', or any custom string you want
my_artifact = wandb.Artifact(name="my-trained-model", type="model")

# 2. Add a specific file to the artifact
my_artifact.add_file("model_weights.pth")

# 3. Alternatively, you can add an entire directory
# my_artifact.add_dir("path/to/dataset/folder")

# 4. Log the artifact to upload it
run.log_artifact(my_artifact)

wandb.finish()
