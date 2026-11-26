"""
remote.py - Modal GPU training wrapper for nanoSASRec

Runs any train_v*.py script on a remote GPU via Modal.
Automatically mounts all root-level .py files and the sasrec/ package.
Dataset zip is downloaded once and persisted in a Modal Volume.
Checkpoints go to a second Volume, so a preempted run resumes instead of starting over.

Usage:
    uv run modal run --detach remote.py --cmd "python -u train_v1.py --dataset ml-1m --epochs 200 --eval-every 20"

Logs are viewable in the Modal dashboard after detaching.
"""

import glob

import modal

app = modal.App("nanoSASRec")

image = modal.Image.debian_slim(python_version="3.12").uv_sync().add_local_python_source("sasrec")

# Mount all root-level .py files (data.py, train_v*.py, etc.)
for f in sorted(glob.glob("*.py")):
    image = image.add_local_file(f, f"/app/{f}")

data_volume = modal.Volume.from_name("nanoSASRec-data", create_if_missing=True)
# Training scripts write resumable checkpoints to /app/checkpoints (their default --checkpoint-dir).
# Modal commits Volume writes in the background every few seconds, so when it preempts a container and
# restarts it with the same command, the new container finds the run's checkpoints and resumes.
checkpoint_volume = modal.Volume.from_name("nanoSASRec-checkpoints", create_if_missing=True, version=2)


@app.function(
    image=image,
    gpu="A10",
    timeout=86400,  # 24 hours
    volumes={"/app/data": data_volume, "/app/checkpoints": checkpoint_volume},
)
def train(cmd: str):
    import re
    import subprocess

    # Extract dataset from the training command
    dataset_match = re.search(r"--dataset\s+(\S+)", cmd)
    dataset = dataset_match.group(1) if dataset_match else "ml-1m"

    subprocess.run(f"python -u data.py --dataset {dataset}", shell=True, cwd="/app", check=True)
    # Commit the downloaded and processed data right away, not only when training finishes.
    # Jobs launched after this point find the dataset on the volume and skip the download.
    data_volume.commit()
    subprocess.run(cmd, shell=True, cwd="/app", check=True)
    data_volume.commit()
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(cmd: str):
    cmd += " --device cuda"
    print(f"Running: {cmd}")
    train.spawn(cmd)
