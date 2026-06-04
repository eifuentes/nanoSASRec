# remote.py
import modal
import sys

app = modal.App("nanoSASRec")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("sasrec")
    .add_local_file("data.py", "/app/data.py")
    .add_local_file("train_v1.py", "/app/train_v1.py")
    .add_local_file("train_v2.py", "/app/train_v2.py")
    .add_local_file("train_v3.py", "/app/train_v3.py")
    .add_local_file("train_v4.py", "/app/train_v4.py")
    .add_local_file("train_v6.py", "/app/train_v6.py")
    .add_local_file("train_v7.py", "/app/train_v7.py")
)

data_volume = modal.Volume.from_name("nanoSASRec-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10",
    timeout=3600,
    volumes={"/app/data": data_volume},
)
def train(cmd: str):
    import subprocess
    subprocess.run("python -u data.py --dataset ml-20m", shell=True, cwd="/app", check=True)
    subprocess.run(cmd, shell=True, cwd="/app", check=True)
    data_volume.commit()

@app.local_entrypoint()
def main(cmd: str):
    cmd += " --device cuda"
    print(f"Running: {cmd}")
    train.remote(cmd)