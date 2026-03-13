"""
Modal app for autodata experiments. Runs training on remote GPUs.

One-time data setup:
    modal run modal_app.py --prepare

Run a training experiment:
    modal run modal_app.py

Run a data exploration script (CPU only):
    modal run modal_app.py --explore-script explore.py

The agent edits data.py locally. This script sends it to Modal,
which processes the data and trains using a remote GPU.
Only data.py (~KB) goes up, only metrics come back.
"""

import os
import sys

import modal

app = modal.App("autodata")

# Persistent storage for raw data shards and tokenizer
volume = modal.Volume.from_name("autodata-cache", create_if_missing=True)
VOLUME_PATH = "/root/.cache/autoresearch"

# Container image with training dependencies
# prepare.py and train.py are baked in (they never change)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "kernels>=0.11.7",
        "numpy>=2.2.6",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
    )
    .add_local_file("prepare.py", "/app/prepare.py")
    .add_local_file("train.py", "/app/train.py")
)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def prepare_data(num_shards: int = 10):
    """One-time: download data shards and train tokenizer on Modal."""
    import subprocess

    result = subprocess.run(
        ["python", "prepare.py", "--num-shards", str(num_shards)],
        capture_output=True,
        text=True,
        cwd="/app",
    )
    volume.commit()
    return {"output": result.stdout + result.stderr, "returncode": result.returncode}


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="H100",
    timeout=900,
)
def train(data_py: str):
    """Run a single 5-minute training experiment with the given data.py."""
    # Write the agent's data.py into the app directory
    with open("/app/data.py", "w") as f:
        f.write(data_py)

    import subprocess

    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True,
        cwd="/app",
        timeout=720,
    )
    return {"output": result.stdout + result.stderr, "returncode": result.returncode}


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def explore(script: str):
    """Run a data exploration script on CPU (no GPU)."""
    with open("/app/explore.py", "w") as f:
        f.write(script)

    import subprocess

    result = subprocess.run(
        ["python", "explore.py"],
        capture_output=True,
        text=True,
        cwd="/app",
        timeout=300,
    )
    return {"output": result.stdout + result.stderr, "returncode": result.returncode}


@app.local_entrypoint()
def main(prepare: bool = False, num_shards: int = 10, explore_script: str = ""):
    if prepare:
        print("Preparing data on Modal (downloading shards + training tokenizer)...")
        result = prepare_data.remote(num_shards)
        print(result["output"])
        sys.exit(result["returncode"])

    if explore_script:
        with open(explore_script) as f:
            script = f.read()
        result = explore.remote(script)
        print(result["output"], end="")
        sys.exit(result["returncode"])

    # Read current data.py and send to Modal for training
    with open("data.py") as f:
        data_py = f.read()

    result = train.remote(data_py)
    print(result["output"], end="")
    sys.exit(result["returncode"])
