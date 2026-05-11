# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from huggingface_hub import hf_hub_download


def setup_data(pickle_name):
    Path("data").mkdir(exist_ok=True)
    if (Path("data") / pickle_name).exists():
        return  # already present locally, skip download
    try:
        hf_hub_download(
            repo_id="facebook/ShapeR-Evaluation",
            filename=pickle_name,
            repo_type="dataset",
            local_dir="./data",  # Optional: forces it to a specific folder instead of cache
        )
    except Exception as e:
        print(f"Error downloading data: {e}")
        print(
            "Are you sure you are logged in to huggingface? (`huggingface_cli login`)"
        )
        exit(1)
