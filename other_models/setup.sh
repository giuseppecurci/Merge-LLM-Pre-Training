#!/bin/bash

source model_merging/model_merging_venv/bin/activate

# Download checkpoints for Pythia, SmolLM3 and Olmo3
python download_olmo3.py
python download_pythia.py
python download_smollm3.py

# Convert to safetensors for faster merging and compatibility
python to_safetensors.py --rm-shards --model pythia --base-ckpt-dir pythia_checkpoints
python to_safetensors.py --rm-shards --model olmo3  --base-ckpt-dir allenai_Olmo-3-1025-7B
python to_safetensors.py --rm-shards --model smollm3 --base-ckpt-dir smollm3_checkpoints