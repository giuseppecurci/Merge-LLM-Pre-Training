import json
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

# ------------------------------------------------------------
# Core loader (auto-detect format)
# ------------------------------------------------------------

def load_snapshot(snapshot_dir: Path):
    """
    Supports:
      - Pythia sharded (.bin + index.json)
      - Pythia single pytorch_model.bin
      - SmolLM3 sharded (.safetensors + index.json)
    Returns full state_dict.
    """

    # ---- SmolLM3 sharded safetensors ----
    st_index = snapshot_dir / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index, "r") as f:
            weight_map = json.load(f)["weight_map"]

        shard_files = sorted(set(weight_map.values()))
        state_dict = {}

        for shard in shard_files:
            shard_path = snapshot_dir / shard
            state_dict.update(load_file(shard_path))

        return state_dict, shard_files, st_index

    # ---- Pythia sharded .bin ----
    bin_index = snapshot_dir / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with open(bin_index, "r") as f:
            weight_map = json.load(f)["weight_map"]

        shard_files = sorted(set(weight_map.values()))
        state_dict = {}

        for shard in shard_files:
            shard_path = snapshot_dir / shard
            shard_dict = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_dict)

        return state_dict, shard_files, bin_index

    # ---- Single pytorch_model.bin ----
    single_bin = snapshot_dir / "pytorch_model.bin"
    if single_bin.exists():
        state_dict = torch.load(single_bin, map_location="cpu")
        return state_dict, [single_bin.name], None

    raise FileNotFoundError(f"No supported weights found in {snapshot_dir}")

# ------------------------------------------------------------
# Conversion
# ------------------------------------------------------------

def convert_snapshot(snapshot_dir: Path, remove_shards: bool):
    output_file = snapshot_dir / "model.safetensors"

    if output_file.exists():
        print(f"Already converted: {output_file}")
        return

    print(f"\nProcessing {snapshot_dir}")

    state_dict, shard_files, index_file = load_snapshot(snapshot_dir)

    if not state_dict:
        raise RuntimeError("Empty state_dict — aborting.")

    save_file(state_dict, output_file)
    print(f"Saved {output_file}")

    if remove_shards:
        print("Removing original shard files...")
        for shard in shard_files:
            shard_path = snapshot_dir / shard
            if shard_path.exists():
                shard_path.unlink()

        if index_file and index_file.exists():
            index_file.unlink()

# ------------------------------------------------------------
# Model traversal
# ------------------------------------------------------------

def process_pythia(base_path: Path, remove_shards: bool):
    for step_dir in sorted(base_path.glob("step*")):
        snapshot_dirs = list(step_dir.glob("models--*/snapshots/*"))
        for snapshot in snapshot_dirs:
            convert_snapshot(snapshot, remove_shards)

def process_smollm3(base_path: Path, remove_shards: bool):
    for stage_dir in sorted(base_path.glob("stage*_step_*")):
        snapshot_dirs = list(stage_dir.glob("models--*/snapshots/*"))
        for snapshot in snapshot_dirs:
            convert_snapshot(snapshot, remove_shards)

def process_olmo(base_path: Path, remove_shards: bool):
    """
    Expected layout:

    base_path/
        stepXXXX/
            models--allenai--Olmo-3-1025-7B/
                snapshots/<hash>/
                    model-*.safetensors
                    model.safetensors.index.json
    """
    for step_dir in sorted(base_path.glob("step*")):
        snapshot_dirs = list(step_dir.glob("models--*/snapshots/*"))
        for snapshot in snapshot_dirs:
            convert_snapshot(snapshot, remove_shards)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unify Pythia, SmolLM3 and Olmo3 checkpoints into single safetensors."
    )

    parser.add_argument(
        "--type",
        choices=["pythia", "smollm3", "olmo3"],
        required=True,
        help="Model type",
    )

    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        required=True,
        help="Base checkpoint directory",
    )

    parser.add_argument(
        "--rm-shards",
        action="store_true",
        help="Remove original shards after conversion",
    )

    args = parser.parse_args()

    if args.type == "pythia":
        process_pythia(args.base_ckpt_dir, args.rm_shards)
    elif args.type == "smollm3":
        process_smollm3(args.base_ckpt_dir, args.rm_shards)
    elif args.type == "olmo3":
        process_olmo(args.base_ckpt_dir, args.rm_shards)

    print("\nAll checkpoints processed.")