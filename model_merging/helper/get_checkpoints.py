import warnings
from pathlib import Path
from typing import Dict

def resolve_checkpoint_paths(
    base_path: str,
    steps: list[int],
    use_safetensors: bool,
) -> Dict[int, Path]:

    base = Path(base_path)
    resolved: Dict[int, Path] = {}

    for step in steps:
        snapshot_path = None

        # ---- Case 0: SmolLM3 support (stageX_step_YYYYYYYY) ----        
        if snapshot_path is None:
            smollm_matches = [
                p for p in base.glob(f"stage*_step_{step}")
                if p.is_dir()
            ]

            if len(smollm_matches) == 1:
                stage_dir = smollm_matches[0]
                snapshot_dirs = list(
                    stage_dir.glob("models--*/snapshots/*")
                )
                snapshot_dirs = [
                    p for p in snapshot_dirs if (p / "config.json").exists()
                ]

                if len(snapshot_dirs) == 1:
                    snapshot_path = snapshot_dirs[0]
                elif len(snapshot_dirs) > 1:
                    raise FileNotFoundError(
                        f"Multiple HF snapshots found for SmolLM3 step {step}: "
                        f"{[str(p) for p in snapshot_dirs]}"
                    )

            elif len(smollm_matches) > 1:
                raise FileNotFoundError(
                    f"Multiple SmolLM3 stage folders found for step {step}: "
                    f"{[str(p) for p in smollm_matches]}"
                )

        # ---- Case 1: original format ----
        if snapshot_path is None:
            expected = base / f"step={step}" / "hf"
            if expected.exists():
                snapshot_path = expected

        # ---- Case 2: HuggingFace / Pythia format ----
        if snapshot_path is None:
            step_dir = base / f"step{step}"
            if step_dir.exists():
                snapshot_dirs = list(
                    step_dir.glob("models--*/snapshots/*")
                )

                snapshot_dirs = [
                    p for p in snapshot_dirs if (p / "config.json").exists()
                ]

                if len(snapshot_dirs) == 1:
                    snapshot_path = snapshot_dirs[0]
                elif len(snapshot_dirs) > 1:
                    raise FileNotFoundError(
                        f"Multiple HF snapshots found for step {step}: "
                        f"{[str(p) for p in snapshot_dirs]}"
                    )

        # ---- Case 3: fuzzy fallback ----
        if snapshot_path is None:
            pattern = f"step={step}"
            candidates = [
                p / "hf"
                for p in base.iterdir()
                if p.is_dir() and p.name.startswith(pattern)
            ]

            if len(candidates) == 1:
                warnings.warn(
                    f"Exact path '{expected}' not found; using '{candidates[0]}' instead."
                )
                snapshot_path = candidates[0]
            elif len(candidates) > 1:
                raise FileNotFoundError(
                    f"Multiple matches for step={step}: {[str(c) for c in candidates]}"
                )
            else:
                print(f"Checkpoint for step={step} not found.")
                while True:
                    choice = input("Ignore or kill merge? [ignore/kill] ").strip().lower()
                    if choice == "ignore":
                        break
                    elif choice == "kill":
                        raise FileNotFoundError(
                            f"Checkpoint directory for step={step} not found "
                            f"and no similar matches in {base}."
                        )
                    else:
                        print("Invalid choice")

        if snapshot_path is None:
            continue

        # ---- Safetensors handling ----
        if use_safetensors:
            st_path = snapshot_path / "model.safetensors"
            if not st_path.exists():
                raise FileNotFoundError(
                    f"Expected safetensors file not found for step {step}: {st_path}"
                )
            resolved[step] = st_path
        else:
            resolved[step] = snapshot_path

    assert len(resolved) >= 2, (
        f"Found only {len(resolved)} checkpoints, but need at least 2 for {base_path}"
    )

    return resolved