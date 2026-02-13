from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os

MODEL_NAME = "allenai/Olmo-3-1025-7B"
BASE_ROOT = Path(os.path.join(os.path.dirname(__file__), MODEL_NAME.replace("/", "_")))
STAGE = "stage1"

N = 15
RANGE_STEP = 1_000 # batch size is double the one used by our model -> adjust halving (2k/2 = 1k)

EARLY_STEP = 286_000
LATE_STEP = 1_413_000

# steps after warmup
early_steps = [EARLY_STEP - RANGE_STEP * i for i in range(N)]

# steps near the end of training
late_steps = [LATE_STEP - RANGE_STEP * i for i in range(N)]

all_steps = early_steps + late_steps


def download_checkpoint(model_name, stage, step):
    revision = f"{stage}-step{step}"
    cache_dir = BASE_ROOT / revision.replace("stage1-", "")

    if cache_dir.exists():
        print(f"[Already Downloaded] {model_name} @ {revision}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {model_name} @ {revision}")

        AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )

        AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )

        print(f"[OK] {model_name} @ {revision}")

    except Exception as e:
        print(f"[FAIL] {model_name} @ {revision}")
        print(f"       {type(e).__name__}: {e}")


print(f"\n=== MODEL {MODEL_NAME} ({STAGE}) ===")
for step in all_steps:
    print(f"\n=== STEP {step} ===")
    download_checkpoint(MODEL_NAME, STAGE, step)
    print("=" * 50)
