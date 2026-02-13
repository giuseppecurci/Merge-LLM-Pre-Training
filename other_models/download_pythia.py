from transformers import GPTNeoXForCausalLM, AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "pythia_checkpoints"

MODELS = [
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

N = 15
RANGE_STEP = 2000
EARLY_STEP = 50000
LAST_STEP = 143000

# steps after warmup
early_steps = [EARLY_STEP - RANGE_STEP * i for i in range(N)]

# steps near the end of training
late_steps = [LAST_STEP - RANGE_STEP * i for i in range(N)]

all_steps = early_steps + late_steps

def download_checkpoint(model_name, step):
    revision = f"step{step}"
    cache_dir = BASE_DIR / model_name.replace("/", "_") / revision
    if cache_dir.exists():
        print(f"[Already Downloaded] {model_name} @ {revision}")
        return 
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {model_name} @ {revision}")
        GPTNeoXForCausalLM.from_pretrained(
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

for model in MODELS:
    print(f"\n=== MODEL {model} ===")
    for step in all_steps:
        print(f"\n=== STEP {step} ===")
        download_checkpoint(model, step)
        print("="*50)
