from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import Path 

BASE_DIR = Path(__file__).parent.parent / "smollm3_checkpoints"

LAST_STABLE_STEP = 3_440_000
LAST_DECAY_STEP = 4_720_000

STAGES = {
    "stable": 1,
    "decay": 3
}

RANGE_STEP = 40_000
N_STABLE = 15
N_DECAY = 13 # no more than 13 availables

stable_steps = [LAST_STABLE_STEP - RANGE_STEP * i for i in range(N_STABLE)]
decay_steps = [LAST_DECAY_STEP - RANGE_STEP * i for i in range(N_DECAY)]

model_name = "HuggingFaceTB/SmolLM3-3B-checkpoints"
print(f"Downloading {N_STABLE + N_DECAY} checkpoints of {model_name}...")
print("Stable steps:")
for stable_step in stable_steps:
    print(f" - {stable_step}")

print("Decay steps:")
for decay_step in decay_steps:
    print(f" - {decay_step}")

for stage in STAGES:
    num_stage = STAGES[stage]
    if stage == "stable":
        steps = stable_steps
    else:
        steps = decay_steps
    
    for step in steps:
        revision = "stage{}-step-{}".format(num_stage, step)

        print(f"Downloading {revision}")
        cache_dir = BASE_DIR / revision.replace("-", "_")

        if os.path.exists(cache_dir):
            print("Skipping already downloaded")
            continue

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            revision=revision,
            cache_dir=cache_dir
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            revision=revision,
            cache_dir=cache_dir
            )

