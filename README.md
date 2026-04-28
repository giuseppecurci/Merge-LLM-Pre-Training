# Merge-LLM-Pre-Training

Official repository of the paper  
**"Practical Guidelines for Model Merging in LLM Pre-Training"** (ACL, 2026)

Checkpoints of the Villanova model available on [Hugging Face](https://huggingface.co/VillanovaAI/Villanova-2B-checkpoints).

---

# Environment Setup

Two separate environments are required:

- **Model merging environment**
- **Evaluation environment**

Keeping them separate ensures reproducibility and avoids dependency conflicts.

---

## Model Merging Environment

```bash
python -m venv model_merging/model_merging_venv
source model_merging/model_merging_venv/bin/activate

cd model_merging/mergekit-llm-pretraining
pip install -e .
cd ..
pip install -r requirements_merge.txt
```

## Evaluation Environment

```bash
python -m venv evaluation/harnessvenv
source evaluation/harnessvenv/bin/activate

cd evaluation
pip install -r requirements_eval.txt
```

# Download Other Models

```bash
chmod +x other_models/setup.sh
./other_models/setup.sh
```

# Merging

Activate the merging environment before running merge commands:

```bash
source model_merging/model_merging_venv/bin/activate
```

---

## Using Helpers

To simplify running merging experiments, we provide a set of helper scripts that automatically traverse checkpoint directories and generate the required merging configuration files.

If you plan to use these utilities, ensure that your checkpoint directory is organized as follows:

```
checkpoints/
└── model/
    └── checkpoints/
        └── step=*/
            └── hf/
```
This structure is required for the scripts to correctly locate and process the model checkpoints. Otherwise you can simply create the config files manually and run the merges following [MergeKit](https://github.com/arcee-ai/mergekit) instructions.

## Linear, Non-Linear and TV Methods

```bash
python model_merging/helper/prepare_merge \
    --ckpts-path [ckpt-path] \
    --async-write \
    --start-step [starting-step] \
    --end-step [end-step] \
    --range-step [range-step] \          # 1k Olmo, 2k Pythia, 40k Smollm3, 2k our models
    --merge-strategy [merge-strategy] \  # one of: linear, ema, wma, slerp, ties
    --alpha [0.2] \                      # used for EMA
    --pull-secondary-model [0.4]         # used for SLERP
```

---

## Synthetic Diversity Methods

Use one of the `--inject-noise` and `--ckpt-dropout 0.2`.

```bash
python model_merging/helper/prepare_merge \
    ...
    --merge-strategy linear \
    --ensemble-merge-strategy linear \
    --inject-noise \                   # noise injection
    --ckpt-dropout [0.2]               # checkpoint dropout
```

---

## Run the Merge

You have two options:

1. Run:
   ```bash
   python model_merging/helper/merge_all.py
   ```

2. Navigate to each `model_merging/merged_models` directory and execute the corresponding `merge.sh` script manually.

---

# Evaluation

Navigate to:

```
evaluation/experiments
```

Create a folder named after the model.

Inside that folder, create a `test_eval.yaml` file with the following structure:

```yaml
methods:
  - name: [merge_method_tested]
    steps:
      - step: [step]
        model_path: [abs_model_path]
```

For Pythia and Olmo models, add the following field to the YAML file to ignore non-English datasets:

```yaml
specific_ignored_tasks:
  - hellaswag_de
  - hellaswag_es
  - hellaswag_fr
  - hellaswag_it
  - xcopa_it
  - xnli_de
  - xnli_es
  - xnli_fr
```

Run the evaluation:

```bash
python evaluate.py
```

# Divergence Computation

Activate the merging environment and move to the divergence directory:

```bash
source model_merging/model_merging_venv/bin/activate
cd plots/divergence
```

---

## Computation

The scripts `compute_divergence_over_training.py` (weights) and `compute_divergence_over_training_extra.py` (activations, logits, probabilities) illustrates how checkpoint divergence is computed throughout training.

---

## Plots

### RMS Plots

We additionally provide the precomputed divergence data to enable inspection and plot reproduction.

To reproduce the RMS plots, run:

```bash
python plot_div_compare.py \
    --jsons our_240-300k/divergence.json our_240-300k_dcos/divergence.json \
    --labels "Stable LR" "Decay LR" \
    --out our_240-300k_stable_vs_decay
```

```bash
python plot_div_compare.py \
    --jsons our_512-640k/divergence.json our_512-640k_dcos/divergence.json \
    --labels "Stable LR" "Decay LR" \
    --out our_512-640k_stable_vs_decay
```

```bash
python plot_div_compare.py \
    --jsons our_1000-1100k/divergence.json our_1000-1100k_dcos/divergence.json \
    --labels "Stable LR" "Decay LR" \
    --out our_1000-1100k_stable_vs_decay
```

## Activations, logits, and probabilities

For divergence measured in activation, logit, or probability space, use `plot_div_compare_extra.py` with the same structure as above. You can select the representation space and metric via:

```
--space [activations | logits | probabilities]
--metric [cosine | kl]
```

**Example**:
```bash
python plot_div_compare_extra.py \
    --jsons ... \
    --labels "Stable LR" "Decay LR" \
    --space activations \
    --metric cosine \
    --out ...
```
