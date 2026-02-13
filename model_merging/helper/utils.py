from pathlib import Path
from moving_avg import *
import yaml
    
def number_to_ordinal_word(n: int) -> str:
    ones = {
        0: "",
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth"
    }

    teens = {
        10: "tenth",
        11: "eleventh",
        12: "twelfth",
        13: "thirteenth",
        14: "fourteenth",
        15: "fifteenth",
        16: "sixteenth",
        17: "seventeenth",
        18: "eighteenth",
        19: "nineteenth"
    }

    tens = {
        2: "twentieth",
        3: "thirtieth",
        4: "fortieth",
        5: "fiftieth",
        6: "sixtieth",
        7: "seventieth",
        8: "eightieth",
        9: "ninetieth"
    }

    compound_tens = {
        2: "twenty",
        3: "thirty",
        4: "forty",
        5: "fifty",
        6: "sixty",
        7: "seventy",
        8: "eighty",
        9: "ninety"
    }

    if n < 10:
        return ones[n]
    if 10 <= n < 20:
        return teens[n]
    if n < 100:
        t, o = divmod(n, 10)
        if o == 0:
            return tens[t]
        return f"{compound_tens[t]}-{ones[o]}"

    # For larger numbers: "101st" vs "one hundred first"
    # If you want full word form, implement below
    raise NotImplementedError(f"Only supports 0–99 in word form instead got {n}")
        
def validate_inputs(
    start_step: int,
    end_step: int,
    range_step: int,
    merge_strategy: str,
    alpha: float | None = None,
    pull_secondary_model: float | None = None,
    inject_noise: bool = False,
    lambda_weight: float = None,
    density: float = None
):
    """Validate parameters for mergekit merging."""
    valid_strategies = [
        "slerp", "linear", "ema", "wma",  "ties"
        ]
    if merge_strategy not in valid_strategies:
        raise ValueError(
            f"The strategy used is not implemented. Use any of {valid_strategies}"
        )

    if start_step < 0 or end_step <= 0 or range_step <= 0:
        raise ValueError(
            f"Invalid input: start_step ({start_step}) < 0, "
            f"end_step ({end_step}) <= 0, or range_step ({range_step}) <= 0"
        )

    if range_step > end_step - start_step:
        raise ValueError(
            f"The step range ({range_step}) must be smaller than end_step - start_step "
            f"({end_step - start_step})"
        )

    if "ema" in merge_strategy and (alpha is None or not (0 < alpha < 1)):
        raise ValueError(f"{merge_strategy} requires alpha between 0 and 1 (got {alpha})")

    if merge_strategy == "slerp" and pull_secondary_model is None:
        raise ValueError(f"{merge_strategy} merge requires pull_secondary_model value")
    
    if merge_strategy == "ties":
        if lambda_weight is None or density is None:
            raise ValueError(f"{merge_strategy} merge requires lambda_weight and density")
        assert lambda_weight > 0, f"lambda_weight must be >0, but got {lambda_weight}"
        assert 0 < density <= 1, f"density must be between (0,1], but got {density}"

    if inject_noise:
        merge_strat_allow_noise = ["linear", "ema", "wma"]
        assert merge_strategy in merge_strat_allow_noise, \
            f"{merge_strategy} doesn't support noise injection. Use any of {merge_strat_allow_noise}"
    
def write_mergekit_yaml(
    checkpoints: dict[int, Path],
    outfolder: str,
    merge_strategy: str,
    merge_multi_stage: bool = False,
    yaml_name: str = "mergekit.yaml",
    alpha: float | None = None,
    pull_secondary_model: float | None = None,
    dtype: str | None = None,
    inject_noise: bool = False,
    noise_scale: int = 0.0,
    lambda_weight: float = None,
    density: float = None
) -> Path:
   
    out = Path(outfolder)
    yaml_path = out / yaml_name

    if yaml_path.exists():
        return yaml_path, True

    data = {    
        "dtype": dtype,
        "parameters": dict()
    }
    
    MERGE_REQUIRE_BASE = {
        "slerp",
        "ties",
    }

    if merge_multi_stage:
        # requires hf folder, not just the .safetensors file
        for step, checkpoint_path in checkpoints.items():
            if checkpoint_path.name.endswith(".safetensors"):
                checkpoints[step] = checkpoint_path.parent
    
    merge_strategy = merge_strategy.lower()
    if merge_strategy in ("ema", "wma"):
        data["merge_method"] = "linear"
    else:
        data["merge_method"] = merge_strategy
    
    weights = None
    if "ema" in merge_strategy:
        weights = compute_ema_weights(checkpoints, alpha)
    elif "wma" in merge_strategy:
        weights = compute_wma_weights(checkpoints)
    elif "linear" in merge_strategy:
        weights = compute_linear_weights(checkpoints)

    if merge_strategy == "slerp": 
        data["parameters"] = {
            "t": [
                {"value": pull_secondary_model}
            ] 
        }
    elif merge_strategy == "ties":
        data["parameters"] = {
            "lambda": lambda_weight,
            "normalize": True
        }
    elif merge_strategy in ("linear", "wma", "ema"):
        pass
    else:
        raise ValueError(f"Invalid merge_strategy: {merge_strategy}")
                
    
    if data["merge_method"] in MERGE_REQUIRE_BASE:
        data["base_model"] = str(checkpoints[min(checkpoints)])

    if inject_noise:
        data["parameters"].update({
            "inject_noise": inject_noise,
            "noise_scale" : noise_scale
        })

    if merge_multi_stage:
        steps_sorted = sorted(checkpoints.items(), reverse=False)
        
        first_step, first_path = steps_sorted[0]
        second_step, second_path = steps_sorted[1]
        last_step = steps_sorted[-1][0]

        # prepare all stage docs
        docs = []
        prev_name = None
         
        for curr_idx, (step, path) in enumerate(steps_sorted):
            
            # skip second checkpoint (included in the first one)
            if curr_idx == 1: continue
            elif curr_idx == 0: curr_idx += 1

            stage_name = f"{number_to_ordinal_word(curr_idx)}-merge"

            if curr_idx == 1:
                # First stage merges the first two checkpoints
                stage = data.copy()
                stage["name"] = stage_name
                stage_models = [
                    {"model": str(first_path)},
                    {"model": str(second_path)}
                ]
                if weights:
                    stage_models[0]["parameters"] = {"weight": float(weights[first_step])}
                    stage_models[1]["parameters"] = {"weight": float(weights[second_step])}
                stage["models"] = stage_models
                prev_name = stage_name
                docs.append(stage)
            else:

                # Subsequent stages: merge one checkpoint at a time into previous stage
                stage = {
                    "name": stage_name,
                    "dtype": dtype,
                    "merge_method": data["merge_method"],
                    "models" : []
                }

                if step == last_step:
                    stage["name"] = "merged_model"

                if "parameters" in data:
                    stage["parameters"] = data["parameters"]
                
                # link previous stage if strategy requires base_model
                if prev_name is not None and merge_strategy == "slerp":
                    stage["base_model"] = prev_name
                else:
                    stage["models"].append({"model": prev_name})

                # model list for this stage
                model_entry = {"model": str(path)}
                if merge_strategy in ("wma", "ema", "linear"):
                    model_entry["parameters"] = {"weight": float(weights[step])}
                
                stage["models"].append(model_entry)

                docs.append(stage)
                prev_name = stage_name

        # write all docs separated by ---
        with open(yaml_path, "w") as f:
            yaml.safe_dump_all(docs, f, sort_keys=False)

        return yaml_path, False

    else:
        models_section = []
        
        if merge_strategy == "ties":
            steps_sorted = sorted(checkpoints.items(), reverse=True)
        else:
            steps_sorted = sorted(checkpoints.items())

        for step, path in steps_sorted:
            
            if str(path) == data.get("base_model", ""):
                continue

            entry = {"model": str(path)}

            if merge_strategy in ("linear", "ema", "wma") or merge_strategy == "ties":
                entry["parameters"] = {"weight": float(weights[step])}

            if merge_strategy == "ties":
                entry["parameters"]["density"] = density

            models_section.append(entry)
        
        data["models"] = models_section

        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    return yaml_path, False