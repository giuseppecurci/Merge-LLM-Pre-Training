#!/usr/bin/env python3
import os
import argparse
import yaml
import sys
from pathlib import Path 
import shutil

BASE_DIR = Path(__file__).parent.parent.parent

def already_exists(steps_path):
    merged_model_path = os.path.join(steps_path, "merged_model")
    if os.path.exists(merged_model_path):
        return True
    else:
        False

def run_jobs(jobs):
    for script in jobs:
        steps_path = os.path.dirname(script)

        if already_exists(steps_path):
            print(f"Skipping (already exists): {steps_path}")
            continue

        try:
            run_merge(script)
        except:
            print(f"Merge failed: {script}")
            print(f"Removing merged model folder")
            merged_model_path = os.path.join(steps_path, "merged_model")
            shutil.rmtree(merged_model_path)

def run_missing_jobs(missing_per_job):
    for missing_script in missing_per_job:
        steps_path = Path(missing_script).parent
        merged_model_path = steps_path / "merged_model"

        if merged_model_path.exists():
            continue

        try:
            still_missing = find_missing_models(steps_path)
            if still_missing:
                print(f"Still missing models, skipping retry: {steps_path}")
                continue

            run_merge(str(missing_script))

        except Exception:
            print(f"Retry merge failed: {missing_script}")

def find_missing_models(steps_path):
    yaml_path = os.path.join(steps_path, "mergekit.yaml")
    if not os.path.exists(yaml_path):
        return ["mergekit.yaml not found"]

    missing = []

    with open(yaml_path) as f:
        docs = yaml.safe_load_all(f)  
        for i, doc in enumerate(docs):
            if not doc:
                continue
            for entry in doc.get("models", []):
                model_rel = entry.get("model")
                if model_rel is None:
                    continue

                model_abs = os.path.join(BASE_DIR, model_rel)
                if not os.path.exists(model_abs):
                    missing.append(model_abs)
            if i == 0:
                if "base_model" in doc:
                    base_model_path = doc["base_model"]
                    if not os.path.exists(base_model_path): missing.append(base_model_path)

    return missing

def run_merge(script_path):
    print(f"Running merge script: {script_path}")
    ret = os.system(f"bash {script_path}")
    if ret != 0:
        print(f"Warning: {script_path} exited with code {ret}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_models",
        default="model_merging/merged_models",
        help="Relative path inside model_merging containing merged models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merge without running them"
    )
    args = parser.parse_args()

    root = os.path.join(BASE_DIR, args.merged_models)
    
    merge_jobs = []
    missing_per_job = {}
    
    if args.dry_run:
        print(f"Running the following merge from {root}:")
        count = 0
    
    for model_name in os.listdir(root):
        model_path = os.path.join(root, model_name)
        for merge_strategy in os.listdir(model_path):
            strategy_path = os.path.join(model_path, merge_strategy)
            for steps_range in os.listdir(strategy_path):
                steps_path = os.path.join(strategy_path, steps_range)

                if steps_path.endswith("intermediate"):
                    continue 

                merged_model_path = os.path.join(steps_path, "merged_model")
                merge_script = os.path.join(steps_path, "merge.sh")

                if os.path.exists(merged_model_path):
                    continue

                if args.dry_run:
                    print(f"- {model_name}/{merge_strategy}/{steps_range}")
                    count += 1
                    continue
                
                missing = find_missing_models(steps_path)

                if missing:
                    missing_per_job[merge_script] = missing
                    continue

                merge_jobs.append(merge_script)

    if args.dry_run:
        print(f"Number of merge to run: {count}")
        return

    # Sequentially run all merge jobs
    try:
        light_jobs = []
        heavy_jobs = []

        for script in merge_jobs:
            # heavy merge: contains "noisy_scale" or "ties" in the path/name
            if "noisy_scale" in script or "ties" in script:
                heavy_jobs.append(script)
            else:
                light_jobs.append(script)

        run_jobs(light_jobs)
        
        run_missing_jobs(missing_per_job)

        run_jobs(heavy_jobs)      

        run_missing_jobs(missing_per_job)      

    except KeyboardInterrupt:
        print("\nCtrl+C received. Exiting sequentially running jobs.")
        sys.exit(1)

if __name__ == "__main__":
    main()