import yaml
import subprocess
import argparse
import os 
import warnings 

BASE_DIR = os.path.dirname(__file__)

def parse_args():
    parser = argparse.ArgumentParser(description='Esegui valutazioni LM con accelerate')
    parser.add_argument('--dry-run', action="store_true", help="Prints info experiments without running them")
    parser.add_argument('--devices', type=str, default=None, help="Comma separated list of GPU ids to use. If not set, it will wait for free GPUs.")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    exp_path = os.path.join(BASE_DIR, "experiments")
    general_and_tasks_path = os.path.join(BASE_DIR, "general_and_tasks.yaml")

    with open(general_and_tasks_path) as f:
        general_and_tasks = yaml.safe_load(f)
    model_format = general_and_tasks['model_format']
    batch_size = str(general_and_tasks['batch_size'])
    device = str(general_and_tasks['device'])
    device_ids = args.devices
    tasks = general_and_tasks['tasks']
    num_few_shots = general_and_tasks["num_few_shots"]

    models_to_eval = []
    models_to_not_eval = []
    for model_name in os.listdir(exp_path):
        model_path = os.path.join(exp_path, model_name)
        if not os.path.isdir(model_path): continue
        methods_file_path = os.path.join(model_path, "test_eval.yaml")
        with open(methods_file_path) as f:
            methods_file = yaml.safe_load(f)
        
        specific_ignored_tasks = methods_file.get("specific_ignored_tasks", [])
        specific_tasks = [task for task in tasks if task not in specific_ignored_tasks]
        specific_tasks = ','.join(specific_tasks)

        if not len(specific_tasks) > 0:
            raise ValueError(f"{methods_file_path} is ignoring all tasks. No tasks to eval.") 

        for method_steps in methods_file["methods"]:
            method_name = method_steps["name"]
            method_path = os.path.join(model_path, method_name)
            for step_model_path in method_steps["steps"]:
                step, model_method_path = step_model_path["step"], step_model_path["model_path"]
                out_path = os.path.join(method_path, step, str(num_few_shots))
                if os.path.exists(out_path): 
                    continue
                else:
                    if not os.path.exists(model_method_path): 
                        warnings.warn(f"{model_method_path} does not exist. Fix {methods_file_path}")
                        continue
                    models_to_eval.append({
                        "out_path" : out_path,
                        "model_path" : model_method_path,
                        "specific_tasks": specific_tasks
                        })

    print(f"Running {len(models_to_eval)} experiments.")
    print(f" - num few shots: {num_few_shots}")
    print(f" - batch size: {batch_size}")
    print(f" - default tasks ({len(tasks)}): {tasks}")
    print(f" - device: {device}")
    for model_to_eval in models_to_eval:
        num_specific_tasks = len(model_to_eval["specific_tasks"].split(","))
        print(
            f"- Output path : {model_to_eval['out_path']}\n"
            f"  Model path  : {model_to_eval['model_path']}\n"
            f"  Tasks ({num_specific_tasks}) : {model_to_eval['specific_tasks']}\n"
        )
    print("="*50)
    
    if args.dry_run:
        return

    for model_to_eval in models_to_eval:
        
        if device_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

        model_args = f"pretrained={model_to_eval['model_path']},dtype=float16" 
        out_path = model_to_eval["out_path"]
        specific_tasks = model_to_eval["specific_tasks"]

        cmd = [
            'accelerate', 'launch',
            '-m', 'lm_eval',
            '--model', model_format,
            '--model_args', model_args,
            '--output_path', out_path,
            '--tasks', specific_tasks,
            '--batch_size', batch_size,
            '--device', device,
            '--num_fewshot', num_few_shots
        ]
        
        subprocess.run(cmd)

        actual_result_path = os.path.join(out_path, os.listdir(out_path)[0])
        json_results = os.path.join(actual_result_path, os.listdir(actual_result_path)[0])
        subprocess.run(["mv", json_results, out_path])
        subprocess.run(["rmdir", actual_result_path])

if __name__ == '__main__':
    main()