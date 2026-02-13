import yaml
import subprocess
import argparse
import os 
import warnings 

def parse_args():
    parser = argparse.ArgumentParser(description='Esegui valutazioni LM con accelerate')
    parser.add_argument('--exp-path', default="experiments", type=str, 
                        help='Percorso al file di configurazione YAML')
    parser.add_argument('--general-tasks', default="experiments/general_and_tasks.yaml", type=str, 
                        help="Path defining tasks and other settings such as batch_size, device...")
    parser.add_argument('--dry-run', action="store_true", help="Prints info experiments without running them")
    parser.add_argument('--devices', type=str, default=None, help="Comma separated list of GPU ids to use. If not set, it will wait for free GPUs.")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    with open(args.general_tasks) as f:
        general_and_tasks = yaml.safe_load(f)
    model_format = general_and_tasks['model_format']
    batch_size = str(general_and_tasks['batch_size'])
    device = str(general_and_tasks['device'])
    device_ids = args.devices
    tasks = general_and_tasks['tasks']
    num_few_shots = general_and_tasks["num_few_shots"]

    models_to_eval = []
    models_to_not_eval = []
    exp_path = args.exp_path 
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
                step, model_method_path, to_eval = step_model_path["step"], step_model_path["model_path"], step_model_path.get("to_eval", True)
                out_path = os.path.join(method_path, step, str(num_few_shots))
                if os.path.exists(out_path): 
                    continue
                elif not to_eval:
                    models_to_not_eval.append({
                        "out_path": out_path,
                        "model_path": model_method_path,
                        "specific_tasks": specific_tasks,
                        })
                else:
                    if not os.path.exists(model_method_path): 
                        warnings.warn(f"{model_method_path} does not exist. Fix {methods_file_path}")
                        continue
                    models_to_eval.append({
                        "out_path" : out_path,
                        "model_path" : model_method_path,
                        "specific_tasks": specific_tasks
                        })

    if models_to_not_eval and not args.dry_run:
        print(f"There are {len(models_to_not_eval)} experiments that are set to no eval")
        while True:
            choice = input("Choose action [show/ignore/eval_all]: ").strip().lower()
            if choice == "show":
                print("The following methods are set to no eval") 
                for model_to_not_eval in models_to_not_eval:
                    print(f"- {model_to_not_eval['out_path']} \n    - {model_to_not_eval['model_path']}")
                print("="*50)
            elif choice == "ignore":
                break
            elif choice == "eval_all":
                models_to_eval.extend(models_to_not_eval)
                models_to_not_eval.clear()
                break
            else:
                print("Invalid answer")

    print(f"Running {len(models_to_eval)} experiments. Not evaluating {len(models_to_not_eval)} methods.")
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