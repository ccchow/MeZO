#!/usr/bin/env python3
"""
MeZO Paper Reproduction Script

This script runs the exact experiments from "Fine-Tuning Language Models with Just Forward Passes"
following the hyperparameter grids in Tables 15-16 and targeting Tables 18-23 for results.

Usage:
    # RoBERTa-large experiments (100k steps)
    python run_mezo_paper_experiments.py --model_family roberta --model_name roberta-large --task sst2
    
    # OPT experiments (20k steps)  
    python run_mezo_paper_experiments.py --model_family opt --model_name facebook/opt-13b --task multirc
    
    # Run full grid search
    python run_mezo_paper_experiments.py --model_family roberta --model_name roberta-large --grid_search
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import itertools

# Exact hyperparameters from original MeZO paper (large_models/README.md)
PAPER_HYPERPARAMETERS = {
    "roberta": {
        "learning_rates": [1e-7, 1e-6, 1e-5],
        "batch_size": 64,
        "eps": 1e-3,
        "max_steps": 100000,
        "eval_steps": 10000,
        "lr_scheduler": "linear",
        "tasks": ["SST2", "SST5", "TREC", "MNLI", "SNLI", "RTE"],
        "num_train": 1000,  # Original paper: 1000 training examples
        "num_dev": 500,     # Original paper: 500 validation examples
        "num_eval": 1000    # Original paper: 1000 test examples
    },
    "opt": {
        "learning_rates": [1e-7, 1e-6],  # Full parameter MeZO from paper
        "batch_size": 16,
        "eps": 1e-3,
        "max_steps": 20000,
        "eval_steps": 4000,
        "lr_scheduler": "constant", 
        "tasks": ["MultiRC", "BoolQ", "Copa", "WiC", "WSC", "CB", "ReCoRD", "RTE", "SST2", "SQuAD", "DROP"],
        "num_train": 1000,  # Original paper: 1000 training examples
        "num_dev": 500,     # Original paper: 500 validation (100 for CB/Copa)
        "num_eval": 1000,   # Original paper: 1000 test examples
        # Prefix-tuning and LoRA variants (from paper Table)
        "prefix_lr": [1e-3, 1e-2],
        "prefix_eps": 1e-1,
        "lora_lr": [5e-5, 1e-4], 
        "lora_eps": 1e-2
    }
}

# Task-specific dataset configurations (matching original paper)
TASK_CONFIGS = {
    # GLUE tasks (used in both RoBERTa and OPT experiments)
    "SST2": {"dataset": "glue", "dataset_config": "sst2", "text_column": "sentence"},
    "SST5": {"dataset": "SetFit/sst5", "text_column": "text"},
    "TREC": {"dataset": "trec", "text_column": "text"},
    "MNLI": {"dataset": "glue", "dataset_config": "mnli", "text_column": "premise"},
    "SNLI": {"dataset": "snli", "text_column": "premise"},
    "RTE": {"dataset": "glue", "dataset_config": "rte", "text_column": "sentence1"},
    
    # SuperGLUE tasks (OPT experiments)
    "MultiRC": {"dataset": "super_glue", "dataset_config": "multirc", "text_column": "paragraph"},
    "BoolQ": {"dataset": "super_glue", "dataset_config": "boolq", "text_column": "passage"},
    "Copa": {"dataset": "super_glue", "dataset_config": "copa", "text_column": "premise", "special": "small_dev"},
    "WiC": {"dataset": "super_glue", "dataset_config": "wic", "text_column": "sentence1"},
    "WSC": {"dataset": "super_glue", "dataset_config": "wsc", "text_column": "text"},
    "CB": {"dataset": "super_glue", "dataset_config": "cb", "text_column": "premise", "special": "small_dev"},
    "ReCoRD": {"dataset": "super_glue", "dataset_config": "record", "text_column": "passage"},
    
    # QA tasks 
    "SQuAD": {"dataset": "squad", "text_column": "context"},
    "DROP": {"dataset": "drop", "text_column": "passage"},
}


def run_experiment(
    model_name: str,
    model_family: str,
    task: str,
    learning_rate: float,
    output_base_dir: str,
    seed: int = 42,
    use_lora: bool = False,
    dry_run: bool = False
) -> Tuple[str, Dict]:
    """Run a single MeZO experiment"""
    
    # Get task configuration
    task_config = TASK_CONFIGS.get(task, {})
    paper_config = PAPER_HYPERPARAMETERS[model_family]
    
    # Create experiment identifier
    exp_id = f"{model_family}_{task}_lr{learning_rate:.0e}_seed{seed}"
    if use_lora:
        exp_id += "_lora"
    
    output_dir = os.path.join(output_base_dir, exp_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "accelerate_mezo.py",
        "--model_name", model_name,
        "--dataset", task_config.get("dataset", task),
        "--text_column", task_config.get("text_column", "text"),
        "--output_dir", output_dir,
        "--batch_size", str(paper_config["batch_size"]),
        "--learning_rate", str(learning_rate),
        "--zo_eps", str(paper_config["eps"]),
        "--max_steps", str(paper_config["max_steps"]),
        "--eval_steps", str(paper_config["eval_steps"]),
        "--lr_scheduler_type", paper_config["lr_scheduler"],
        "--seed", str(seed),
        "--memory_logging",
        "--paper_hparams",
        "--model_family", model_family,
        "--task_name", task,
        "--logging_steps", "100",
        "--save_steps", str(paper_config["eval_steps"])
    ]
    
    # Add dataset config if specified
    if task_config.get("dataset_config"):
        cmd.extend(["--dataset_config", task_config["dataset_config"]])
    
    # Add exact paper dataset sizes (1000 train, 500 dev, 1000 test)
    cmd.extend(["--max_train_samples", str(paper_config["num_train"])])
    
    # Handle special cases for small datasets (CB, Copa)
    if task_config.get("special") == "small_dev":
        # CB and Copa use only 100 dev examples in the paper
        print(f"Using reduced dev set for {task} (100 examples)")
    
    # Add LoRA configuration
    if use_lora:
        cmd.extend([
            "--use_lora",
            "--lora_r", "16",
            "--lora_alpha", "32",
            "--lora_dropout", "0.05",
            "--lora_target", "q_proj", "v_proj", "k_proj", "o_proj"
        ])
    
    # Save experiment configuration
    exp_config = {
        "model_name": model_name,
        "model_family": model_family,
        "task": task,
        "learning_rate": learning_rate,
        "seed": seed,
        "use_lora": use_lora,
        "command": " ".join(cmd),
        "paper_config": paper_config,
        "task_config": task_config
    }
    
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return exp_id, exp_config
    
    print(f"Running experiment: {exp_id}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the experiment
    start_time = time.time()
    
    # Redirect output to log file
    log_file = os.path.join(output_dir, "training.log")
    with open(log_file, 'w') as f:
        process = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    end_time = time.time()
    
    # Update experiment config with results
    exp_config["status"] = "completed" if process.returncode == 0 else "failed"
    exp_config["return_code"] = process.returncode
    exp_config["runtime_seconds"] = end_time - start_time
    exp_config["runtime_hours"] = (end_time - start_time) / 3600
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    print(f"Experiment {exp_id} {'completed' if process.returncode == 0 else 'failed'} "
          f"in {exp_config['runtime_hours']:.2f} hours")
    
    return exp_id, exp_config


def run_grid_search(
    model_name: str,
    model_family: str,
    tasks: List[str],
    output_base_dir: str,
    seeds: List[int] = [42, 43, 44],
    use_lora: bool = False,
    dry_run: bool = False
) -> Dict:
    """Run full hyperparameter grid search"""
    
    paper_config = PAPER_HYPERPARAMETERS[model_family]
    learning_rates = paper_config["learning_rates"]
    
    all_experiments = []
    results = {
        "model_name": model_name,
        "model_family": model_family,
        "tasks": tasks,
        "learning_rates": learning_rates,
        "seeds": seeds,
        "use_lora": use_lora,
        "experiments": {}
    }
    
    # Generate all combinations
    for task, lr, seed in itertools.product(tasks, learning_rates, seeds):
        exp_id, exp_config = run_experiment(
            model_name=model_name,
            model_family=model_family,
            task=task,
            learning_rate=lr,
            output_base_dir=output_base_dir,
            seed=seed,
            use_lora=use_lora,
            dry_run=dry_run
        )
        
        results["experiments"][exp_id] = exp_config
        all_experiments.append((exp_id, exp_config))
    
    # Save grid search summary
    summary_path = os.path.join(output_base_dir, f"grid_search_summary_{model_family}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGrid search completed. Summary saved to: {summary_path}")
    print(f"Total experiments: {len(all_experiments)}")
    
    # Print summary
    completed = sum(1 for _, config in all_experiments if config.get("status") == "completed")
    failed = sum(1 for _, config in all_experiments if config.get("status") == "failed")
    
    print(f"Completed: {completed}, Failed: {failed}")
    
    return results


def create_slurm_job_script(
    model_name: str,
    model_family: str,
    task: str,
    learning_rate: float,
    output_dir: str,
    job_name: str,
    partition: str = "gpu",
    gpus: int = 1,
    mem: str = "80G",
    time: str = "24:00:00"
) -> str:
    """Create SLURM job script for cluster execution"""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err

# Load environment
module load cuda/11.8
source activate mezo_env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Run experiment
cd /home/lei/MeZO

python accelerate_mezo.py \\
    --model_name {model_name} \\
    --dataset {TASK_CONFIGS.get(task, {}).get("dataset", task)} \\
    --text_column {TASK_CONFIGS.get(task, {}).get("text_column", "text")} \\
    --output_dir {output_dir} \\
    --batch_size {PAPER_HYPERPARAMETERS[model_family]["batch_size"]} \\
    --learning_rate {learning_rate} \\
    --zo_eps {PAPER_HYPERPARAMETERS[model_family]["eps"]} \\
    --max_steps {PAPER_HYPERPARAMETERS[model_family]["max_steps"]} \\
    --eval_steps {PAPER_HYPERPARAMETERS[model_family]["eval_steps"]} \\
    --lr_scheduler_type {PAPER_HYPERPARAMETERS[model_family]["lr_scheduler"]} \\
    --memory_logging \\
    --paper_hparams \\
    --model_family {model_family} \\
    --task_name {task}
"""
    
    if TASK_CONFIGS.get(task, {}).get("dataset_config"):
        script_content += f" \\\n    --dataset_config {TASK_CONFIGS[task]['dataset_config']}"
    
    return script_content


def main():
    parser = argparse.ArgumentParser(description="Run MeZO paper reproduction experiments")
    parser.add_argument("--model_name", type=str, required=True, 
                       help="HuggingFace model name")
    parser.add_argument("--model_family", type=str, required=True,
                       choices=["roberta", "opt"],
                       help="Model family (roberta or opt)")
    parser.add_argument("--task", type=str, default=None,
                       help="Single task to run")
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Multiple tasks to run")
    parser.add_argument("--grid_search", action="store_true",
                       help="Run full hyperparameter grid search")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Single learning rate (if not doing grid search)")
    parser.add_argument("--output_dir", type=str, default="./mezo_paper_experiments",
                       help="Base output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                       help="Random seeds to use")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA fine-tuning")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without running")
    parser.add_argument("--create_slurm", action="store_true",
                       help="Create SLURM job scripts instead of running")
    parser.add_argument("--slurm_partition", type=str, default="gpu",
                       help="SLURM partition")
    parser.add_argument("--slurm_gpus", type=int, default=1,
                       help="Number of GPUs for SLURM job")
    
    args = parser.parse_args()
    
    # Determine tasks to run
    if args.grid_search:
        tasks = PAPER_HYPERPARAMETERS[args.model_family]["tasks"]
    elif args.tasks:
        tasks = args.tasks
    elif args.task:
        tasks = [args.task]
    else:
        parser.error("Must specify --task, --tasks, or --grid_search")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== MeZO Paper Reproduction ===")
    print(f"Model: {args.model_name} ({args.model_family})")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")
    print(f"Grid search: {args.grid_search}")
    print(f"Use LoRA: {args.use_lora}")
    print("=" * 35)
    
    if args.create_slurm:
        # Create SLURM job scripts
        learning_rates = PAPER_HYPERPARAMETERS[args.model_family]["learning_rates"]
        if args.learning_rate:
            learning_rates = [args.learning_rate]
        
        script_dir = os.path.join(args.output_dir, "slurm_scripts")
        os.makedirs(script_dir, exist_ok=True)
        
        for task, lr, seed in itertools.product(tasks, learning_rates, args.seeds):
            job_name = f"mezo_{args.model_family}_{task}_lr{lr:.0e}_seed{seed}"
            if args.use_lora:
                job_name += "_lora"
            
            exp_output_dir = os.path.join(args.output_dir, job_name)
            os.makedirs(exp_output_dir, exist_ok=True)
            
            script_content = create_slurm_job_script(
                args.model_name, args.model_family, task, lr, exp_output_dir,
                job_name, args.slurm_partition, args.slurm_gpus
            )
            
            script_path = os.path.join(script_dir, f"{job_name}.sh")
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            print(f"Created SLURM script: {script_path}")
        
        print(f"\nTo submit all jobs:")
        print(f"for script in {script_dir}/*.sh; do sbatch $script; done")
        
    elif args.grid_search:
        # Run full grid search
        run_grid_search(
            model_name=args.model_name,
            model_family=args.model_family,
            tasks=tasks,
            output_base_dir=args.output_dir,
            seeds=args.seeds,
            use_lora=args.use_lora,
            dry_run=args.dry_run
        )
    else:
        # Run individual experiments
        learning_rates = PAPER_HYPERPARAMETERS[args.model_family]["learning_rates"]
        if args.learning_rate:
            learning_rates = [args.learning_rate]
        
        for task, lr, seed in itertools.product(tasks, learning_rates, args.seeds):
            run_experiment(
                model_name=args.model_name,
                model_family=args.model_family,
                task=task,
                learning_rate=lr,
                output_base_dir=args.output_dir,
                seed=seed,
                use_lora=args.use_lora,
                dry_run=args.dry_run
            )


if __name__ == "__main__":
    main()
