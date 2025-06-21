#!/usr/bin/env python3
"""
MeZO Experiment Launcher

Easy launcher for common MeZO paper reproduction experiments.
Uses predefined configurations from experiment_configs.json.

Usage:
    python launch_experiment.py --config roberta_sst2_reproduction
    python launch_experiment.py --config quick_test
    python launch_experiment.py --list-configs
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

def load_configs():
    """Load experiment configurations"""
    # Try to load paper-exact configs first
    paper_config_path = Path(__file__).parent / "paper_exact_configs.json"
    if paper_config_path.exists():
        with open(paper_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Fallback to original configs
    config_path = Path(__file__).parent / "experiment_configs.json"
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_configs(configs):
    """List available experiment configurations"""
    print("📋 Available experiment configurations:")
    print("=" * 50)
    
    for name, config in configs.items():
        desc = config.get('description', 'No description')
        runtime = config.get('expected_runtime_hours', 'Unknown')
        memory = config.get('expected_memory_gb', 'Unknown')
        
        print(f"\n🔬 {name}")
        print(f"   Description: {desc}")
        print(f"   Runtime: {runtime} hours")
        print(f"   Memory: {memory} GB")
        
        if 'paper_target' in config:
            print(f"   Target: {config['paper_target']}")

def build_command(config):
    """Build command from configuration"""
    command = config.get('command', '')
    args = config.get('args', {})
    
    cmd_parts = command.split()
    
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.extend([f"--{key}", str(value)])
    
    return cmd_parts

def run_experiment(config_name, config, dry_run=False):
    """Run an experiment"""
    print(f"🚀 Running experiment: {config_name}")
    print(f"📝 Description: {config.get('description', 'No description')}")
    
    # Check for runtime/memory expectations
    runtime = config.get('expected_runtime_hours')
    memory = config.get('expected_memory_gb')
    
    if runtime:
        print(f"⏱️  Expected runtime: {runtime} hours")
    if memory:
        print(f"💾 Expected memory: {memory} GB")
    
    # Build command
    cmd = build_command(config)
    
    if not cmd:
        print("❌ Invalid configuration - no command specified")
        return False
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    if dry_run:
        print("🏃 DRY RUN - command not executed")
        return True
    
    # Create output directory if specified
    args = config.get('args', {})
    output_dir = args.get('output_dir')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    # Confirm before running long experiments (skip if --auto flag is set)
    if runtime and runtime > 1 and not dry_run:
        print(f"\n⚠️  This experiment will run for ~{runtime} hours.")
        if os.getenv('MEZO_AUTO_RUN') != '1':
            response = input("Continue? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Experiment cancelled")
                return False
        else:
            print("🤖 Auto-run mode enabled, starting experiment...")
    
    # Run the command
    try:
        print(f"\n🏃 Starting experiment...")
        result = subprocess.run(cmd, check=True)
        print(f"✅ Experiment completed successfully!")
        
        if output_dir:
            print(f"📊 Results saved to: {output_dir}")
            print(f"📈 Analyze with: python analyze_mezo_results.py --results_dir {output_dir} --create_plots")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Experiment error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch MeZO paper reproduction experiments")
    parser.add_argument("--config", type=str, help="Configuration name to run")
    parser.add_argument("--list-configs", action="store_true", help="List available configurations")
    parser.add_argument("--dry-run", action="store_true", help="Show command without running")
    
    args = parser.parse_args()
    
    # Load configurations
    configs = load_configs()
    
    if not configs:
        print("❌ No configurations found")
        return 1
    
    if args.list_configs:
        list_configs(configs)
        return 0
    
    if not args.config:
        print("❌ Please specify a configuration with --config or use --list-configs")
        return 1
    
    if args.config not in configs:
        print(f"❌ Configuration '{args.config}' not found")
        print(f"Available: {', '.join(configs.keys())}")
        return 1
    
    # Run the experiment
    config = configs[args.config]
    success = run_experiment(args.config, config, dry_run=args.dry_run)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
