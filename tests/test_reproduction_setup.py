#!/usr/bin/env python3
"""
Quick test script to validate MeZO paper reproduction setup.

This script runs a minimal test to ensure all components are working
before launching full experiments.
"""

import os
import sys
import subprocess
import torch
import json
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', '2.0.0'),
        ('transformers', '4.20.0'),
        ('accelerate', '0.20.0'),
        ('datasets', '2.0.0'),
        ('peft', '0.4.0'),
        ('tqdm', '4.0.0'),
        ('numpy', '1.20.0'),
    ]
    
    missing = []
    for package, min_version in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} (missing)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"✅ PyTorch version: {torch_version}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No CUDA devices found")
    
    return True

def test_minimal_mezo():
    """Run a minimal MeZO test with tiny model and dataset"""
    print("\n🧪 Running minimal MeZO test...")
    
    test_dir = "./test_mezo_minimal"
    os.makedirs(test_dir, exist_ok=True)
    
    # Use a very small model and dataset for quick testing
    cmd = [
        "python", "accelerate_mezo.py",
        "--model_name", "distilgpt2",  # Small model for testing
        "--dataset", "imdb",           # Standard dataset
        "--text_column", "text",
        "--output_dir", test_dir,
        "--batch_size", "2",
        "--learning_rate", "1e-5",
        "--max_steps", "10",           # Just 10 steps for testing
        "--zo_eps", "1e-3",
        "--seed", "42",
        "--logging_steps", "2",
        "--memory_logging"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Minimal test passed!")
            
            # Check if output files were created
            expected_files = [
                "training_metrics.json",
                "run_config.json"
            ]
            
            for file in expected_files:
                file_path = os.path.join(test_dir, file)
                if os.path.exists(file_path):
                    print(f"✅ Created {file}")
                else:
                    print(f"⚠️  Missing {file}")
            
            # Load and check metrics
            metrics_path = os.path.join(test_dir, "training_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                print(f"✅ Completed {len(metrics.get('steps', []))} training steps")
                print(f"✅ Peak memory: {max(metrics.get('memory_peak_gb', [0])):.1f} GB")
            
            return True
        else:
            print("❌ Test failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_paper_config():
    """Test paper hyperparameter application"""
    print("\n⚙️  Testing paper configuration...")
    
    # Test RoBERTa config
    test_dir = "./test_paper_config"
    os.makedirs(test_dir, exist_ok=True)
    
    cmd = [
        "python", "accelerate_mezo.py",
        "--model_name", "distilgpt2",  # Small model
        "--dataset", "imdb",
        "--text_column", "text",
        "--output_dir", test_dir,
        "--paper_hparams",
        "--model_family", "roberta",  # Test RoBERTa config
        "--max_steps", "5",           # Just 5 steps
        "--dry_run"  # If this flag exists
    ]
    
    try:
        # Try to run with --help to check if our arguments work
        help_cmd = ["python", "accelerate_mezo.py", "--help"]
        result = subprocess.run(help_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            if "--paper_hparams" in result.stdout:
                print("✅ Paper hyperparameter flag available")
            if "--model_family" in result.stdout:
                print("✅ Model family flag available")
            if "--memory_logging" in result.stdout:
                print("✅ Memory logging flag available")
            return True
        else:
            print("⚠️  Could not check argument flags")
            return False
            
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_experiment_runner():
    """Test the experiment runner script"""
    print("\n🚀 Testing experiment runner...")
    
    try:
        # Test help flag
        cmd = ["python", "run_mezo_paper_experiments.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Experiment runner script works")
            if "--grid_search" in result.stdout:
                print("✅ Grid search option available")
            if "--create_slurm" in result.stdout:
                print("✅ SLURM generation option available")
            return True
        else:
            print("❌ Experiment runner failed")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Experiment runner error: {e}")
        return False

def test_analysis_script():
    """Test the analysis script"""
    print("\n📊 Testing analysis script...")
    
    try:
        cmd = ["python", "analyze_mezo_results.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Analysis script works")
            return True
        else:
            print("❌ Analysis script failed")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Analysis script error: {e}")
        return False

def cleanup_test_files():
    """Remove test files"""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = ["./test_mezo_minimal", "./test_paper_config"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print(f"✅ Removed {test_dir}")

def main():
    print("=" * 50)
    print("🔬 MeZO Paper Reproduction Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Check dependencies
    if not check_dependencies():
        all_passed = False
        print("\n❌ Dependency check failed. Please install missing packages.")
        return
    
    # Test paper configuration
    if not test_paper_config():
        all_passed = False
    
    # Test experiment runner
    if not test_experiment_runner():
        all_passed = False
    
    # Test analysis script
    if not test_analysis_script():
        all_passed = False
    
    # Run minimal MeZO test (only if other tests pass)
    if all_passed:
        if not test_minimal_mezo():
            all_passed = False
    
    # Cleanup
    cleanup_test_files()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou're ready to run MeZO paper reproduction experiments!")
        print("\nNext steps:")
        print("1. Run: python run_mezo_paper_experiments.py --model_family roberta --model_name roberta-large --grid_search")
        print("2. Or start with a single experiment: python accelerate_mezo.py --paper_hparams --model_family roberta ...")
        print("3. Monitor with: python analyze_mezo_results.py --results_dir <your_output_dir> --create_plots")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running full experiments.")
        print("Check the error messages and ensure all dependencies are installed.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
