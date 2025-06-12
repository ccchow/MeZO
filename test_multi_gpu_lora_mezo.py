#!/usr/bin/env python3
"""
Multi-GPU LoRA MeZO Test Script

This script tests the multi-GPU functionality of your LoRA MeZO trainer.
Run this after successful single GPU testing.
"""

import os
import subprocess
import tempfile
import json
import time

def create_test_dataset(filepath, num_samples=100):
    """Create a larger test dataset for multi-GPU testing"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of data to train effectively.",
        "Transformers have revolutionized the field of artificial intelligence.",
        "Zero-order optimization methods can be memory efficient for large models.",
        "LoRA enables parameter-efficient fine-tuning of language models.",
        "Distributed training allows scaling to multiple GPUs and nodes.",
        "The attention mechanism is a key component of transformer architectures.",
        "Language models can generate coherent and contextually relevant text.",
        "Multi-GPU training can significantly accelerate model training.",
        "Parameter synchronization is crucial for distributed training.",
        "FSDP enables efficient memory usage across multiple devices.",
        "Gradient clipping helps stabilize training with large models.",
        "Learning rate scheduling improves training convergence."
    ]
    
    test_data = []
    for i in range(num_samples):
        text = sample_texts[i % len(sample_texts)]
        test_data.append({"text": f"Sample {i}: {text}"})
    
    with open(filepath, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return filepath

def run_multi_gpu_test():
    """Run multi-GPU LoRA MeZO test"""
    print("🚀 Starting Multi-GPU LoRA MeZO Test")
    
    # Check GPU count
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        gpu_count = int(result.stdout.strip())
        print(f"🔍 Detected {gpu_count} GPUs")
        
        if gpu_count < 2:
            print("⚠️  Need at least 2 GPUs for multi-GPU testing")
            print("   Running single GPU test instead...")
            gpu_count = 1
    except Exception:
        print("⚠️  Could not detect GPU count, assuming single GPU")
        gpu_count = 1
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        dataset_file = os.path.join(temp_dir, "multi_gpu_dataset.json")
        create_test_dataset(dataset_file, num_samples=200)
        print(f"📁 Created test dataset: {dataset_file}")
        
        output_dir = os.path.join(temp_dir, "multi_gpu_output")
        
        if gpu_count >= 2:
            # Multi-GPU command
            cmd = [
                "accelerate", "launch",
                "--num_processes", str(min(gpu_count, 2)),  # Use max 2 GPUs for testing
                "--multi_gpu",
                "accelerate_mezo.py",
                "--model_name", "distilgpt2",
                "--dataset", "json",
                "--dataset_path", dataset_file,
                "--use_lora",
                "--lora_r", "8",
                "--lora_alpha", "16",
                "--batch_size", "4",
                "--max_steps", "20",
                "--max_length", "128",
                "--output_dir", output_dir,
                "--learning_rate", "1e-4",  # Smaller LR for stability
                "--zo_eps", "1e-4"          # Larger epsilon for stability
            ]
            test_name = f"Multi-GPU Test ({min(gpu_count, 2)} GPUs)"
        else:
            # Single GPU fallback
            cmd = [
                "python3", "accelerate_mezo.py",
                "--model_name", "distilgpt2",
                "--dataset", "json",
                "--dataset_path", dataset_file,
                "--use_lora",
                "--lora_r", "8",
                "--lora_alpha", "16",
                "--batch_size", "4",
                "--max_steps", "20",
                "--max_length", "128",
                "--output_dir", output_dir,
                "--learning_rate", "1e-4",
                "--zo_eps", "1e-4"
            ]
            test_name = "Single GPU Test (Fallback)"
        
        print(f"\n{'='*80}")
        print(f"🔥 Running: {test_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd="/home/lei/MeZO", capture_output=True, text=True, timeout=600)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n📊 RESULTS")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Exit code: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ SUCCESS: Multi-GPU test completed!")
                
                # Check output files
                expected_files = ["adapter_config.json", "adapter_model.safetensors", "base_model.txt"]
                missing_files = []
                for file in expected_files:
                    if not os.path.exists(os.path.join(output_dir, file)):
                        missing_files.append(file)
                
                if missing_files:
                    print(f"⚠️  Missing output files: {missing_files}")
                else:
                    print("✅ All expected output files created")
                
                # Show key metrics from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if any(keyword in line for keyword in ['LoRA enabled:', 'Trainable parameter count:', 'Loss', 'Grad']):
                        print(f"   {line.strip()}")
                
            else:
                print("❌ FAILED: Multi-GPU test failed!")
                print("STDOUT:")
                print(result.stdout)
                print("STDERR:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT: Test took too long (>10 minutes)")
        except Exception as e:
            print(f"💥 ERROR: {e}")

def main():
    """Main entry point"""
    print("🎯 Multi-GPU LoRA MeZO Testing")
    print("="*50)
    
    # Check if accelerate is available
    try:
        subprocess.run(['accelerate', '--version'], capture_output=True, check=True)
        print("✅ Accelerate is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Accelerate is not available or not installed")
        print("   Install with: pip install accelerate")
        return 1
    
    # Check if we're in the right directory
    if not os.path.exists("accelerate_mezo.py"):
        print("❌ accelerate_mezo.py not found in current directory")
        return 1
    
    run_multi_gpu_test()
    
    print("\n🎯 Multi-GPU testing complete!")
    print("   Check the output above for results.")
    
    return 0

if __name__ == "__main__":
    exit(main())
