#!/usr/bin/env python3
"""
End-to-end validation script for fine-tuning models on FineWeb dataset
and evaluating text generation quality.
"""

import os
import subprocess
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import numpy as np
import argparse
from datetime import datetime


def calculate_perplexity(model, tokenizer, dataset, batch_size=8, max_samples=1000):
    """Calculate perplexity on a dataset"""
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare data
    def tokenize_fn(examples):
        # Handle both single text and batch of texts
        texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            max_length=512,
            padding=False,
            return_tensors=None  # Don't return tensors yet
        )
        return tokenized
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    # Filter valid examples
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    
    # Limit samples
    if len(tokenized) > max_samples:
        tokenized = tokenized.select(range(max_samples))
    
    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
    
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Get the loss
            loss = outputs.loss
            
            # Count non-padding tokens
            attention_mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def generate_samples(model, tokenizer, prompts, max_length=100, temperature=0.8, top_p=0.9):
    """Generate text samples from prompts"""
    model.eval()
    device = next(model.parameters()).device
    
    generated_texts = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def run_fine_tuning(model_name, output_dir, max_steps=100, batch_size=4, learning_rate=1e-5, zo_eps=1e-3):
    """Run fine-tuning using accelerate_mezo.py"""
    cmd = [
        "python", "accelerate_mezo.py",
        "--model_name", model_name,
        "--dataset", "HuggingFaceFW/fineweb",
        "--dataset_config", "sample-10BT",  # Use smaller sample for validation
        "--output_dir", output_dir,
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--zo_eps", str(zo_eps),
        "--max_steps", str(max_steps),
        "--max_train_samples", "1000",  # Limit samples for faster validation
        "--streaming",  # Enable streaming for large dataset
        "--max_length", "512"
    ]
    
    print(f"Running fine-tuning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Fine-tuning output:")
        print(result.stdout)
        if result.stderr:
            print("Fine-tuning errors/warnings:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Fine-tuning failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuning on FineWeb dataset")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./fineweb_validation", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--zo_eps", type=float, default=1e-3, help="ZO epsilon")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only evaluate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    model_output_dir = os.path.join(run_dir, "model")
    
    # Step 1: Fine-tune the model
    if not args.skip_training:
        print("\n=== Step 1: Fine-tuning model on FineWeb ===")
        success = run_fine_tuning(
            args.model_name,
            model_output_dir,
            args.max_steps,
            args.batch_size,
            args.learning_rate,
            args.zo_eps
        )
        
        if not success:
            print("Fine-tuning failed. Exiting.")
            return
    else:
        print("\n=== Skipping training, using existing model ===")
        model_output_dir = os.path.join(args.output_dir, "model")
    
    # Step 2: Load models for evaluation
    print("\n=== Step 2: Loading models for evaluation ===")
    
    # Load base model
    print(f"Loading base model: {args.model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Set padding token if needed
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from: {model_output_dir}")
    ft_tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    ft_model = AutoModelForCausalLM.from_pretrained(model_output_dir)
    
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)
    ft_model = ft_model.to(device)
    
    # Step 3: Load evaluation dataset
    print("\n=== Step 3: Loading evaluation dataset ===")
    eval_dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        streaming=True
    )
    
    # Convert to regular dataset for evaluation
    eval_samples = []
    for i, sample in enumerate(eval_dataset):
        eval_samples.append(sample)
        if i >= 500:  # Use 500 samples for evaluation
            break
    
    from datasets import Dataset
    eval_dataset = Dataset.from_list(eval_samples)
    
    # Step 4: Calculate perplexity
    print("\n=== Step 4: Calculating perplexity ===")
    
    print("Calculating base model perplexity...")
    base_perplexity = calculate_perplexity(base_model, base_tokenizer, eval_dataset)
    
    print("Calculating fine-tuned model perplexity...")
    ft_perplexity = calculate_perplexity(ft_model, ft_tokenizer, eval_dataset)
    
    # Step 5: Generate sample texts
    print("\n=== Step 5: Generating sample texts ===")
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to successful machine learning is",
        "In the world of technology,",
        "Scientists have discovered that"
    ]
    
    print("\nBase model generations:")
    base_generations = generate_samples(base_model, base_tokenizer, prompts)
    for i, (prompt, generation) in enumerate(zip(prompts, base_generations)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generation: {generation}")
    
    print("\n" + "="*50 + "\n")
    
    print("Fine-tuned model generations:")
    ft_generations = generate_samples(ft_model, ft_tokenizer, prompts)
    for i, (prompt, generation) in enumerate(zip(prompts, ft_generations)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generation: {generation}")
    
    # Step 6: Save results
    print("\n=== Step 6: Saving results ===")
    
    results = {
        "model_name": args.model_name,
        "training_config": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "zo_eps": args.zo_eps
        },
        "evaluation_results": {
            "base_perplexity": float(base_perplexity),
            "finetuned_perplexity": float(ft_perplexity),
            "perplexity_improvement": float(base_perplexity - ft_perplexity),
            "perplexity_improvement_percent": float((base_perplexity - ft_perplexity) / base_perplexity * 100)
        },
        "generation_samples": {
            "prompts": prompts,
            "base_generations": base_generations,
            "finetuned_generations": ft_generations
        }
    }
    
    results_file = os.path.join(run_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Base model perplexity: {base_perplexity:.2f}")
    print(f"Fine-tuned model perplexity: {ft_perplexity:.2f}")
    print(f"Improvement: {base_perplexity - ft_perplexity:.2f} ({(base_perplexity - ft_perplexity) / base_perplexity * 100:.1f}%)")
    
    if ft_perplexity < base_perplexity:
        print("\n✅ Fine-tuning successful! The model improved on the FineWeb dataset.")
    else:
        print("\n⚠️  Fine-tuning did not improve perplexity. Consider adjusting hyperparameters.")


if __name__ == "__main__":
    main()
