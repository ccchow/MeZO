import argparse
import json
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
import os
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from datasets import (
    load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.optimization import get_scheduler
from accelerate import Accelerator
from peft import (                        # PEFT = LoRA for HF models
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)


@dataclass
class Arguments:
    model_name: str
    dataset: str
    text_column: str = "text"
    output_dir: str = "./output"
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 3
    max_length: int = 512
    zo_eps: float = 1e-3
    seed: int = 42
    # If set, limits the number of training steps
    max_steps: Optional[int] = None
    dataset_path: Optional[str] = None  # Path to dataset file
    streaming: bool = False  # Enable dataset streaming
    dataset_split: str = "train"  # Dataset split to use
    dataset_config: Optional[str] = None  # Dataset configuration name
    # Limit training samples for streaming
    max_train_samples: Optional[int] = None
    # Probability for masking tokens (if using MLM)
    mlm_probability: float = 0.15
    block_size: Optional[int] = None  # Block size for grouping texts
    # NEW ────────────────────────────────────────────────────────────
    # LoRA‑specific flags (defaults match common practice)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: Optional[List[str]] = None  # e.g. ["q_proj", "v_proj"]
    # When resuming / inference, path to an existing adapter
    lora_path: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_epochs if set)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset file (for JSON datasets)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable dataset streaming for large datasets"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (useful for streaming)"
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability for masking tokens (if using MLM)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for grouping texts"
    )
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA fine‑tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="Module names to apply LoRA to")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Load an existing LoRA adapter from this folder")

    args = parser.parse_args()
    return Arguments(**vars(args))


def prepare_dataloader(args, tokenizer):
    # Handle different dataset types
    if args.dataset == "json":
        # Load from JSON file
        # Determine JSON file path
        if args.dataset_path and os.path.exists(args.dataset_path):
            json_path = args.dataset_path
        else:
            # Fall back to environment variable or common locations
            json_path = os.environ.get('DATASET_PATH', 'dataset.json')
            if not os.path.exists(json_path):
                # Try common locations
                possible_paths = [
                    'dataset.json',
                    'data.json',
                    '/tmp/dataset.json']
                if args.dataset_path:
                    possible_paths.append(args.dataset_path)
                for path in possible_paths:
                    if os.path.exists(path):
                        json_path = path
                        break
                else:
                    raise ValueError(
                        f"JSON dataset file not found. Tried: {possible_paths}")

        print(f"Loading JSON dataset from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'train' in data:
            train_data = data['train']
        elif isinstance(data, list):
            train_data = data
        else:
            raise ValueError(
                "JSON file must contain either a list of samples or a dict with 'train' key")

        # Convert to Dataset
        dataset = {"train": Dataset.from_list(train_data)}

    else:
        # Load Hugging Face dataset with optional streaming and config
        load_kwargs = {
            "streaming": args.streaming
        }

        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config

        dataset = load_dataset(args.dataset, **load_kwargs)

    # Handle dataset splits
    if args.streaming:
        # For streaming datasets, we work with the specified split directly
        if isinstance(dataset, (IterableDatasetDict, DatasetDict)):
            try:
                train_dataset = dataset[args.dataset_split]
            except KeyError as exc:
                available_splits = list(dataset.keys())
                raise ValueError(
                    f"Split '{args.dataset_split}' not found. "
                    f"Available splits: {available_splits}"
                ) from exc
        else:
            # Single dataset case
            train_dataset = dataset

        # Apply sample limit for streaming datasets
        if (args.max_train_samples and
                isinstance(train_dataset, IterableDataset)):
            train_dataset = train_dataset.take(args.max_train_samples)

        # Convert streaming dataset to regular dataset for easier handling
        print(f"Loading streaming dataset split '{args.dataset_split}'...")
        if args.max_train_samples:
            print(f"Limiting to {args.max_train_samples} samples")

        # Collect samples from streaming dataset
        samples = []

        for example in enumerate(train_dataset):
            samples.append(example[1])  # example[1] is the actual data

            # Break if we've collected enough samples
            if (args.max_train_samples and
                    len(samples) >= args.max_train_samples):
                break

            # For very large limits, break early to avoid memory issues
            if len(samples) >= 10000:  # Safety limit
                print("Reached safety limit of 10000 samples")
                break

        print(f"Collected {len(samples)} samples from streaming dataset")
        dataset = {"train": Dataset.from_list(samples)}

    else:
        # For non-streaming datasets, use the standard approach
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            try:
                train_split = dataset[args.dataset_split]
            except KeyError as exc:
                available_splits = list(dataset.keys())
                raise ValueError(
                    f"Split '{args.dataset_split}' not found. "
                    f"Available splits: {available_splits}"
                ) from exc
        else:
            # Single dataset case
            train_split = dataset

        # Apply sample limit for regular datasets
        if args.max_train_samples and isinstance(train_split, Dataset):
            max_samples = min(args.max_train_samples, len(train_split))
            train_split = train_split.select(range(max_samples))
            print(f"Limited dataset to {max_samples} samples")

        # Ensure we have the right split structure
        if args.dataset_split != "train" or not isinstance(dataset, dict):
            dataset = {"train": train_split}

    if "train" not in dataset:
        raise ValueError("Dataset must have a train split")

    # Set block size for text processing
    if args.block_size is None:
        block_size = args.max_length
    else:
        block_size = min(args.block_size, args.max_length)

    def tokenize_fn(examples):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            max_length=block_size,
            padding=False  # Don't pad here, let the collator handle it
        )

    # Handle tokenization for dict vs DatasetDict
    if isinstance(dataset, dict):
        tokenized = {}
        for split_name, split_dataset in dataset.items():
            if isinstance(split_dataset, Dataset):
                tokenized[split_name] = split_dataset.map(
                    tokenize_fn, batched=True)
    else:
        tokenized = dataset.map(tokenize_fn, batched=True)

    # Remove text columns to avoid issues with the dataloader
    if isinstance(tokenized, dict) and "train" in tokenized:
        train_dataset = tokenized["train"]
        if isinstance(train_dataset, Dataset):
            columns_to_remove = [col for col in train_dataset.column_names
                                 if col not in ["input_ids", "attention_mask"]]
            if columns_to_remove:
                for split_name in tokenized:
                    if isinstance(tokenized[split_name], Dataset):
                        tokenized[split_name] = tokenized[split_name].remove_columns(
                            columns_to_remove)

    # Get the train dataset
    if isinstance(tokenized, dict):
        train_dataset = tokenized["train"]
    else:
        train_dataset = tokenized
        tokenized = {"train": train_dataset}

    print(f"Dataset size: {len(train_dataset)}")

    # Filter out empty sequences
    def is_valid_example(example):
        return len(example["input_ids"]) > 0

    train_dataset = train_dataset.filter(is_valid_example)
    print(f"Final dataset size after filtering empty sequences: {len(train_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError(
            "No valid examples remaining after filtering empty sequences")

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to True for MLM with BERT-like models
        return_tensors="pt"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True  # Drop incomplete batches to avoid size mismatches
    )

    return dataloader


def zo_perturb_parameters(named_params, eps, scaling_factor, z_vectors):
    """Apply perturbation using pre-generated z vectors"""
    for (_, param), z in zip(named_params, z_vectors):
        param.data.add_(z, alpha=scaling_factor * eps)


def zo_forward(model, inputs):
    model.eval()
    with torch.inference_mode():
        try:
            outputs = model(**inputs)
            loss = outputs.loss
            return loss.detach()
        except RuntimeError as e:
            if "batch_size" in str(e):
                print(f"Batch size error in forward pass: {e}")
                # Return a dummy loss to continue training
                return torch.tensor(
                    0.0, device=next(
                        model.parameters()).device)
            else:
                raise e


def zo_step(model, inputs, named_params, eps, device):
    zo_random_seed = np.random.randint(1000000000)
    
    # Perturb +eps
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=1)
    loss1 = zo_forward(model, inputs)
    
    # Perturb -2*eps (to get to -eps from +eps)
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=-2)
    loss2 = zo_forward(model, inputs)
    
    # Restore to original
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=1)
    
    projected_grad = ((loss1 - loss2) / (2 * eps)).item();
    
    return loss1, projected_grad, zo_random_seed  # Return seed, not z_vectors

def perturb_parameters(named_params, eps, random_seed, scaling_factor):
    """Perturb parameters using a specific random seed"""
    torch.manual_seed(random_seed)
    
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.shape,
                        device=param.device, dtype=param.dtype)
        param.data.add_(z, alpha=scaling_factor * eps)


def synchronize_params(model, accelerator):
    """Synchronize model parameters across all processes"""
    if accelerator.num_processes > 1:
        # Use accelerator's built-in synchronization
        with accelerator.no_sync(model):
            pass
        # Force synchronization by doing a dummy forward/backward
        dummy_input = next(
            iter(
                model.parameters())).new_zeros(
            1, requires_grad=True)
        (dummy_input.sum() * 0).backward()
        accelerator.wait_for_everyone()


def zo_update(named_params, lr_scheduler, projected_grad, zo_random_seed, lr, weight_decay):
    """Update parameters by regenerating the same z vectors"""
    projected_grad = max(min(projected_grad, 1000.0), -1000.0)
    current_lr = lr_scheduler.get_last_lr()[0] * lr
    
    # Reset to same seed to regenerate same z vectors
    torch.manual_seed(zo_random_seed)
    
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.shape,
                        device=param.device, dtype=param.dtype)
        
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data.mul_(1 - current_lr * weight_decay)
            param.data.add_(z, alpha=-current_lr * projected_grad)
        else:
            param.data.add_(z, alpha=-current_lr * projected_grad)
    
    lr_scheduler.step()


def zo_step_with_sync(model, inputs, named_params, eps, device, accelerator):
    """ZO step with proper synchronization for distributed training"""
    # Ensure all processes use the same random seed
    zo_random_seed = np.random.randint(1000000000)
    if accelerator.num_processes > 1:
        # Create tensor on CPU first, then move to device
        zo_random_seed_tensor = torch.tensor(zo_random_seed, dtype=torch.long)
        zo_random_seed_tensor = zo_random_seed_tensor.to(device)
        zo_random_seed_tensor = accelerator.gather(zo_random_seed_tensor)[0]
        zo_random_seed = zo_random_seed_tensor.item()

    # Use the synchronized seed for all operations
    torch.manual_seed(zo_random_seed)

    # Synchronize before perturbation
    accelerator.wait_for_everyone()

    # First perturbation: +eps
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=1)
    accelerator.wait_for_everyone()

    loss1 = zo_forward(model, inputs)

    # Second perturbation: -2*eps (to get to -eps)
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=-2)
    accelerator.wait_for_everyone()

    loss2 = zo_forward(model, inputs)

    # Restore to original: +eps
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=1)
    accelerator.wait_for_everyone()

    # Calculate projected gradient
    projected_grad = ((loss1 - loss2) / (2 * eps)).item()

    return loss1, projected_grad, zo_random_seed


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()

    # Load tokenizer and configure padding token properly
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token for models that don't have one
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Add a new pad token if no eos_token exists
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

    # Load model for causal language modeling
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")

    model.gradient_checkpointing_enable()

    # ─── LoRA integration ───────────────────────────────────────────────
    if args.use_lora:
        if args.lora_path is not None and os.path.isdir(args.lora_path):
            # Resume / inference on an existing adapter
            print(f"Loading LoRA adapter from {args.lora_path}")
            model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                is_trainable=True   # keep adapter params requires_grad=True
            )
        else:
            # Fresh LoRA setup
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target,
                inference_mode=False,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            print(
                f"LoRA enabled ⇒ r={args.lora_r}, "
                f"alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
                f"targets={args.lora_target}"
            )

        # Note: PEFT automatically freezes base parameters, but LoRA params stay trainable
        # No need for manual freezing - PEFT handles this correctly
    # ────────────────────────────────────────────────────────────────────


    # Apply FSDP to transformer layers only if running in distributed mode
    # if (torch.distributed.is_available() and
    #         torch.distributed.is_initialized()):
    #     if (hasattr(model, 'transformer') and
    #             hasattr(model.transformer, 'h')):
    #         # GPT-2, GPT-Neo, GPT-J, etc.
    #         for layer in model.transformer.h:
    #             fully_shard(layer)
    #     elif (hasattr(model, 'model') and
    #           hasattr(model.model, 'layers')):
    #         # LLaMA, Mistral, Phi, etc.
    #         for layer in model.model.layers:
    #             fully_shard(layer)
    #     elif (hasattr(model, 'encoder') and
    #           hasattr(model.encoder, 'layer')):
    #         # BERT, RoBERTa, etc.
    #         for layer in model.encoder.layer:
    #             fully_shard(layer)

    #     fully_shard(model)

    # Resize token embeddings if we added new tokens
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Update model config to match tokenizer padding token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare data loader
    train_dataloader = prepare_dataloader(args, tokenizer)

    print(f"Model loaded for text generation: {args.model_name}")

    # Create optimizer and scheduler (needed for lr_scheduler.step())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=1.0,
        weight_decay=0.0)


    # Calculate total training steps
    if args.max_steps is not None:
        total_training_steps = args.max_steps
    else:
        total_training_steps = len(train_dataloader) * args.num_epochs

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )

    # Prepare with accelerator BEFORE getting named parameters
    # model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, lr_scheduler
    # )

    # Get named parameters AFTER accelerator.prepare()
    named_params = [(n, p)
                    for n, p in model.named_parameters() if p.requires_grad]

    # Get the device from the model
    device = next(model.parameters()).device

    # Set model to training mode
    model.train()

    global_step = 0
    max_steps = args.max_steps if args.max_steps is not None else float('inf')

    for epoch in range(args.num_epochs):
        if global_step >= max_steps:
            break

        for step, batch in enumerate(train_dataloader):
            if global_step >= max_steps:
                break

            try:
                # Move batch to the correct device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Use synchronized version for multi-GPU
                if accelerator.num_processes > 1:
                    loss, grad, seed = zo_step_with_sync(
                        model, batch, named_params, args.zo_eps, device, accelerator)
                else:
                    loss, grad, seed = zo_step(
                        model, batch, named_params, args.zo_eps, device)

                # Skip update if gradient is NaN or too large
                if not (
                    torch.isfinite(
                        torch.tensor(grad)) and abs(grad) < 1e6):
                    if accelerator.is_main_process and step % 10 == 0:
                        print(f"Epoch {epoch} Step {step} Global {global_step} Skipping update - invalid gradient: {grad}")
                    continue

                zo_update(
                    named_params,
                    lr_scheduler,
                    grad,
                    seed,  # Pass seed instead of z_vectors
                    args.learning_rate,
                    args.weight_decay)

                # Synchronize after update in multi-GPU setting
                if accelerator.num_processes > 1:
                    accelerator.wait_for_everyone()

                global_step += 1

                if accelerator.is_main_process and step % 10 == 0:
                    print(f"Epoch {epoch} Step {step} Global {global_step} Loss {loss.item():.4f} Grad {grad:.4f}")

            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error at step {step}: {e}")
                continue

        # Keep this - ensures all processes finish the epoch
        accelerator.wait_for_everyone()

    # Save the model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        if args.use_lora:
            # Save only the adapter (compact, recommended)
            unwrapped_model.save_pretrained(args.output_dir)
            # Record the base model id for convenience
            with open(os.path.join(args.output_dir, "base_model.txt"), "w") as f:
                f.write(args.model_name + "\n")
            print(f"LoRA adapter saved to {args.output_dir}")
        else:
            # Original full‑model save
            unwrapped_model.save_pretrained(args.output_dir)
            print(f"Model saved to {args.output_dir}")

        tokenizer.save_pretrained(args.output_dir)
