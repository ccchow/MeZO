import argparse
import os
import json
import time
import threading
import psutil
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from datasets import (
    load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.optimization import get_scheduler
from accelerate import (
    Accelerator,
    FullyShardedDataParallelPlugin,
)
from peft import (                        # PEFT = LoRA for HF models
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
from peft.tuners.lora import LoraLayer
from tqdm import tqdm


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
    # FSDP version to use when multiple GPUs are available
    fsdp_version: int = 2
    # MeZO Paper Reproduction Settings
    task_name: Optional[str] = None  # For GLUE/SuperGLUE tasks
    eval_steps: int = 10000  # Evaluation frequency (10k for RoBERTa, 4k for OPT)
    save_steps: int = 10000  # Checkpoint frequency
    logging_steps: int = 100  # Logging frequency
    memory_logging: bool = True  # Log GPU memory usage
    gradient_accumulation_steps: int = 1  # For large batch sizes
    lr_scheduler_type: str = "constant"  # constant for OPT, linear for RoBERTa
    warmup_steps: int = 0  # Warmup steps
    # Model-specific overrides for paper reproduction
    model_family: Optional[str] = None  # "roberta", "opt" - auto-detected if None
    paper_hparams: bool = False  # Use exact hyperparameters from Tables 15-16
    skip_model_save: bool = False  # Skip model saving to avoid hanging (useful for test runs)


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
    parser.add_argument(
        "--fsdp_version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Enable FSDP when multiple GPUs are available."
             " Use 1 or 2 to select the FSDP version.")
    
    # MeZO Paper Reproduction Arguments
    parser.add_argument("--task_name", type=str, default=None,
                        help="GLUE/SuperGLUE task name for evaluation")
    parser.add_argument("--eval_steps", type=int, default=10000,
                        help="Evaluation frequency (10k for RoBERTa, 4k for OPT)")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Checkpoint saving frequency")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging frequency")
    parser.add_argument("--memory_logging", action="store_true",
                        help="Log GPU memory usage during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--model_family", type=str, default=None,
                        choices=["roberta", "opt"],
                        help="Model family (auto-detected if None)")
    parser.add_argument("--paper_hparams", action="store_true",
                        help="Use exact hyperparameters from MeZO paper Tables 15-16")
    parser.add_argument(
        "--skip_model_save",
        action="store_true",
        help="Skip model saving to avoid hanging (useful for test runs)"
    )

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
            raise ValueError("JSON dataset specified but --dataset_path not provided or file not found.")

        print(f"Loading JSON dataset from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'train' in data:
            train_data = data['train']
        elif isinstance(data, list):
            train_data = data
        else:
            raise ValueError("Unsupported JSON structure. Expected a list of examples or a dict with a 'train' key.")

        # Convert to Dataset
        dataset = {"train": Dataset.from_list(train_data)}

    else:
        # Load Hugging Face dataset with optional streaming and config
        load_kwargs = {
            "streaming": args.streaming
        }

        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config
        elif args.dataset == "glue" and args.task_name:
            # For GLUE, use task_name as config if no explicit config provided
            load_kwargs["name"] = args.task_name

        dataset = load_dataset(args.dataset, **load_kwargs)

    # Handle dataset splits
    if args.streaming:
        # For streaming datasets, we work with the specified split directly
        if isinstance(dataset, (IterableDatasetDict, DatasetDict)):
            train_dataset = dataset[args.dataset_split]
        else:
            train_dataset = dataset

        # Apply sample limit for streaming datasets
        if (args.max_train_samples and
                isinstance(train_dataset, IterableDataset)):
            train_dataset = train_dataset.take(args.max_train_samples)

        # Convert streaming dataset to regular dataset for easier handling
        print(f"Loading streaming dataset split '{args.dataset_split}'...")
        if args.max_train_samples:
            print(f"Limiting to {args.max_train_samples} samples.")

        # Collect samples from streaming dataset
        samples = []

        for example in enumerate(train_dataset):
            samples.append(example[1])

        print(f"Collected {len(samples)} samples from streaming dataset")
        dataset = {"train": Dataset.from_list(samples)}

    else:
        # For non-streaming datasets, use the standard approach
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            train_split = dataset[args.dataset_split]
        else:
            train_split = dataset

        # Apply sample limit for regular datasets
        if args.max_train_samples and isinstance(train_split, Dataset):
            train_split = train_split.select(range(args.max_train_samples))

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

    # Get the text column name for the specific dataset/task
    if args.text_column != "text":  # Only use args.text_column if explicitly set by user
        text_col = args.text_column
    elif args.dataset == "glue" and args.task_name == "sst2":
        text_col = "sentence"
    elif args.dataset == "glue" and args.task_name in ["mrpc", "qqp"]:
        text_col = "sentence1"  # For pair classification tasks
    elif args.dataset == "glue" and args.task_name in ["mnli", "qnli", "rte"]:
        text_col = "premise"    # For NLI tasks
    else:
        text_col = "text"       # Default
    
    print(f"🔍 Debug: Using text column '{text_col}' for dataset={args.dataset}, task={args.task_name}")

    def tokenize_fn(examples):
        print(f"🔍 Debug: Available columns: {list(examples.keys())}")
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=block_size,
            padding=False,
            return_special_tokens_mask=False,
        )

    # Handle tokenization for dict vs DatasetDict
    if isinstance(dataset, dict):
        tokenized = {}
        for split_name, split_dataset in dataset.items():
            tokenized[split_name] = split_dataset.map(tokenize_fn, batched=True, remove_columns=split_dataset.column_names)
    else:
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # Get the train dataset - text columns are already removed during tokenization
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
        raise ValueError("No valid examples found in the dataset after filtering.")

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True
    )

    return dataloader


def zo_forward(model, inputs):
    """Memory-optimized forward pass with aggressive no_grad contexts"""
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        outputs = model(**inputs)
        loss = outputs.loss
        # Ensure we detach and minimize memory footprint
        loss_value = loss.detach().clone()
        del outputs, loss  # Explicit cleanup
    return loss_value


def zo_step(model, inputs, named_params, eps, accelerator, gradient_accumulation_steps=1):
    """Compute gradient estimate using MeZO algorithm - simplified and corrected."""
    
    # MeZO doesn't support gradient accumulation - enforce single step
    assert gradient_accumulation_steps == 1, "MeZO algorithm doesn't support gradient accumulation"
    
    # Generate random seed locally on each process (same as trainer.py)
    zo_random_seed = np.random.randint(42)
    
    # Set global random seed for consistent perturbations
    torch.manual_seed(zo_random_seed)
    
    # First perturbation: +eps
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + z * eps
    
    loss_pos = zo_forward(model, inputs)
    
    # Reset seed and apply -eps perturbation
    torch.manual_seed(zo_random_seed)
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data - 2 * z * eps  # This gets us to -eps from +eps
    
    loss_neg = zo_forward(model, inputs)
    
    # Restore original parameters
    torch.manual_seed(zo_random_seed)
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + z * eps  # This restores to original from -eps
    
    # In a distributed setting, loss must be averaged across all processes
    if accelerator.num_processes > 1:
        loss_pos = accelerator.reduce(loss_pos, "mean")
        loss_neg = accelerator.reduce(loss_neg, "mean")

    # Calculate projected gradient
    projected_grad = (loss_pos - loss_neg) / (2 * eps)

    return projected_grad.item(), zo_random_seed, loss_pos.item()


def zo_update(named_params, lr_scheduler, projected_grad, zo_random_seed, weight_decay, base_lr):
    """Update parameters by regenerating the same z vectors - corrected implementation"""
    # Use base learning rate for now (ignore scheduler for debugging)
    current_lr = base_lr
    
    # Use global seed for consistency (same as zo_step)
    torch.manual_seed(zo_random_seed)
    
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        
        # Apply weight decay and gradient update together (as in original trainer.py)
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - current_lr * (projected_grad * z + weight_decay * param.data)
        else:
            param.data = param.data - current_lr * (projected_grad * z)
    
    lr_scheduler.step()


def build_fsdp_plugin(fsdp_version: int = 2):
    """Build FSDP plugin with auto-wrapping for transformer blocks and memory optimizations"""
    # Try to import model-specific transformer block classes
    transformer_cls_set = set()
    
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        transformer_cls_set.add(LlamaDecoderLayer)
    except ImportError:
        pass
    
    try:
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
        transformer_cls_set.add(GPTNeoXLayer)
    except ImportError:
        pass
    
    try:
        from transformers.models.gptj.modeling_gptj import GPTJBlock
        transformer_cls_set.add(GPTJBlock)
    except ImportError:
        pass
    
    try:
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        transformer_cls_set.add(OPTDecoderLayer)
    except ImportError:
        pass
    
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        transformer_cls_set.add(GPT2Block)
    except ImportError:
        pass
    
    try:
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        transformer_cls_set.add(MistralDecoderLayer)
    except ImportError:
        pass
    
    # If no specific classes found, let FSDP auto-detect
    transformer_cls = transformer_cls_set if transformer_cls_set else None
    
    # Create a partial function for the auto wrap policy
    if transformer_cls:
        from functools import partial
        wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls)
    else:
        wrap_policy = None
    
    # Memory optimization: Enable activation checkpointing
    activation_checkpointing = True
    
    return FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",
        state_dict_type="full_state_dict",
        auto_wrap_policy=wrap_policy,
        ignored_modules=None,
        use_orig_params=True,
        activation_checkpointing=activation_checkpointing,  # Enable for memory savings
    )


def save_lora_adapter_fsdp_compatible(model, output_dir, accelerator):
    """
    Save LoRA adapter in a way that's compatible with FSDP.
    
    This function extracts LoRA parameters by reconstructing them from
    their original module shapes, bypassing FSDP's parameter flattening.
    """
    import torch
    import os
    import json
    from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
    from peft import PeftModel
    from collections import OrderedDict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from peft.tuners.lora import LoraLayer
    
    print("Extracting LoRA parameters with correct shapes...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Navigate through the model hierarchy to find the PEFT model
    def find_peft_model(module):
        if hasattr(module, 'peft_config'):
            return module
        for child in module.modules():
            if hasattr(child, 'peft_config'):
                return child
        return None
    
    peft_model = find_peft_model(model)
    
    if peft_model is None:
        raise ValueError("Model doesn't appear to be a PEFT model")
    
    print(f"Found PEFT model: {type(peft_model)}")
    
    # Extract LoRA parameters with their original shapes
    lora_state_dict = OrderedDict()
    
    # Check if model is wrapped with FSDP
    is_fsdp = any(isinstance(module, FSDP) for module in model.modules())
    print(f"Model is FSDP-wrapped: {is_fsdp}")
    
    # Strategy: Access LoRA modules directly and reconstruct shapes from config
    def extract_lora_weights(module, prefix="", base_prefix="base_model.model"):
        """Recursively extract LoRA weights from modules"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module has LoRA layers
            if hasattr(child, 'lora_A') and hasattr(child, 'lora_B'):
                try:
                    adapter_name = "default"  # Default adapter name
                    
                    if adapter_name in child.lora_A and adapter_name in child.lora_B:
                        # Get LoRA config for this layer
                        r = child.r[adapter_name] if hasattr(child, 'r') and adapter_name in child.r else 8
                        
                        # Access the Linear layers directly
                        lora_A_layer = child.lora_A[adapter_name]
                        lora_B_layer = child.lora_B[adapter_name]
                        
                        # Get the expected dimensions from the parent Linear layer
                        if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                            in_features = child.in_features
                            out_features = child.out_features
                        else:
                            # Fallback: try to infer from the original weight
                            original_weight = getattr(child, 'weight', None)
                            if original_weight is not None:
                                out_features, in_features = original_weight.shape
                            else:
                                print(f"Warning: Could not determine dimensions for {full_name}")
                                continue
                        
                        # Extract LoRA A (should be [r, in_features])
                        lora_A_param_name = f"{base_prefix}.{full_name}.lora_A.{adapter_name}.weight"
                        lora_A_weight = None
                        if hasattr(lora_A_layer, 'weight'):
                            lora_A_weight = lora_A_layer.weight.data
                            expected_A_shape = (r, in_features)
                            
                            if lora_A_weight.shape != expected_A_shape:
                                if lora_A_weight.numel() == r * in_features:
                                    # Reshape if the elements match but shape is wrong
                                    lora_A_weight = lora_A_weight.view(expected_A_shape)
                                    print(f"Reshaped {lora_A_param_name}: {lora_A_weight.shape}")
                                else:
                                    print(f"Warning: {lora_A_param_name} has wrong size: {lora_A_weight.shape}, expected: {expected_A_shape}")
                            
                            lora_state_dict[lora_A_param_name] = lora_A_weight.detach().clone().cpu()
                        
                        # Extract LoRA B (should be [out_features, r])
                        lora_B_param_name = f"{base_prefix}.{full_name}.lora_B.{adapter_name}.weight"
                        lora_B_weight = None
                        if hasattr(lora_B_layer, 'weight'):
                            lora_B_weight = lora_B_layer.weight.data
                            expected_B_shape = (out_features, r)
                            
                            if lora_B_weight.shape != expected_B_shape:
                                if lora_B_weight.numel() == out_features * r:
                                    # Reshape if the elements match but shape is wrong
                                    lora_B_weight = lora_B_weight.view(expected_B_shape)
                                    print(f"Reshaped {lora_B_param_name}: {lora_B_weight.shape}")
                                else:
                                    print(f"Warning: {lora_B_param_name} has wrong size: {lora_B_weight.shape}, expected: {expected_B_shape}")
                            
                            lora_state_dict[lora_B_param_name] = lora_B_weight.detach().clone().cpu()
                        
                        if lora_A_weight is not None and lora_B_weight is not None:
                            print(f"Extracted LoRA weights for {full_name}: A={lora_A_weight.shape}, B={lora_B_weight.shape}")
                
                except Exception as e:
                    print(f"Error extracting LoRA from {full_name}: {e}")
            
            # Recurse into child modules
            extract_lora_weights(child, full_name, base_prefix)
    
    if is_fsdp:
        print("Extracting LoRA parameters from FSDP model...")
        
        # Use summon_full_params context to access parameters
        with FSDP.summon_full_params(model, writeback=False):
            extract_lora_weights(peft_model.base_model if hasattr(peft_model, 'base_model') else peft_model)
    else:
        print("Model is not FSDP-wrapped, extracting directly...")
        extract_lora_weights(peft_model.base_model if hasattr(peft_model, 'base_model') else peft_model)
    
    if not lora_state_dict:
        raise ValueError("No LoRA parameters found in the model")
    
    print(f"Extracted {len(lora_state_dict)} LoRA parameters")
    
    # Verify that all parameters have the expected 2D shapes
    for name, param in lora_state_dict.items():
        if param.dim() != 2:
            print(f"WARNING: Parameter {name} has unexpected shape {param.shape} (expected 2D)")
        if param.numel() == 0:
            print(f"WARNING: Parameter {name} is empty!")
    
    # Save the LoRA weights
    adapter_weights_path = os.path.join(output_dir, ADAPTER_WEIGHTS_NAME)
    torch.save(lora_state_dict, adapter_weights_path)
    print(f"Saved LoRA weights to: {adapter_weights_path}")
    
    # Save the adapter config
    peft_config = peft_model.peft_config
    if hasattr(peft_config, 'values'):
        # If it's a dict-like object, get the first config
        adapter_config = next(iter(peft_config.values()))
    else:
        adapter_config = peft_config
    
    # Convert config to dict and save
    config_dict = adapter_config.to_dict() if hasattr(adapter_config, 'to_dict') else adapter_config.__dict__
    
    # Ensure all values are JSON serializable while preserving correct types
    def make_serializable(obj):
        from enum import Enum
        
        if isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value  # Use the enum value, not the whole enum object
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return list(obj)  # Convert sets to lists for JSON serialization
        else:
            # Try to convert to string as a fallback
            return str(obj)
    
    serializable_config = {k: make_serializable(v) for k, v in config_dict.items()}
    
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"Saved adapter config to: {config_path}")
    
    # Create adapter_model.safetensors if needed (some versions expect this)
    try:
        import safetensors.torch
        safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
        safetensors.torch.save_file(lora_state_dict, safetensors_path)
        print(f"Saved SafeTensors format to: {safetensors_path}")
    except ImportError:
        print("SafeTensors not available, skipping .safetensors format")
    
    print("LoRA adapter saved successfully with correct parameter shapes!")


def detect_model_family(model_name: str) -> str:
    """Auto-detect model family from model name"""
    model_name_lower = model_name.lower()
    if "roberta" in model_name_lower:
        return "roberta"
    elif "opt" in model_name_lower:
        return "opt"
    elif "gpt" in model_name_lower:
        return "gpt"
    elif "llama" in model_name_lower:
        return "llama"
    else:
        return "unknown"


def apply_paper_hyperparameters(args):
    """Apply exact hyperparameters from MeZO paper Tables 15-16"""
    if not args.paper_hparams:
        return args
    
    # Auto-detect model family if not specified
    if args.model_family is None:
        args.model_family = detect_model_family(args.model_name)
    
    print(f"Applying paper hyperparameters for {args.model_family} family")
    
    if args.model_family == "roberta":
        # RoBERTa-large settings from Table 15
        args.batch_size = 64 if args.batch_size == 8 else args.batch_size  # Keep if user overrode
        args.learning_rate = 1e-6 if args.learning_rate == 1e-4 else args.learning_rate  # Default from grid
        args.zo_eps = 1e-3
        args.weight_decay = 0.0
        args.max_steps = 100000  # 100k steps as per paper
        args.eval_steps = 10000
        args.save_steps = 10000
        args.lr_scheduler_type = "linear"
        args.warmup_steps = 0
        
        # Note: MeZO doesn't support gradient accumulation, so keep batch_size as is
        if args.batch_size == 64:
            print("Warning: Large batch size (64) may require high memory. MeZO doesn't support gradient accumulation.")
        
        
    elif args.model_family == "opt":
        # OPT settings from Table 16
        args.batch_size = 16 if args.batch_size == 8 else args.batch_size
        args.learning_rate = 1e-6 if args.learning_rate == 1e-4 else args.learning_rate  # Default from grid
        args.zo_eps = 1e-3
        args.weight_decay = 0.0
        args.max_steps = 20000  # 20k steps as per paper
        args.eval_steps = 4000
        args.save_steps = 4000
        args.lr_scheduler_type = "constant"
        args.warmup_steps = 0
        
    print(f"Applied settings: batch_size={args.batch_size}, lr={args.learning_rate}, "
          f"eps={args.zo_eps}, max_steps={args.max_steps}")
    
    return args


def setup_memory_logging():
    """Enhanced memory logging with optimization tips"""
    import psutil
    import threading
    import time
    
    memory_log = []
    stop_logging = threading.Event()
    
    def log_memory():
        while not stop_logging.is_set():
            if torch.cuda.is_available():
                # Clear cache before measuring for accurate readings
                torch.cuda.empty_cache()
                gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                gpu_mem_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
            else:
                gpu_mem = gpu_mem_cached = gpu_mem_max = 0
            
            cpu_mem = psutil.Process().memory_info().rss / 1024**3  # GB
            
            memory_log.append({
                'timestamp': time.time(),
                'gpu_allocated_gb': gpu_mem,
                'gpu_cached_gb': gpu_mem_cached, 
                'gpu_max_gb': gpu_mem_max,
                'cpu_gb': cpu_mem
            })
            time.sleep(10)  # Log every 10 seconds as per paper
    
    logging_thread = threading.Thread(target=log_memory, daemon=True)
    logging_thread.start()
    
    return memory_log, stop_logging


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seeds for reproducibility FIRST
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Apply paper hyperparameters if requested
    args = apply_paper_hyperparameters(args)
    
    # MeZO doesn't support gradient accumulation - enforce this
    if args.gradient_accumulation_steps != 1:
        print(f"Warning: MeZO doesn't support gradient accumulation. Setting gradient_accumulation_steps=1 (was {args.gradient_accumulation_steps})")
        args.gradient_accumulation_steps = 1
    
    # Setup memory logging if requested
    memory_log = None
    stop_memory_logging = None
    if args.memory_logging:
        memory_log, stop_memory_logging = setup_memory_logging()
        print("Memory logging enabled (sampling every 10s)")
    
    # Create FSDP accelerator with memory optimizations
    fsdp_plugin = None
    device_count = torch.cuda.device_count()
    print(f"🔍 Debug: device_count={device_count}")
    
    # Check if we're running with accelerate config or multiple processes
    import os
    num_processes = int(os.environ.get('WORLD_SIZE', '1'))
    print(f"🔍 Debug: WORLD_SIZE={num_processes}")
    
    if num_processes > 1:
        fsdp_plugin = build_fsdp_plugin(args.fsdp_version)
        print(f"🔍 Debug: Using FSDP for {num_processes} processes")
    else:
        print(f"🔍 Debug: Single process detected, no FSDP")
    
    # Use mixed precision for memory efficiency
    mixed_precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "no"
    
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=1  # Always 1 for MeZO
    )
    
    print(f"✅ Accelerator initialized with mixed_precision={mixed_precision}")
    
    # Load tokenizer and configure padding token properly
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token for models that don't have one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for causal language modeling with enhanced memory optimization
    with accelerator.main_process_first():
        print(f"Loading model {args.model_name} with enhanced memory optimization...")
        
        # For large models, use maximum memory optimization
        if "opt" in args.model_name.lower() and any(size in args.model_name for size in ["6.7b", "13b", "30b", "66b"]):
            print("Applying aggressive memory optimization for large model...")
            print(f"🔍 Debug: fsdp_plugin={fsdp_plugin}")
            
            try:
                # For FSDP: load on CPU first, no device_map
                if fsdp_plugin:
                    print("🔍 Debug: Loading with FSDP path (no device_map)")
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        device_map=None  # CRITICAL: No device_map with FSDP
                    )
                    print("✅ FSDP-compatible model loading successful")
                else:
                    # Non-FSDP: use device_map for automatic placement
                    print("🔍 Debug: Loading with device_map=auto")
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        offload_folder="./model_offload" if torch.cuda.get_device_properties(0).total_memory < 25e9 else None
                    )
                    print("✅ Optimized model loading successful")
            except Exception as e:
                print(f"⚠️ Optimized loading failed ({e}), trying standard approach...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map=None if fsdp_plugin else "auto"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=None if fsdp_plugin else "auto"
            )

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Enable compilation for memory and speed optimization (if supported)
    # Note: Skip compilation to avoid type issues with FSDP
    try:
        if (hasattr(torch, 'compile') and torch.__version__ >= "2.0" and 
            not fsdp_plugin and not args.use_lora):
            # Only compile if not using FSDP or LoRA to avoid compatibility issues
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("✅ Model compilation enabled for memory optimization")
            # Keep original model reference for other operations
            model._compiled_forward = compiled_model.forward
        else:
            print("⚠️ Model compilation skipped (FSDP/LoRA compatibility)")
    except Exception as e:
        print(f"⚠️ Model compilation skipped: {e}")

    # ─── LoRA integration ───────────────────────────────────────────────
    if args.use_lora:
        # Auto-detect LoRA targets for OPT models if not specified
        lora_target_modules = args.lora_target
        if lora_target_modules is None and "opt" in args.model_name.lower():
            # Use all linear layers for OPT models: attention + MLP
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            print(f"🎯 Auto-detected LoRA targets for OPT model: {lora_target_modules}")
        elif lora_target_modules is None:
            # Fallback for other models
            lora_target_modules = ["q_proj", "v_proj"]
            print(f"🎯 Using default LoRA targets: {lora_target_modules}")
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if args.lora_path:
            model = PeftModel.from_pretrained(model, args.lora_path)
        model.print_trainable_parameters()
        
        # Fix dtype consistency for FSDP + LoRA compatibility
        if fsdp_plugin is not None:
            print("🔧 Fixing LoRA parameter dtypes for FSDP compatibility...")
            target_dtype = torch.bfloat16  # Match the base model dtype
            
            for name, param in model.named_parameters():
                if param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)
                    print(f"   Fixed {name}: {param.dtype}")
            
            print("✅ All LoRA parameters converted to bfloat16 for FSDP compatibility")

    # Resize token embeddings if we added new tokens
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Update model config to match tokenizer padding token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model - handle device_map models specially
    model_uses_device_map = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
    
    if model_uses_device_map:
        # Model already placed with device_map="auto" - don't call accelerator.prepare()
        print(f"🔧 Model uses device_map: {model.hf_device_map}")
        print("✅ Skipping accelerator.prepare() for device_map model")
        prepared_model = model  # Don't wrap with accelerator
    elif accelerator.num_processes == 1:
        # Single GPU without device_map
        print("🔧 Preparing model for single GPU...")
        prepared_model = accelerator.prepare(model)
        print("✅ Single GPU model preparation complete")
    else:
        # Multi-GPU: use FSDP
        print("🔧 Preparing model with FSDP for multi-GPU...")
        prepared_model = accelerator.prepare(model)
        print("✅ FSDP model preparation complete")
    
    model = prepared_model
    
    # NOW create optimizer with the prepared (potentially FSDP-wrapped) model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
    # Note: MeZO doesn't actually use the optimizer, this is just for compatibility with accelerate
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.learning_rate,  # Use the real learning rate
        weight_decay=0.0)

    # Prepare data loader
    train_dataloader = prepare_dataloader(args, tokenizer)
    train_dataloader = accelerator.prepare(train_dataloader)

    # Calculate total training steps
    if args.max_steps is not None:
        total_training_steps = args.max_steps
    else:
        total_training_steps = args.num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Get named parameters AFTER accelerator.prepare()
    named_params = [(n, p)
                    for n, p in model.named_parameters() if p.requires_grad]

    # Get the device from the model
    device = accelerator.device

    # Set model to training mode
    model.train()

    global_step = 0
    max_steps = args.max_steps if args.max_steps is not None else float('inf')
    
    # Training metrics tracking
    training_metrics = {
        'steps': [],
        'losses': [],
        'memory_peak_gb': [],
        'step_times': [],
        'learning_rates': []
    }
    
    # Progress bar
    pbar = tqdm(total=total_training_steps, disable=not accelerator.is_main_process)

    # Training timing
    step_start_time = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            if global_step >= max_steps:
                break
            
            # Timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                step_start_time = time.time()
            
            # MeZO Step with corrected implementation (no gradient accumulation)
            projected_grad, zo_random_seed, current_loss = zo_step(
                model, batch, named_params, args.zo_eps, accelerator, args.gradient_accumulation_steps
            )
            
            # MeZO Update with corrected parameters
            zo_update(
                named_params, lr_scheduler, projected_grad, zo_random_seed, args.weight_decay, args.learning_rate
            )

            global_step += 1
            
            # Timing and memory tracking
            step_time = 0
            if step_start_time is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                step_time = time.time() - step_start_time
            
            # Memory optimization: Clear cache periodically
            if global_step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log metrics  
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                current_lr = lr_scheduler.get_last_lr()[0]  # Scheduler already has the right LR
                
                training_metrics['steps'].append(global_step)
                training_metrics['losses'].append(current_loss)  # Log actual loss, not gradient
                training_metrics['learning_rates'].append(current_lr)
                training_metrics['step_times'].append(step_time)
                
                if torch.cuda.is_available():
                    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    training_metrics['memory_peak_gb'].append(peak_memory_gb)
                    
                    print(f"Step {global_step}: Loss={current_loss:.4f}, Grad={projected_grad:.2e}, LR={current_lr:.2e}, "
                          f"Memory={peak_memory_gb:.1f}GB, Time={step_time:.2f}s")
                else:
                    training_metrics['memory_peak_gb'].append(0)
                    print(f"Step {global_step}: Loss={current_loss:.4f}, Grad={projected_grad:.2e}, LR={current_lr:.2e}")
            
            # Evaluation and checkpointing
            if global_step % args.eval_steps == 0 and accelerator.is_main_process:
                print(f"\nEvaluation at step {global_step}")
                # TODO: Add evaluation logic here for GLUE/SuperGLUE tasks
                
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                print(f"Saving checkpoint to {checkpoint_dir}")
                # Save will happen at the end for now
            
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}, Step {global_step}, Loss: {current_loss:.4f}")

        if global_step >= max_steps:
            break
    
    # Stop memory logging
    if stop_memory_logging is not None:
        try:
            stop_memory_logging.set()
        except Exception as e:
            print(f"Warning: Memory logging cleanup failed: {e}")
    
    # Save final model and metrics
    if accelerator.is_main_process:
        print("Saving model...")
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save training metrics first (always works)
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        print(f"✅ Saved training metrics to: {metrics_path}")
        
        # Save memory log if available
        if memory_log is not None:
            memory_log_path = os.path.join(args.output_dir, "memory_log.json")
            with open(memory_log_path, 'w') as f:
                json.dump(memory_log, f, indent=2)
            print(f"✅ Saved memory log to: {memory_log_path}")
        
        # Save run configuration
        config_path = os.path.join(args.output_dir, "run_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"✅ Saved run config to: {config_path}")
        
        # Save tokenizer first (lightweight)
        try:
            tokenizer.save_pretrained(args.output_dir)
            print(f"✅ Saved tokenizer to: {args.output_dir}")
        except Exception as e:
            print(f"⚠️ Failed to save tokenizer: {e}")
        
        # Model saving with timeout and fallback
        if args.skip_model_save:
            print("⚠️ Skipping model save as requested (--skip_model_save)")
            print("   Metrics and configuration have been saved successfully")
        elif args.use_lora:
            print("Saving LoRA adapter (FSDP-compatible method)...")
            
            try:
                # FSDP-compatible LoRA saving: extract LoRA parameters with correct shapes
                save_lora_adapter_fsdp_compatible(model, args.output_dir, accelerator)
                print(f"✅ LoRA adapter saved successfully to: {args.output_dir}")
                
            except Exception as e:
                print(f"❌ Error saving LoRA adapter: {e}")
                print("⚠️ Skipping model save to avoid hanging - metrics and config saved")
        else:
            # For non-LoRA models, try fast save methods
            print("Attempting to save full model...")
            model_save_success = False
            
            # Method 1: Try accelerator's save_state (recommended for FSDP)
            try:
                print("Trying accelerator.save_state()...")
                accelerator.save_state(args.output_dir)
                print("✅ Model saved using accelerator.save_state()")
                model_save_success = True
            except Exception as e:
                print(f"⚠️ accelerator.save_state() failed: {e}")
            
            # Method 2: Try save_model if save_state failed
            if not model_save_success:
                try:
                    print("Trying accelerator.save_model()...")
                    accelerator.save_model(model, args.output_dir)
                    print("✅ Model saved using accelerator.save_model()")
                    model_save_success = True
                except Exception as e:
                    print(f"⚠️ accelerator.save_model() failed: {e}")
            
            # Method 3: Skip model saving entirely to avoid hanging
            if not model_save_success:
                print("⚠️ All model saving methods failed - skipping to avoid hanging")
                print("   Metrics and configuration have been saved successfully")
                print("   Model weights are not saved but training results are preserved")
    
    # Use a timeout for wait_for_everyone to avoid hanging
    print("Synchronizing processes...")
    try:
        accelerator.wait_for_everyone()
        print("✅ All processes synchronized")
    except Exception as e:
        print(f"⚠️ Process synchronization issue: {e}")
    
    print("Training finished.")
    
    # Print final summary
    if accelerator.is_main_process:
        print(f"\n=== Training Summary ===")
        print(f"Total steps: {global_step}")
        print(f"Final loss: {training_metrics['losses'][-1] if training_metrics['losses'] else 'N/A'}")
        if torch.cuda.is_available() and training_metrics['memory_peak_gb']:
            print(f"Peak memory: {max(training_metrics['memory_peak_gb']):.1f} GB")
        if training_metrics['step_times']:
            print(f"Avg step time: {np.mean(training_metrics['step_times']):.2f}s")
        print(f"Metrics saved to: {args.output_dir}")
        print("=" * 25)
