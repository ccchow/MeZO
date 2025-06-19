import argparse
import os
import json
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
    # FSDP version to use when multiple GPUs are available
    fsdp_version: int = 2


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
    parser.add_argument(
        "--fsdp_version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Enable FSDP when multiple GPUs are available."
             " Use 1 or 2 to select the FSDP version.")

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

    def tokenize_fn(examples):
        return tokenizer(
            examples[args.text_column],
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
    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs)
        loss = outputs.loss
    return loss.detach()


def perturb_parameters(named_params, eps, random_seed, scaling_factor):
    """Perturb parameters using a specific random seed"""
    # Create a generator for reproducible randomness
    device = next(iter(named_params))[1].device if named_params else torch.device('cpu')
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=generator)
        param.data.add_(z * scaling_factor * eps)


def zo_step(model, inputs, named_params, eps, accelerator):
    """Compute gradient estimate using MeZO algorithm - simplified version without broadcasting."""
    # Generate random seed locally on each process (same as trainer.py)
    zo_random_seed = np.random.randint(1000000000)

    # First perturbation: +eps
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=1)
    loss1 = zo_forward(model, inputs)
    
    # In a distributed setting, loss must be averaged across all processes
    if accelerator.num_processes > 1:
        loss1 = accelerator.reduce(loss1, "mean")

    # Second perturbation: -2*eps (to get to -eps)
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=-2)
    loss2 = zo_forward(model, inputs)

    if accelerator.num_processes > 1:
        loss2 = accelerator.reduce(loss2, "mean")

    # Restore to original parameters
    perturb_parameters(named_params, eps, zo_random_seed, scaling_factor=1)

    # Calculate projected gradient
    projected_grad = (loss1 - loss2) / (2 * eps)

    return projected_grad.item(), zo_random_seed


def zo_update(named_params, lr_scheduler, projected_grad, zo_random_seed, lr, weight_decay, eps):
    """Update parameters by regenerating the same z vectors"""
    current_lr = lr_scheduler.get_last_lr()[0] * lr
    
    # Reset to same seed to regenerate same z vectors
    device = next(iter(named_params))[1].device if named_params else torch.device('cpu')
    generator = torch.Generator(device=device)
    generator.manual_seed(zo_random_seed)
    
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=generator)
        
        # Apply weight decay
        if weight_decay > 0 and 'bias' not in name and 'LayerNorm' not in name:
            param.data.mul_(1 - current_lr * weight_decay)
        
        # Apply gradient update
        param.data.add_(-current_lr * projected_grad * z)
    
    lr_scheduler.step()


def build_fsdp_plugin(fsdp_version: int = 2):
    """Build FSDP plugin with auto-wrapping for transformer blocks"""
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
    
    return FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",
        state_dict_type="full_state_dict",
        auto_wrap_policy=wrap_policy,
        ignored_modules=None,
        use_orig_params=True,
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


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create FSDP accelerator
    fsdp_plugin = None
    if torch.cuda.device_count() > 1:
        fsdp_plugin = build_fsdp_plugin(args.fsdp_version)
    accelerator = Accelerator(
        mixed_precision="no",
        fsdp_plugin=fsdp_plugin
    )
    
    # Load tokenizer and configure padding token properly
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token for models that don't have one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for causal language modeling
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model.gradient_checkpointing_enable()

    # ─── LoRA integration ───────────────────────────────────────────────
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if args.lora_path:
            model = PeftModel.from_pretrained(model, args.lora_path)
        model.print_trainable_parameters()

    # Resize token embeddings if we added new tokens
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Update model config to match tokenizer padding token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model with FSDP BEFORE creating optimizer
    model = accelerator.prepare(model)
    
    # NOW create optimizer with the prepared (potentially FSDP-wrapped) model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=1.0, # We use a dummy LR here and apply the real one manually in zo_update
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
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
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
    
    # Progress bar
    pbar = tqdm(total=total_training_steps, disable=not accelerator.is_main_process)

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            if global_step >= max_steps:
                break
            
            # MeZO Step
            projected_grad, zo_random_seed = zo_step(
                model, batch, named_params, args.zo_eps, accelerator
            )
            
            # MeZO Update
            zo_update(
                named_params, lr_scheduler, projected_grad, zo_random_seed, args.learning_rate, args.weight_decay, args.zo_eps
            )

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}, Grad: {projected_grad:.2e}")

        if global_step >= max_steps:
            break
    
    # Save the model
    if accelerator.is_main_process:
        print("Saving model...")
        
        if args.use_lora:
            print("Saving LoRA adapter (FSDP-compatible method)...")
            
            try:
                # FSDP-compatible LoRA saving: extract LoRA parameters with correct shapes
                save_lora_adapter_fsdp_compatible(model, args.output_dir, accelerator)
                print(f"✅ LoRA adapter saved successfully to: {args.output_dir}")
                
                # Save tokenizer
                tokenizer.save_pretrained(args.output_dir)
                
            except Exception as e:
                print(f"❌ Error saving LoRA adapter: {e}")
                print("Trying fallback method...")
                try:
                    # Fallback: try saving before unwrapping (may still be corrupted)
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
                    # Last resort: unwrapped model (likely corrupted)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                
        else:
            # For non-LoRA models, use standard unwrapping
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    accelerator.wait_for_everyone()
    print("Training finished.")