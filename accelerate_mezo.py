import argparse
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from accelerate import Accelerator


@dataclass
class Arguments:
    model_name: str
    dataset: str
    text_column: str = "text"
    label_column: str = "label"
    output_dir: str = "./output"
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 3
    max_length: int = 512
    zo_eps: float = 1e-3
    seed: int = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return Arguments(**vars(args))


def prepare_dataloader(args, tokenizer):
    dataset = load_dataset(args.dataset)
    if "train" not in dataset:
        raise ValueError("Dataset must have a train split")

    def tokenize_fn(ex):
        return tokenizer(ex[args.text_column], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(args.label_column, "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized["train"], batch_size=args.batch_size, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))


def zo_perturb_parameters(named_params, eps, random_seed=None, scaling_factor=1):
    torch.manual_seed(random_seed if random_seed is not None else 0)
    for _, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * eps


def zo_forward(model, inputs):
    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs)
        loss = outputs.loss
    return loss.detach()


def zo_step(model, inputs, named_params, eps):
    zo_random_seed = np.random.randint(1000000000)
    zo_perturb_parameters(named_params, eps, scaling_factor=1)
    loss1 = zo_forward(model, inputs)
    zo_perturb_parameters(named_params, eps, scaling_factor=-2)
    loss2 = zo_forward(model, inputs)
    projected_grad = ((loss1 - loss2) / (2 * eps)).item()
    zo_perturb_parameters(named_params, eps, random_seed=zo_random_seed, scaling_factor=1)
    return loss1, projected_grad, zo_random_seed


def zo_update(named_params, lr_scheduler, projected_grad, zo_random_seed, lr, weight_decay, eps):
    torch.manual_seed(zo_random_seed)
    for name, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
        else:
            param.data = param.data - lr * (projected_grad * z)
    lr_scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    train_dataloader = prepare_dataloader(args, tokenizer)

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.num_epochs)

    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            loss, grad, seed = zo_step(model, batch, named_params, args.zo_eps)
            zo_update(named_params, lr_scheduler, grad, seed, args.learning_rate, args.weight_decay, args.zo_eps)
            if accelerator.is_main_process and step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
