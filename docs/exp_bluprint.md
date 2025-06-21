Below is a complete, **step‑by‑step experimental blueprint** for reproducing the main results reported in *Fine‑Tuning Language Models with Just Forward Passes* (MeZO) while leveraging your new PyTorch FSDP v2 implementation.  The plan follows the original paper as closely as possible and adds the practical details you need to run on modern hardware, manage sharding with FSDP2, and obtain verifiable numbers.

> **In one sentence:** run MeZO for 100 k (RoBERTa‑large) or 20 k (OPT) forward‑only steps using the hyper‑parameter grids in Tables 15–16, shard the model/optimizer with FSDP2, evaluate on the same prompt‑based SuperGLUE/GLUE subsets, and log memory, wall‑clock, and accuracy so they match the paper’s Tables 18–23.

---

## 1 Goals & Target Tables

| Model family               | Size(s)                      | Tasks to match                           | Figure/table to reproduce                                    |
| -------------------------- | ---------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| **RoBERTa‑large** (∼350 M) | single size                  | SST‑2, SST‑5, TREC, MNLI, SNLI, RTE      | Table 18 & Fig. 2([arxiv.org][1])                            |
| **OPT**                    | 13 B (optionally 30 B, 66 B) | 8 SuperGLUE cls/mch + 2 QA (SQuAD, DROP) | Fig. 1, Table 1, 20, 22 & 23([arxiv.org][1], [arxiv.org][1]) |
| **Memory & time**          | all sizes                    | MultiRC (token length ∼400)              | Table 22 & 23 (mem & speed‑up)([arxiv.org][1])               |

---

## 2 Hardware & Software Environment

### 2.1 Compute cluster

| Component       | Minimum spec to hit paper numbers                 | Notes                                                             |
| --------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| GPU             | 1 × A100‑80 GB for ≤30 B; 2 × A100‑80 GB for 66 B | Matches paper’s single‑GPU claim and memory table([arxiv.org][1]) |
| NV‑linked nodes | Optional but recommended for >30 B FT baseline    | Needed only for back‑prop baseline; MeZO itself stays on ≤2 GPUs  |
| Storage         | ≥2 TB SSD                                         | to cache HF models & checkpoints                                  |

### 2.2 Key software versions

```text
PyTorch          = 2.4 or newer  (includes FSDP2)                    :contentReference[oaicite:5]{index=5}
torch.distributed = built‑in NCCL backend
transformers      = 4.49  (TP & device_map features)                :contentReference[oaicite:6]{index=6}
accelerate        = 0.28 (for big‑model dispatch utilities)          :contentReference[oaicite:7]{index=7}
datasets          = 2.19
bitsandbytes      = 0.43 (optional 8‑bit weights for checkpoints)
```

> **Why FSDP v2?** FSDP2 stores parameters, gradients and optimizer shards off‑rank until needed, gives *activation recomputation free* sharding, and now supports low‑precision parameter groups and mixed‑precision optimizers out‑of‑the‑box.([docs.pytorch.org][2], [pytorch.org][3], [pytorch.org][4], [pytorch.org][5], [vldb.org][6])

---

## 3 Data & Prompt Preparation

### 3.1 Datasets and sampling

* **RoBERTa‑large experiments**
  *Take the six GLUE‑style sets shown in Appendix E.1* – sample either **k = 16** or **k = 512** examples *per class* for train **and** validation; keep the full test split for reporting.([arxiv.org][1])

* **OPT experiments**
  Use the SuperGLUE collection plus SST‑2, SQuAD v1.1 and DROP; randomly sample **1 000 / 500 / 1 000** for train/val/test as in the paper.([arxiv.org][1], [arxiv.org][1])

### 3.2 Prompt templates

Reuse the exact templates in Tables 13–14 (masking for RoBERTa, verbalizers or answer strings for OPT).([arxiv.org][1])

> **Tip:** Put all templates into a JSONL so they can be loaded by both MeZO and back‑prop baselines for identical data pipelines.

---

## 4 Model Checkpoints

| Family        | Pre‑trained IDs                          | HF load arguments                                                                                                                                                          |
| ------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RoBERTa‑large | `roberta-large`                          | `torch_dtype=torch.float16`                                                                                                                                                |
| OPT series    | `facebook/opt-13b`, `opt-30b`, `opt-66b` | `torch_dtype=torch.bfloat16` + `device_map="auto"` for inference baseline in memory table; but **disable** device\_map when training under FSDP2 to avoid nested sharding. |

---

## 5 Implementing MeZO with FSDP2

1. **Wrap the root module in `FSDP`**

   ```python
   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
   model = FSDP(model, sync_module_states=True, mixed_precision=True,
                sharding_strategy="FULL_SHARD_AUTO_WRAP")
   ```

   *This keeps parameters & grads sharded, yet MeZO only needs params in forward, so RAM stays close to inference.*([docs.pytorch.org][2], [pytorch.org][4])

2. **Perturb in‑place**
   *Sample a Rademacher (or Gaussian) direction `u`, scale by ε, and add to each *local shard* in an `autocast` forward‑only context.*

   ```python
   for p in model.parameters(recurse=True):
       if p.requires_grad: p.data.add_(eps * torch.sign(torch.randn_like(p)))
   loss_pos = fwd_pass()
   ...
   p.data.sub_(2*eps*...)   # negative perturbation
   loss_neg = fwd_pass()
   grad_est = (loss_pos - loss_neg) / (2*eps)
   ```

3. **Optimizer step**
   Use *in‑place SGD* with shard‑aware parameter groups; no optimizer state is gathered until `optimizer.step()`, so memory remains sharded.

4. **Synchronize perturbations**
   Because each rank holds different param shards, broadcast the *scalar* `(loss_pos‑loss_neg)` and accumulate the per‑parameter finite‑difference contribution locally; this avoids a full `all_gather` of parameter vectors.

5. **Mixed precision**
   FSDP2’s `mixed_precision=True` automatically converts model shards to **bfloat16** buffers during forward; choose **float32** for optimiser states to match the paper’s exact grid.

---

## 6 Hyper‑parameters (faithful to Tables 15 & 16)

### 6.1 RoBERTa‑large (100 k MeZO steps)

| Setting            | Value                                                       |
| ------------------ | ----------------------------------------------------------- |
| Batch size         | **64** (micro‑batch = 8, grad‑acc = 8 if GPU memory <80 GB) |
| Learning rate      | {1e‑7, 1e‑6, 1e‑5}                                          |
| ε (perturb radius) | 1 e‑3                                                       |
| Weight decay       | 0                                                           |
| Eval & checkpoint  | every 10 k steps                                            |

*(Prefix/LoRA variants: lr grids {1e‑2…} and ε=1e‑1 exactly as in Table 15.)*

### 6.2 OPT‑13B / 30B / 66B (20 k MeZO steps)

| Setting           | Value           |
| ----------------- | --------------- |
| Batch size        | 16              |
| LR grid           | {1e‑6, 1e‑7}    |
| ε                 | 1 e‑3           |
| Prefix / LoRA LRs | per Table 16    |
| Tokens per prefix | 5               |
| Validation        | every 4 k steps |

*(Use constant LR schedule; FT baselines use 5 epochs + linear warm‑up.)*

---

## 7 Baselines for Comparison

1. **Full‑parameter back‑prop (Adam & SGD)**
   *Wrap with FSDP2 + activation checkpointing OFF* to match the paper’s “12 × memory” numbers. Hyper‑parameter grids are in Table 15–16.
2. **Zero‑shot & 32‑shot ICL** (same prompts, no weight updates).
3. **Linear probing & head‑tuning** (scipy logistic regression or head‑only SGD) as in Appendix E.4.([arxiv.org][1])

---

## 8 Evaluation & Logging

| Metric type         | Implementation                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Classification & MC | accuracy                                                                                    |
| QA (SQuAD)          | EM & F1 from `squad_v1_metric.py`                                                           |
| DROP                | official `drop_eval.py`                                                                     |
| Memory              | `torch.cuda.mem_allocated()` + `nvidia‑smi` sampling (every 10 s) as in E.7([arxiv.org][1]) |
| Wall‑clock          | `torch.cuda.Event` per step; aggregate GPU‑hours.                                           |

Save *all* raw logs plus a CSV summarising **(task, model size, method, best‑val, test‑metric, peak‑mem, step‑time)**.

---

## 9 Expected Reference Numbers

| Task              | Paper MeZO                  | Paper FT   | Your run (target) |
| ----------------- | --------------------------- | ---------- | ----------------- |
| SST‑2 (RoBERTa)   | 90.5 ± 1.2                  | 91.9 ± 1.8 | 89–92             |
| MultiRC (OPT‑13B) | 67.2                        | 66.2       | 65–68             |
| Peak mem 30 B     | 58 GB (MeZO) vs 315 GB (FT) | same order |                   |

These correspond to Tables 18, 20 & 22; match within ±1 pp accuracy and ±5 % memory to claim reproduction.([arxiv.org][1])

---

## 10 Troubleshooting & Tips

* **Slow convergence?** Increase batch size first; the paper notes larger batches help MeZO more than tuning ε.
* **Divergence on 66 B?** Start from `opt‑66b` with BF16 weights and lower LR to 5 e‑7; the authors flagged extra tuning was needed.([arxiv.org][1])
* **FSDP2 “CPU offload stall”**: disable `cpu_offload` when you already shard grads; perf penalty > memory gain on A100‑80 GB.([docs.pytorch.org][2])
* **Checkpoint size**: with MeZO there is *no* optimizer state → checkpoints are ½ model weights; keep only last/best to save disk.
* **Prefix‑tuning stability**: initialise prefixes with *real token activations*, not random; see Table 17 ablation.([arxiv.org][1])

---

## 11 Reproducibility Package

1. `env.yml` – conda spec with exact versions above.
2. `run_mezo.py` – main script accepting `--model`, `--task`, `--fsdp` flags.
3. `slurm_job.sh` – launches one job per `(task, seed)` on your cluster.
4. `analysis.ipynb` – aggregates logs and compares against paper tables.

---

### Closing note

Following this recipe — datasets, prompts, hyper‑parameters, and careful FSDP2 sharding — should let you replicate the headline MeZO results on a single or dual A100 node while confirming the dramatic memory savings the paper reports. Good luck, and let me know if any step needs deeper clarification!

[1]: https://arxiv.org/html/2305.17333v3 "Fine-Tuning Language Models with Just Forward Passes"
[2]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=chatgpt.com "Getting Started with Fully Sharded Data Parallel (FSDP2)"
[3]: https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/?utm_source=chatgpt.com "Introducing PyTorch Fully Sharded Data Parallel (FSDP) API"
[4]: https://pytorch.org/docs/stable/fsdp.html?utm_source=chatgpt.com "FullyShardedDataParallel — PyTorch 2.7 documentation"
[5]: https://pytorch.org/blog/maximizing-training/?utm_source=chatgpt.com "Maximizing training throughput using PyTorch FSDP"
[6]: https://www.vldb.org/pvldb/vol16/p3848-huang.pdf?utm_source=chatgpt.com "[PDF] PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"