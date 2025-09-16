import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune
from pathlib import Path


device = torch.device("cuda")
HUGGING_FACE_API_KEY = 'key'


def compute_neuron_pair_importance(gate_weight, up_weight):
  gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(torch.min(gate_weight, dim=1).values)
  up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(torch.min(up_weight, dim=1).values)
  importance_scores = gate_max_abs + up_max_abs
  return importance_scores


def prune_neuron_pairs_unstructured(mlp, prune_frac):

    for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        prune.l1_unstructured(proj, name="weight", amount=prune_frac)
        prune.remove(proj, "weight")

    return mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.gate_proj.out_features


def prune_neuron_pairs(mlp, prune_frac):
    with torch.no_grad():
        gate_w = mlp.gate_proj.weight.data
        up_w   = mlp.up_proj.weight.data

        imp = gate_w.abs().max(dim=1).values + up_w.abs().max(dim=1).values

        orig = gate_w.size(0)
        n_prune = int(prune_frac * orig)
        keep_idx = torch.topk(imp, orig - n_prune, largest=True).indices.sort().values

        dtype = gate_w.dtype
        device = gate_w.device
        new_gate = torch.nn.Linear(mlp.gate_proj.in_features, keep_idx.size(0), bias=False, dtype=dtype).to(device)
        new_up   = torch.nn.Linear(mlp.up_proj.in_features, keep_idx.size(0), bias=False, dtype=dtype).to(device)
        new_down = torch.nn.Linear(keep_idx.size(0), mlp.down_proj.out_features, bias=False, dtype=dtype).to(device)

        new_gate.weight.copy_(gate_w[keep_idx])
        new_up.weight.copy_(up_w[keep_idx])
        new_down.weight.copy_(mlp.down_proj.weight.data[:, keep_idx])

        return new_gate, new_up, new_down, keep_idx.size(0)


def update_model(model, prune_percent, structured=True):
   
   new_intermediate_size = None

   for idx, layer in enumerate(model.model.layers):

       mlp = layer.mlp

       if structured:
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(mlp, prune_percent)
       else:
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs_unstructured(mlp, prune_percent)

       mlp.gate_proj = new_gate_proj
       mlp.up_proj = new_up_proj
       mlp.down_proj = new_down_proj

       if new_intermediate_size is None:
           new_intermediate_size = new_size

   model.config.intermediate_size = new_intermediate_size

   return model

def load_model(model_path: str):
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=HUGGING_FACE_API_KEY,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=HUGGING_FACE_API_KEY, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem0 = torch.cuda.memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    return model, tokenizer, mem0
