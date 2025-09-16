import os
devices = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = devices
import gc, csv, re, math, time, statistics
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import pruning

HUGGING_FACE_API_KEY = 'key'

DEVICE = torch.device("cuda")

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, quant: str):
    print(f"Loading model {model_name} -> {DEVICE}")

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        token=HUGGING_FACE_API_KEY,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if quant == "none":
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
            token=HUGGING_FACE_API_KEY,
            device_map="auto",
            max_memory={0: "90GiB"},
            torch_dtype=torch.bfloat16,
        )

    elif quant == "4":
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
            token=HUGGING_FACE_API_KEY,
            device_map="auto",
            max_memory={0: "90GiB"},
            quantization_config=qconf,
        )

    elif quant == "8":
        qconf = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
            token=HUGGING_FACE_API_KEY,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={0: "90GiB"},
            quantization_config=qconf,
        )
    else:
        raise ValueError(f"quant must be 'none', '4', or '8', got {quant}")
    
    print("\nLoaded Model: ", model_name, "\n Quantization: ", quant)

    mdl.eval()
    if mdl.config:
        mdl.config.pad_token_id = tok.pad_token_id

    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()

    return mdl, tok, mem0



def calculate_logs(
    prompts: List[str],
    model,
    tokenizer,
    mem_before: float,
    mem_readings: list,
    use_cache: bool = True,
) -> Tuple[List[float], List[float], float]:
    enc = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    bos = torch.full((enc.input_ids.size(0), 1), tokenizer.bos_token_id, device=DEVICE)
    ids = torch.cat([bos, enc.input_ids], dim=1)

    with torch.inference_mode():
        logits = model(ids, use_cache=use_cache).logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)

    target = ids[:, 1:]
    mask = (target != tokenizer.pad_token_id).float()
    tok_logp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1) * mask

    log_sum = tok_logp.sum(1)
    tok_count = mask.sum(1)
    avg_logp = log_sum / tok_count
    ppl = torch.exp(-avg_logp)

    torch.cuda.synchronize()

    mem_total = 0.0
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_b, total_b = torch.cuda.mem_get_info()
            mem_total += (total_b - free_b) / 1e6

    mem_readings.append(mem_total)

    log_sum_val = log_sum.tolist()
    if len(prompts) == 1:
        log_sum_val = log_sum_val[0]

    return avg_logp.tolist(), ppl.tolist(), log_sum_val

def evaluate_bias(df: pd.DataFrame, model, tokenizer, mem_before: float, tag: str):
    csv_file = f"row_metrics_{tag}_pruning_en.csv"
    header = [
        "row_idx",
        "avg_ppl_sent_more",
        "avg_ppl_sent_less",
        "logp_sent_more",
        "logp_sent_less",
        "ll_sent_more",
        "ll_sent_less",
    ]

    with open(csv_file, mode="w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        mem_readings = []
        ppls_more, ppls_less = [], []
        logps_more, logps_less = [], []
        lls_more, lls_less = [], []
        logp_diff_sum = 0.0
        row_buf = []

        for i in tqdm(range(len(df)), desc="Processing prompts"):
            sent_more = df.iloc[i]["sent_more"]
            sent_less = df.iloc[i]["sent_less"]

            avg_logp_more, ppl_more, ll_more = calculate_logs(
                [sent_more], model, tokenizer, mem_before, mem_readings
            )
            avg_logp_less, ppl_less, ll_less = calculate_logs(
                [sent_less], model, tokenizer, mem_before, mem_readings
            )

            ppls_more.append(ppl_more[0])
            ppls_less.append(ppl_less[0])
            logps_more.append(avg_logp_more[0])
            logps_less.append(avg_logp_less[0])
            lls_more.append(ll_more)
            lls_less.append(ll_less)

            row_buf.append(
                [
                    i,
                    ppl_more[0],
                    ppl_less[0],
                    avg_logp_more[0],
                    avg_logp_less[0],
                    ll_more,
                    ll_less,
                ]
            )

            if len(row_buf) == 100 or i == len(df) - 1:
                writer.writerows(row_buf)
                row_buf = []

            logp_diff_sum += avg_logp_more[0] - avg_logp_less[0]

    avg_mem = statistics.mean(mem_readings) if mem_readings else 0.0
    stats = dict(
        avg_logp_diff=logp_diff_sum / len(df),
        mem_avg=avg_mem,
        ppl_more=statistics.mean(ppls_more),
        ppl_less=statistics.mean(ppls_less),
        avg_ll_more=statistics.mean(lls_more),
        avg_ll_less=statistics.mean(lls_less),
        avg_logps_sent_more=statistics.mean(logps_more),
        avg_logps_sent_less=statistics.mean(logps_less),
    )
    return stats

def execute(model_path: str, dataset_path: str, quant='none', prune = 0.0, structured=True):
    
    set_seed()
    model, tokenizer, mem_before = load_model(model_path, quant)
    if prune > 0.0:
        model = pruning.update_model(model=model, prune_percent=prune, structured=structured)
        print("model pruned with ", prune*100, " %\n")
    df = pd.read_csv(dataset_path)

    tag = f"{Path(model_path).name}_{Path(dataset_path).stem}"
    t0 = time.time()
    stats = evaluate_bias(df, model, tokenizer, mem_before, tag)
    minutes = (time.time() - t0) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    if dataset_path == "datasets/crows_pairs.csv":
        out_file = "results/crows/results_crowspairs_on_fly_en.csv"
    elif dataset_path == "datasets/crows_pairs_translated_fully_diacriticized.csv":
        out_file = "results/crows/results_crowspairs_on_fly_ar_diacriticized.csv"
    else:
        out_file = "results/crows/results_crowspairs_on_fly_de.csv"
    with open(out_file, mode="a", newline="") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(
                [
                    "model",
                    "prune_percentage",
                    "avg_logp_diff",
                    "time_min",
                    "avg_mem_MB",
                    "peak_mem_MB",
                    "ppl_more",
                    "ppl_less",
                    "avg_logps_sent_more",
                    "avg_logps_sent_less",
                    "avg_ll_more",
                    "avg_ll_less",
                    "quantization",
                ]
            )
        writer.writerow(
            [
                Path(model_path).name,
                prune,
                round(stats["avg_logp_diff"], 3),
                round(minutes, 1),
                int(stats["mem_avg"]),
                int(peak_mem),
                stats["ppl_more"],
                stats["ppl_less"],
                stats["avg_logps_sent_more"],
                stats["avg_logps_sent_less"],
                stats["avg_ll_more"],
                stats["avg_ll_less"],
                quant,
            ]
        )

    print(f"{Path(model_path).name}:    Δlogp={stats['avg_logp_diff']:.3f}   time={minutes:.1f}m")
