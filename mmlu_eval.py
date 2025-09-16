
import os


devices = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = devices
import gc
import csv
import math
import random
import statistics
import time
from pathlib import Path
import pruning
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


HUGGING_FACE_API_KEY = "key"

DEVICE = "cuda"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seq_avg_prob(model, seq_ids, prefix_ids_tensor):
    """Return exp‑averaged probability and negative log‑likelihood for seq_ids."""
    full_ids = torch.cat([
        prefix_ids_tensor,
        torch.tensor(seq_ids, dtype=torch.long, device=prefix_ids_tensor.device),
    ])
    with torch.no_grad():
        logits = model(full_ids.unsqueeze(0), use_cache=False).logits[0]

    total_lp = 0.0
    offset = prefix_ids_tensor.size(0)
    for i, tok_id in enumerate(seq_ids):
        lp = F.log_softmax(logits[offset + i - 1], dim=-1)[tok_id]
        total_lp += lp.item()
    mean_lp = total_lp / len(seq_ids)
    return math.exp(mean_lp), -mean_lp

def load_mmlu(lang: str):
    if lang == 'AR':
        path = "datasets/mmlu_ar_diacritics_cleaned.csv"
        df = pd.read_csv(path)
        import ast
        df["choices"] = df["choices"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        df["answer"] = df["answer"].apply(lambda x: letter2idx.get(x, x))
        return df
    elif lang == 'DE':
        path = "datasets/mmlu_de.csv"
        df = pd.read_csv(path)
        import ast
        df["choices"] = df["choices"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        df["answer"] = df["answer"].apply(lambda x: letter2idx.get(x, x))
        return df
    elif lang == 'EN':
        return datasets.load_dataset("cais/mmlu", "all", split="test", token=HUGGING_FACE_API_KEY, trust_remote_code=True)

def load_model(model_name: str, quant: str = "none"):
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
            low_cpu_mem_usage=True,
            max_memory={0: "90GiB", 1: "90GiB"},
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
            low_cpu_mem_usage=True,
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
            low_cpu_mem_usage=True,
            max_memory={0: "90GiB"},
            quantization_config=qconf,
        )
    else:
        raise ValueError(f"quant must be 'none', '4', or '8', got {quant}")

    mdl.eval()
    if mdl.config:
        mdl.config.pad_token_id = tok.pad_token_id

    return mdl, tok


def evaluate(model, tokenizer, dataset, lang: str, pre_prompt: str):
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False) for c in "ABCD"]

    results, mem_readings, resp_lens = [], [], []
    mem_start = torch.cuda.memory_allocated(device=DEVICE)
    correct = 0

    for idx in tqdm(range(len(dataset)), desc="MMLU"):

        if lang == 'EN':
            row = dataset[idx]
        else:
            row = dataset.iloc[idx]
        choices = row["choices"]
        if not isinstance(choices, list) or len(choices) != 4:
            continue

        question, subject, gt_idx = row["question"], row["subject"], int(row["answer"])
        if lang == 'AR':
            instruction = pre_prompt + ("\n\nإليك السؤال:\n\n")
        elif lang == 'DE':
            instruction = pre_prompt + ("\n\nHier ist die Frage:\n\n")
        else:
            instruction = pre_prompt + ("\n\nHere is the question:\n\n")

        opts_block = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        user_prompt = f"{instruction}{question}\nOptions:\n{opts_block}\nAnswer:"

        chat = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "ANSWER:"},
        ]
        prefix_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(DEVICE).squeeze(0)
        if prefix_ids[-1] == tokenizer.eos_token_id:
            prefix_ids = prefix_ids[:-1]

        probs = [seq_avg_prob(model, ids, prefix_ids)[0] for ids in choice_ids]
        pred_idx = int(np.argmax(probs))
        correct += int(pred_idx == gt_idx)

        gen = model.generate(prefix_ids.unsqueeze(0), max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
        ans_text = tokenizer.decode(gen[0][prefix_ids.size(0):], skip_special_tokens=True)
        resp_lens.append(len(tokenizer.encode(ans_text, add_special_tokens=False)))

        mem_readings.append((torch.cuda.memory_allocated(device=DEVICE) - mem_start) / 1e6)

        results.append({
            "idx": idx,
            "Subject": subject,
            "Correct": int(pred_idx == gt_idx),
            "Predicted": chr(65 + pred_idx),
            "Ground_truth": gt_idx,
        })

    total = len(results) or 1
    accuracy = correct * 100 / total
    return results, accuracy, statistics.mean(mem_readings) if mem_readings else 0.0, statistics.mean(resp_lens) if resp_lens else 0.0




def clean_gpu_mem(*objs):
    for o in objs:
        del o
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def run(model_path, lang, structured=True, quant="none", prune=0.0):

    if lang == 'AR':
        mmlu_name = "mmlu_ar_diacritics_cleaned.csv"
        pre_prompt = "ستُقَدَّم لك مسألة رياضية أو سؤال مع أربعة خيارات مُقترحة كإجابات محتملة. اختر الإجابة الصحيحة من الخيارات المقدمة فقط. لا تُضِف أي كلمات إضافية في إجابتك. يجب أن تبدأ إجابتك بـ 'ANSWER:'."
    elif lang == 'DE':
        mmlu_name = "mmlu_de.csv"
        pre_prompt = "Du erhältst eine Mathematikaufgabe oder eine Frage mit vier Antwortmöglichkeiten. Wähle ausschließlich die korrekte Antwort aus den gegebenen Optionen. Füge deiner Antwort keine zusätzlichen Wörter hinzu. Deine Antwort muss mit ‚ANSWER:‘ beginnen."
    elif lang == 'EN':
        mmlu_name = "cais_mmlu_all"
        pre_prompt = "You will be given a math problem or a question with four choices provided as possible answers to the problem. Select the correct answer from the options provided only. Do not include any additional words in your answer. Your answer should start with ‘ANSWER:’."

    if quant == "8":
        quant_type = "8bit"
    elif quant == "4":
        quant_type = "4bit"
    else:
        quant_type = "no_quant"


    model, tokenizer = load_model(model_path, quant)
    print(prune)
    if prune > 0.0:
        model = pruning.update_model(model=model, prune_percent=prune, structured=structured)
        print("model pruned with ", prune*100, " %\n")
    
    dataset = load_mmlu(lang)

    start = time.time()
    results, acc, mem_avg, resp_len_avg = evaluate(model, tokenizer, dataset, lang, pre_prompt)
    runtime_min = (time.time() - start) / 60


    p = Path(model_path)
    safe_model = f"{p.parent.name}-{p.name}"

    if prune > 0.0:
        if structured:
            summary_path = f"results/mmlu_pruning/remaining/results_{mmlu_name}_{safe_model}_pruning_structured_{prune}_diacritics.csv"
        else:
            summary_path = f"results/mmlu_pruning/remaining/results_{mmlu_name}_{safe_model}_pruning_unstructured_{prune}_diacritics.csv"

    elif quant != 'none':
        print(quant_type)
        summary_path = f"results/mmlu_pruning/remaining/results_{mmlu_name}_{safe_model}_{quant_type}_diacritics.csv"
    else:
        summary_path = f"results/mmlu_pruning/remaining/results_{mmlu_name}_{safe_model}_diacritics.csv"

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    is_new = not os.path.exists(summary_path) or os.stat(summary_path).st_size == 0
    
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "model", "computation_time", "mem_usage", "mem_usage_max", "response_length", "accuracy", "lang","quant_type"
            ])
        w.writerow([
            safe_model,
            runtime_min,
            mem_avg,
            torch.cuda.max_memory_allocated() / 1e6,
            resp_len_avg,
            acc,
            lang,
            quant_type,
        ])
    print(f"Summary appended to {summary_path}\nAccuracy: {acc:.2f}%")

    torch.cuda.empty_cache()
    gc.collect()