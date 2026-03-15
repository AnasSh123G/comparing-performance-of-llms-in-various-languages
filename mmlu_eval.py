"""
MMLU (Massive Multitask Language Understanding) evaluation.

Evaluates model accuracy on multiple-choice questions across many subjects,
supporting English, Arabic, and German datasets.
"""

import os
import gc
import csv
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import datasets
from tqdm import tqdm

import pruning
import utils


DEVICE = "cuda"


def seq_avg_prob(model, seq_ids, prefix_ids_tensor):
    # Return exp-averaged probability and negative log-likelihood for *seq_ids*.
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
    # Load the MMLU dataset for the given language.
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
        return datasets.load_dataset(
            "cais/mmlu", "all", split="test",
            token=utils.HUGGING_FACE_API_KEY, trust_remote_code=True,
        )


def _get_pre_prompt(lang: str) -> str:
    # Return the instruction pre-prompt for the given language.
    prompts = {
        'AR': "ستُقَدَّم لك مسألة رياضية أو سؤال مع أربعة خيارات مُقترحة كإجابات محتملة. اختر الإجابة الصحيحة من الخيارات المقدمة فقط. لا تُضِف أي كلمات إضافية في إجابتك. يجب أن تبدأ إجابتك بـ 'ANSWER:'.",
        'DE': "Du erhältst eine Mathematikaufgabe oder eine Frage mit vier Antwortmöglichkeiten. Wähle ausschließlich die korrekte Antwort aus den gegebenen Optionen. Füge deiner Antwort keine zusätzlichen Wörter hinzu. Deine Antwort muss mit ‚ANSWER:' beginnen.",
        'EN': "You will be given a math problem or a question with four choices provided as possible answers to the problem. Select the correct answer from the options provided only. Do not include any additional words in your answer. Your answer should start with 'ANSWER:'.",
    }
    return prompts[lang]


def _get_mmlu_name(lang: str) -> str:
    # Return the MMLU dataset name component for output file naming.
    names = {'AR': "mmlu_ar_diacritics_cleaned.csv", 'DE': "mmlu_de.csv", 'EN': "cais_mmlu_all"}
    return names[lang]


def _get_question_intro(lang: str) -> str:
    # Return the localised 'Here is the question' connector.
    intros = {'AR': "\n\nإليك السؤال:\n\n", 'DE': "\n\nHier ist die Frage:\n\n", 'EN': "\n\nHere is the question:\n\n"}
    return intros[lang]


def evaluate(model, tokenizer, dataset, lang: str, pre_prompt: str):
    # Evaluate the model on the MMLU dataset, returning results and summary stats.
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
        instruction = pre_prompt + _get_question_intro(lang)

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
    return (
        results,
        accuracy,
        statistics.mean(mem_readings) if mem_readings else 0.0,
        statistics.mean(resp_lens) if resp_lens else 0.0,
    )


def _build_summary_path(model_path, mmlu_name, prune, structured, quant):
    p = Path(model_path)
    safe_model = f"{p.parent.name}-{p.name}"
    base = "results/mmlu_pruning/remaining"

    if prune > 0.0:
        suffix = "structured" if structured else "unstructured"
        return f"{base}/results_{mmlu_name}_{safe_model}_pruning_{suffix}_{prune}_diacritics.csv"
    elif quant != 'none':
        quant_type = utils.quant_type_label(quant)
        return f"{base}/results_{mmlu_name}_{safe_model}_{quant_type}_diacritics.csv"
    else:
        return f"{base}/results_{mmlu_name}_{safe_model}_diacritics.csv"


def run(model_path, lang, structured=True, quant="none", prune=0.0):
    # Run the full MMLU evaluation pipeline.
    pre_prompt = _get_pre_prompt(lang)
    mmlu_name = _get_mmlu_name(lang)
    quant_type = utils.quant_type_label(quant)

    model, tokenizer = utils.load_model(model_path, quant)[:2]
    print(prune)
    if prune > 0.0:
        model = pruning.update_model(model=model, prune_percent=prune, structured=structured)
        print("model pruned with ", prune*100, " %\n")

    dataset = load_mmlu(lang)

    start = time.time()
    results, acc, mem_avg, resp_len_avg = evaluate(model, tokenizer, dataset, lang, pre_prompt)
    runtime_min = (time.time() - start) / 60

    summary_path = _build_summary_path(model_path, mmlu_name, prune, structured, quant)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    is_new = not os.path.exists(summary_path) or os.stat(summary_path).st_size == 0

    p = Path(model_path)
    safe_model = f"{p.parent.name}-{p.name}"

    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "model", "computation_time", "mem_usage", "mem_usage_max",
                "response_length", "accuracy", "lang", "quant_type",
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