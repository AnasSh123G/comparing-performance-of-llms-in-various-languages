import os
import gc
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


HUGGING_FACE_API_KEY = "key"


def configure_devices(device_ids: str) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    return "cuda"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_quant_config(quant: str) -> BitsAndBytesConfig | None:
    if quant == "none":
        return None
    if quant == "4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quant == "8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    raise ValueError(f"quant must be 'none', '4', or '8', got {quant}")


def load_model(
    model_name: str,
    quant: str = "none",
    max_memory: dict | None = None,
    hf_token: str | None = None,
):

    token = hf_token or HUGGING_FACE_API_KEY
    if max_memory is None:
        max_memory = {0: "90GiB"}

    print(f"Loading model {model_name}  (quant={quant})")

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        token=token,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qconf = _build_quant_config(quant)
    dtype = torch.float16 if quant == "8" else torch.bfloat16

    extra_kwargs = {}
    if qconf is not None:
        extra_kwargs["quantization_config"] = qconf

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False,
        token=token,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        torch_dtype=dtype,
        **extra_kwargs,
    )

    mdl.eval()
    if mdl.config:
        mdl.config.pad_token_id = tok.pad_token_id

    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()

    return mdl, tok, mem0


def clean_gpu_mem(*objs) -> None:
    for o in objs:
        del o
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def quant_type_label(quant: str) -> str:
    return {"4": "4bit", "8": "8bit"}.get(quant, "no_quant")
