#!/usr/bin/env python3
"""
main.py — Unified CLI entry point for all LLM evaluation benchmarks.

Usage examples
--------------
    python main.py crowspairs --model meta-llama/Llama-3-8B --dataset datasets/crows_pairs.csv
    python main.py discrim   --model meta-llama/Llama-3-8B --lang EN
    python main.py mmlu      --model meta-llama/Llama-3-8B --lang AR --quant 4
"""

import argparse

import utils


def cmd_crowspairs(args):
    # CrowS-Pairs
    import crowspairs_eval
    crowspairs_eval.execute(
        model_path=args.model,
        dataset_path=args.dataset,
        quant=args.quant,
        prune=args.prune,
        structured=args.structured,
    )


def cmd_discrim(args):
    # Discrim-Eval
    import discrim_eval
    discrim_eval.get_scores(
        model_id=args.model,
        lang=args.lang,
        structured=args.structured,
        quant=args.quant,
        prune=args.prune,
        diacritics=args.diacritics,
    )


def cmd_mmlu(args):
    # MMLU
    import mmlu_eval
    mmlu_eval.run(
        model_path=args.model,
        lang=args.lang,
        structured=args.structured,
        quant=args.quant,
        prune=args.prune,
    )




def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--quant", choices=["none", "4", "8"], default="none",
                        help="Quantisation level (default: none)")
    parser.add_argument("--prune", type=float, default=0.0,
                        help="Pruning fraction, e.g. 0.2 for 20%% (default: 0.0)")
    parser.add_argument("--structured", action=argparse.BooleanOptionalAction, default=True,
                        help="Use structured pruning (default: True)")
    parser.add_argument("--devices", default="2",
                        help="Comma-separated CUDA device IDs (default: '2')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified entry point for LLM evaluation benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # crowspairs
    cp = subparsers.add_parser("crowspairs", help="CrowS-Pairs bias evaluation")
    _add_common_args(cp)
    cp.add_argument("--dataset", required=True,
                     help="Path to the CrowS-Pairs CSV (e.g. datasets/crows_pairs.csv)")
    cp.set_defaults(func=cmd_crowspairs)

    # discrim
    de = subparsers.add_parser("discrim", help="Discrim-Eval bias evaluation")
    _add_common_args(de)
    de.add_argument("--lang", choices=["EN", "AR", "DE"], required=True,
                     help="Evaluation language")
    de.add_argument("--diacritics", action="store_true", default=False,
                     help="Use diacritised Arabic dataset")
    de.set_defaults(func=cmd_discrim)

    # mmlu
    mm = subparsers.add_parser("mmlu", help="MMLU accuracy evaluation")
    _add_common_args(mm)
    mm.add_argument("--lang", choices=["EN", "AR", "DE"], required=True,
                     help="Evaluation language")
    mm.set_defaults(func=cmd_mmlu)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    utils.configure_devices(args.devices)
    utils.set_seed(args.seed)

    args.func(args)


if __name__ == "__main__":
    main()
