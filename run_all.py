"""
run_all.py

Executa os 3 casos de uso sobre os documentos em docs/ e salva em outputs/.

Exemplos:
    python run_all.py --preset lower
    python run_all.py --preset higher
    python run_all.py --model Qwen/Qwen2.5-VL-3B-Instruct --quant 4bit --max-pages 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import DEFAULT_DOCS, DEFAULT_OUTPUTS, OUTPUTS_DIR, add_runtime_args, runtime_from_args
from extract_transformers import (
    extract_cnh,
    extract_invoice,
    extract_long_document,
    set_runtime,
    set_show_progress,
)
from tqdm import tqdm

try:
    from qwen_vl_utils import process_vision_info  # noqa: F401
except ImportError:
    process_vision_info = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["cnh", "fatura", "longdoc"],
        action="append",
        help="Executa apenas os casos indicados (pode repetir)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desativa barras de progresso",
    )
    add_runtime_args(parser)
    args = parser.parse_args()

    if process_vision_info is None:
        sys.exit("Instale qwen-vl-utils: pip install qwen-vl-utils")

    set_show_progress(not args.no_progress)

    config = runtime_from_args(args)
    set_runtime(config)

    casos = args.only or ["cnh", "fatura", "longdoc"]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Modelo: {config.model_id}")
    print(f"Quant:  {config.quant}")
    print(f"Pixels: min={config.min_pixels}, max={config.max_pixels}")
    if config.max_pages:
        print(f"Longdoc limitado a {config.max_pages} página(s)")
    print()

    for caso in tqdm(casos, desc="Casos", unit="caso", disable=args.no_progress):
        doc = DEFAULT_DOCS[caso]
        out = DEFAULT_OUTPUTS[caso]
        tqdm.write(f"[{caso}] {doc.name} -> {out.name}")

        if caso == "cnh":
            resultado = extract_cnh(doc)
            out.write_text(
                json.dumps(resultado, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif caso == "fatura":
            resultado = extract_invoice(doc)
            out.write_text(
                json.dumps(resultado, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            markdown = extract_long_document(doc)
            out.write_text(markdown, encoding="utf-8")

        tqdm.write(f"  OK: {out}")

    print("Concluído.")


if __name__ == "__main__":
    main()
