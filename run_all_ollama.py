"""
run_all_ollama.py

Executa os 3 casos de uso via Ollama sobre os documentos em docs/ e salva em outputs/.

Pré-requisitos:
    - App Ollama instalado e em execução
    - ollama pull qwen2.5vl:3b

Exemplos:
    python run_all_ollama.py
    python run_all_ollama.py --model qwen2.5vl:7b
    python run_all_ollama.py --only cnh --only fatura
    python run_all_ollama.py --max-pages 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from config import DEFAULT_DOCS, DEFAULT_OUTPUTS, OUTPUTS_DIR, add_runtime_args, runtime_from_args
from extract_ollama import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_MODEL_3B,
    OLLAMA_MODEL_7B,
    extract_cnh,
    extract_invoice,
    extract_long_document,
    resolve_ollama_model,
    set_runtime,
    set_show_progress,
)
from tqdm import tqdm


def _resolve_runtime_model(args: argparse.Namespace) -> str:
    if args.model:
        return resolve_ollama_model(args.model)
    if os.getenv("OLLAMA_MODEL"):
        return resolve_ollama_model(os.getenv("OLLAMA_MODEL"))
    if args.preset == "lower":
        return OLLAMA_MODEL_3B
    if args.preset == "higher":
        return OLLAMA_MODEL_7B
    return DEFAULT_OLLAMA_MODEL


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

    set_show_progress(not args.no_progress)

    try:
        import ollama  # noqa: F401
    except ImportError:
        sys.exit("Instale ollama: pip install ollama")

    config = runtime_from_args(args)
    ollama_model = _resolve_runtime_model(args)
    set_runtime(config, ollama_model)

    casos = args.only or ["cnh", "fatura", "longdoc"]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Modelo Ollama: {ollama_model}")
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
