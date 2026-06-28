"""
extract_ollama.py

POC com Qwen2.5-VL via Ollama para os 3 casos de uso do desafio:

  1) CNH (documento estruturado)         -> extract_cnh()
  2) Documento extenso (Claude 3 paper) -> extract_long_document()
  3) Fatura de energia (layout complexo) -> extract_invoice()
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

from ollama import chat
from PIL import Image
from tqdm import tqdm

from config import (
    DEFAULT_DOCS,
    DEFAULT_OUTPUTS,
    OUTPUTS_DIR,
    MODEL_3B,
    MODEL_7B,
    MODEL_7B_AWQ,
    RuntimeConfig,
    add_runtime_args,
    resolve_doc,
    runtime_from_args,
)
from schemas import CNH_SCHEMA, FATURA_ENERGIA_SCHEMA, FATURA_EXTRACTION_PROMPT

OLLAMA_MODEL_3B = "qwen2.5vl:3b"
OLLAMA_MODEL_7B = "qwen2.5vl:7b"
DEFAULT_OLLAMA_MODEL = OLLAMA_MODEL_3B

_HF_TO_OLLAMA: dict[str, str] = {
    MODEL_3B: OLLAMA_MODEL_3B,
    MODEL_7B: OLLAMA_MODEL_7B,
    MODEL_7B_AWQ: OLLAMA_MODEL_7B,
}

_runtime: RuntimeConfig | None = None
_model_id: str = DEFAULT_OLLAMA_MODEL
_show_progress: bool = True


def set_show_progress(show: bool) -> None:
    global _show_progress
    _show_progress = show


def resolve_ollama_model(model_id: str | None = None) -> str:
    if model_id is None:
        return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return _HF_TO_OLLAMA.get(model_id, model_id)


def set_runtime(config: RuntimeConfig, model_id: str | None = None) -> None:
    global _runtime, _model_id
    _runtime = config
    _model_id = resolve_ollama_model(model_id)


def get_runtime() -> RuntimeConfig:
    if _runtime is None:
        set_runtime(RuntimeConfig())
    return _runtime  # type: ignore[return-value]


def get_model_id() -> str:
    return _model_id


def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def load_pages(path: str | Path) -> list[bytes]:
    path = Path(path)
    config = get_runtime()

    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        if _show_progress:
            tqdm.write(f"Convertendo PDF: {path.name}...")
        pages = convert_from_path(str(path), dpi=config.pdf_dpi)
        if config.max_pages is not None:
            pages = pages[: config.max_pages]
        iterator = tqdm(
            pages,
            desc="Preparando páginas",
            unit="pág",
            leave=False,
            disable=not _show_progress,
        )
        return [_image_to_bytes(page) for page in iterator]

    return [_image_to_bytes(Image.open(path).convert("RGB"))]


def _run_inference(
    image_bytes: bytes,
    prompt: str,
    *,
    schema: dict | None = None,
    max_new_tokens: int = 2048,
    desc: str | None = "Inferência",
) -> str:
    kwargs: dict = {
        "model": get_model_id(),
        "messages": [
            {"role": "user", "content": prompt, "images": [image_bytes]},
        ],
        "options": {"temperature": 0, "num_predict": max_new_tokens},
    }
    if schema is not None:
        kwargs["format"] = schema

    if not _show_progress or desc is None:
        response = chat(**kwargs)
        return response.message.content or ""

    kwargs["stream"] = True
    parts: list[str] = []
    with tqdm(
        total=max_new_tokens,
        desc=desc,
        unit="tok",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for chunk in chat(**kwargs):
            if chunk.message.content:
                parts.append(chunk.message.content)
            if chunk.eval_count is not None:
                pbar.n = min(chunk.eval_count, max_new_tokens)
                pbar.refresh()
            if chunk.done:
                if chunk.eval_count is not None:
                    pbar.total = chunk.eval_count
                    pbar.n = chunk.eval_count
                pbar.refresh()
                break

    return "".join(parts)


def _safe_json_parse(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def extract_cnh(image_path: str | Path) -> dict:
    pages = load_pages(image_path)
    prompt = (
        "Você é um sistema de extração de dados de documentos brasileiros. "
        "Extraia da CNH (Carteira Nacional de Habilitação) os seguintes campos: "
        "nome, cpf, data_nascimento, data_emissao, filiacao_pai, filiacao_mae. "
        f"Responda APENAS com um JSON válido seguindo este schema, sem nenhum "
        f"texto antes ou depois: {json.dumps(CNH_SCHEMA, ensure_ascii=False)}"
    )
    raw = _run_inference(
        pages[0], prompt, schema=CNH_SCHEMA, max_new_tokens=1024, desc="CNH"
    )
    return _safe_json_parse(raw)


def extract_long_document(pdf_path: str | Path) -> str:
    pages = load_pages(pdf_path)
    prompt_pagina = (
        "Transcreva o conteúdo textual desta página mantendo a organização "
        "do layout (títulos, parágrafos e, principalmente, tabelas) em "
        "formato Markdown. Tabelas devem virar tabelas Markdown. "
        "Se houver gráficos, diagramas ou imagens não-textuais, descreva "
        "objetivamente o que eles mostram em uma seção "
        "'> **Figura:** ...' imediatamente após o ponto onde aparecem.\n\n"
        "IMPORTANTE — distinção de rodapés:\n"
        "- Referências bibliográficas (citadas como [N] no corpo do texto) "
        "devem ser mantidas como [N] onde aparecem no texto.\n"
        "- Notas de rodapé de produto/interface (que aparecem abaixo de uma "
        "linha horizontal no final da página e NÃO são citadas no texto "
        "corrido) devem ser transcritas numa seção separada ao final, "
        "precedida por '---\\n> **Nota de rodapé:**', sem reutilizar a "
        "numeração [N] das referências bibliográficas."
    )

    markdown_partes = []
    total = len(pages)
    for i, page_bytes in enumerate(
        tqdm(
            pages,
            desc="Páginas",
            unit="pág",
            disable=not _show_progress,
        ),
        start=1,
    ):
        texto_pagina = _run_inference(
            page_bytes,
            prompt_pagina,
            max_new_tokens=2048,
            desc=f"Página {i}/{total}",
        )
        markdown_partes.append(f"<!-- página {i} -->\n{texto_pagina}")

    return "\n\n".join(markdown_partes)


def extract_invoice(image_path: str | Path) -> dict:
    pages = load_pages(image_path)
    raw = _run_inference(
        pages[0],
        FATURA_EXTRACTION_PROMPT,
        schema=FATURA_ENERGIA_SCHEMA,
        max_new_tokens=2048,
        desc="Fatura",
    )
    return _safe_json_parse(raw)


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
    parser.add_argument("caso", choices=["cnh", "fatura", "longdoc"])
    parser.add_argument(
        "arquivo",
        nargs="?",
        help=f"Caminho do documento (default: docs/). Opções: {list(DEFAULT_DOCS)}",
    )
    parser.add_argument("--out", help="Caminho de saída (usado em 'longdoc')")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desativa barras de progresso",
    )
    add_runtime_args(parser)
    args = parser.parse_args()

    set_show_progress(not args.no_progress)

    config = runtime_from_args(args)
    ollama_model = _resolve_runtime_model(args)
    set_runtime(config, ollama_model)
    doc_path = resolve_doc(args.caso, args.arquivo)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.caso == "cnh":
        resultado = extract_cnh(doc_path)
        conteudo = json.dumps(resultado, ensure_ascii=False, indent=2)
        out_path = Path(args.out) if args.out else DEFAULT_OUTPUTS["cnh"]
        out_path.write_text(conteudo, encoding="utf-8")
        print(conteudo)
        print(f"\nSalvo em {out_path}", file=sys.stderr)
    elif args.caso == "fatura":
        resultado = extract_invoice(doc_path)
        conteudo = json.dumps(resultado, ensure_ascii=False, indent=2)
        out_path = Path(args.out) if args.out else DEFAULT_OUTPUTS["fatura"]
        out_path.write_text(conteudo, encoding="utf-8")
        print(conteudo)
        print(f"\nSalvo em {out_path}", file=sys.stderr)
    elif args.caso == "longdoc":
        markdown = extract_long_document(doc_path)
        out_path = Path(args.out) if args.out else DEFAULT_OUTPUTS["longdoc"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Markdown salvo em {out_path}")


if __name__ == "__main__":
    try:
        from ollama import chat as _chat  # noqa: F401
    except ImportError:
        sys.exit("Instale ollama: pip install ollama")
    main()
