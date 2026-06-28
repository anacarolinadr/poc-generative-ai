"""
extract_vllm_single_step.py

Demonstra a arquitetura de 1 passo (extração + formatação) via vLLM:

  - CNH e fatura: guided_json (JSON garantido pelo schema)
  - longdoc: inferência livre em Markdown (guided_json não se aplica a texto longo)

PASSO 1 — Suba o servidor (Linux/Colab com GPU):

    pip install -r requirements/requirements-vllm.txt

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \\
        --port 8000 --host 0.0.0.0 \\
        --quantization awq --dtype float16 \\
        --limit-mm-per-prompt image=1

PASSO 2 — Rode este script como cliente:

    python extract_vllm_single_step.py cnh
    python extract_vllm_single_step.py fatura docs/Documento\\ 2.jpg
    python extract_vllm_single_step.py longdoc --max-pages 3 --out outputs/documento_extenso.md
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import requests

from config import (
    DEFAULT_DOCS,
    DEFAULT_OUTPUTS,
    OUTPUTS_DIR,
    MODEL_7B_AWQ,
    RuntimeConfig,
    add_runtime_args,
    resolve_doc,
    runtime_from_args,
)
from schemas import CNH_SCHEMA, FATURA_ENERGIA_SCHEMA, FATURA_EXTRACTION_PROMPT

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.getenv("QWEN_MODEL_ID", MODEL_7B_AWQ)

LONGDOC_PROMPT = (
    "Transcreva o conteúdo textual desta página mantendo a organização "
    "do layout (títulos, parágrafos e, principalmente, tabelas) em "
    "formato Markdown. Tabelas devem virar tabelas Markdown. "
    "Se houver gráficos, diagramas ou imagens não-textuais, descreva "
    "objetivamente o que eles mostram em uma seção "
    "'> **Figura:** ...' imediatamente após o ponto onde aparecem."
)


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def _encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _encode_image(path: str | Path) -> str:
    path = Path(path)
    return _encode_image_bytes(path.read_bytes(), _mime_for_path(path))


def _load_pages(path: str | Path, config: RuntimeConfig) -> list[tuple[Path, bytes, str]]:
    path = Path(path)
    pages: list[tuple[Path, bytes, str]] = []

    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=config.pdf_dpi)
        if config.max_pages is not None:
            images = images[: config.max_pages]

        for image in images:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
            pages.append((path, buffer.getvalue(), "image/jpeg"))
        return pages

    mime = _mime_for_path(path)
    pages.append((path, path.read_bytes(), mime))
    return pages


def _chat_completion(
    *,
    model: str,
    instrucao: str,
    image_data_url: str,
    guided_json: dict | None = None,
    max_tokens: int = 2048,
) -> str:
    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instrucao},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if guided_json is not None:
        payload["guided_json"] = guided_json

    resp = requests.post(VLLM_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_structured(
    image_path: str | Path,
    schema: dict,
    instrucao: str,
    *,
    model: str = DEFAULT_MODEL,
) -> dict:
    content = _chat_completion(
        model=model,
        instrucao=instrucao,
        image_data_url=_encode_image(image_path),
        guided_json=schema,
        max_tokens=1024,
    )
    return json.loads(content)


def extract_long_document(
    pdf_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    config: RuntimeConfig | None = None,
) -> str:
    config = config or RuntimeConfig()
    pages = _load_pages(pdf_path, config)

    markdown_partes = []
    for i, (_, page_bytes, mime) in enumerate(pages, start=1):
        data_url = _encode_image_bytes(page_bytes, mime)
        texto = _chat_completion(
            model=model,
            instrucao=LONGDOC_PROMPT,
            image_data_url=data_url,
            guided_json=None,
            max_tokens=2048,
        )
        markdown_partes.append(f"<!-- página {i} -->\n{texto}")

    return "\n\n".join(markdown_partes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caso", choices=["cnh", "fatura", "longdoc"])
    parser.add_argument(
        "arquivo",
        nargs="?",
        help=f"Caminho do documento (default: docs/). Opções: {list(DEFAULT_DOCS)}",
    )
    parser.add_argument("--vllm-url", default=None, help="URL do endpoint vLLM")
    parser.add_argument("--out", help="Caminho de saída (usado em 'longdoc')")
    add_runtime_args(parser)
    args = parser.parse_args()

    global VLLM_URL
    if args.vllm_url:
        VLLM_URL = args.vllm_url

    runtime = runtime_from_args(args)
    model = runtime.model_id
    if args.model is None and args.preset is None and not os.getenv("QWEN_MODEL_ID"):
        model = DEFAULT_MODEL
    doc_path = resolve_doc(args.caso, args.arquivo)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.caso == "cnh":
        schema, instrucao = CNH_SCHEMA, (
            "Extraia da CNH: nome, cpf, data_nascimento, data_emissao, "
            "filiacao_pai, filiacao_mae."
        )
        resultado = extract_structured(doc_path, schema, instrucao, model=model)
        conteudo = json.dumps(resultado, ensure_ascii=False, indent=2)
        out_path = Path(args.out) if args.out else DEFAULT_OUTPUTS["cnh"]
        out_path.write_text(conteudo, encoding="utf-8")
        print(conteudo)
        print(f"\nSalvo em {out_path}", file=sys.stderr)
    elif args.caso == "fatura":
        schema, instrucao = FATURA_ENERGIA_SCHEMA, FATURA_EXTRACTION_PROMPT
        resultado = extract_structured(doc_path, schema, instrucao, model=model)
        conteudo = json.dumps(resultado, ensure_ascii=False, indent=2)
        out_path = Path(args.out) if args.out else DEFAULT_OUTPUTS["fatura"]
        out_path.write_text(conteudo, encoding="utf-8")
        print(conteudo)
        print(f"\nSalvo em {out_path}", file=sys.stderr)
    else:
        markdown = extract_long_document(doc_path, model=model, config=runtime)
        out_path = Path(args.out) if args.out else DEFAULT_OUTPUTS["longdoc"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Markdown salvo em {out_path}")


if __name__ == "__main__":
    main()
