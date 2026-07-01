"""
extract_transformers.py

POC com Qwen2.5-VL (transformers) para os 3 casos de uso do desafio:

  1) CNH (documento estruturado)         -> extract_cnh()
  2) Documento extenso (Claude 3 paper) -> extract_long_document()
  3) Fatura de energia (layout complexo) -> extract_invoice()
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import (
    DEFAULT_DOCS,
    DEFAULT_OUTPUTS,
    LONGDOC_MAX_NEW_TOKENS,
    OUTPUTS_DIR,
    RuntimeConfig,
    add_runtime_args,
    resolve_doc,
    runtime_from_args,
)
from schemas import CNH_SCHEMA, FATURA_ENERGIA_SCHEMA, FATURA_EXTRACTION_PROMPT, LONGDOC_PAGE_PROMPT, normalize_longdoc_page

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

_runtime: RuntimeConfig | None = None
_model = None
_processor = None
_show_progress: bool = True


def set_show_progress(show: bool) -> None:
    global _show_progress
    _show_progress = show


def set_runtime(config: RuntimeConfig) -> None:
    global _runtime, _model, _processor
    _runtime = config
    _model = None
    _processor = None


def get_runtime() -> RuntimeConfig:
    if _runtime is None:
        set_runtime(RuntimeConfig())
    return _runtime  # type: ignore[return-value]


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    config = get_runtime()
    processor_kwargs = {
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
    }

    load_kwargs: dict = {"device_map": config.device_map}

    if config.quant == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["torch_dtype"] = torch.float16
    elif config.is_awq:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = "auto"

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        **load_kwargs,
    )
    _processor = AutoProcessor.from_pretrained(config.model_id, **processor_kwargs)
    return _model, _processor


def load_pages(path: str | Path) -> list[Image.Image]:
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
        return [page.convert("RGB") for page in iterator]

    return [Image.open(path).convert("RGB")]


def _run_inference(
    image: Image.Image,
    prompt: str,
    *,
    schema: dict | None = None,
    max_new_tokens: int = 2048,
    desc: str | None = "Inferência",
) -> str:
    del schema, desc

    model, processor = load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


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
    config = get_runtime()
    page_offset = (config.page_range[0] - 1) if config.page_range else 0

    markdown_partes = []
    total = len(pages)
    for i, page in enumerate(
        tqdm(
            pages,
            desc="Páginas",
            unit="pág",
            disable=not _show_progress,
        ),
        start=1,
    ):
        pdf_page = page_offset + i
        texto_pagina = normalize_longdoc_page(
            _run_inference(
                page,
                LONGDOC_PAGE_PROMPT,
                max_new_tokens=LONGDOC_MAX_NEW_TOKENS,
                desc=f"Página {pdf_page} ({i}/{total})",
            )
        )
        markdown_partes.append(f"<!-- página {pdf_page} -->\n{texto_pagina}")

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
    set_runtime(runtime_from_args(args))
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
    if process_vision_info is None:
        sys.exit("Instale qwen-vl-utils: pip install qwen-vl-utils")
    main()
