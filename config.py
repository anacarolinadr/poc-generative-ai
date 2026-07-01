"""
Configuração central da POC Qwen2.5-VL.

Presets:
  - lower:  3B em 4-bit (~6 GB VRAM)
  - higher: 7B-AWQ (~6–8 GB; GPU consumer / L4 / Colab T4 16 GB)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_7B_AWQ = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_DOCS: dict[str, Path] = {
    "cnh": DOCS_DIR / "Documento 1.jpeg",
    "fatura": DOCS_DIR / "Documento 2.jpg",
    "longdoc": DOCS_DIR / "Documento 3.pdf",
}

DEFAULT_OUTPUTS: dict[str, Path] = {
    "cnh": OUTPUTS_DIR / "cnh.json",
    "fatura": OUTPUTS_DIR / "fatura.json",
    "longdoc": OUTPUTS_DIR / "documento_extenso.md",
}

PATCH_SIZE = 28

LONGDOC_MAX_NEW_TOKENS = 8192
LONGDOC_NUM_CTX = 32768


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str = MODEL_7B
    quant: Literal["none", "4bit"] = "none"
    min_pixels: int = 256 * PATCH_SIZE * PATCH_SIZE
    max_pixels: int = 768 * PATCH_SIZE * PATCH_SIZE
    device_map: str = "auto"
    max_pages: int | None = None
    page_range: tuple[int, int] | None = None
    pdf_dpi: int = 150

    @property
    def is_awq(self) -> bool:
        return self.model_id.endswith("-AWQ")


PRESETS: dict[str, RuntimeConfig] = {
    "lower": RuntimeConfig(
        model_id=MODEL_3B,
        quant="4bit",
        max_pixels=512 * PATCH_SIZE * PATCH_SIZE,
        pdf_dpi=120,
    ),
    "higher": RuntimeConfig(
        model_id=MODEL_7B_AWQ,
        quant="none",
        max_pixels=1280 * PATCH_SIZE * PATCH_SIZE,
        pdf_dpi=150,
    ),
}


def parse_page_range(value: str) -> tuple[int, int]:
    """Converte '5' ou '3-7' em (first, last) 1-indexed."""
    value = value.strip()
    if "-" in value:
        first_str, last_str = value.split("-", 1)
        first, last = int(first_str.strip()), int(last_str.strip())
    else:
        first = last = int(value)
    if first < 1 or last < 1:
        raise ValueError(f"Páginas devem ser >= 1, recebido: {value!r}")
    if first > last:
        raise ValueError(f"Página inicial ({first}) não pode ser maior que a final ({last})")
    return first, last


def resolve_doc(caso: str, arquivo: str | Path | None = None) -> Path:
    if arquivo is not None:
        path = Path(arquivo)
    else:
        if caso not in DEFAULT_DOCS:
            raise ValueError(f"Caso desconhecido: {caso}")
        path = DEFAULT_DOCS[caso]
    if not path.exists():
        raise FileNotFoundError(f"Documento não encontrado: {path}")
    return path


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Preset de hardware (lower=3B 4-bit, higher=7B-AWQ)",
    )
    parser.add_argument("--model", help="ID do modelo no Hugging Face")
    parser.add_argument(
        "--quant",
        choices=["none", "4bit"],
        help="Quantização via bitsandbytes (recomendado para 3B)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        help=f"Mínimo de pixels visuais (default: 256*{PATCH_SIZE}²)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Máximo de pixels visuais (reduza para caber em 6 GB VRAM)",
    )
    parser.add_argument("--device-map", default=None, help="device_map do transformers")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limita páginas processadas em longdoc (útil para testes rápidos)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Página(s) do PDF a processar. Ex: '5' ou '3-7'",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=None,
        help="DPI ao converter PDF em imagens",
    )


def runtime_from_args(args: argparse.Namespace) -> RuntimeConfig:
    base = PRESETS.get(args.preset, RuntimeConfig()) if getattr(args, "preset", None) else RuntimeConfig()

    model_id = args.model or os.getenv("QWEN_MODEL_ID") or base.model_id
    quant = args.quant or os.getenv("QWEN_QUANT") or base.quant
    if quant not in ("none", "4bit"):
        quant = "none"

    min_pixels = args.min_pixels if args.min_pixels is not None else base.min_pixels
    max_pixels = args.max_pixels if args.max_pixels is not None else base.max_pixels
    device_map = args.device_map or base.device_map
    max_pages = args.max_pages if args.max_pages is not None else base.max_pages
    pdf_dpi = args.pdf_dpi if args.pdf_dpi is not None else base.pdf_dpi
    page_range = base.page_range
    if getattr(args, "pages", None):
        page_range = parse_page_range(args.pages)

    return RuntimeConfig(
        model_id=model_id,
        quant=quant,  # type: ignore[arg-type]
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        device_map=device_map,
        max_pages=max_pages,
        page_range=page_range,
        pdf_dpi=pdf_dpi,
    )
