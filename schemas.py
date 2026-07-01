"""
Schemas JSON usados nas extrações estruturadas.
"""

import json
import re

# Caso 1 — CNH (documento estruturado)
CNH_SCHEMA = {
    "type": "object",
    "properties": {
        "nome": {"type": "string"},
        "cpf": {"type": "string", "description": "Formato: 000.000.000-00"},
        "data_nascimento": {"type": "string", "description": "Formato: DD/MM/AAAA"},
        "data_emissao": {"type": "string", "description": "Formato: DD/MM/AAAA"},
        "filiacao_pai": {"type": "string"},
        "filiacao_mae": {"type": "string"},
    },
    "required": [
        "nome",
        "cpf",
        "data_nascimento",
        "data_emissao",
        "filiacao_pai",
        "filiacao_mae",
    ],
    "additionalProperties": False,
}

# Caso 3 — Fatura de energia (layout complexo, mas semi-estruturado)
FATURA_ENERGIA_SCHEMA = {
    "type": "object",
    "properties": {
        "titular": {"type": "string"},
        "numero_cliente": {"type": "string"},
        "endereco_instalacao": {"type": "string"},
        "mes_referencia": {"type": "string"},
        "data_vencimento": {"type": "string"},
        "valor_total": {"type": "string"},
        "consumo_kwh": {"type": "string"},
        "historico_consumo": {
            "type": "array",
            "description": (
                "Histórico de consumo em kWh por mês (ex: JUN'14, 160). "
                "NÃO incluir valores em R$, ICMS, PIS, COFINS nem componentes de cobrança."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "mes": {"type": "string", "description": "Mês de referência (ex: JUN'14)"},
                    "consumo_kwh": {"type": "string", "description": "Consumo em kWh"},
                },
                "required": ["mes", "consumo_kwh"],
            },
        },
        "componentes_cobranca": {
            "type": "array",
            "description": (
                "Composição da fatura em R$: Geração de Energia, Transmissão, "
                "Distribuição, Encargos Setoriais, etc. NÃO incluir impostos (ICMS/PIS/COFINS)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string"},
                    "valor": {"type": "string", "description": "Valor em R$"},
                },
                "required": ["nome", "valor"],
            },
        },
        "tributos": {
            "type": "array",
            "description": (
                "Impostos da fatura em R$: ICMS, PIS, COFINS (base, alíquota e valor). "
                "NÃO incluir Geração, Transmissão, Distribuição nem meses de consumo."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string"},
                    "valor": {"type": "string", "description": "Valor em R$"},
                },
                "required": ["nome", "valor"],
            },
        },
    },
    "required": [
        "titular",
        "numero_cliente",
        "mes_referencia",
        "data_vencimento",
        "valor_total",
        "consumo_kwh",
    ],
    "additionalProperties": True,
}

FATURA_EXTRACTION_PROMPT = (
    "Extraia os dados desta fatura de energia elétrica brasileira, preservando a "
    "associação entre rótulos e valores mesmo quando estiverem em blocos/colunas "
    "separados no layout.\n\n"
    "ATENÇÃO — três seções distintas (NÃO misture):\n"
    "1) historico_consumo: apenas meses e consumo em kWh (ex: JUN'14, 160). "
    "NÃO inclua valores em R$, ICMS, PIS, COFINS nem componentes de cobrança.\n"
    "2) componentes_cobranca: composição da fatura em R$ — Geração de Energia, "
    "Transmissão, Distribuição, Encargos Setoriais, etc.\n"
    "3) tributos: apenas impostos (ICMS, PIS, COFINS) com valores em R$.\n\n"
    f"Responda APENAS com um JSON válido seguindo este schema, sem texto extra: "
    f"{json.dumps(FATURA_ENERGIA_SCHEMA, ensure_ascii=False)}"
)

LONGDOC_PAGE_PROMPT = (
    "Transcreva o conteúdo textual desta página mantendo a organização "
    "do layout (títulos, parágrafos e, principalmente, tabelas) em "
    "formato Markdown. Tabelas devem virar tabelas Markdown. "
    "Responda com Markdown puro, sem envolver a saída em blocos ```.\n"
    "Preserve a numeração de seções do documento nos headings "
    "(ex.: '# 5.4 Behavioral Design', '## 5.4.1 Refusals').\n"
    "Se houver gráficos, diagramas ou imagens não-textuais, descreva "
    "objetivamente o que eles mostram em blockquote "
    "'> **Figura:** ...' imediatamente após o ponto onde aparecem "
    "(não use '**Figure:**' sem blockquote).\n\n"
    "IMPORTANTE — fidelidade de tabelas:\n"
    "- Cada linha visível do PDF vira uma linha da tabela Markdown.\n"
    "- Subcategorias indentadas (ex.: '→ Art & Design', '→ Business') "
    "devem ser linhas separadas, NÃO agrupadas com <br> numa única célula.\n"
    "- Não fundir linhas, omitir colunas nem simplificar a estrutura.\n\n"
    "IMPORTANTE — distinção entre referências bibliográficas e notas de rodapé:\n"
    "- Citações [N] no corpo do texto (parágrafos, tabelas) devem permanecer "
    "como [N] inline onde aparecem. Nunca mova para uma seção de rodapé.\n"
    "- Se um [N] já foi citado no corpo da página, NÃO o repita no final — "
    "mesmo que também apareça abaixo de uma linha horizontal (lista de "
    "referências ou continuação de bibliografia).\n"
    "- Blocos numerados [N] com autor, título ou URL no rodapé da página são "
    "referências bibliográficas: omita essa lista; mantenha apenas as citações "
    "[N] inline no texto transcrito.\n"
    "- Use a seção '---\\n> **Nota de rodapé:**' para notas explicativas "
    "abaixo de linha horizontal que NÃO são referências bibliográficas:\n"
    "  • condições de benchmark (†, 0-shot);\n"
    "  • legendas de tabela (ex.: [56] explicando scores GPT citado só no "
    "rodapé da tabela);\n"
    "  • atribuição de fonte de gráfico em texto corrido SEM formato "
    "'[N] Autor, título, URL' — ex.: 'Spring 2023 and 2022 Global Attitudes "
    "survey...' abaixo de um gráfico DEVE ser preservada nessa seção.\n"
    "- Não confunda: texto de survey/fonte de gráfico (sem autor/URL) é nota "
    "de rodapé; bloco '[58] Autor et al., ...' ou '[59] ...' com URL é "
    "bibliografia — omita.\n\n"
    "IMPORTANTE — término da transcrição:\n"
    "- Encerre imediatamente após o último parágrafo ou elemento visível do "
    "corpo da página (tabelas, listas, blockquote de figura).\n"
    "- NÃO inclua listas de referências bibliográficas do rodapé — nem "
    "parcialmente (ex.: '[58] Author et al., ...').\n"
    "- Se descrever uma figura em '> **Figura:**', complete a descrição "
    "inteira antes de encerrar.\n"
    "- Complete todas as frases até o ponto final; nunca interrompa no meio "
    "de uma sentença ou de uma citação inline [N]."
)


def normalize_longdoc_page(text: str) -> str:
    """Remove code fences, converte \\n literais e limpa artefatos de corte."""
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = re.sub(r"^```(?:markdown|md)?\s*\n?", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    text = text.replace("\\n", "\n")
    return _strip_longdoc_tail_artifacts(text).strip()


def _strip_longdoc_tail_artifacts(text: str) -> str:
    """Remove linhas finais incompletas (figura ou bibliografia omitida pelo prompt)."""
    lines = text.split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.startswith("> **Figura") and ": " not in last:
            lines.pop()
            continue
        if re.match(r">\s*\*\*Nota de rodapé:\*\*\s*\[\d+\]", last, re.I):
            lines.pop()
            continue
        if re.match(r"\[\d+\]\s*Author", last, re.I):
            lines.pop()
            continue
        break
    return "\n".join(lines)
