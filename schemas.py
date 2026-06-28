"""
Schemas JSON usados nas extrações estruturadas.
"""

import json

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
