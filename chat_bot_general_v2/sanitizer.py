"""
sanitizer.py
============
Pos-processamento do output da LLM antes de exibir ao usuario.

Natureza diferente dos guardrails:
  - Guardrail  -> BLOQUEIA a resposta inteira
  - Sanitizer  -> TRANSFORMA a resposta, substituindo trechos especificos

Tres camadas de substituicao aplicadas em sequencia:
  [1] PII automatico   -- regex para CPF, CNPJ, email, telefone, cartao,
                          CEP, IP, chaves de API
  [2] Palavras fixas   -- dicionario de termos -> substitutos (case-insensitive)
  [3] Regex customizado-- padroes avancados definidos em CUSTOM_REGEX_RULES

Todas as substituicoes sao SILENCIOSAS -- o usuario ve o texto limpo
sem saber que houve substituicao. O TurnRecord registra quais categorias
foram aplicadas para auditoria.

Uso:
    result = sanitize(text)
    if result.was_sanitized:
        # exibe result.sanitized_text
        # loga result.applied_rules para o TurnRecord
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Substituto padrao para PII -- silencioso mas auditavel internamente
# ---------------------------------------------------------------------------
_PII_PLACEHOLDER = "[REDACTED]"


# ---------------------------------------------------------------------------
# Regras PII automaticas -- padroes brasileiros + internacionais comuns
# Cada entrada: (nome_da_regra, padrao_regex, substituto)
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, str, str]] = [

    # CPF: 000.000.000-00 ou 00000000000
    ("CPF",
     r"\b\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\s]?\d{2}\b",
     _PII_PLACEHOLDER),

    # CNPJ: 00.000.000/0000-00 ou variacoes sem pontuacao
    ("CNPJ",
     r"\b\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[\/\s]?\d{4}[-\s]?\d{2}\b",
     _PII_PLACEHOLDER),

    # Cartao de credito: 16 digitos em grupos de 4 (Visa, Master, etc.)
    ("CARTAO_CREDITO",
     r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
     _PII_PLACEHOLDER),

    # Email: padrao RFC simplificado
    ("EMAIL",
     r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
     _PII_PLACEHOLDER),

    # Telefone brasileiro: (11) 99999-9999 ou variacoes
    ("TELEFONE",
     r"\b(?:\+55\s?)?(?:\(\d{2}\)|\d{2})[\s\-]?\d{4,5}[\s\-]?\d{4}\b",
     _PII_PLACEHOLDER),

    # CEP: 00000-000 ou 00000000
    ("CEP",
     r"\b\d{5}[\-\s]?\d{3}\b",
     _PII_PLACEHOLDER),

    # Endereco IP v4: 192.168.0.1
    ("IP_V4",
     r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
     _PII_PLACEHOLDER),

    # Chaves de API OpenAI: sk-... (20+ chars alfanumericos)
    ("API_KEY_OPENAI",
     r"\bsk-[A-Za-z0-9]{20,}\b",
     _PII_PLACEHOLDER),

    # Tokens HuggingFace: hf_...
    ("API_KEY_HF",
     r"\bhf_[A-Za-z0-9]{10,}\b",
     _PII_PLACEHOLDER),

    # JWT tokens: header.payload.signature
    ("JWT_TOKEN",
     r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b",
     _PII_PLACEHOLDER),
]


# ---------------------------------------------------------------------------
# Palavras fixas -- substitutos exatos (case-insensitive)
# Formato: "termo_original": "substituto"
# Expanda esta lista sem mudar nenhuma funcao.
# ---------------------------------------------------------------------------
WORD_SUBSTITUTIONS: dict[str, str] = {
    # Exemplos didaticos -- substitua pelos termos reais do seu contexto
    # "concorrente_x":    "[empresa]",
    # "sistema_legado":   "sistema atual",
    # "salario":          "[informacao confidencial]",

    # Termos ofensivos residuais com grafia alternativa (leetspeak)
    "p0rra":   "***",
    "merrd4":  "***",
    "c4ralho": "***",
}


# ---------------------------------------------------------------------------
# Regras regex customizadas -- para padroes avancados alem de PII
# Cada entrada: (nome_da_regra, padrao_regex, substituto)
# ---------------------------------------------------------------------------
CUSTOM_REGEX_RULES: list[tuple[str, str, str]] = [
    # Numero de processo judicial brasileiro: 0000000-00.0000.0.00.0000
    ("PROCESSO_JUDICIAL",
     r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b",
     _PII_PLACEHOLDER),

    # Numero de conta bancaria simplificado: 00000-0
    ("CONTA_BANCARIA",
     r"\b\d{5,6}-\d{1}\b",
     _PII_PLACEHOLDER),
]


# ---------------------------------------------------------------------------
# Resultado da sanitizacao
# ---------------------------------------------------------------------------
@dataclass
class SanitizeResult:
    """
    Resultado imutavel de uma operacao de sanitizacao.
    Registrado no TurnRecord para auditoria completa.
    """
    sanitized_text: str                        # texto apos substituicoes
    was_sanitized:  bool                       # houve alguma substituicao?
    applied_rules:  list  = field(default_factory=list)   # regras aplicadas
    rule_counts:    dict  = field(default_factory=dict)   # contagem por regra


# ---------------------------------------------------------------------------
# Engine de sanitizacao -- funcoes privadas
# ---------------------------------------------------------------------------

def _apply_pii_patterns(text: str, applied: list, counts: dict) -> str:
    """
    Aplica os padroes PII automaticos em sequencia.
    Registra em applied e counts quais regras foram acionadas.
    """
    for rule_name, pattern, replacement in _PII_PATTERNS:
        new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if n > 0:
            text = new_text
            applied.append(f"PII:{rule_name}")
            counts[f"PII:{rule_name}"] = counts.get(f"PII:{rule_name}", 0) + n
    return text


def _apply_word_substitutions(text: str, applied: list, counts: dict) -> str:
    """
    Aplica substituicoes de palavras fixas (case-insensitive).
    Usa word boundary \b para nao substituir dentro de palavras maiores.
    """
    for word, replacement in WORD_SUBSTITUTIONS.items():
        pattern  = r"\b" + re.escape(word) + r"\b"
        new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if n > 0:
            text = new_text
            rule = f"WORD:{word}"
            applied.append(rule)
            counts[rule] = counts.get(rule, 0) + n
    return text


def _apply_custom_regex(text: str, applied: list, counts: dict) -> str:
    """
    Aplica regras regex customizadas definidas em CUSTOM_REGEX_RULES.
    """
    for rule_name, pattern, replacement in CUSTOM_REGEX_RULES:
        new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if n > 0:
            text = new_text
            applied.append(f"REGEX:{rule_name}")
            counts[f"REGEX:{rule_name}"] = counts.get(f"REGEX:{rule_name}", 0) + n
    return text


# ---------------------------------------------------------------------------
# Funcao publica -- unica interface usada pelo app.py
# ---------------------------------------------------------------------------

def sanitize(text: str) -> SanitizeResult:
    """
    Aplica o pipeline completo de sanitizacao no texto.

    Ordem de aplicacao:
      [1] PII automatico   -- padroes brasileiros + internacionais
      [2] Palavras fixas   -- WORD_SUBSTITUTIONS (case-insensitive)
      [3] Regex customizado-- CUSTOM_REGEX_RULES

    Parametros
    ----------
    text : texto original (output da LLM antes de exibir ao usuario)

    Retorna
    -------
    SanitizeResult com texto sanitizado, flag de mudanca e log de regras
    """
    if not text:
        return SanitizeResult(sanitized_text=text, was_sanitized=False)

    applied: list = []
    counts:  dict = {}

    # Camada 1 -- PII automatico
    text = _apply_pii_patterns(text, applied, counts)

    # Camada 2 -- Palavras fixas
    text = _apply_word_substitutions(text, applied, counts)

    # Camada 3 -- Regex customizado
    text = _apply_custom_regex(text, applied, counts)

    return SanitizeResult(
        sanitized_text = text,
        was_sanitized  = len(applied) > 0,
        applied_rules  = applied,
        rule_counts    = counts,
    )


# ---------------------------------------------------------------------------
# Utilitario de diagnostico -- lista todas as regras ativas
# ---------------------------------------------------------------------------

def sanitizer_rules_summary() -> dict:
    """
    Retorna um resumo de todas as regras ativas por categoria.
    Util para exibir no painel de controle ou em logs de inicializacao.
    """
    return {
        "pii_patterns":       [name for name, _, _ in _PII_PATTERNS],
        "word_substitutions": list(WORD_SUBSTITUTIONS.keys()),
        "custom_regex":       [name for name, _, _ in CUSTOM_REGEX_RULES],
        "total_rules":        len(_PII_PATTERNS) + len(WORD_SUBSTITUTIONS) + len(CUSTOM_REGEX_RULES),
    }
