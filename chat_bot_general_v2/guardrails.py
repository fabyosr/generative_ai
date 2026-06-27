"""
guardrails.py
=============
Pipeline de moderação leve — otimizado para Streamlit Cloud (zero dependências
pesadas, zero chamadas de rede adicionais).

Stack atual:
  [1] better-profanity  — léxico EN nativo + lista customizada PT-BR (~1ms)
  [2] OpenAI Moderation API — gratuita, sem tokens, quando provedor = openai

Camadas futuras (comentadas, prontas para habilitar em ambiente com mais recursos):
  [3] Detoxify   — requer compatibilidade transformers==4.30.x
  [4] LlamaGuard — requer token HF e latência aceitável

A principal linha de defesa é o system prompt em config.py (SAFETY POLICY),
que instrui o modelo a recusar conteúdo nocivo independentemente dessas camadas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from better_profanity import profanity

# ---------------------------------------------------------------------------
# Lista customizada PT-BR
# ---------------------------------------------------------------------------
PTBR_PROFANITY_LIST: list[str] = [
    # Palavrões comuns
    "merda", "porra", "caralho", "fodase", "foda-se",
    "buceta", "cu", "puta", "vagabunda", "arrombado", "babaca",
    "imbecil", "idiota", "otario", "otário", "desgraça", "desgraçado",
    # Abreviações ofensivas
    "fdp", "vsf", "tnc", "vtnc", "pqp", "krl",
    # Termos de ódio / discriminação
    "macaco", "nego", "crioulo",
    "viado", "viadinho", "sapatão", "traveco",
]

# ---------------------------------------------------------------------------
# Mensagens padrão ao usuário
# ---------------------------------------------------------------------------
BLOCK_MESSAGE_INPUT  = "⚠️ Sua mensagem foi bloqueada por violar as diretrizes de uso."
BLOCK_MESSAGE_OUTPUT = "⚠️ A resposta foi bloqueada por conter conteúdo inadequado."

# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

class GuardrailLayer(str, Enum):
    PROFANITY  = "better-profanity"
    OPENAI_MOD = "openai-moderation"
    SYSTEM_PROMPT = "system-prompt"   # reservado para log — o modelo recusou via SP
    NONE       = "none"


@dataclass
class GuardrailResult:
    safe:     bool
    layer:    GuardrailLayer
    category: str    # categoria detectada ou "safe"
    score:    float  # confiança [0.0 – 1.0]
    reason:   str    # texto legível para auditoria


# ---------------------------------------------------------------------------
# Inicialização lazy — feita uma única vez por processo
# ---------------------------------------------------------------------------
_profanity_initialized = False


def _init_profanity() -> None:
    global _profanity_initialized
    if not _profanity_initialized:
        profanity.load_censor_words(whitelist_words=[])
        profanity.add_censor_words(PTBR_PROFANITY_LIST)
        _profanity_initialized = True


# ---------------------------------------------------------------------------
# Camada 1 — better-profanity (léxico EN + PT-BR)
# ---------------------------------------------------------------------------

def _check_profanity(text: str) -> GuardrailResult:
    """Verificação léxica instantânea. Bloqueia palavrões EN e PT-BR."""
    _init_profanity()
    if profanity.contains_profanity(text):
        return GuardrailResult(
            safe     = False,
            layer    = GuardrailLayer.PROFANITY,
            category = "profanity",
            score    = 1.0,
            reason   = "Palavra proibida detectada (léxico EN/PT-BR).",
        )
    return GuardrailResult(
        safe=True, layer=GuardrailLayer.NONE,
        category="safe", score=0.0, reason="",
    )


# ---------------------------------------------------------------------------
# Camada 2 — OpenAI Moderation API (gratuita, ~150ms)
# Ativada automaticamente quando o provedor ativo for OpenAI.
# Categorias: hate, hate/threatening, harassment, self-harm,
#             sexual, sexual/minors, violence, violence/graphic
# ---------------------------------------------------------------------------

def _check_openai_moderation(text: str) -> GuardrailResult:
    """
    Chama o endpoint gratuito /v1/moderations da OpenAI.
    Não consome tokens nem aparece no billing.
    Retorna resultado seguro (fail-open) em caso de erro de rede.
    """
    try:
        from openai import OpenAI

        client   = OpenAI()   # usa OPENAI_API_KEY do ambiente automaticamente
        response = client.moderations.create(input=text)
        result   = response.results[0]

        if result.flagged:
            # Identifica a categoria com maior score
            scores  = result.category_scores.model_dump()
            top_cat = max(scores, key=scores.get)
            top_score = float(scores[top_cat])

            return GuardrailResult(
                safe     = False,
                layer    = GuardrailLayer.OPENAI_MOD,
                category = top_cat,
                score    = top_score,
                reason   = f"OpenAI Moderation flagged: {top_cat} ({top_score:.2f})",
            )

        return GuardrailResult(
            safe=True, layer=GuardrailLayer.OPENAI_MOD,
            category="safe", score=0.0,
            reason="OpenAI Moderation: safe.",
        )

    except Exception as exc:
        # Fail-open: qualquer falha (rede, key ausente) não bloqueia o usuário
        return GuardrailResult(
            safe=True, layer=GuardrailLayer.OPENAI_MOD,
            category="error", score=0.0,
            reason=f"OpenAI Moderation indisponível: {exc}",
        )


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def check_input(text: str, provider: str = "") -> GuardrailResult:
    """
    Pipeline de moderação do INPUT do usuário.

    Fluxo:
      1. better-profanity (sempre)
      2. OpenAI Moderation API (só se provider == 'openai')

    O system prompt de segurança em config.py atua como terceira camada
    diretamente no modelo, cobrindo casos semânticos que léxico não pega.

    Parâmetros
    ----------
    text     : mensagem do usuário
    provider : chave do provedor ativo ('openai' | 'ollama' | 'hf_endpoint')
    """
    # Camada 1 — léxico
    result = _check_profanity(text)
    if not result.safe:
        return result

    # Camada 2 — OpenAI Moderation (somente quando provedor é OpenAI)
    if provider == "openai":
        result = _check_openai_moderation(text)
        if not result.safe:
            return result

    return GuardrailResult(
        safe=True, layer=GuardrailLayer.NONE,
        category="safe", score=0.0, reason="Todas as camadas aprovaram.",
    )


def check_output(text: str, provider: str = "") -> GuardrailResult:
    """
    Moderação do OUTPUT da LLM.
    Mesma stack do input — o modelo já foi instruído pelo system prompt,
    então o output tende a ser mais seguro; léxico + moderation são suficientes.
    """
    result = _check_profanity(text)
    if not result.safe:
        return result

    if provider == "openai":
        result = _check_openai_moderation(text)
        if not result.safe:
            return result

    return GuardrailResult(
        safe=True, layer=GuardrailLayer.NONE,
        category="safe", score=0.0, reason="Output aprovado.",
    )


# ---------------------------------------------------------------------------
# Utilitário de diagnóstico
# ---------------------------------------------------------------------------

def guardrail_status(provider: str = "") -> dict:
    """
    Retorna disponibilidade de cada camada para exibir na sidebar.
    """
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "system_prompt":   True,                          # sempre ativo
        "better_profanity": True,                         # sempre ativo
        "openai_moderation": provider == "openai" and openai_key,
        # Camadas futuras:
        "detoxify":    False,   # desabilitado: incompatibilidade Python 3.12+
        "llamaguard":  False,   # desabilitado: reservado para ambiente dedicado
    }
