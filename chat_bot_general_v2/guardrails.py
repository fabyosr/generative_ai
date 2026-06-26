"""
guardrails.py
=============
Pipeline de moderação em 3 camadas, aplicado em sequência crescente de custo:

  [1] better-profanity  — léxico EN nativo + lista customizada PT-BR
                          ~1ms, zero rede, primeira barreira óbvia

  [2] Detoxify          — modelo BERT local (multilíngue)
                          ~50ms, zero rede, classifica toxicidade com score
                          · score > DETOXIFY_BLOCK  → bloqueia direto
                          · score > DETOXIFY_REVIEW → escala para LlamaGuard

  [3] LlamaGuard        — modelo Meta fine-tunado para moderação via HF API
                          ~500ms, usa token HuggingFace já existente no projeto
                          acionado SOMENTE na zona cinza do Detoxify

Uso:
    result = check_input("mensagem do usuário")
    if not result.safe:
        # exibe aviso, não envia à LLM

    result = check_output("resposta da LLM")
    if not result.safe:
        # substitui resposta por mensagem padrão

Nenhuma dependência de Streamlit → 100% testável de forma isolada.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from better_profanity import profanity
from detoxify import Detoxify

# ---------------------------------------------------------------------------
# Configuração — altere thresholds aqui sem tocar em lógica
# ---------------------------------------------------------------------------

# Detoxify: bloqueia direto acima deste score
DETOXIFY_BLOCK_THRESHOLD: float = 0.70

# Detoxify: zona cinza — escala para LlamaGuard entre os dois valores
DETOXIFY_REVIEW_THRESHOLD: float = 0.40

# LlamaGuard no HuggingFace Inference API
LLAMAGUARD_REPO = "meta-llama/LlamaGuard-7b"

# Lista customizada PT-BR para better-profanity (didática — expanda conforme necessário)
PTBR_PROFANITY_LIST: list[str] = [
    "merda", "porra", "caralho", "fodase", "foda-se", "viado",
    "buceta", "cu", "puta", "vagabunda", "arrombado", "babaca",
    "imbecil", "idiota", "otario", "otário", "desgraça", "desgraçado",
    "fdp", "vsf", "tnc", "vtnc", "pqp", "krl",
    # termos de ódio / discriminação
    "macaco", "nego", "crioulo",          # contexto racial ofensivo
    "viadinho", "sapatão", "traveco",     # homofóbicos
]

# Mensagens padrão exibidas ao usuário quando bloqueado
BLOCK_MESSAGE_INPUT  = "⚠️ Sua mensagem foi bloqueada por violar as diretrizes de uso."
BLOCK_MESSAGE_OUTPUT = "⚠️ A resposta foi bloqueada por conter conteúdo inadequado."


# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

class GuardrailLayer(str, Enum):
    PROFANITY  = "better-profanity"
    DETOXIFY   = "detoxify"
    LLAMAGUARD = "llamaguard"
    NONE       = "none"          # nenhuma camada bloqueou


@dataclass
class GuardrailResult:
    safe:       bool
    layer:      GuardrailLayer   # qual camada tomou a decisão
    category:   str              # categoria detectada ou "safe"
    score:      float            # score de confiança [0.0 – 1.0]
    reason:     str              # texto legível para log/auditoria


# ---------------------------------------------------------------------------
# Inicialização — feita uma única vez por processo (lazy + cached)
# ---------------------------------------------------------------------------

_profanity_initialized = False
_detoxify_model: Detoxify | None = None


def _init_profanity() -> None:
    """Carrega better-profanity com a lista PT-BR customizada."""
    global _profanity_initialized
    if not _profanity_initialized:
        profanity.load_censor_words(whitelist_words=[])   # mantém lista EN nativa
        profanity.add_censor_words(PTBR_PROFANITY_LIST)  # adiciona PT-BR
        _profanity_initialized = True


def _get_detoxify() -> Detoxify:
    """Carrega o modelo Detoxify na primeira chamada e reutiliza nas demais."""
    global _detoxify_model
    if _detoxify_model is None:
        # 'multilingual' cobre PT melhor que 'original' (que é só EN)
        _detoxify_model = Detoxify("multilingual")
    return _detoxify_model


# ---------------------------------------------------------------------------
# Camada 1 — better-profanity (léxico)
# ---------------------------------------------------------------------------

def _check_profanity(text: str) -> GuardrailResult:
    """
    Verificação léxica rápida.
    Retorna bloqueio imediato se encontrar palavra na lista EN ou PT-BR.
    """
    _init_profanity()
    flagged = profanity.contains_profanity(text)
    if flagged:
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
# Camada 2 — Detoxify (ML local)
# ---------------------------------------------------------------------------

def _check_detoxify(text: str) -> tuple[GuardrailResult, bool]:
    """
    Classifica toxicidade com modelo BERT local.

    Retorna
    -------
    (result, needs_escalation)
        needs_escalation=True quando score está na zona cinza
        → deve ser escalado para LlamaGuard
    """
    model   = _get_detoxify()
    scores  = model.predict(text)        # dict com várias categorias

    # Pega a categoria com maior score
    top_cat   = max(scores, key=scores.get)
    top_score = float(scores[top_cat])

    if top_score >= DETOXIFY_BLOCK_THRESHOLD:
        return GuardrailResult(
            safe     = False,
            layer    = GuardrailLayer.DETOXIFY,
            category = top_cat,
            score    = top_score,
            reason   = f"Detoxify bloqueou: {top_cat} ({top_score:.2f})",
        ), False

    if top_score >= DETOXIFY_REVIEW_THRESHOLD:
        # Zona cinza — passa para LlamaGuard decidir
        return GuardrailResult(
            safe     = True,   # provisoriamente seguro até LlamaGuard confirmar
            layer    = GuardrailLayer.DETOXIFY,
            category = top_cat,
            score    = top_score,
            reason   = f"Zona cinza: {top_cat} ({top_score:.2f}) → escalando para LlamaGuard",
        ), True

    return GuardrailResult(
        safe=True, layer=GuardrailLayer.NONE,
        category="safe", score=top_score, reason="",
    ), False


# ---------------------------------------------------------------------------
# Camada 3 — LlamaGuard (HuggingFace API) — só na zona cinza
# ---------------------------------------------------------------------------

def _check_llamaguard(text: str) -> GuardrailResult:
    """
    Chama LlamaGuard via HuggingFace Inference API.
    Só é acionado quando Detoxify está na zona cinza.

    LlamaGuard responde "safe" ou "unsafe\nS<N>" onde S<N> é a categoria:
        S1  Violência / crimes
        S2  Conteúdo sexual explícito
        S3  Armas
        S4  Substâncias controladas
        S5  Conteúdo com menores
        S6  Privacidade / dados pessoais
        S7  Ódio / discriminação
        S8  Automutilação
        S9  Desinformação
        S10 Atividades ilegais
        S11 Código malicioso
        S12 Terrorismo / extremismo
        S13 Regulamentado (financeiro, jurídico, médico)
    """
    try:
        import requests

        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        if not hf_token:
            # Sem token → assume seguro para não bloquear indevidamente
            return GuardrailResult(
                safe=True, layer=GuardrailLayer.LLAMAGUARD,
                category="unknown", score=0.0,
                reason="Token HF ausente — LlamaGuard ignorado.",
            )

        api_url  = f"https://api-inference.huggingface.co/models/{LLAMAGUARD_REPO}"
        headers  = {"Authorization": f"Bearer {hf_token}"}
        payload  = {"inputs": f"[INST] Task: Check safety.\nUser: {text} [/INST]"}

        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        raw = response.json()
        # HF retorna lista de dicts com 'generated_text'
        generated = raw[0].get("generated_text", "safe") if isinstance(raw, list) else "safe"
        generated = generated.strip().lower()

        if generated.startswith("unsafe"):
            # Extrai categoria (ex: "unsafe\ns7" → "S7")
            parts    = generated.split()
            category = parts[1].upper() if len(parts) > 1 else "unsafe"
            return GuardrailResult(
                safe     = False,
                layer    = GuardrailLayer.LLAMAGUARD,
                category = category,
                score    = 1.0,
                reason   = f"LlamaGuard bloqueou: categoria {category}",
            )

        return GuardrailResult(
            safe=True, layer=GuardrailLayer.LLAMAGUARD,
            category="safe", score=0.0, reason="LlamaGuard: seguro.",
        )

    except Exception as exc:
        # Falha na API → fail-open (não bloqueia) para não degradar a UX
        return GuardrailResult(
            safe=True, layer=GuardrailLayer.LLAMAGUARD,
            category="error", score=0.0,
            reason=f"LlamaGuard indisponível: {exc}",
        )


# ---------------------------------------------------------------------------
# Funções públicas — únicas interfaces usadas pelo app.py
# ---------------------------------------------------------------------------

def check_input(text: str) -> GuardrailResult:
    """
    Executa o pipeline completo de moderação no INPUT do usuário.
    Camadas: better-profanity → Detoxify → LlamaGuard (zona cinza)
    """
    # Camada 1 — léxico
    result = _check_profanity(text)
    if not result.safe:
        return result

    # Camada 2 — ML local
    result, needs_escalation = _check_detoxify(text)
    if not result.safe:
        return result

    # Camada 3 — LlamaGuard (somente zona cinza)
    if needs_escalation:
        result = _check_llamaguard(text)

    return result


def check_output(text: str) -> GuardrailResult:
    """
    Moderação do OUTPUT da LLM.
    Usa apenas Detoxify (sem LlamaGuard) — output tende a ser mais seguro
    e rodar LlamaGuard duas vezes por turno dobraria as chamadas à HF API.
    """
    # Camada 1 — léxico
    result = _check_profanity(text)
    if not result.safe:
        return result

    # Camada 2 — Detoxify com threshold ligeiramente mais permissivo no output
    model   = _get_detoxify()
    scores  = model.predict(text)
    top_cat = max(scores, key=scores.get)
    top_score = float(scores[top_cat])

    if top_score >= DETOXIFY_BLOCK_THRESHOLD:
        return GuardrailResult(
            safe     = False,
            layer    = GuardrailLayer.DETOXIFY,
            category = top_cat,
            score    = top_score,
            reason   = f"Output bloqueado pelo Detoxify: {top_cat} ({top_score:.2f})",
        )

    return GuardrailResult(
        safe=True, layer=GuardrailLayer.NONE,
        category="safe", score=top_score, reason="",
    )
