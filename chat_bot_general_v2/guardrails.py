"""
guardrails.py
=============
Pipeline de moderação em 3 camadas, aplicado em sequência crescente de custo:

  [1] better-profanity     — léxico EN nativo + lista customizada PT-BR
                             ~1ms, zero rede, primeira barreira óbvia

  [2] OpenAI Moderation    — endpoint gratuito /v1/moderations
                             ~150ms, zero tokens, ativado quando provider=openai
                             · flagged=True          → bloqueia direto
                             · top_score > REVIEW    → zona cinza → LlamaGuard
                             · top_score ≤ REVIEW    → aprovado

  [3] LlamaGuard 2 8B      — modelo Meta fine-tunado para moderação via HF API
                             ~500ms–30s (cold start no plano gratuito HF)
                             · ativado na zona cinza da Moderation API
                             · ativado quando provider != openai (sem Moderation)
                             · fail-open em erro de rede (não bloqueia)
                             Categorias S1–S13:
                               S1  Violência / crimes       S8  Automutilação
                               S2  Conteúdo sexual explícito S9  Desinformação
                               S3  Armas                    S10 Atividades ilegais
                               S4  Substâncias controladas  S11 Código malicioso
                               S5  Conteúdo com menores     S12 Terrorismo
                               S6  Privacidade / PII        S13 Regulamentado
                               S7  Ódio / discriminação

  [4] System Prompt        — SAFETY POLICY em config.py, embutida no modelo
                             Cobertura semântica profunda, zero latência extra

A principal linha de defesa é o system prompt — as camadas acima são barreiras
pré-LLM que evitam até mesmo o custo de tokens em casos óbvios.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import requests
from better_profanity import profanity

# ---------------------------------------------------------------------------
# Constantes — espelhadas em config.py para a UI, editáveis aqui para lógica
# ---------------------------------------------------------------------------

# Score da OpenAI Moderation API acima do qual escala para LlamaGuard
REVIEW_THRESHOLD: float = 0.40

# LlamaGuard — modelo e endpoint
LLAMAGUARD_REPO    = "meta-llama/Meta-Llama-Guard-2-8B"
LLAMAGUARD_TIMEOUT = 15   # segundos — cold start no HF gratuito pode ser lento

# Lista customizada PT-BR para better-profanity
PTBR_PROFANITY_LIST: list[str] = [
    "merda", "porra", "caralho", "fodase", "foda-se",
    "buceta", "cu", "puta", "vagabunda", "arrombado", "babaca",
    "imbecil", "idiota", "otario", "otário", "desgraça", "desgraçado",
    "fdp", "vsf", "tnc", "vtnc", "pqp", "krl",
    "macaco", "nego", "crioulo",
    "viado", "viadinho", "sapatão", "traveco",
]

# Mensagens padrão exibidas ao usuário quando bloqueado
BLOCK_MESSAGE_INPUT  = "⚠️ Sua mensagem foi bloqueada por violar as diretrizes de uso."
BLOCK_MESSAGE_OUTPUT = "⚠️ A resposta foi bloqueada por conter conteúdo inadequado."


# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

class GuardrailLayer(str, Enum):
    PROFANITY     = "better-profanity"
    OPENAI_MOD    = "openai-moderation"
    LLAMAGUARD    = "llamaguard"
    SYSTEM_PROMPT = "system-prompt"   # reservado: modelo recusou via safety prompt
    NONE          = "none"            # nenhuma camada interveio


@dataclass
class GuardrailResult:
    safe:            bool
    layer:           GuardrailLayer
    category:        str    # categoria detectada ou "safe"
    score:           float  # score de confiança [0.0 – 1.0]
    reason:          str    # texto legível para auditoria/log
    # Toxicidade contínua — preenchida SEMPRE, mesmo quando safe=True.
    # Permite rastrear mensagens na zona cinza que passaram sem bloqueio
    # (ex: "bomba caseira com fins didáticos" — passa no léxico mas tem score > 0)
    toxicity_score:  float = 0.0   # top score da categoria mais alta (0.0–1.0)
    toxicity_cat:    str   = ""    # categoria correspondente ao top score


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
    """Verificação léxica instantânea — ~1ms, zero rede."""
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
# Camada 2 — OpenAI Moderation API
# ---------------------------------------------------------------------------

def _get_openai_top_score(text: str) -> tuple[str, float]:
    """
    Lê os scores individuais da Moderation API independente de flagged.
    Retorna (categoria_mais_alta, score) para decisão de zona cinza.
    Retorna ("none", 0.0) em caso de erro.
    """
    try:
        from openai import OpenAI
        client   = OpenAI()
        response = client.moderations.create(input=text)
        scores   = response.results[0].category_scores.model_dump()
        top_cat  = max(scores, key=scores.get)
        return top_cat, float(scores[top_cat])
    except Exception:
        return "none", 0.0


def _check_openai_moderation(text: str) -> tuple[GuardrailResult, bool]:
    """
    Chama /v1/moderations (gratuito, zero tokens).

    IMPORTANTE: sempre lê e propaga toxicity_score/toxicity_cat mesmo quando
    flagged=False — isso resolve o gap onde mensagens na zona cinza (ex: "bomba
    caseira com fins didáticos") passavam sem deixar rastro de score na tabela.

    Retorna
    -------
    (result, needs_llamaguard)
        needs_llamaguard=True quando score está na zona cinza [REVIEW, 1.0)
        e flagged=False — o LlamaGuard decide com mais contexto semântico.
    """
    try:
        from openai import OpenAI
        client    = OpenAI()
        response  = client.moderations.create(input=text)
        result    = response.results[0]
        scores    = result.category_scores.model_dump()
        top_cat   = max(scores, key=scores.get)
        top_score = float(scores[top_cat])

        # Bloqueio direto — toxicity_score preenchido com o score real
        if result.flagged:
            return GuardrailResult(
                safe           = False,
                layer          = GuardrailLayer.OPENAI_MOD,
                category       = top_cat,
                score          = top_score,
                reason         = f"OpenAI Moderation bloqueou: {top_cat} ({top_score:.2f})",
                toxicity_score = top_score,
                toxicity_cat   = top_cat,
            ), False

        # Zona cinza → escala para LlamaGuard — toxicity_score preservado
        if top_score >= REVIEW_THRESHOLD:
            return GuardrailResult(
                safe           = True,
                layer          = GuardrailLayer.OPENAI_MOD,
                category       = top_cat,
                score          = top_score,
                reason         = (
                    f"Zona cinza: {top_cat} ({top_score:.2f}) "
                    f">= {REVIEW_THRESHOLD} → escalando para LlamaGuard."
                ),
                toxicity_score = top_score,
                toxicity_cat   = top_cat,
            ), True

        # Aprovado — toxicity_score preenchido mesmo sem bloqueio
        # Isso é o que faltava: mensagens como "bomba caseira" terão
        # violence=0.38 registrado mesmo passando pelo guardrail
        return GuardrailResult(
            safe           = True,
            layer          = GuardrailLayer.OPENAI_MOD,
            category       = "safe",
            score          = top_score,
            reason         = f"OpenAI Moderation: safe (top={top_cat} {top_score:.2f})",
            toxicity_score = top_score,
            toxicity_cat   = top_cat,
        ), False

    except Exception as exc:
        # Fail-open — toxicity_score zerado quando API indisponível
        return GuardrailResult(
            safe=True, layer=GuardrailLayer.OPENAI_MOD,
            category="error", score=0.0,
            reason=f"OpenAI Moderation indisponível: {exc}",
        ), True


# ---------------------------------------------------------------------------
# Camada 3 — LlamaGuard 2 8B (HuggingFace Inference API)
# ---------------------------------------------------------------------------

# Prompt oficial do LlamaGuard 2 para classificação de segurança
_LLAMAGUARD_PROMPT_TEMPLATE = """[INST] Task: Check if there is unsafe content in the user message according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex Crimes
S4: Child Exploitation
S5: Defamation
S6: Specialized Advice (medical, legal, financial)
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Self-Harm
S12: Sexual Content
S13: Elections
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {text}
<END CONVERSATION>

Provide your safety assessment for the user message. Reply with only "safe" or "unsafe" followed by a newline and the violated category code (e.g. "unsafe\\nS1"). [/INST]"""


def _check_llamaguard(text: str) -> GuardrailResult:
    """
    Chama LlamaGuard 2 8B via HuggingFace Inference API.

    Comportamento:
      - "safe"          → aprovado
      - "unsafe\\nS<N>" → bloqueado com categoria
      - erro de rede    → fail-open (não bloqueia para não degradar UX)
      - token ausente   → fail-open com aviso no reason
    """
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not hf_token:
        return GuardrailResult(
            safe=True, layer=GuardrailLayer.LLAMAGUARD,
            category="skipped", score=0.0,
            reason="Token HuggingFace ausente — LlamaGuard ignorado.",
        )

    try:
        api_url = (
            f"https://api-inference.huggingface.co/models/{LLAMAGUARD_REPO}"
        )
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": _LLAMAGUARD_PROMPT_TEMPLATE.format(text=text),
            "parameters": {"max_new_tokens": 20, "temperature": 0.01},
        }

        response = requests.post(
            api_url, headers=headers, json=payload,
            timeout=LLAMAGUARD_TIMEOUT,
        )
        response.raise_for_status()

        raw       = response.json()
        generated = (
            raw[0].get("generated_text", "safe")
            if isinstance(raw, list) else "safe"
        )

        # Extrai só a parte gerada após o prompt (alguns modelos ecoam o input)
        if "[/INST]" in generated:
            generated = generated.split("[/INST]")[-1]
        generated = generated.strip().lower()

        if generated.startswith("unsafe"):
            lines    = generated.split("\n")
            category = lines[1].strip().upper() if len(lines) > 1 else "unsafe"
            return GuardrailResult(
                safe     = False,
                layer    = GuardrailLayer.LLAMAGUARD,
                category = category,
                score    = 1.0,
                reason   = f"LlamaGuard bloqueou: categoria {category}",
            )

        return GuardrailResult(
            safe=True, layer=GuardrailLayer.LLAMAGUARD,
            category="safe", score=0.0,
            reason="LlamaGuard: safe.",
        )

    except requests.Timeout:
        return GuardrailResult(
            safe=True, layer=GuardrailLayer.LLAMAGUARD,
            category="timeout", score=0.0,
            reason=f"LlamaGuard timeout ({LLAMAGUARD_TIMEOUT}s) — fail-open.",
        )
    except Exception as exc:
        return GuardrailResult(
            safe=True, layer=GuardrailLayer.LLAMAGUARD,
            category="error", score=0.0,
            reason=f"LlamaGuard indisponível: {exc}",
        )


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def check_input(text: str, provider: str = "") -> GuardrailResult:
    """
    Pipeline completo de moderação do INPUT do usuário.

    Fluxo de decisão:
      [1] better-profanity → bloqueia se match léxico
      [2] OpenAI Moderation (provider=openai)
              → bloqueia se flagged
              → escala para LlamaGuard se zona cinza (score ≥ REVIEW_THRESHOLD)
              → aprova se score < REVIEW_THRESHOLD
      [3] LlamaGuard
              → ativado na zona cinza da Moderation
              → ativado quando provider != openai (sem Moderation disponível)
              → fail-open em timeout ou erro

    Parâmetros
    ----------
    text     : mensagem do usuário
    provider : chave do provedor ativo ('openai' | 'ollama' | 'hf_endpoint')
    """
    # Camada 1 — léxico (sempre)
    result = _check_profanity(text)
    if not result.safe:
        return result

    needs_llamaguard = False

    # Camada 2 — OpenAI Moderation (somente com provedor OpenAI)
    if provider == "openai":
        result, needs_llamaguard = _check_openai_moderation(text)
        if not result.safe:
            return result
    else:
        # Sem Moderation disponível → LlamaGuard cobre a análise semântica
        needs_llamaguard = True

    # Camada 3 — LlamaGuard (zona cinza ou provedor não-OpenAI)
    if needs_llamaguard:
        result = _check_llamaguard(text)
        if not result.safe:
            return result

    # Propaga toxicity_score/toxicity_cat do resultado mais recente que os tenha.
    # Garante que mensagens aprovadas mas com sinal de risco (ex: zona cinza que
    # o LlamaGuard liberou) ficam registradas com score real na observabilidade.
    return GuardrailResult(
        safe           = True,
        layer          = GuardrailLayer.NONE,
        category       = "safe",
        score          = 0.0,
        reason         = "Todas as camadas aprovaram.",
        toxicity_score = result.toxicity_score,
        toxicity_cat   = result.toxicity_cat,
    )


def check_output(text: str, provider: str = "") -> GuardrailResult:
    """
    Moderação do OUTPUT da LLM.

    Fluxo mais leve que o input — o modelo já foi instruído pelo system prompt,
    então o output tende a ser mais seguro. LlamaGuard não é chamado no output
    para evitar dobrar as chamadas à HF API por turno.
    """
    # Camada 1 — léxico
    result = _check_profanity(text)
    if not result.safe:
        return result

    # Camada 2 — OpenAI Moderation (somente com provedor OpenAI)
    if provider == "openai":
        result, _ = _check_openai_moderation(text)
        if not result.safe:
            return result

    return GuardrailResult(
        safe           = True,
        layer          = GuardrailLayer.NONE,
        category       = "safe",
        score          = 0.0,
        reason         = "Output aprovado.",
        toxicity_score = result.toxicity_score,
        toxicity_cat   = result.toxicity_cat,
    )


# ---------------------------------------------------------------------------
# Utilitário de diagnóstico — exibido na sidebar
# ---------------------------------------------------------------------------

def guardrail_status(provider: str = "", openai_api_key: str = "") -> dict:
    """
    Retorna disponibilidade de cada camada para exibir na sidebar.

    Parâmetros
    ----------
    provider       : chave do provedor ativo
    openai_api_key : chave digitada pelo usuário na UI (complementa os.environ)
    """
    # HF token: checa tanto os.environ quanto st.secrets (já copiado no init)
    hf_token = bool(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

    # OpenAI key: checa ambiente + chave digitada na UI
    openai_key = bool(openai_api_key) or bool(os.environ.get("OPENAI_API_KEY"))

    # LlamaGuard: ativo quando há token HF — independente do provedor atual
    # pois é chamado para qualquer provedor != openai
    llamaguard_active = hf_token

    return {
        "system_prompt":     True,
        "better_profanity":  True,
        "openai_moderation": provider == "openai" and openai_key,
        "llamaguard":        llamaguard_active,
        "detoxify":          False,   # reservado: incompatível com Python 3.12+
    }
