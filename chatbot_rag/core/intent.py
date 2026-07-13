"""
=============================================================================
core/intent.py — Classificador de Intenção (Intent Router)
=============================================================================
Estratégia em duas camadas:
    1. Heurística local — regex MUITO restrita, apenas padrões inequívocos.
       Princípio: na dúvida, NÃO classifica como chitchat. Delega ao LLM.
    2. Classificador LLM — fallback seguro para RAG_QUERY em caso de erro.

Problema corrigido:
    A versão anterior classificava queries curtas (≤2 palavras sem "?")
    como chitchat pela heurística, mesmo que fossem perguntas sobre
    documentos ("valor", "prazo", "contrato"). Isso fazia perguntas
    objetivas e diretas caírem em chitchat e nunca chegarem ao RAG.

    Correção: a heurística agora só classifica como chitchat se o texto
    bater EXATAMENTE em um padrão de saudação/agradecimento. Qualquer
    coisa ambígua vai para o LLM, que tem fallback seguro para RAG_QUERY.
=============================================================================
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage


class IntentType(str, Enum):
    CHITCHAT  = "chitchat"
    RAG_QUERY = "rag_query"
    FOLLOWUP  = "followup"


@dataclass
class IntentResult:
    intent:         IntentType
    method:         str        = "heuristic"
    confidence:     float      = 1.0
    latency_ms:     float      = 0.0
    raw_llm_output: str | None = None


# ---------------------------------------------------------------------------
# Heurística RESTRITA — apenas padrões inequivocamente não-RAG
# ---------------------------------------------------------------------------

# Saudações puras — matches exatos, sem espaço para ambiguidade
_GREETINGS = re.compile(
    r"^(oi|olá|ola|oii|oiii|hey|hi|hello|e aí|eai|oi tudo bem"
    r"|bom dia|boa tarde|boa noite|good morning|good afternoon)[!?.,\s]*$",
    re.IGNORECASE,
)

# Agradecimentos e encerramentos puros
_ACKNOWLEDGEMENTS = re.compile(
    r"^(obrigad[ao]|obg|valeu|vlw|thanks|thank you"
    r"|tchau|até logo|até mais|ate logo|bye|até|flw|falou)[!?.,\s]*$",
    re.IGNORECASE,
)

# Followup explícito — frases que só fazem sentido referindo-se à resposta anterior
_FOLLOWUP = re.compile(
    r"^(pode (repetir|explicar melhor|detalhar|elaborar|continuar|exemplificar)"
    r"|explica melhor|mais detalhes|pode dar um exemplo|dê um exemplo"
    r"|não entendi|como assim|pode reformular|em outras palavras"
    r"|repete|repita)[?!.\s]*$",
    re.IGNORECASE,
)


def _heuristic_classify(message: str) -> IntentType | None:
    """
    Classifica apenas padrões inequívocos. Retorna None para qualquer dúvida.

    NUNCA classifica como chitchat baseado apenas no comprimento da mensagem.
    Queries curtas como "prazo", "valor total", "quem assinou" são perguntas
    legítimas sobre documentos e devem ir para o LLM classificar.
    """
    stripped = message.strip()

    # Só considera saudação se bater exatamente no padrão completo
    if _GREETINGS.fullmatch(stripped):
        return IntentType.CHITCHAT

    if _ACKNOWLEDGEMENTS.fullmatch(stripped):
        return IntentType.CHITCHAT

    if _FOLLOWUP.fullmatch(stripped):
        return IntentType.FOLLOWUP

    # TUDO mais vai para o LLM — incluindo queries curtas e ambíguas
    return None


# ---------------------------------------------------------------------------
# Classificador LLM — prompt reforçado com exemplos
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """Você é um classificador de intenção para um chatbot RAG especializado em documentos.

Classifique a mensagem em UMA das categorias:

RAG_QUERY → qualquer pergunta ou interesse no conteúdo dos documentos enviados.
  Exemplos: "qual o prazo?", "valor do contrato", "quem é o responsável",
            "me fale sobre X", "o que diz sobre Y", "como funciona Z",
            "qual a política de reembolso", "me explique sobre isso"

FOLLOWUP → pedido de esclarecimento ou continuação da resposta ANTERIOR do assistente.
  Exemplos: "pode elaborar?", "explica melhor", "dê um exemplo disso", "repete"

CHITCHAT → saudação, agradecimento ou conversa completamente fora do contexto de documentos.
  Exemplos: "oi", "obrigado", "tchau", "bom dia"

REGRA CRÍTICA: em caso de dúvida, sempre responda RAG_QUERY.
Queries curtas sobre temas específicos (preço, prazo, nome, data) são RAG_QUERY.

Responda APENAS com uma palavra: RAG_QUERY, FOLLOWUP ou CHITCHAT."""


def _llm_classify(message: str, chat_history: list, llm) -> IntentResult:
    """
    Classifica via LLM com fallback seguro para RAG_QUERY.

    O fallback é RAG_QUERY (não chitchat) para garantir que perguntas
    sobre documentos nunca sejam descartadas por erro do classificador.
    """
    from langchain_core.messages import SystemMessage

    t0     = time.perf_counter()
    recent = chat_history[-4:] if len(chat_history) >= 4 else chat_history

    messages = [
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        *recent,
        HumanMessage(content=f'Classifique: "{message}"'),
    ]

    raw    = ""
    intent = IntentType.RAG_QUERY   # fallback conservador — nunca perde uma query RAG

    try:
        response = llm.invoke(messages)
        raw      = response.content.strip().upper()

        if "CHITCHAT" in raw and "RAG" not in raw:
            intent     = IntentType.CHITCHAT
            confidence = 0.9
        elif "FOLLOWUP" in raw and "RAG" not in raw:
            intent     = IntentType.FOLLOWUP
            confidence = 0.9
        elif "RAG_QUERY" in raw or "RAG" in raw:
            intent     = IntentType.RAG_QUERY
            confidence = 0.9
        else:
            # Resposta inesperada → fallback conservador
            intent     = IntentType.RAG_QUERY
            confidence = 0.6

    except Exception:
        intent     = IntentType.RAG_QUERY
        confidence = 0.6

    return IntentResult(
        intent         = intent,
        method         = "llm",
        confidence     = confidence,
        latency_ms     = round((time.perf_counter() - t0) * 1000, 1),
        raw_llm_output = raw,
    )


def classify_intent(message: str, chat_history: list, llm) -> IntentResult:
    """
    Classifica a intenção em duas camadas.

    Camada 1 (heurística): apenas padrões inequívocos — saudações e
    agradecimentos exatos. Qualquer dúvida vai para a camada 2.

    Camada 2 (LLM): prompt com exemplos e regra explícita de que dúvida
    = RAG_QUERY. Fallback de erro também = RAG_QUERY.
    """
    t0 = time.perf_counter()

    heuristic = _heuristic_classify(message.strip())
    if heuristic is not None:
        return IntentResult(
            intent     = heuristic,
            method     = "heuristic",
            confidence = 1.0,
            latency_ms = round((time.perf_counter() - t0) * 1000, 1),
        )

    return _llm_classify(message, chat_history, llm)
