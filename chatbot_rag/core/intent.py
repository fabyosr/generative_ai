"""
=============================================================================
core/intent.py — Classificador de Intenção (Intent Router)
=============================================================================
Responsabilidade:
    Determinar a intenção de cada mensagem do usuário antes de decidir
    se o pipeline RAG completo deve ser executado.

Estratégia em duas camadas (custo crescente):
    1. Heurística local  — regex + comprimento. Gratuita, instantânea.
                           Captura casos óbvios: saudações, agradecimentos.
    2. Classificador LLM — chamada leve com max_tokens=10 e output
                           estruturado. Resolve ambiguidades que a heurística
                           não consegue tratar (ex: "pode detalhar?").

Tipos de intenção (IntentType):
    CHITCHAT   — saudações, agradecimentos, despedidas.
                 Ação: resposta direta via LLM, sem RAG.
    RAG_QUERY  — perguntas sobre o conteúdo dos documentos.
                 Ação: pipeline RAG completo.
    FOLLOWUP   — pedidos de esclarecimento sobre a resposta anterior
                 ("pode explicar melhor?", "dê um exemplo").
                 Ação: RAG reutilizando o último contexto recuperado,
                 sem nova busca vetorial.

Métricas coletadas por IntentResult:
    - intent:        tipo classificado
    - method:        "heuristic" | "llm"
    - confidence:    float 0–1 (heurística = 1.0, LLM = 0.0–1.0)
    - latency_ms:    tempo de classificação em milissegundos
    - raw_llm_output: resposta bruta do LLM (para debug)
=============================================================================
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Enumeração de intenções
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    CHITCHAT  = "chitchat"   # saudação / conversa sem necessidade de RAG
    RAG_QUERY = "rag_query"  # pergunta sobre documentos → pipeline completo
    FOLLOWUP  = "followup"   # pedido de esclarecimento → reutiliza último contexto


# ---------------------------------------------------------------------------
# Resultado da classificação (inclui métricas)
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """
    Encapsula o resultado da classificação de intenção com métricas.

    Attributes:
        intent:         Tipo de intenção classificado.
        method:         "heuristic" se resolvido localmente, "llm" se usou modelo.
        confidence:     Nível de confiança 0.0–1.0.
                        Heurística sempre retorna 1.0.
                        LLM retorna 0.9 (classificação explícita) ou
                        0.6 (fallback por timeout/erro).
        latency_ms:     Tempo total de classificação em ms.
        raw_llm_output: Resposta bruta do LLM (None se heurística resolveu).
    """
    intent:         IntentType
    method:         str          = "heuristic"
    confidence:     float        = 1.0
    latency_ms:     float        = 0.0
    raw_llm_output: str | None   = None


# ---------------------------------------------------------------------------
# Padrões de heurística — expressões regulares por categoria
# ---------------------------------------------------------------------------

# Saudações, cumprimentos e abertura de conversa
_GREETINGS = re.compile(
    r"^(oi|olá|ola|oii|hey|hi|hello|e aí|eai|bom dia|boa tarde|boa noite|tudo bem"
    r"|tudo bom|como vai|como você está|como vc está)[!?.,:…]*$",
    re.IGNORECASE,
)

# Agradecimentos, confirmações e encerramentos
_ACKNOWLEDGEMENTS = re.compile(
    r"^(obrigad[ao]|obg|valeu|vlw|thanks|thank you|ok|certo|entendi|compreendi"
    r"|perfeito|ótimo|otimo|legal|show|massa|bacana|muito bom|excelente"
    r"|tchau|até|até logo|ate logo|bye|falou|flw)[!?.,:…]*$",
    re.IGNORECASE,
)

# Pedidos de esclarecimento sobre a resposta anterior (followup)
_FOLLOWUP = re.compile(
    r"^(pode (repetir|explicar|detalhar|elaborar|continuar|exemplificar)"
    r"|explica melhor|mais detalhes|mais detalhado|pode dar um exemplo"
    r"|dê um exemplo|da um exemplo|não entendi|nao entendi"
    r"|como assim|o que quer dizer|o que significa isso"
    r"|pode reformular|em outras palavras)[?!.…]*$",
    re.IGNORECASE,
)

# Comprimento máximo de tokens para ser considerado chitchat pela heurística
_MAX_CHITCHAT_WORDS = 6


# ---------------------------------------------------------------------------
# Camada 1 — Heurística local
# ---------------------------------------------------------------------------

def _heuristic_classify(message: str) -> IntentType | None:
    """
    Tenta classificar a mensagem sem chamar nenhum LLM.

    Retorna o IntentType se a classificação for confiável,
    ou None se a mensagem for ambígua e precisar do LLM.

    Args:
        message: Texto do usuário (já stripped).

    Returns:
        IntentType | None: Classificação ou None se inconclusivo.
    """
    stripped = message.strip()
    words    = stripped.split()

    # Mensagem muito curta sem interrogação → provável chitchat
    if len(words) <= 2 and "?" not in stripped:
        if _GREETINGS.match(stripped) or _ACKNOWLEDGEMENTS.match(stripped):
            return IntentType.CHITCHAT

    # Saudação mesmo com mais palavras
    if _GREETINGS.match(stripped):
        return IntentType.CHITCHAT

    # Agradecimento
    if _ACKNOWLEDGEMENTS.match(stripped):
        return IntentType.CHITCHAT

    # Pedido de esclarecimento explícito
    if _FOLLOWUP.match(stripped):
        return IntentType.FOLLOWUP

    # Ambíguo — delega ao LLM
    return None


# ---------------------------------------------------------------------------
# Camada 2 — Classificador LLM
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """Você é um classificador de intenção para um chatbot RAG de documentos.
Classifique a mensagem do usuário em exatamente uma das categorias:

- CHITCHAT   → saudação, agradecimento, conversa genérica sem relação com documentos
- RAG_QUERY  → pergunta ou solicitação sobre o conteúdo de documentos
- FOLLOWUP   → pedido de esclarecimento, repetição ou elaboração da resposta anterior

Responda APENAS com uma palavra: CHITCHAT, RAG_QUERY ou FOLLOWUP.
Não adicione pontuação, explicação ou qualquer outro texto."""


def _llm_classify(message: str, chat_history: list, llm) -> IntentResult:
    """
    Classifica a intenção usando uma chamada leve ao LLM.

    Usa um prompt minimalista com output de uma palavra para minimizar
    latência e consumo de tokens. Em caso de erro ou resposta inválida,
    faz fallback seguro para RAG_QUERY.

    Args:
        message:      Texto do usuário.
        chat_history: Histórico recente (últimas 4 mensagens) para contexto.
        llm:          Instância de LLM (BaseChatModel).

    Returns:
        IntentResult com method="llm".
    """
    from langchain_core.messages import SystemMessage

    t0 = time.perf_counter()

    # Inclui as últimas 2 trocas do histórico para o LLM ter contexto
    # de followup ("pode explicar melhor?" só faz sentido com histórico)
    recent = chat_history[-4:] if len(chat_history) >= 4 else chat_history

    messages = [
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        *recent,
        HumanMessage(content=f"Mensagem a classificar: {message}"),
    ]

    raw = ""
    intent = IntentType.RAG_QUERY  # fallback seguro

    try:
        response = llm.invoke(messages)
        raw      = response.content.strip().upper()

        if "CHITCHAT" in raw:
            intent     = IntentType.CHITCHAT
            confidence = 0.9
        elif "FOLLOWUP" in raw:
            intent     = IntentType.FOLLOWUP
            confidence = 0.9
        elif "RAG_QUERY" in raw:
            intent     = IntentType.RAG_QUERY
            confidence = 0.9
        else:
            # Resposta fora do esperado → fallback conservador
            intent     = IntentType.RAG_QUERY
            confidence = 0.6

    except Exception:
        # Erro de rede, timeout, etc. → fallback conservador
        intent     = IntentType.RAG_QUERY
        confidence = 0.6

    latency_ms = (time.perf_counter() - t0) * 1000

    return IntentResult(
        intent         = intent,
        method         = "llm",
        confidence     = confidence,
        latency_ms     = round(latency_ms, 1),
        raw_llm_output = raw,
    )


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------

def classify_intent(
    message:      str,
    chat_history: list,
    llm,
) -> IntentResult:
    """
    Classifica a intenção da mensagem do usuário em duas camadas.

    Fluxo:
        1. Heurística local  → se conclusivo, retorna imediatamente (0ms)
        2. Classificador LLM → para casos ambíguos

    Args:
        message:      Texto do usuário.
        chat_history: Lista de AIMessage/HumanMessage da sessão.
        llm:          Instância de LLM para a camada 2.

    Returns:
        IntentResult com intent, method, confidence, latency_ms.
    """
    t0 = time.perf_counter()

    # --- Camada 1: heurística ---
    heuristic_result = _heuristic_classify(message.strip())

    if heuristic_result is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        return IntentResult(
            intent     = heuristic_result,
            method     = "heuristic",
            confidence = 1.0,
            latency_ms = round(latency_ms, 1),
        )

    # --- Camada 2: LLM ---
    return _llm_classify(message, chat_history, llm)
