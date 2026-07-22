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
    system_prompt:  str        = None
    method:         str        = "heuristic"
    confidence:     float      = 1.0
    latency_ms:     float      = 0.0
    raw_llm_output: str | None = None


# ---------------------------------------------------------------------------
# Heurística RESTRITA — apenas padrões inequivocamente não-RAG
# ---------------------------------------------------------------------------

# Token de abertura de saudação (início da frase)
_GREETING_OPEN = re.compile(
    r"^(oi|olá|ola|oii|oiii|hey|hi|hello|e aí|eai"
    r"|bom dia|boa tarde|boa noite|good morning|good afternoon|good evening)",
    re.IGNORECASE,
)

# Conteúdo puramente social — o que pode vir DEPOIS de uma saudação
_SOCIAL_TAIL = re.compile(
    r"^[,!\s]*(tudo bem|tudo bom|tudo certo|td bem|td bom|blz|beleza"
    r"|como vai|como você está|como vc está|como tá|como ta|como estas"
    r"|how are you|how's it going|all good|tudo)[?!.,\s]*$",
    re.IGNORECASE,
)

# Saudações simples completas
_GREETINGS_EXACT = re.compile(
    r"^(oi|olá|ola|oii|oiii|hey|hi|hello|e aí|eai"
    r"|bom dia|boa tarde|boa noite|good morning|good afternoon|good evening"
    r"|tudo bem|tudo bom|tudo certo|td bem|blz|beleza)[!?.,\s]*$",
    re.IGNORECASE,
)

# Agradecimentos e encerramentos puros
_ACKNOWLEDGEMENTS = re.compile(
    r"^(obrigad[ao]|obg|valeu|vlw|thanks|thank you|muito obrigad[ao]"
    r"|tchau|até logo|até mais|ate logo|até a próxima|bye|até|flw|falou"
    r"|entendido|ok|certo|perfeito|ótimo|otimo|show|legal)[!?.,\s]*$",
    re.IGNORECASE,
)

# Followup explícito
_FOLLOWUP = re.compile(
    r"^(pode (repetir|explicar melhor|detalhar|elaborar|continuar|exemplificar)"
    r"|explica melhor|mais detalhes|pode dar um exemplo|dê um exemplo|da um exemplo"
    r"|não entendi|nao entendi|como assim|pode reformular|em outras palavras"
    r"|repete|repita|continua|continue)[?!.\s]*$",
    re.IGNORECASE,
)


def _heuristic_classify(message: str) -> IntentType | None:
    """
    Classifica padrões inequívocos sem chamar LLM.

    Lógica para saudações compostas como "oi como vai?":
        1. Detecta se inicia com token de saudação
        2. Verifica se o restante é conteúdo puramente social
        3. Se sim → CHITCHAT; se não → None (LLM decide)

    Exemplos:
        "oi"             → CHITCHAT (saudação exata)
        "oi como vai?"   → CHITCHAT (saudação + social)
        "oi tudo bem?"   → CHITCHAT (saudação + social)
        "oi qual o prazo?" → None  (saudação + conteúdo informacional → LLM)
        "prazo"          → None    (conteúdo ambíguo → LLM)
        "obrigado"       → CHITCHAT
        "pode repetir?"  → FOLLOWUP
    """
    stripped = message.strip()

    # 1. Saudação ou frase social exata
    if _GREETINGS_EXACT.fullmatch(stripped):
        return IntentType.CHITCHAT

    # 2. Agradecimento ou encerramento
    if _ACKNOWLEDGEMENTS.fullmatch(stripped):
        return IntentType.CHITCHAT

    # 3. Saudação composta: inicia com token de saudação
    m = _GREETING_OPEN.match(stripped)
    if m:
        tail = stripped[m.end():].strip()
        if not tail:
            # Só a saudação, sem mais nada
            return IntentType.CHITCHAT
        if _SOCIAL_TAIL.fullmatch(tail):
            # Saudação + conteúdo puramente social → chitchat
            return IntentType.CHITCHAT
        # Saudação + conteúdo informacional → delega ao LLM
        return None

    # 4. Followup explícito
    if _FOLLOWUP.fullmatch(stripped):
        return IntentType.FOLLOWUP

    # 5. Qualquer outro caso → LLM decide
    return None


def _build_classifier_prompt(doc_knowledge: str = "") -> str:
    """
    Constrói o system prompt do classificador com contexto dos documentos.

    Se doc_knowledge for fornecido (resumo dos temas dos documentos),
    adiciona uma instrução sobre quais assuntos pertencem à base de
    conhecimento — o classificador fica mais preciso ao saber o domínio.

    Args:
        doc_knowledge: Saída de format_for_classifier() (pode ser "").

    Returns:
        str: System prompt completo para o classificador.
    """

    # ---------------------------------------------------------------------------
    # Classificador LLM — prompt reforçado com exemplos
    # ---------------------------------------------------------------------------

    _CLASSIFIER_SYSTEM = f"""You are an expert Intent Classifier for a RAG-powered chatbot specialized in analyzing and answering questions about uploaded documents.

                            Your job is to analyze the user's message and classify it into **exactly one** of the following three categories:

                            ### RAG_QUERY
                            Use this when the user is asking for information, facts, summaries, explanations, or performing any action that requires consulting the document collection.
                            - Questions about content, deadlines, values, responsibilities, clauses, definitions, processes, etc.
                            - Requests like "what does the document say about...", "summarize...", "extract...", "compare...", "who is responsible for...", "what is the deadline for...".
                            - Short, specific queries about topics present in the documents (price, deadline, name, date, clause, condition, article, etc.).
                            - Any query that would benefit from grounding in the uploaded documents.

                            ### FOLLOWUP
                            Use this when the user is clearly continuing, refining, or asking for clarification on the **previous response** from the assistant.
                            - Examples: "Can you elaborate?", "Explain that better", "Give me an example", "What about the other contract?", "Go deeper on that point", "Why is that?", "Next item".

                            ### CHITCHAT
                            Use this for casual conversation, greetings, thanks, farewells, or any message completely unrelated to the documents.
                            - Examples: "Hi", "Hello", "Thank you", "Goodbye", "Good morning", "How are you?", jokes, small talk.

                            ### CRITICAL RULES

                            1. **Default to RAG_QUERY** in case of any doubt or ambiguity. It is much better to query the documents than to miss relevant information.
                            2. Short or poorly written messages about specific topics (prices, deadlines, names, clauses, articles, obligations, values, dates, etc.) should almost always be classified as **RAG_QUERY**.
                            3. If the user is referring to content that is likely present in the documents (even implicitly), classify as **RAG_QUERY**.
                            4. Only classify as FOLLOWUP if it is very clear that the user is reacting to the assistant’s last message.

                            ### CONTEXT
                            Below is a summary in Brazilian portuguese of the documents currently available in the knowledge base:

                            {doc_knowledge}

                            Use this context to better understand the domain and topics the user might be asking about.

                            ### RESPONSE FORMAT
                            Respond with **ONLY** one of the following words. Do not add any explanation, punctuation, or extra text:

                            RAG_QUERY  
                            FOLLOWUP  
                            CHITCHAT"""

    return _CLASSIFIER_SYSTEM


def _llm_classify(
    message:        str,
    chat_history:   list,
    llm,
    doc_knowledge:  str = "",
) -> IntentResult:
    """
    Classifica via LLM com system prompt contextualizado pelos documentos.

    Args:
        message:       Texto do usuário.
        chat_history:  Histórico recente da conversa.
        llm:           Instância de LLM (pode ser o classificador dedicado).
        doc_knowledge: Contexto dos temas dos documentos (opcional).

    Returns:
        IntentResult com method="llm".
    """
    from langchain_core.messages import SystemMessage

    t0     = time.perf_counter()
    recent = chat_history[-4:] if len(chat_history) >= 4 else chat_history

    system_prompt = _build_classifier_prompt(doc_knowledge)

    messages = [
        SystemMessage(content=system_prompt),
        *recent,
        HumanMessage(content=f'Classifique: "{message}"'),
    ]

    raw    = ""
    intent = IntentType.RAG_QUERY

    try:
        response = llm.invoke(messages)
        raw      = response.content.strip().upper()

        if "CHITCHAT" in raw and "RAG" not in raw:
            intent, confidence = IntentType.CHITCHAT,  0.9
        elif "FOLLOWUP" in raw and "RAG" not in raw:
            intent, confidence = IntentType.FOLLOWUP,  0.9
        elif "RAG" in raw:
            intent, confidence = IntentType.RAG_QUERY, 0.9
        else:
            intent, confidence = IntentType.RAG_QUERY, 0.6

    except Exception:
        intent, confidence = IntentType.RAG_QUERY, 0.6

    return IntentResult(
        intent         = intent,
        system_prompt  = system_prompt,
        method         = "llm",
        confidence     = confidence,
        latency_ms     = round((time.perf_counter() - t0) * 1000, 1),
        raw_llm_output = raw,
    )


def classify_intent(
    message:        str,
    chat_history:   list,
    llm,
    classifier_llm  = None,
    doc_knowledge:  str = "",
) -> IntentResult:
    """
    Classifica a intenção em duas camadas.

    Parâmetros novos:
        classifier_llm: Modelo dedicado para classificação — independente
                        do modelo principal. Se None, usa o llm principal.
                        Recomendado: modelo leve (Qwen2.5-3B, gpt-4o-mini).

        doc_knowledge:  Resumo dos temas dos documentos gerado por
                        doc_summary.format_for_classifier(). Injetado no
                        system prompt do classificador para que ele saiba
                        quais assuntos estão na base de conhecimento.

    Args:
        message:        Texto do usuário.
        chat_history:   Lista de mensagens da sessão.
        llm:            Modelo principal (fallback se classifier_llm=None).
        classifier_llm: Modelo leve dedicado ao classificador (opcional).
        doc_knowledge:  Contexto dos documentos para o classificador.

    Returns:
        IntentResult com intent, method, confidence, latency_ms.
    """
    t0 = time.perf_counter()

    # Camada 1: heurística — apenas padrões inequívocos
    heuristic = _heuristic_classify(message.strip())
    if heuristic is not None:
        return IntentResult(
            intent     = heuristic,
            method     = "heuristic",
            confidence = 1.0,
            latency_ms = round((time.perf_counter() - t0) * 1000, 1),
        )

    # Camada 2: LLM — usa classificador dedicado se disponível
    active_llm = classifier_llm if classifier_llm is not None else llm
    return _llm_classify(message, chat_history, active_llm, doc_knowledge)
