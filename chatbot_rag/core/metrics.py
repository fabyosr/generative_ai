"""
=============================================================================
core/metrics.py — Coleta e Cálculo de Métricas de Observabilidade
=============================================================================
Responsabilidade:
    Centralizar toda lógica de medição do sistema: tokens consumidos,
    tamanho do histórico, metadados dos documentos recuperados pelo RAG,
    latência de resposta e metadados do modelo utilizado.

Estratégia de estimativa de tokens:
    - OpenAI/Groq: usa `tiktoken` (cl100k_base) para contagem precisa.
    - HuggingFace:  estimativa via heurística (1 token ≈ 4 caracteres),
                    já que tokenizadores variam por modelo.

Todas as funções retornam dicionários simples, sem dependência de Streamlit,
facilitando testes unitários e reuso em outros contextos.
=============================================================================
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Estimativa de tokens
# ---------------------------------------------------------------------------

def _estimate_tokens_heuristic(text: str) -> int:
    """
    Estima a quantidade de tokens via heurística de caracteres.

    Regra empírica amplamente usada: 1 token ≈ 4 caracteres em inglês/português.

    Args:
        text: Texto a ser estimado.

    Returns:
        int: Estimativa de tokens.
    """
    return max(1, len(text) // 4)


def _count_tokens_tiktoken(text: str) -> int:
    """
    Conta tokens com precisão usando tiktoken (OpenAI cl100k_base).

    Compatível com modelos GPT-3.5, GPT-4, GPT-4o e Groq (LLaMA, Mixtral).

    Args:
        text: Texto a ser tokenizado.

    Returns:
        int: Número exato de tokens.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback silencioso se tiktoken não estiver instalado
        return _estimate_tokens_heuristic(text)


def count_tokens(text: str, provider: str) -> int:
    """
    Conta tokens de um texto usando a estratégia adequada ao provedor.

    Args:
        text:     Texto a tokenizar.
        provider: Provedor LLM ("openai" | "groq" | "hf_hub").

    Returns:
        int: Contagem de tokens (precisa ou estimada).
    """
    if provider in ("openai", "groq"):
        return _count_tokens_tiktoken(text)
    return _estimate_tokens_heuristic(text)


# ---------------------------------------------------------------------------
# Métricas do histórico de conversa
# ---------------------------------------------------------------------------

def compute_history_metrics(chat_history: list, provider: str) -> dict:
    """
    Calcula métricas sobre o histórico de mensagens da sessão.

    Percorre o chat_history (lista de AIMessage/HumanMessage) e acumula:
    - Contagem total de mensagens (Human + AI separadamente)
    - Total de tokens estimados/contados no contexto atual
    - Tokens por papel (human_tokens, ai_tokens)

    Args:
        chat_history: Lista de mensagens LangChain (AIMessage/HumanMessage).
        provider:     Provedor LLM para escolha da estratégia de tokenização.

    Returns:
        dict: {
            "total_messages":  int,
            "human_messages":  int,
            "ai_messages":     int,
            "total_tokens":    int,
            "human_tokens":    int,
            "ai_tokens":       int,
        }
    """
    total_tokens  = 0
    human_tokens  = 0
    ai_tokens     = 0
    human_count   = 0
    ai_count      = 0

    for msg in chat_history:
        tokens = count_tokens(msg.content, provider)
        total_tokens += tokens

        if isinstance(msg, HumanMessage):
            human_tokens += tokens
            human_count  += 1
        elif isinstance(msg, AIMessage):
            ai_tokens += tokens
            ai_count  += 1

    return {
        "total_messages": human_count + ai_count,
        "human_messages": human_count,
        "ai_messages":    ai_count,
        "total_tokens":   total_tokens,
        "human_tokens":   human_tokens,
        "ai_tokens":      ai_tokens,
    }


# ---------------------------------------------------------------------------
# Métricas de recuperação RAG
# ---------------------------------------------------------------------------

def compute_rag_metrics(context_docs: list, query: str, provider: str) -> dict:
    """
    Extrai e calcula métricas dos documentos recuperados pelo RAG.

    A partir dos documentos no `result['context']` da chain, coleta:
    - Quantidade de chunks recuperados
    - Fontes únicas (nomes de arquivo distintos)
    - Páginas referenciadas
    - Tamanho médio dos chunks (em tokens)
    - Tokens totais de contexto injetados no prompt

    Args:
        context_docs: Lista de Document objetos retornados pelo retriever.
        query:        Pergunta do usuário (para calcular tokens da query).
        provider:     Provedor LLM para tokenização.

    Returns:
        dict: {
            "chunks_retrieved":   int,
            "unique_sources":     list[str],
            "pages_referenced":   list[int | str],
            "avg_chunk_tokens":   float,
            "total_context_tokens": int,
            "query_tokens":       int,
            "chunks_detail":      list[dict],  # metadados por chunk
        }
    """
    if not context_docs:
        return {
            "chunks_retrieved":     0,
            "unique_sources":       [],
            "pages_referenced":     [],
            "avg_chunk_tokens":     0.0,
            "total_context_tokens": 0,
            "query_tokens":         count_tokens(query, provider),
            "chunks_detail":        [],
        }

    import os

    chunks_detail         = []
    unique_sources        = set()
    pages_referenced      = []
    total_context_tokens  = 0

    for idx, doc in enumerate(context_docs):
        chunk_tokens = count_tokens(doc.page_content, provider)
        total_context_tokens += chunk_tokens

        source_path  = doc.metadata.get("source", "desconhecido")
        source_name  = os.path.basename(source_path)
        page         = doc.metadata.get("page", "N/A")

        unique_sources.add(source_name)
        pages_referenced.append(page)

        chunks_detail.append({
            "index":        idx,
            "source":       source_name,
            "page":         page,
            "tokens":       chunk_tokens,
            "char_count":   len(doc.page_content),
            "preview":      doc.page_content[:120] + "…" if len(doc.page_content) > 120 else doc.page_content,
        })

    avg_chunk_tokens = total_context_tokens / len(context_docs) if context_docs else 0.0

    return {
        "chunks_retrieved":     len(context_docs),
        "unique_sources":       sorted(unique_sources),
        "pages_referenced":     pages_referenced,
        "avg_chunk_tokens":     round(avg_chunk_tokens, 1),
        "total_context_tokens": total_context_tokens,
        "query_tokens":         count_tokens(query, provider),
        "chunks_detail":        chunks_detail,
    }


# ---------------------------------------------------------------------------
# Métricas do modelo (metadados LLM)
# ---------------------------------------------------------------------------

def extract_llm_metadata(provider: str, model: str, temperature: float) -> dict:
    """
    Consolida metadados do modelo em uso, incluindo janela de contexto real.

    A janela de contexto é obtida dinamicamente do MODEL_CATALOG em
    core/models.py — cada modelo tem seu valor exato em vez de um
    placeholder genérico por provedor.

    Args:
        provider:    Provedor LLM selecionado.
        model:       Nome/ID do modelo.
        temperature: Temperatura configurada.

    Returns:
        dict com: provider, provider_label, provider_icon, model,
                  temperature, context_window (int), context_window_label (str),
                  price_input, price_output, tier.
    """
    from core.models import get_model_info

    provider_info = {
        "hf_hub": {"label": "HuggingFace Hub", "icon": "🤗"},
        "openai": {"label": "OpenAI",           "icon": "🟢"},
        "groq":   {"label": "Groq",             "icon": "⚡"},
    }
    info       = provider_info.get(provider, {"label": provider, "icon": "🤖"})
    model_info = get_model_info(provider, model)

    ctx = model_info["context_window"]
    # Label legível: 4096 → "4k", 131072 → "131k", 163840 → "160k"
    if ctx >= 1_000:
        ctx_label = f"{ctx // 1_000}k tokens"
    else:
        ctx_label = f"{ctx} tokens"

    price_input, price_output = model_info["price"]

    return {
        "provider":          provider,
        "provider_label":    info["label"],
        "provider_icon":     info["icon"],
        "model":             model,
        "temperature":       temperature,
        "context_window":    ctx,           # int — usado para cálculo de %
        "context_window_label": ctx_label,  # str — exibição amigável
        "price_input":       price_input,   # USD por 1K tokens input
        "price_output":      price_output,  # USD por 1K tokens output
        "tier":              model_info["tier"],
    }


# ---------------------------------------------------------------------------
# Timer de latência
# ---------------------------------------------------------------------------

class LatencyTimer:
    """
    Utilitário simples para medir latência de operações.

    Uso:
        timer = LatencyTimer()
        timer.start()
        # ... operação ...
        elapsed = timer.stop()  # retorna segundos (float)
    """

    def __init__(self):
        self._start: float | None = None

    def start(self) -> None:
        """Inicia o cronômetro."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """
        Para o cronômetro e retorna o tempo decorrido em segundos.

        Returns:
            float: Segundos decorridos desde start().
        """
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self._start = None
        return round(elapsed, 2)
