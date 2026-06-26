"""
metrics.py
==========
Toda a lógica de cálculo de métricas fica aqui:
  - extração de tokens/metadados de cada chunk do stream
  - cálculo de custo por chamada
  - estimativa de tokens da janela de contexto (histórico)

Nenhuma dependência de Streamlit → 100 % testável de forma isolada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from config import DEFAULT_METRICS, PRICING


# ---------------------------------------------------------------------------
# Estrutura de dados que acumula metadados durante o streaming
# ---------------------------------------------------------------------------
@dataclass
class StreamMetadata:
    input_tokens:     int   = 0
    output_tokens:    int   = 0
    reasoning_tokens: int   = 0
    finish_reason:    str   = "stop"
    system_fingerprint: str = "N/A"

    def update_from_chunk(self, chunk: Any) -> None:
        """Extrai e acumula metadados de um chunk LangChain."""
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            meta = chunk.usage_metadata
            self.input_tokens  = meta.get("input_tokens",  self.input_tokens)
            self.output_tokens = meta.get("output_tokens", self.output_tokens)
            details = meta.get("input_token_details", {})
            self.reasoning_tokens = details.get("reasoning", self.reasoning_tokens)

        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
            rmeta = chunk.response_metadata
            if "finish_reason" in rmeta:
                self.finish_reason = str(rmeta["finish_reason"])
            if "system_fingerprint" in rmeta:
                self.system_fingerprint = str(rmeta["system_fingerprint"])


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------

def apply_token_fallback(meta: StreamMetadata, user_query: str, response: str) -> StreamMetadata:
    """
    Quando o provedor não devolve tokens nativos (ex.: HuggingFace gratuito),
    faz uma estimativa simples baseada em contagem de caracteres.
    """
    if meta.input_tokens == 0:
        meta.input_tokens  = (len(user_query) + len(response)) // 4
        meta.output_tokens = len(response) // 4
    return meta


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Retorna o custo estimado em USD para uma única chamada."""
    return (
        (input_tokens  / 1_000_000) * PRICING["input_per_million"]
        + (output_tokens / 1_000_000) * PRICING["output_per_million"]
    )


def build_metrics_dict(
    meta: StreamMetadata,
    latency: float,
    full_response: str,
) -> dict:
    """
    Monta o dicionário de métricas que será salvo no session_state,
    calculando também tokens/s.
    """
    tokens_per_sec = (
        meta.output_tokens / latency
        if latency > 0 and meta.output_tokens > 0
        else len(full_response.split()) / max(latency, 1e-6)
    )
    return {
        "last_input_tokens":  meta.input_tokens,
        "last_output_tokens": meta.output_tokens,
        "latency":            latency,
        "tokens_per_sec":     tokens_per_sec,
        "finish_reason":      meta.finish_reason,
        "system_fingerprint": meta.system_fingerprint,
        "reasoning_tokens":   meta.reasoning_tokens,
    }


# ---------------------------------------------------------------------------
# Métricas da Janela de Contexto (histórico do chat)
# ---------------------------------------------------------------------------

def estimate_history_tokens(history: list[BaseMessage]) -> int:
    """
    Estima o total de tokens no histórico de mensagens usando a heurística
    chars / 4, amplamente adotada para modelos GPT-like.
    """
    total_chars = sum(len(msg.content) for msg in history)
    return total_chars // 4


def count_history_messages(history: list[BaseMessage]) -> int:
    """Retorna a quantidade de mensagens no histórico (sem contar a inicial do bot)."""
    return len(history)
