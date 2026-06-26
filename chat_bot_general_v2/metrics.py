"""
metrics.py
==========
Toda a lógica de cálculo de métricas fica aqui:
  - extração de tokens/metadados de cada chunk do stream
  - cálculo de custo por chamada
  - estimativa de tokens da janela de contexto (histórico)
  - registro imutável de cada turno (TurnRecord) para o log de observabilidade

Nenhuma dependência de Streamlit → 100 % testável de forma isolada.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage

from config import DEFAULT_METRICS, PRICING


# ---------------------------------------------------------------------------
# Estrutura de dados que acumula metadados durante o streaming
# ---------------------------------------------------------------------------
@dataclass
class StreamMetadata:
    input_tokens:       int = 0
    output_tokens:      int = 0
    reasoning_tokens:   int = 0
    finish_reason:      str = "stop"
    system_fingerprint: str = "N/A"
    message_id:         str = ""   # id nativo do provedor, se disponível

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
            # Tenta extrair id nativo do provedor (OpenAI: "id", Ollama: "id")
            if "id" in rmeta and not self.message_id:
                self.message_id = str(rmeta["id"])

        # Fallback: id gerado localmente se o provedor não retornar um
        if not self.message_id:
            self.message_id = f"local-{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Registro imutável de um turno completo (usuário + LLM)
# ---------------------------------------------------------------------------
@dataclass
class TurnRecord:
    """
    Snapshot completo de um turno de conversa.
    Registrado no momento do evento — nunca reconstruído depois.
    Todas as colunas da tabela de observabilidade derivam daqui.
    """
    # Identifiers
    turn_number:        int
    id_session:         str   # uuid4 gerado no início da sessão
    id_message:         str   # id nativo do provedor ou uuid4 de fallback

    # Provedor / configuração ativa no turno
    provider:           str
    model_label:        str
    personality:        str
    temperature:        float

    # Mensagem do usuário
    user_datetime:      datetime
    user_message:       str
    user_tokens:        int   # estimativa chars ÷ 4

    # Resposta da LLM
    llm_datetime:       datetime
    llm_message:        str
    llm_input_tokens:   int   # tokens de entrada reportados pelo provedor
    llm_output_tokens:  int   # tokens de saída reportados pelo provedor
    llm_reasoning_tokens: int

    # Performance
    latency_s:          float
    tokens_per_sec:     float
    finish_reason:      str
    system_fingerprint: str

    # Custo
    turn_cost_usd:      float


# ---------------------------------------------------------------------------
# Funções utilitárias — sem efeitos colaterais
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
    """Monta o dicionário de métricas para o session_state (sidebar KPIs)."""
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


def build_turn_record(
    *,
    turn_number:    int,
    id_session:     str,
    meta:           StreamMetadata,
    provider:       str,
    model_label:    str,
    personality:    str,
    temperature:    float,
    user_query:     str,
    user_ts:        datetime,
    llm_ts:         datetime,
    full_response:  str,
    latency:        float,
) -> TurnRecord:
    """
    Constrói o TurnRecord completo após o fim do stream.
    Centraliza toda a lógica de montagem — app.py só chama esta função.
    """
    tokens_per_sec = (
        meta.output_tokens / latency
        if latency > 0 and meta.output_tokens > 0
        else len(full_response.split()) / max(latency, 1e-6)
    )
    turn_cost = calculate_cost(meta.input_tokens, meta.output_tokens)

    return TurnRecord(
        turn_number          = turn_number,
        id_session           = id_session,
        id_message           = meta.message_id,
        provider             = provider,
        model_label          = model_label,
        personality          = personality,
        temperature          = temperature,
        user_datetime        = user_ts,
        user_message         = user_query,
        user_tokens          = len(user_query) // 4,
        llm_datetime         = llm_ts,
        llm_message          = full_response,
        llm_input_tokens     = meta.input_tokens,
        llm_output_tokens    = meta.output_tokens,
        llm_reasoning_tokens = meta.reasoning_tokens,
        latency_s            = latency,
        tokens_per_sec       = tokens_per_sec,
        finish_reason        = meta.finish_reason,
        system_fingerprint   = meta.system_fingerprint,
        turn_cost_usd        = turn_cost,
    )


def turn_log_to_dataframe(turn_log: list[TurnRecord]):
    """
    Converte a lista de TurnRecord em DataFrame pronto para exibição.
    Tipos corretos, colunas renomeadas para português, datetimes formatados.
    """
    import pandas as pd

    if not turn_log:
        return pd.DataFrame()

    rows = []
    for t in turn_log:
        rows.append({
            "# Turno":              t.turn_number,
            "ID Sessão":            t.id_session,
            "ID Mensagem":          t.id_message,
            "Provedor":             t.model_label,
            "Personalidade":        t.personality,
            "Temperature":          t.temperature,
            # Usuário
            "Data/Hora Usuário":    t.user_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "Mensagem Usuário":     t.user_message[:120] + "…" if len(t.user_message) > 120 else t.user_message,
            "Tokens Usuário":       t.user_tokens,
            # LLM
            "Data/Hora LLM":        t.llm_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "Resposta LLM":         t.llm_message[:120] + "…" if len(t.llm_message) > 120 else t.llm_message,
            "Tokens Input LLM":     t.llm_input_tokens,
            "Tokens Output LLM":    t.llm_output_tokens,
            "Tokens Raciocínio":    t.llm_reasoning_tokens,
            # Performance
            "Latência (s)":         round(t.latency_s, 3),
            "Tokens/s":             round(t.tokens_per_sec, 1),
            "Finish Reason":        t.finish_reason,
            "System Fingerprint":   t.system_fingerprint,
            # Custo
            "Custo Turno (USD)":    round(t.turn_cost_usd, 6),
        })

    df = pd.DataFrame(rows)

    # Garante tipos numéricos corretos para ordenação e filtros
    int_cols = ["# Turno", "Tokens Usuário", "Tokens Input LLM",
                "Tokens Output LLM", "Tokens Raciocínio"]
    float_cols = ["Temperature", "Latência (s)", "Tokens/s", "Custo Turno (USD)"]
    for c in int_cols:
        df[c] = df[c].astype(int)
    for c in float_cols:
        df[c] = df[c].astype(float)

    return df


# ---------------------------------------------------------------------------
# Métricas da Janela de Contexto (histórico do chat)
# ---------------------------------------------------------------------------

def estimate_history_tokens(history: list[BaseMessage]) -> int:
    """
    Estima o total de tokens no histórico usando a heurística chars ÷ 4,
    amplamente adotada para modelos GPT-like.
    """
    return sum(len(msg.content) for msg in history) // 4


def count_history_messages(history: list[BaseMessage]) -> int:
    """Retorna a quantidade de mensagens no histórico."""
    return len(history)
