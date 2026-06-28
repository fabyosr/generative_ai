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
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from langchain_core.messages import BaseMessage

from config import (
    DEFAULT_METRICS,
    PRICING,
    MODEL_CONTEXT_LIMITS,
    CONTEXT_WINDOW_ALERT_PCT,
    FINISH_REASON_TRUNCATED,
    COST_PROJECTION_SCALES,
)

if TYPE_CHECKING:
    from guardrails import GuardrailResult


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

    # Auditoria de guardrails
    input_flagged:      bool    # input foi bloqueado?
    input_gr_layer:     str     # camada que decidiu (profanity/detoxify/llamaguard/none)
    input_gr_category:  str     # categoria detectada
    input_gr_score:     float   # score de confiança
    output_flagged:     bool    # output foi bloqueado?
    output_gr_layer:    str
    output_gr_category: str
    output_gr_score:    float


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
    turn_number:      int,
    id_session:       str,
    meta:             StreamMetadata,
    provider:         str,
    model_label:      str,
    personality:      str,
    temperature:      float,
    user_query:       str,
    user_ts:          datetime,
    llm_ts:           datetime,
    full_response:    str,
    latency:          float,
    input_gr_result,               # GuardrailResult
    output_gr_result,              # GuardrailResult
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
        input_flagged        = not input_gr_result.safe,
        input_gr_layer       = input_gr_result.layer.value,
        input_gr_category    = input_gr_result.category,
        input_gr_score       = input_gr_result.score,
        output_flagged       = not output_gr_result.safe,
        output_gr_layer      = output_gr_result.layer.value,
        output_gr_category   = output_gr_result.category,
        output_gr_score      = output_gr_result.score,
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

            # ── Grupo A — métricas derivadas (calculadas na hora de exibir) ──
            # Nenhum campo novo no TurnRecord — derivadas dos campos existentes

            # % da janela de contexto consumida neste turno
            # calculada com context_window_pct() usando limite do provedor
            "Context Window %":     context_window_pct(t.llm_input_tokens, t.provider),

            # Razão output/input — detecta contexto inflado (< 0.05 = atenção)
            "Token Efficiency":     token_efficiency_ratio(t.llm_input_tokens, t.llm_output_tokens),

            # Resposta cortada: finish_reason = length ou max_tokens
            "Resposta Cortada":     "✂️ Sim" if is_response_truncated(t.finish_reason) else "✅ Ok",

            # Guardrails — auditoria
            "Input Bloqueado":      "🚫 Sim" if t.input_flagged  else "✅ Ok",
            "Input GR Layer":       t.input_gr_layer,
            "Input GR Categoria":   t.input_gr_category,
            "Input GR Score":       round(t.input_gr_score, 3),
            "Output Bloqueado":     "🚫 Sim" if t.output_flagged else "✅ Ok",
            "Output GR Layer":      t.output_gr_layer,
            "Output GR Categoria":  t.output_gr_category,
            "Output GR Score":      round(t.output_gr_score, 3),
        })

    df = pd.DataFrame(rows)

    # Garante tipos numéricos corretos para ordenação e filtros
    int_cols = ["# Turno", "Tokens Usuário", "Tokens Input LLM",
                "Tokens Output LLM", "Tokens Raciocínio"]
    float_cols = ["Temperature", "Latência (s)", "Tokens/s", "Custo Turno (USD)", "Input GR Score", "Output GR Score", "Context Window %", "Token Efficiency"]
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


# ---------------------------------------------------------------------------
# Grupo A — métricas derivadas (calculadas na hora de exibir, sem mudar
# a assinatura de build_turn_record ou a estrutura do TurnRecord)
# ---------------------------------------------------------------------------

def context_window_pct(input_tokens: int, provider: str) -> float:
    """
    Percentual da janela de contexto consumida neste turno.
    Usa MODEL_CONTEXT_LIMITS do config para o denominador correto por modelo.
    Retorna 0.0 se input_tokens não estiver disponível.
    """
    limit = MODEL_CONTEXT_LIMITS.get(provider, 8_000)
    if limit <= 0 or input_tokens <= 0:
        return 0.0
    return round(min((input_tokens / limit) * 100, 100.0), 1)


def token_efficiency_ratio(input_tokens: int, output_tokens: int) -> float:
    """
    Razão output / input tokens.
    Valores muito baixos (< 0.05) indicam que o contexto acumulado domina
    o custo sem gerar output proporcional — sinal de janela inflada.
    Retorna 0.0 quando input_tokens = 0.
    """
    if input_tokens <= 0:
        return 0.0
    return round(output_tokens / input_tokens, 3)


def is_response_truncated(finish_reason: str) -> bool:
    """
    Retorna True quando o modelo parou por limite de tokens (resposta cortada).
    Finish reasons que indicam truncamento: 'length', 'max_tokens'.
    """
    return finish_reason.lower() in FINISH_REASON_TRUNCATED


def session_duration_seconds(session_start: datetime) -> float:
    """Retorna quantos segundos se passaram desde o início da sessão."""
    return (datetime.now(timezone.utc) - session_start).total_seconds()


def format_session_duration(total_seconds: float) -> str:
    """Formata duração de sessão em mm:ss ou hh:mm:ss legível."""
    total = int(total_seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def guardrail_block_rate(turn_log: list) -> dict[str, float]:
    """
    Taxa de bloqueio por camada de guardrail sobre o total de turnos.
    Retorna dicionário {layer: pct} para exibição no dashboard.
    Ignora turnos com input_gr_layer = 'none' (não bloqueados).
    """
    if not turn_log:
        return {}
    total = len(turn_log)
    counts: dict[str, int] = {}
    for t in turn_log:
        if t.input_flagged:
            counts[t.input_gr_layer] = counts.get(t.input_gr_layer, 0) + 1
        if t.output_flagged:
            key = f"output/{t.output_gr_layer}"
            counts[key] = counts.get(key, 0) + 1
    return {layer: round((n / total) * 100, 1) for layer, n in counts.items()}


def cost_projection(cost_per_turn_usd: float) -> dict[str, float]:
    """
    Projeta custo mensal estimado para diferentes escalas de uso.
    Assume mesma distribuição de tokens e provedores do histórico atual.
    """
    return {
        f"{scale:,} interações/mês": round(cost_per_turn_usd * scale, 4)
        for scale in COST_PROJECTION_SCALES
    }


def build_session_kpis(
    turn_log:      list,
    session_start: datetime,
    provider:      str,
) -> dict:
    """
    Agrega todos os KPIs do Grupo A em um único dicionário para a UI.
    Chamado uma vez por render — sem side effects.
    """
    total_turns = len(turn_log)

    if total_turns == 0:
        return {
            "session_duration":    "00:00",
            "total_turns":         0,
            "context_window_pct":  0.0,
            "context_window_alert": False,
            "token_efficiency":    0.0,
            "truncated_turns":     0,
            "truncated_pct":       0.0,
            "block_rate":          {},
            "avg_cost_per_turn":   0.0,
            "cost_projection":     {},
        }

    last = turn_log[-1]
    ctx_pct       = context_window_pct(last.llm_input_tokens, provider)
    efficiency    = token_efficiency_ratio(last.llm_input_tokens, last.llm_output_tokens)
    truncated     = [t for t in turn_log if is_response_truncated(t.finish_reason)]
    avg_cost      = sum(t.turn_cost_usd for t in turn_log) / total_turns
    block_rate    = guardrail_block_rate(turn_log)
    projection    = cost_projection(avg_cost)
    duration      = format_session_duration(session_duration_seconds(session_start))

    return {
        "session_duration":     duration,
        "total_turns":          total_turns,
        "context_window_pct":   ctx_pct,
        "context_window_alert": ctx_pct >= CONTEXT_WINDOW_ALERT_PCT,
        "token_efficiency":     efficiency,
        "truncated_turns":      len(truncated),
        "truncated_pct":        round(len(truncated) / total_turns * 100, 1),
        "block_rate":           block_rate,
        "avg_cost_per_turn":    round(avg_cost, 6),
        "cost_projection":      projection,
    }
