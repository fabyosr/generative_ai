"""
app.py
======
Camada exclusiva de UI (Streamlit).
Não contém lógica de negócio, cálculo de métricas nem instanciação de modelos.
Toda a lógica está em: config.py | llm_factory.py | chat_chain.py | metrics.py
"""

import os
import time

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chat_chain import build_stream
from config import DEFAULT_METRICS, LLM_PROVIDERS, PERSONALITIES
from metrics import (
    StreamMetadata,
    apply_token_fallback,
    build_metrics_dict,
    calculate_cost,
    count_history_messages,
    estimate_history_tokens,
)

# ===========================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ===========================================================================
st.set_page_config(
    page_title="Virtual Assistant 🤖",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
    <style>
        .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
        .stChatInputContainer { border-radius: 20px; }
        h1 {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 700;
            background: linear-gradient(45deg, #1E90FF, #12005e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 25px;
        }
        [data-testid="stMetricValue"] { font-size: 20px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Virtual Assistant 🤖")


# ===========================================================================
# 2. INICIALIZAÇÃO DO SESSION STATE
# ===========================================================================
def _init_session_state() -> None:
    """Garante que todas as chaves do session_state existam na primeira execução."""

    if "api_configured" not in st.session_state:
        try:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            st.session_state.api_configured = True
        except Exception:
            st.warning("Token do HuggingFace não encontrado.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Oi, sou seu assistente virtual! Como posso ajudar você?")
        ]

    if "metrics" not in st.session_state:
        st.session_state.metrics = DEFAULT_METRICS.copy()

    if "accumulated_cost" not in st.session_state:
        st.session_state.accumulated_cost = 0.0

    if "latency_history" not in st.session_state:
        st.session_state.latency_history = []


_init_session_state()


# ===========================================================================
# 3. SIDEBAR — CONTROLES E DASHBOARD
# ===========================================================================
def _render_sidebar() -> tuple[str, str, float]:
    """
    Renderiza a sidebar e retorna as escolhas do usuário.

    Retorna
    -------
    (personality_key, provider_key, temperature)
    """
    with st.sidebar:
        st.header("⚙️ Painel de Controle")

        personality = st.selectbox(
            "🎭 Personalidade do Bot:",
            options=list(PERSONALITIES.keys()),
        )

        provider = st.selectbox(
            "Selecione o Provedor de LLM:",
            options=list(LLM_PROVIDERS.keys()),
            format_func=lambda k: LLM_PROVIDERS[k],
        )

        temperature = st.slider(
            "Criatividade (Temperature):",
            min_value=0.0, max_value=1.0, value=0.1, step=0.1,
        )

        st.markdown("---")

        # ── Cost Monitor ────────────────────────────────────────────────────
        st.metric(
            label="💰 Custo Acumulado da Sessão",
            value=f"USD {st.session_state.accumulated_cost:.5f}",
            help="Custo estimado baseado no consumo real de tokens (Tabela GPT-4o-mini)",
        )

        st.markdown("---")

        # ── Monitor de Metadados ─────────────────────────────────────────────
        with st.expander("📊 Monitor de Metadados Reais", expanded=True):
            m = st.session_state.metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📥 Input Tokens",       m["last_input_tokens"])
                st.metric("⏱️ Latência",            f"{m['latency']:.2f}s")
                st.metric("🧠 Tokens de Raciocínio", m["reasoning_tokens"])
            with col2:
                st.metric("📤 Output Tokens",  m["last_output_tokens"])
                st.metric("⚡ Velocidade",      f"{m['tokens_per_sec']:.1f} t/s")
                st.metric("🛑 Fim do Stream",   m["finish_reason"])

        # ── Janela de Contexto (NOVA FEATURE) ───────────────────────────────
        with st.expander("🧠 Janela de Contexto Atual", expanded=True):
            history = st.session_state.chat_history
            ctx_tokens   = estimate_history_tokens(history)
            ctx_messages = count_history_messages(history)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="💬 Mensagens no Histórico",
                    value=ctx_messages,
                    help="Total de mensagens (usuário + assistente) na memória do chat.",
                )
            with col2:
                st.metric(
                    label="🔢 Tokens Acumulados",
                    value=ctx_tokens,
                    help="Estimativa de tokens enviados ao modelo via MessagesPlaceholder (chars ÷ 4).",
                )

        # ── Histórico de Latência ────────────────────────────────────────────
        with st.expander("📈 Histórico de Latência", expanded=True):
            if st.session_state.latency_history:
                df = pd.DataFrame(st.session_state.latency_history, columns=["Latência (s)"])
                st.line_chart(df, height=120)
                st.caption(f"Fingerprint: {st.session_state.metrics['system_fingerprint']}")
            else:
                st.caption("Envie mensagens para mapear a latência.")

        # ── Reset ────────────────────────────────────────────────────────────
        if st.button("Limpar Tudo", use_container_width=True):
            st.session_state.chat_history = [
                AIMessage(content="Oi! Tudo reiniciado. Como posso ajudar?")
            ]
            st.session_state.metrics         = DEFAULT_METRICS.copy()
            st.session_state.accumulated_cost = 0.0
            st.session_state.latency_history  = []
            st.rerun()

    return personality, provider, temperature


personality_key, provider_key, temperature = _render_sidebar()


# ===========================================================================
# 4. RENDERIZAÇÃO DO HISTÓRICO DE CHAT
# ===========================================================================
def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        is_ai   = isinstance(message, AIMessage)
        role    = "assistant" if is_ai else "user"
        avatar  = "🤖" if is_ai else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.content)


_render_chat_history()


# ===========================================================================
# 5. PROCESSAMENTO DA MENSAGEM DO USUÁRIO
# ===========================================================================
def _process_user_message(user_query: str) -> None:
    """Executa o stream, coleta métricas e atualiza o session_state."""

    # Adiciona mensagem do usuário ao histórico e exibe imediatamente
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # ── Streaming da resposta ────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        placeholder  = st.empty()
        full_response = ""
        stream_meta   = StreamMetadata()

        start_time = time.time()

        stream = build_stream(
            user_query   = user_query,
            chat_history = st.session_state.chat_history,
            provider     = provider_key,
            temperature  = temperature,
            system_prompt= PERSONALITIES[personality_key],
        )

        for chunk in stream:
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")
            stream_meta.update_from_chunk(chunk)

        end_time = time.time()
        placeholder.markdown(full_response)   # remove cursor de digitação

    # ── Pós-processamento de métricas ────────────────────────────────────────
    latency = end_time - start_time

    stream_meta = apply_token_fallback(stream_meta, user_query, full_response)

    call_cost = calculate_cost(stream_meta.input_tokens, stream_meta.output_tokens)
    st.session_state.accumulated_cost += call_cost

    st.session_state.latency_history.append(latency)
    st.session_state.metrics = build_metrics_dict(stream_meta, latency, full_response)

    # ── Salva resposta no histórico ──────────────────────────────────────────
    st.session_state.chat_history.append(AIMessage(content=full_response))

    # Força re-render para atualizar gráfico e KPIs da sidebar
    st.rerun()


# ===========================================================================
# 6. INPUT DO USUÁRIO
# ===========================================================================
user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query and user_query.strip():
    _process_user_message(user_query)
