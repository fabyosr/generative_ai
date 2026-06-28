"""
app.py
======
Camada exclusiva de UI (Streamlit).
Não contém lógica de negócio, cálculo de métricas nem instanciação de modelos.
Toda a lógica está em: config.py | llm_factory.py | chat_chain.py | metrics.py
"""

import os
import time
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chat_chain import build_stream
from config import DEFAULT_METRICS, LLM_PROVIDERS, PERSONALITIES
from guardrails import (
    GuardrailLayer,
    GuardrailResult,
    BLOCK_MESSAGE_INPUT,
    BLOCK_MESSAGE_OUTPUT,
    check_input,
    check_output,
    guardrail_status,
)
from llm_factory import is_provider_available
from metrics import (
    StreamMetadata,
    apply_token_fallback,
    build_metrics_dict,
    build_turn_record,
    calculate_cost,
    count_history_messages,
    estimate_history_tokens,
    turn_log_to_dataframe,
)

# ===========================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ===========================================================================
st.set_page_config(
    page_title="Virtual Assistant 🤖",
    page_icon="🤖",
    layout="wide",
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
        [data-testid="stDataFrame"] td { white-space: nowrap; }

        /* Chat container com scroll interno — input fica fixo abaixo */
        .chat-history-container {
            overflow-y: auto;
            padding-right: 8px;
        }
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
            st.session_state.api_configured = False

    if "id_session" not in st.session_state:
        st.session_state.id_session = str(uuid4())

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

    if "turn_log" not in st.session_state:
        st.session_state.turn_log = []

    # Chave OpenAI digitada pelo usuário na UI
    if "openai_api_key_input" not in st.session_state:
        st.session_state.openai_api_key_input = ""


_init_session_state()


# ===========================================================================
# 3. SIDEBAR — CONTROLES E DASHBOARD
# ===========================================================================
def _render_sidebar() -> tuple[str, str, float]:
    """Renderiza a sidebar e retorna (personality_key, provider_key, temperature)."""

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

        # ── API Key OpenAI (apenas quando provider = openai) ────────────────
        if provider == "openai":
            openai_key_input = st.text_input(
                "🔑 OpenAI API Key:",
                value=st.session_state.openai_api_key_input,
                type="password",
                placeholder="sk-...",
                help="Cole sua chave da OpenAI. Não é armazenada permanentemente.",
            )
            if openai_key_input:
                st.session_state.openai_api_key_input = openai_key_input
                os.environ["OPENAI_API_KEY"] = openai_key_input
        else:
            openai_key_input = ""

        # ── Aviso Ollama ─────────────────────────────────────────────────────
        if provider == "ollama":
            available, reason = is_provider_available("ollama")
            if not available:
                st.error(
                    f"⚠️ **Ollama indisponível**\n\n{reason}",
                    icon="🚫",
                )

        temperature = st.slider(
            "Criatividade (Temperature):",
            min_value=0.0, max_value=1.0, value=0.1, step=0.1,
        )

        st.markdown("---")

        # ── Cost Monitor ─────────────────────────────────────────────────────
        st.metric(
            label="💰 Custo Acumulado da Sessão",
            value=f"USD {st.session_state.accumulated_cost:.5f}",
            help="Custo estimado baseado no consumo real de tokens (Tabela GPT-4o-mini)",
        )

        st.markdown("---")

        # ── Monitor de Metadados ──────────────────────────────────────────────
        with st.expander("📊 Monitor de Metadados Reais", expanded=True):
            m = st.session_state.metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📥 Input Tokens",     m["last_input_tokens"])
                st.metric("⏱️ Latência",          f"{m['latency']:.2f}s")
                st.metric("🧠 Tokens Raciocínio", m["reasoning_tokens"])
            with col2:
                st.metric("📤 Output Tokens", m["last_output_tokens"])
                st.metric("⚡ Velocidade",     f"{m['tokens_per_sec']:.1f} t/s")
                st.metric("🛑 Fim do Stream",  m["finish_reason"])

        # ── Janela de Contexto ────────────────────────────────────────────────
        with st.expander("🧠 Janela de Contexto Atual", expanded=True):
            history      = st.session_state.chat_history
            ctx_tokens   = estimate_history_tokens(history)
            ctx_messages = count_history_messages(history)
            col1, col2   = st.columns(2)
            with col1:
                st.metric(
                    "💬 Mensagens no Histórico", ctx_messages,
                    help="Total de mensagens (usuário + assistente) na memória do chat.",
                )
            with col2:
                st.metric(
                    "🔢 Tokens Acumulados", ctx_tokens,
                    help="Estimativa de tokens enviados ao modelo via MessagesPlaceholder (chars ÷ 4).",
                )

        # ── Histórico de Latência ─────────────────────────────────────────────
        with st.expander("📈 Histórico de Latência", expanded=True):
            if st.session_state.latency_history:
                df_lat = pd.DataFrame(
                    st.session_state.latency_history, columns=["Latência (s)"]
                )
                st.line_chart(df_lat, height=120)
                st.caption(f"Fingerprint: {st.session_state.metrics['system_fingerprint']}")
            else:
                st.caption("Envie mensagens para mapear a latência.")

        # ── Status Guardrails ─────────────────────────────────────────────────
        with st.expander("🛡️ Status dos Guardrails", expanded=False):
            gs = guardrail_status(
                provider       = provider,
                openai_api_key = openai_key_input,
            )
            st.markdown("**Camadas ativas:**")
            c1, c2 = st.columns(2)
            c1.metric("🔒 System Prompt", "✅ ON" if gs["system_prompt"]    else "❌ OFF")
            c2.metric("📝 Léxico PT+EN",  "✅ ON" if gs["better_profanity"] else "❌ OFF")
            c3, c4 = st.columns(2)
            c3.metric("🤖 OpenAI Mod",    "✅ ON" if gs["openai_moderation"] else "⚠️ OFF")
            c4.metric("🦙 LlamaGuard",    "✅ ON" if gs["llamaguard"]        else "⚠️ OFF")

            if not gs["openai_moderation"]:
                st.caption("⚠️ OpenAI Moderation inativa — disponível somente com provedor OpenAI.")
            if gs["llamaguard"]:
                st.caption(
                    "🦙 LlamaGuard ativo — acionado na zona cinza da Moderation "
                    "ou quando provedor não é OpenAI."
                )
            else:
                st.caption("⚠️ LlamaGuard inativo — configure HUGGINGFACEHUB_API_TOKEN para habilitar.")

            st.markdown("---")
            log         = st.session_state.turn_log
            if not log:
                st.caption("Nenhum turno registrado ainda.")
            else:
                total       = len(log)
                inp_blocked = sum(1 for t in log if t.input_flagged)
                out_blocked = sum(1 for t in log if t.output_flagged)
                col1, col2, col3 = st.columns(3)
                col1.metric("🔁 Turnos",          total)
                col2.metric("🚫 Input Bloqueado",  inp_blocked)
                col3.metric("🚫 Output Bloqueado", out_blocked)

                flagged_turns = [t for t in log if t.input_flagged or t.output_flagged]
                if flagged_turns:
                    st.markdown("**Últimas detecções:**")
                    for t in flagged_turns[-3:]:
                        if t.input_flagged:
                            st.warning(
                                f"Turno {t.turn_number} · Input · "
                                f"{t.input_gr_layer} · {t.input_gr_category} "
                                f"({t.input_gr_score:.2f})",
                                icon="🚫",
                            )
                        if t.output_flagged:
                            st.warning(
                                f"Turno {t.turn_number} · Output · "
                                f"{t.output_gr_layer} · {t.output_gr_category} "
                                f"({t.output_gr_score:.2f})",
                                icon="🚫",
                            )

        # ── Reset ─────────────────────────────────────────────────────────────
        if st.button("Limpar Tudo", width="stretch"):
            st.session_state.chat_history        = [AIMessage(content="Oi! Tudo reiniciado. Como posso ajudar?")]
            st.session_state.metrics             = DEFAULT_METRICS.copy()
            st.session_state.accumulated_cost    = 0.0
            st.session_state.latency_history     = []
            st.session_state.turn_log            = []
            st.session_state.id_session          = str(uuid4())
            st.session_state.openai_api_key_input = ""
            st.rerun()

    return personality, provider, temperature


personality_key, provider_key, temperature = _render_sidebar()


# ===========================================================================
# 4. TABS PRINCIPAIS
# ===========================================================================
tab_chat, tab_log = st.tabs(["💬 Chat", "📋 Log de Observabilidade"])


# ===========================================================================
# 5. TAB CHAT
# ===========================================================================
with tab_chat:

    # ── Container com scroll interno para o histórico ────────────────────────
    # st.container(height=...) mantém o histórico em área rolável
    # e o st.chat_input fica fixo abaixo do container, fora do scroll
    history_container = st.container(height=520, border=False)

    with history_container:
        for message in st.session_state.chat_history:
            is_ai  = isinstance(message, AIMessage)
            role   = "assistant" if is_ai else "user"
            avatar = "🤖" if is_ai else "👤"
            with st.chat_message(role, avatar=avatar):
                st.markdown(message.content)

    # ── Input fixo abaixo do container ───────────────────────────────────────
    user_query = st.chat_input("Digite sua mensagem aqui...")

    if user_query and user_query.strip():

        # ── Bloqueia se Ollama indisponível ──────────────────────────────────
        if provider_key == "ollama":
            available, reason = is_provider_available("ollama")
            if not available:
                st.error(f"⚠️ {reason}")
                st.stop()

        user_ts = datetime.now(timezone.utc)

        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Re-renderiza histórico + nova mensagem dentro do container
        with history_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)

        # ── Guardrail INPUT ───────────────────────────────────────────────────
        input_gr = check_input(user_query, provider=provider_key)

        if not input_gr.safe:
            with history_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.warning(BLOCK_MESSAGE_INPUT)

            _safe_gr = GuardrailResult(
                safe=True, layer=GuardrailLayer.NONE,
                category="safe", score=0.0, reason="",
            )
            turn = build_turn_record(
                turn_number      = len(st.session_state.turn_log) + 1,
                id_session       = st.session_state.id_session,
                meta             = StreamMetadata(),
                provider         = provider_key,
                model_label      = LLM_PROVIDERS[provider_key],
                personality      = personality_key,
                temperature      = temperature,
                user_query       = user_query,
                user_ts          = user_ts,
                llm_ts           = user_ts,
                full_response    = BLOCK_MESSAGE_INPUT,
                latency          = 0.0,
                input_gr_result  = input_gr,
                output_gr_result = _safe_gr,
            )
            st.session_state.turn_log.append(turn)
            st.session_state.chat_history.append(AIMessage(content=BLOCK_MESSAGE_INPUT))
            st.rerun()

        # ── Stream da resposta ────────────────────────────────────────────────
        with history_container:
            with st.chat_message("assistant", avatar="🤖"):
                placeholder   = st.empty()
                full_response = ""
                stream_meta   = StreamMetadata()

                start_time = time.time()

                try:
                    stream = build_stream(
                        user_query    = user_query,
                        chat_history  = st.session_state.chat_history,
                        provider      = provider_key,
                        temperature   = temperature,
                        system_prompt = PERSONALITIES[personality_key],
                        api_key       = st.session_state.openai_api_key_input,
                    )

                    for chunk in stream:
                        if chunk.content:
                            full_response += chunk.content
                            placeholder.markdown(full_response + "▌")
                        stream_meta.update_from_chunk(chunk)

                except Exception as exc:
                    error_msg = f"❌ Erro ao chamar o modelo: `{exc}`"
                    placeholder.error(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))
                    st.rerun()

                end_time = time.time()
                llm_ts   = datetime.now(timezone.utc)

                # ── Guardrail OUTPUT ──────────────────────────────────────────
                output_gr = check_output(full_response, provider=provider_key)
                if not output_gr.safe:
                    placeholder.warning(BLOCK_MESSAGE_OUTPUT)
                    full_response = BLOCK_MESSAGE_OUTPUT
                else:
                    placeholder.markdown(full_response)

        # ── Pós-processamento ─────────────────────────────────────────────────
        latency     = end_time - start_time
        stream_meta = apply_token_fallback(stream_meta, user_query, full_response)
        call_cost   = calculate_cost(stream_meta.input_tokens, stream_meta.output_tokens)

        st.session_state.accumulated_cost += call_cost
        st.session_state.latency_history.append(latency)
        st.session_state.metrics = build_metrics_dict(stream_meta, latency, full_response)

        turn = build_turn_record(
            turn_number      = len(st.session_state.turn_log) + 1,
            id_session       = st.session_state.id_session,
            meta             = stream_meta,
            provider         = provider_key,
            model_label      = LLM_PROVIDERS[provider_key],
            personality      = personality_key,
            temperature      = temperature,
            user_query       = user_query,
            user_ts          = user_ts,
            llm_ts           = llm_ts,
            full_response    = full_response,
            latency          = latency,
            input_gr_result  = input_gr,
            output_gr_result = output_gr,
        )
        st.session_state.turn_log.append(turn)
        st.session_state.chat_history.append(AIMessage(content=full_response))
        st.rerun()


# ===========================================================================
# 6. TAB LOG DE OBSERVABILIDADE
# ===========================================================================
with tab_log:
    st.subheader("📋 Log de Observabilidade por Turno")

    turn_log = st.session_state.turn_log

    if not turn_log:
        st.info("Nenhuma interação registrada ainda. Envie uma mensagem na aba Chat.")
    else:
        df = turn_log_to_dataframe(turn_log)

        # ── KPIs ──────────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🔁 Turnos",           len(turn_log))
        k2.metric("📥 Total Input Tok",  int(df["Tokens Input LLM"].sum()))
        k3.metric("📤 Total Output Tok", int(df["Tokens Output LLM"].sum()))
        k4.metric("⏱️ Latência Média",   f"{df['Latência (s)'].mean():.2f}s")
        k5.metric("💰 Custo Total",      f"USD {df['Custo Turno (USD)'].sum():.5f}")

        st.markdown("---")

        # ── Filtros ───────────────────────────────────────────────────────────
        with st.expander("🔍 Filtros", expanded=False):
            col_f1, col_f2, col_f3 = st.columns(3)

            providers_disp = df["Provedor"].unique().tolist()
            filtro_provider = col_f1.multiselect(
                "Provedor", providers_disp, default=providers_disp
            )

            personalities_disp = df["Personalidade"].unique().tolist()
            filtro_personality = col_f2.multiselect(
                "Personalidade", personalities_disp, default=personalities_disp
            )

            lat_min = float(df["Latência (s)"].min())
            lat_max = float(df["Latência (s)"].max())
            filtro_latencia = (
                col_f3.slider("Latência máxima (s)", lat_min, lat_max, lat_max, step=0.1)
                if lat_min < lat_max else lat_max
            )

            df_filtrado = df[
                df["Provedor"].isin(filtro_provider)
                & df["Personalidade"].isin(filtro_personality)
                & (df["Latência (s)"] <= filtro_latencia)
            ]

        # ── Tabela ────────────────────────────────────────────────────────────
        st.dataframe(
            df_filtrado,
            use_container_width=True,
            hide_index=True,
            column_config={
                "# Turno":           st.column_config.NumberColumn(width="small"),
                "ID Sessão":         st.column_config.TextColumn(width="medium"),
                "ID Mensagem":       st.column_config.TextColumn(width="medium"),
                "Mensagem Usuário":  st.column_config.TextColumn(width="large"),
                "Resposta LLM":      st.column_config.TextColumn(width="large"),
                "Latência (s)":      st.column_config.NumberColumn(format="%.3f", width="small"),
                "Tokens/s":          st.column_config.NumberColumn(format="%.1f",  width="small"),
                "Custo Turno (USD)": st.column_config.NumberColumn(format="%.6f", width="small"),
                "Temperature":       st.column_config.NumberColumn(format="%.1f",  width="small"),
                "Input Bloqueado":   st.column_config.TextColumn(width="small"),
                "Input GR Score":    st.column_config.NumberColumn(format="%.3f",  width="small"),
                "Output Bloqueado":  st.column_config.TextColumn(width="small"),
                "Output GR Score":   st.column_config.NumberColumn(format="%.3f",  width="small"),
            },
        )

        st.caption(
            f"Exibindo {len(df_filtrado)} de {len(df)} turnos  •  "
            f"ID Sessão atual: `{st.session_state.id_session}`"
        )

        # ── Export CSV ────────────────────────────────────────────────────────
        csv_bytes = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Exportar CSV",
            data      = csv_bytes,
            file_name = f"observabilidade_{st.session_state.id_session[:8]}.csv",
            mime      = "text/csv",
            width     = "stretch",
        )
