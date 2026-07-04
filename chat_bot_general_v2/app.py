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
from sanitizer import sanitize, sanitizer_rules_summary
from text_analytics import analyze as text_analyze, analytics_status
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
    build_session_kpis,
    build_turn_record,
    calculate_cost,
    count_history_messages,
    estimate_history_tokens,
    is_response_truncated,
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

    # Timestamp de início da sessão — usado para duração e KPIs do Grupo A
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now(timezone.utc)


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

        # ── Grupo A — Métricas de Sessão e Qualidade ────────────────────────
        # Todas as métricas deste bloco são DERIVADAS de dados já existentes
        # no session_state — nenhuma chamada de rede, nenhum efeito colateral.
        # build_session_kpis() agrega tudo em um único dict para a UI.
        with st.expander("📐 Sessão & Qualidade", expanded=True):

            # Coleta todos os KPIs do Grupo A de uma vez
            kpis = build_session_kpis(
                turn_log      = st.session_state.turn_log,
                session_start = st.session_state.session_start,
                provider      = provider,
            )

            # ── Linha 1: visão geral da sessão ──────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "⏳ Duração",
                kpis["session_duration"],
                help="Tempo decorrido desde o início ou último reset da sessão.",
            )
            c2.metric(
                "🔁 Turnos",
                kpis["total_turns"],
                help="Total de interações registradas nesta sessão.",
            )
            c3.metric(
                "💸 Custo Médio/Turno",
                f"USD {kpis['avg_cost_per_turn']:.6f}",
                help="Média do custo por interação — base para projeção de escala.",
            )

            # ── Linha 2: saúde da janela de contexto e qualidade ────────────
            c4, c5, c6 = st.columns(3)

            # Context window %: input_tokens do último turno ÷ limite do modelo
            ctx_pct = kpis["context_window_pct"]
            c4.metric(
                "🪟 Context Window",
                f"{ctx_pct:.1f}%",
                help=(
                    "% da janela de contexto consumida no último turno. "
                    "Acima de 80% o modelo pode começar a 'esquecer' mensagens antigas."
                ),
            )

            # Token efficiency: razão output/input — detecta contexto inflado
            c5.metric(
                "⚖️ Token Efficiency",
                f"{kpis['token_efficiency']:.3f}",
                help=(
                    "Razão output ÷ input tokens. "
                    "Valores < 0.05 indicam que o contexto acumulado domina o custo "
                    "sem gerar output proporcional."
                ),
            )

            # Respostas cortadas: finish_reason = 'length' ou 'max_tokens'
            c6.metric(
                "✂️ Respostas Cortadas",
                f"{kpis['truncated_turns']} ({kpis['truncated_pct']:.1f}%)",
                help=(
                    "Turnos onde finish_reason=length — "
                    "resposta interrompida pelo limite de tokens do modelo."
                ),
            )

            # Alerta visual quando context window ultrapassa 80%
            if kpis["context_window_alert"]:
                st.warning(
                    f"⚠️ Janela de contexto em **{ctx_pct:.1f}%** — "
                    "considere usar 'Limpar Tudo' para evitar degradação das respostas.",
                )

            # ── Projeção de custo mensal por escala ─────────────────────────
            # Só exibe quando há pelo menos 1 turno para ter custo médio real
            if kpis["cost_projection"]:
                st.markdown("**Projeção de custo mensal (baseada no custo médio atual):**")
                proj_cols = st.columns(len(kpis["cost_projection"]))
                for col, (label, value) in zip(proj_cols, kpis["cost_projection"].items()):
                    col.metric(label, f"USD {value:.2f}")

            # ── Taxa de bloqueio por camada de guardrail ─────────────────────
            # Só exibe quando há bloqueios registrados no turn_log
            if kpis["block_rate"]:
                st.markdown("**Taxa de bloqueio por camada:**")
                for layer, pct in kpis["block_rate"].items():
                    st.progress(
                        min(int(pct), 100),
                        text=f"{layer}: {pct:.1f}% dos turnos",
                    )

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
            log = st.session_state.turn_log
            if not log:
                st.caption("Nenhum turno registrado ainda.")
            else:
                total       = len(log)
                inp_blocked = sum(1 for t in log if t.input_flagged)
                out_blocked = sum(1 for t in log if t.output_flagged)
                # sp_refusal: guardrails aprovaram mas modelo recusou via SP
                sp_refused  = sum(1 for t in log if t.sp_refusal)
                # Toxicidade média dos inputs — inclui mensagens não bloqueadas
                tox_scores  = [t.input_toxicity_score for t in log if t.input_toxicity_score > 0]
                avg_tox     = sum(tox_scores) / len(tox_scores) if tox_scores else 0.0

                # Linha 1 — bloqueios diretos
                col1, col2, col3 = st.columns(3)
                col1.metric("🔁 Turnos",          total)
                col2.metric("🚫 Input Bloqueado",  inp_blocked)
                col3.metric("🚫 Output Bloqueado", out_blocked)

                # Linha 2 — métricas de toxicidade e recusa via SP
                col4, col5 = st.columns(2)
                col4.metric(
                    "🛡️ Recusas via SP",
                    sp_refused,
                    help=(
                        "Turnos onde os guardrails aprovaram mas o modelo recusou "
                        "semanticamente por conta do safety prompt. "
                        "Ex: 'bomba caseira com fins didáticos'."
                    ),
                )
                col5.metric(
                    "☣️ Toxicidade Média",
                    f"{avg_tox:.3f}",
                    help=(
                        "Score médio de toxicidade dos inputs via OpenAI Moderation. "
                        "Inclui mensagens aprovadas — detecta padrões de risco mesmo "
                        "sem bloqueio. Disponível somente com provedor OpenAI."
                    ),
                )

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

        # ── Text Analytics ─────────────────────────────────────────────────
        with st.expander("📊 Text Analytics", expanded=False):
            # Status das dependencias
            astatus = analytics_status()
            c1, c2, c3 = st.columns(3)
            c1.metric("spaCy",          "✅ ON" if astatus["spacy"] else "⚠️ OFF",
                      help=f"Modelo: {astatus['spacy_model']}")
            c2.metric("Embeddings",     "✅ ON" if astatus["sentence_transformers"] else "⚠️ OFF",
                      help=f"Modelo: {astatus['embedding_model']}")
            c3.metric("TextBlob",       "✅ ON" if astatus["textblob"] else "⚠️ OFF")

            if not astatus["spacy"]:
                st.caption("⚠️ spaCy indisponivel -- instale: pip install spacy && python -m spacy download pt_core_news_sm")
            if not astatus["sentence_transformers"]:
                st.caption("⚠️ sentence-transformers indisponivel -- instale: pip install sentence-transformers")

            st.markdown("---")

            # KPIs agregados da sessao
            log = st.session_state.turn_log
            turns_with_ta = [t for t in log if t.text_metrics is not None]
            if not turns_with_ta:
                st.caption("Nenhuma analise disponivel ainda.")
            else:
                # Medias das metricas mais relevantes
                avg_sim    = sum(t.text_metrics.semantic_similarity for t in turns_with_ta) / len(turns_with_ta)
                avg_flesch = sum(t.text_metrics.r_flesch             for t in turns_with_ta) / len(turns_with_ta)
                avg_ttr    = sum(t.text_metrics.r_ttr                for t in turns_with_ta) / len(turns_with_ta)
                avg_novelty= sum(t.text_metrics.response_novelty     for t in turns_with_ta) / len(turns_with_ta)
                followups  = sum(1 for t in turns_with_ta if t.text_metrics.is_followup)
                avg_ms     = sum(t.text_metrics.processing_ms        for t in turns_with_ta) / len(turns_with_ta)

                # Linha 1 -- qualidade semantica
                c1, c2, c3 = st.columns(3)
                c1.metric("🧲 Sim. Semantica", f"{avg_sim:.3f}",
                          help="Media de cosine similarity query↔response. Ideal > 0.60.")
                c2.metric("📖 Flesch Medio",   f"{avg_flesch:.1f}",
                          help="Legibilidade media das respostas (0-100, maior=mais facil).")
                c3.metric("🔵 Response Novelty", f"{avg_novelty:.3f}",
                          help="Conteudo genuinamente novo nas respostas (1 - lexical overlap).")

                # Linha 2 -- riqueza e comportamento
                c4, c5, c6 = st.columns(3)
                c4.metric("📊 TTR Medio",      f"{avg_ttr:.3f}",
                          help="Riqueza vocabular media das respostas (TTR).")
                c5.metric("🔁 Follow-ups",     followups,
                          help="Turnos onde o usuario reformulou a query anterior.")
                c6.metric("⏱️ Analytics/turno",  f"{avg_ms:.0f}ms",
                          help="Tempo medio de processamento do text_analytics.py.")

                # Distribuicao de tipos de query
                qtypes = {}
                for t in turns_with_ta:
                    qt = t.text_metrics.q_type or "desconhecido"
                    qtypes[qt] = qtypes.get(qt, 0) + 1
                if qtypes:
                    st.markdown("**Distribuicao de tipos de query:**")
                    total_qt = sum(qtypes.values())
                    for qt, count in sorted(qtypes.items(), key=lambda x: x[1], reverse=True):
                        st.progress(
                            min(int(count / total_qt * 100), 100),
                            text=f"{qt}: {count} ({count/total_qt*100:.0f}%)",
                        )

        # ── Sanitização ──────────────────────────────────────────────────────
        with st.expander("🧹 Sanitização de Output", expanded=False):
            # Resumo de todas as regras ativas no sanitizer.py
            rules = sanitizer_rules_summary()
            c1, c2, c3 = st.columns(3)
            c1.metric("🔍 PII Patterns",   len(rules["pii_patterns"]))
            c2.metric("📝 Palavras Fixas",  len(rules["word_substitutions"]))
            c3.metric("🔧 Regex Custom",    len(rules["custom_regex"]))
            st.caption(f"Total de regras ativas: {rules['total_rules']}")

            st.markdown("---")

            # Estatísticas de sanitização da sessão atual
            log = st.session_state.turn_log
            if not log:
                st.caption("Nenhum turno registrado ainda.")
            else:
                sanitized_turns = [t for t in log if t.was_sanitized]
                san_pct = len(sanitized_turns) / len(log) * 100

                col1, col2 = st.columns(2)
                col1.metric(
                    "🧹 Turnos Sanitizados",
                    f"{len(sanitized_turns)} ({san_pct:.1f}%)",
                    help="Turnos onde pelo menos uma substituição foi aplicada.",
                )

                # Agrega todas as regras disparadas na sessão
                all_rules: dict[str, int] = {}
                for t in sanitized_turns:
                    for rule, count in t.sanitized_counts.items():
                        all_rules[rule] = all_rules.get(rule, 0) + count

                col2.metric(
                    "🔢 Total Substituições",
                    sum(all_rules.values()),
                    help="Total de ocorrências substituídas em toda a sessão.",
                )

                # Breakdown por regra
                if all_rules:
                    st.markdown("**Regras mais acionadas:**")
                    for rule, count in sorted(
                        all_rules.items(), key=lambda x: x[1], reverse=True
                    )[:5]:   # top 5
                        st.progress(
                            min(count * 10, 100),
                            text=f"{rule}: {count}x",
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
            st.session_state.session_start        = datetime.now(timezone.utc)
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

        # Captura query anterior para follow-up detection no text_analytics
        prev_query = next(
            (m.content for m in reversed(st.session_state.chat_history[:-1])
             if hasattr(m, 'content') and isinstance(m, HumanMessage)),
            None,
        )
        # Posicao do turno na sessao (1-based)
        session_pos = len(st.session_state.turn_log) + 1

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
                    san_result    = None   # não sanitiza resposta de bloqueio
                else:
                    # ── Sanitização silenciosa ─────────────────────────────
                    # Aplicada APÓS o guardrail aprovar e ANTES de exibir.
                    # Substitui PII, palavras fixas e padrões regex no output.
                    # O usuário vê o texto limpo; o TurnRecord registra o log.
                    san_result    = sanitize(full_response)
                    full_response = san_result.sanitized_text
                    placeholder.markdown(full_response)

        # ── Alerta de truncamento (Grupo A) ──────────────────────────────────
        # is_response_truncated() checa se finish_reason indica resposta cortada
        # pelo limite de tokens (finish_reason = 'length' ou 'max_tokens').
        # Exibido como st.warning fora do history_container para não sujar o log
        # de chat — é uma notificação técnica, não parte da conversa.
        if is_response_truncated(stream_meta.finish_reason):
            st.warning(
                "✂️ **Resposta truncada** — o modelo atingiu o limite de tokens "
                f"(`finish_reason={stream_meta.finish_reason}`). "
                "A resposta pode estar incompleta. Considere reformular a pergunta "
                "em partes menores ou limpar o histórico para reduzir o contexto.",
                icon="⚠️",
            )


        # ── Text Analytics ───────────────────────────────────────────────────────────
        # Executado em background apos o stream -- zero impacto na UX.
        # analyze() eh fail-safe: retorna zeros se spaCy/embedder indisponiveis.
        # None eh passado quando o input foi bloqueado (sem resposta real).
        try:
            tx_metrics = text_analyze(
                query            = user_query,
                response         = full_response,
                prev_query       = prev_query,
                session_position = session_pos,
            )
        except Exception:
            tx_metrics = None   # fail-safe: nunca quebra o fluxo principal

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
            sanitize_result  = san_result,   # None quando bloqueado pelo guardrail
            text_metrics     = tx_metrics,   # TextMetrics do text_analytics.py
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
                # Colunas Grupo A — derivadas, sem mudar TurnRecord
                "Context Window %":  st.column_config.NumberColumn(format="%.1f",  width="small",
                                        help="% da janela de contexto do modelo consumida neste turno."),
                "Token Efficiency":  st.column_config.NumberColumn(format="%.3f",  width="small",
                                        help="Razão output÷input tokens. < 0.05 = contexto inflado."),
                "Resposta Cortada":  st.column_config.TextColumn(width="small",
                                        help="finish_reason=length indica resposta interrompida."),
                # Toxicidade contínua — score real mesmo quando safe=True
                "Input Toxicidade":  st.column_config.NumberColumn(format="%.3f", width="small",
                                        help="Score de toxicidade do input (OpenAI Moderation). Preenchido mesmo quando não bloqueado."),
                "Input Tox Categoria": st.column_config.TextColumn(width="small",
                                        help="Categoria com maior score de toxicidade no input."),
                "Output Toxicidade": st.column_config.NumberColumn(format="%.3f", width="small",
                                        help="Score de toxicidade do output (OpenAI Moderation)."),
                # Recusa via System Prompt
                "Recusa via SP":     st.column_config.TextColumn(width="small",
                                        help="Guardrails aprovaram mas o modelo recusou via safety prompt."),
                # Sanitização silenciosa
                "Sanitizado":        st.column_config.TextColumn(width="small",
                                        help="🧹 Sim = pelo menos uma substituição foi aplicada no output."),
                "Regras Sanitização": st.column_config.TextColumn(width="large",
                                        help="Lista de regras aplicadas: PII:CPF, WORD:foo, REGEX:bar."),
                # Text Analytics -- colunas selecionadas
                "Q Palavras":          st.column_config.NumberColumn(width="small", help="Total de palavras na query."),
                "R Palavras":          st.column_config.NumberColumn(width="small", help="Total de palavras na resposta."),
                "R Paragrafos":        st.column_config.NumberColumn(width="small", help="Paragrafos na resposta."),
                "Q Flesch":            st.column_config.NumberColumn(format="%.1f", width="small", help="Legibilidade da query (0-100, maior=mais facil)."),
                "R Flesch":            st.column_config.NumberColumn(format="%.1f", width="small", help="Legibilidade da resposta."),
                "Flesch Gap":          st.column_config.NumberColumn(format="%.1f", width="small", help="Diferenca de legibilidade resposta-query."),
                "R Gunning Fog":       st.column_config.NumberColumn(format="%.1f", width="small", help="Anos de escolaridade para entender a resposta."),
                "R TTR":               st.column_config.NumberColumn(format="%.3f", width="small", help="Type-Token Ratio: riqueza vocabular da resposta."),
                "R MATTR":             st.column_config.NumberColumn(format="%.3f", width="small", help="Moving Average TTR: TTR normalizado por janela de 50 tokens."),
                "R Hedge Ratio":       st.column_config.NumberColumn(format="%.3f", width="small", help="Proporcao de marcadores de incerteza na resposta."),
                "Q Sentimento":        st.column_config.NumberColumn(format="%.3f", width="small", help="Polaridade da query: -1 (neg) a +1 (pos)."),
                "R Sentimento":        st.column_config.NumberColumn(format="%.3f", width="small", help="Polaridade da resposta: -1 (neg) a +1 (pos)."),
                "R Formalidade":       st.column_config.NumberColumn(format="%.3f", width="small", help="Score de formalidade da resposta: 0 (informal) a 1 (formal)."),
                "Q Tipo":              st.column_config.TextColumn(width="small", help="Tipo de query: factual, procedimental, comparativa, etc."),
                "Q Especificidade":    st.column_config.NumberColumn(format="%.3f", width="small", help="Score de especificidade da query: 0 (vaga) a 1 (especifica)."),
                "Sem Similaridade":    st.column_config.NumberColumn(format="%.3f", width="small", help="Cosine similarity semantica entre query e resposta."),
                "Lexical Overlap":     st.column_config.NumberColumn(format="%.3f", width="small", help="Proporcao de palavras da query presentes na resposta."),
                "Response Novelty":    st.column_config.NumberColumn(format="%.3f", width="small", help="Conteudo genuinamente novo na resposta (1 - overlap)."),
                "Effort Ratio":        st.column_config.NumberColumn(format="%.2f",  width="small", help="Palavras resposta / palavras query."),
                "Info Gain":           st.column_config.NumberColumn(format="%.3f", width="small", help="Ganho de informacao: entropia resposta - entropia query."),
                "Follow-up":           st.column_config.TextColumn(width="small", help="Query similar a anterior (reformulacao detectada)."),
                "NER Total":           st.column_config.NumberColumn(width="small", help="Total de entidades nomeadas detectadas na resposta."),
                "Analytics (ms)":      st.column_config.NumberColumn(format="%.1f",  width="small", help="Tempo de processamento do text_analytics.py."),
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

        # ── Glossário de métricas ─────────────────────────────────────────────
        with st.expander("📖 Glossário de Métricas", expanded=False):

            st.markdown("### 🔢 Métricas de Tokens e Custo")
            st.markdown("""
| Métrica | Descrição | Valores de referência |
|---|---|---|
| **Tokens Usuário** | Estimativa de tokens do texto digitado pelo usuário (chars ÷ 4) | Típico: 5–50 |
| **Tokens Input LLM** | Total de tokens enviados ao modelo: system prompt + histórico + query atual. Representa o tamanho real da janela de contexto consumida e o que é cobrado como input. | Cresce a cada turno |
| **Tokens Output LLM** | Tokens gerados pelo modelo na resposta. Cobrados separadamente — custo maior por token que o input. | Típico: 50–500 |
| **Tokens Raciocínio** | Tokens internos de chain-of-thought usados por modelos com raciocínio estendido (ex: o1). Não aparecem na resposta. | 0 na maioria dos modelos |
| **Custo Turno (USD)** | Custo estimado deste turno: `(input_tokens/1M × $0.15) + (output_tokens/1M × $0.60)`. Baseado na tabela GPT-4o-mini. | USD 0.000001–0.001 |
| **Context Window %** | Percentual da janela de contexto do modelo consumida no último turno. Acima de 80% o modelo começa a "esquecer" mensagens antigas. | Alerta: > 80% |
| **Token Efficiency** | Razão `output_tokens ÷ input_tokens`. Valores muito baixos (< 0.05) indicam que o contexto acumulado domina o custo sem gerar output proporcional. | Ideal: 0.10–0.50 |
""")

            st.markdown("### ⏱️ Métricas de Performance")
            st.markdown("""
| Métrica | Descrição | Valores de referência |
|---|---|---|
| **Latência (s)** | Tempo total do início do stream até o último token recebido. | Bom: < 3s. Alerta: > 8s |
| **Tokens/s** | Velocidade de geração: `output_tokens ÷ latência`. Varia conforme carga do servidor. | Típico GPT-4o-mini: 40–80 t/s |
| **Finish Reason** | Motivo pelo qual o modelo parou de gerar. `stop` = resposta completa. `length` = cortada pelo limite de tokens. `content_filter` = bloqueada pelo provedor. | Ideal: stop |
| **Resposta Cortada** | Indica `finish_reason = length` ou `max_tokens`. A resposta foi interrompida antes de terminar. Reformule a pergunta em partes menores ou limpe o histórico. | ✅ Ok / ✂️ Sim |
| **Analytics (ms)** | Tempo de processamento do módulo `text_analytics.py` para calcular as métricas textuais do turno. | Típico: 50–500ms |
""")

            st.markdown("### 🛡️ Métricas de Guardrails e Segurança")
            st.markdown("""
| Métrica | Descrição | Valores de referência |
|---|---|---|
| **Input Bloqueado** | O input do usuário foi bloqueado por alguma camada de guardrail antes de chegar ao modelo. | ✅ Ok / 🚫 Sim |
| **Input GR Layer** | Camada que tomou a decisão de bloqueio: `better-profanity` (léxico), `openai-moderation`, `llamaguard`, ou `none` (passou em todas). | — |
| **Input GR Categoria** | Categoria de risco detectada: `profanity`, `hate`, `violence`, `self-harm`, `sexual`, `S1`–`S13` (LlamaGuard), etc. | safe = sem detecção |
| **Input GR Score** | Score de confiança da camada que bloqueou (0.0–1.0). Score 1.0 = certeza absoluta (léxico). Scores menores indicam decisão probabilística. | 1.0 = léxico. 0.40–1.0 = ML |
| **Input Toxicidade** | Score contínuo da OpenAI Moderation API para a categoria mais tóxica do input. Preenchido **mesmo quando não bloqueado** — permite rastrear mensagens na zona cinza. | 0.0 = seguro. > 0.40 = zona cinza. > 0.70 = alto risco |
| **Input Tox Categoria** | Categoria com maior score de toxicidade no input (ex: `violence`, `hate`, `harassment`). | — |
| **Output Toxicidade** | Score de toxicidade da resposta gerada pelo modelo, checado antes de exibir ao usuário. | Ideal: < 0.10 |
| **Recusa via SP** | Os guardrails aprovaram o input, mas o modelo recusou semanticamente por conta das regras do Safety Prompt (RULE 1–7). Detectado por heurística léxica na resposta. | ✅ Ok / 🛡️ Sim |
""")

            st.markdown("### 🧹 Métricas de Sanitização")
            st.markdown("""
| Métrica | Descrição |
|---|---|
| **Sanitizado** | Pelo menos uma substituição foi aplicada no output antes de exibir ao usuário. O texto exibido pode diferir do texto bruto da LLM. |
| **Regras Sanitização** | Lista de regras aplicadas no turno, ex: `PII:CPF, PII:EMAIL`. Prefixo `PII:` = dado pessoal. `WORD:` = palavra fixa. `REGEX:` = padrão customizado. |
""")

            st.markdown("### 📝 Métricas Textuais — Superfície")
            st.markdown("""
| Métrica | Descrição | Valores de referência |
|---|---|---|
| **Q Palavras** | Total de palavras na mensagem do usuário (query). | Típico: 5–50 |
| **R Palavras** | Total de palavras na resposta do modelo. | Típico: 50–400 |
| **R Parágrafos** | Número de parágrafos na resposta (separados por linha em branco). Respostas bem estruturadas tendem a ter 2–5 parágrafos. | Ideal: 2–6 |
| **Q Flesch** | Índice de legibilidade Flesch da query (0–100). Quanto maior, mais fácil de ler. Queries complexas (técnicas, longas) têm score mais baixo. | 0–30 difícil. 60–100 fácil |
| **R Flesch** | Índice de legibilidade Flesch da resposta. Permite avaliar se o modelo está respondendo em nível de complexidade adequado para o usuário. | Ideal para suporte: > 50 |
| **Flesch Gap** | Diferença `R Flesch − Q Flesch`. Positivo = modelo respondeu mais simplesmente que a pergunta (bom para suporte). Negativo = resposta mais complexa que a pergunta (pode confundir). | Ideal: próximo de 0 ou levemente positivo |
| **R Gunning Fog** | Anos de escolaridade estimados para compreender a resposta. Baseado em percentual de palavras complexas (3+ sílabas). | Ideal: 8–12 anos |
""")

            st.markdown("### 📊 Métricas Textuais — Riqueza e Qualidade")
            st.markdown("""
| Métrica | Descrição | Valores de referência |
|---|---|---|
| **R TTR** | Type-Token Ratio: palavras únicas ÷ total de palavras. Mede diversidade vocabular da resposta. Decresce naturalmente em textos longos. | > 0.70 rico. < 0.40 repetitivo |
| **R MATTR** | Moving Average TTR (janela de 50 tokens). Versão normalizada do TTR que não penaliza textos longos. Métrica mais confiável que o TTR simples para comparações. | > 0.70 rico |
| **R Hedge Ratio** | Proporção de marcadores de incerteza ("talvez", "possivelmente") sobre total de marcadores epistêmicos. Equilíbrio adequado indica calibração: o modelo não afirma com certeza o que é incerto. | Ideal: 0.30–0.70 |
| **Q Sentimento** | Polaridade emocional da query do usuário: −1.0 (muito negativo) a +1.0 (muito positivo). Permite identificar usuários frustrados ou insatisfeitos. | Neutro: −0.1 a +0.1 |
| **R Sentimento** | Polaridade emocional da resposta do modelo. Respostas de assistentes devem ser levemente positivas ou neutras. Valores muito negativos são problemáticos. | Ideal: 0.0 a +0.3 |
| **R Formalidade** | Score de formalidade da resposta (0.0 = informal a 1.0 = formal). Cruzar com a personalidade ativa: "Técnico 💻" deve ter score alto; "Descontraído ☕" pode ter score menor. | Depende da personalidade |
""")

            st.markdown("### 🔗 Métricas de Alinhamento Query/Response")
            st.markdown("""
| Métrica | Descrição | Valores de referência |
|---|---|---|
| **Q Tipo** | Classificação automática do tipo de query: `factual` (o que é X?), `procedimental` (como fazer X?), `comparativa` (X vs Y?), `opinativa` (o que acha de X?), `criativa` (escreva X), `ambigua` (difícil classificar). | — |
| **Q Especificidade** | Score de especificidade da query (0.0–1.0). Queries vagas (pronomes sem referente, sem contexto) tendem a gerar respostas genéricas. | > 0.50 específica |
| **Sem Similaridade** | Cosine similarity entre os embeddings semânticos da query e da resposta. Mede se a resposta está semanticamente alinhada com a pergunta. Valores baixos indicam resposta fora do tema. | Alerta: < 0.50 |
| **Lexical Overlap** | Proporção de palavras da query que aparecem na resposta. Alto overlap pode indicar que o modelo está apenas parafraseando a pergunta sem adicionar valor. | Típico: 0.20–0.60 |
| **Response Novelty** | `1 − Lexical Overlap`. Proporção de conteúdo genuinamente novo na resposta. Complementar ao lexical overlap. | Ideal: > 0.50 |
| **Effort Ratio** | `palavras_resposta ÷ palavras_query`. Mede a proporção de esforço do modelo em relação ao esforço do usuário. Valores muito altos = verbose. Muito baixos = resposta rasa. | Ideal: 3–15x |
| **Info Gain** | Ganho de informação estimado: `entropia_resposta − entropia_query`. Positivo = resposta trouxe mais informação que a pergunta continha. Negativo = resposta circular ou repetitiva. | Ideal: positivo |
| **Follow-up** | A query atual é semanticamente similar à query do turno anterior (similaridade > 0.85). Indica que o usuário reformulou a pergunta — possível sinal de insatisfação com a resposta anterior. | ✅ Ok / 🔁 Sim |
| **NER Total** | Total de entidades nomeadas detectadas na resposta (pessoas, organizações, datas, valores monetários, locais). Respostas com muitas entidades tendem a ser mais informativas e verificáveis. | Depende do contexto |
""")
