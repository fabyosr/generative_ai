"""
=============================================================================
ui/observability.py — Painel de Observabilidade e Métricas
=============================================================================
Responsabilidade:
    Renderizar o painel de observabilidade usando st.expander + st.tabs.

    Organização em abas:
        🧠 Contexto   — tokens acumulados, distribuição Human/AI, barra de uso
        📡 RAG         — chunks recuperados, fontes, tokens de contexto, detalhes
        🤖 Modelo      — provedor, modelo, temperatura, janela de contexto
        ⏱ Desempenho  — latência da última resposta e histórico de latências

    Recebe dicionários de métricas já calculados por core/metrics.py,
    sem nenhuma lógica de cálculo própria.
=============================================================================
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Constante: limite de tokens exibido na barra de contexto
# ---------------------------------------------------------------------------

# Estimativa conservadora usada como referência visual (barra de progresso).
# Não representa o limite real do modelo, apenas serve como âncora visual.
CONTEXT_REFERENCE_TOKENS = 4096


def render_observability_panel(
    history_metrics: dict,
    rag_metrics: dict,
    llm_metadata: dict,
    latency: float,
) -> None:
    """
    Renderiza o painel completo de observabilidade em um st.expander.

    Args:
        history_metrics: Saída de core.metrics.compute_history_metrics().
        rag_metrics:     Saída de core.metrics.compute_rag_metrics().
        llm_metadata:    Saída de core.metrics.extract_llm_metadata().
        latency:         Tempo da última resposta em segundos (float).
    """
    with st.expander("🔍 Observabilidade & Métricas", expanded=False):

        tab_context, tab_rag, tab_model, tab_perf = st.tabs([
            "🧠 Contexto",
            "📡 RAG",
            "🤖 Modelo",
            "⏱ Desempenho",
        ])

        # ---------------------------------------------------------------
        # Aba 1 — Contexto: janela de tokens do histórico
        # ---------------------------------------------------------------
        with tab_context:
            _render_context_tab(history_metrics)

        # ---------------------------------------------------------------
        # Aba 2 — RAG: métricas de recuperação
        # ---------------------------------------------------------------
        with tab_rag:
            _render_rag_tab(rag_metrics)

        # ---------------------------------------------------------------
        # Aba 3 — Modelo: metadados do LLM
        # ---------------------------------------------------------------
        with tab_model:
            _render_model_tab(llm_metadata)

        # ---------------------------------------------------------------
        # Aba 4 — Desempenho: latência
        # ---------------------------------------------------------------
        with tab_perf:
            _render_performance_tab(latency)


# ---------------------------------------------------------------------------
# Renderizadores internos por aba
# ---------------------------------------------------------------------------

def _render_context_tab(m: dict) -> None:
    """
    Aba 🧠 Contexto — exibe métricas da janela de contexto do histórico.

    Mostra:
        - Total de tokens acumulados (com barra de progresso relativa)
        - Distribuição Human vs AI em colunas
        - Contagem de mensagens por papel
    """
    st.markdown("##### Janela de Contexto Atual")

    total   = m["total_tokens"]
    pct     = min(total / CONTEXT_REFERENCE_TOKENS, 1.0)
    pct_str = f"{pct * 100:.1f}%"

    # Barra de uso do contexto
    st.markdown(
        f"**Tokens acumulados:** `{total:,}` "
        f"<span style='font-size:0.78rem; color:var(--text-secondary);'>"
        f"(~{pct_str} de {CONTEXT_REFERENCE_TOKENS:,} ref.)</span>",
        unsafe_allow_html=True,
    )
    st.progress(pct)

    st.markdown("")

    # Distribuição Human / AI
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💬 Total de Mensagens",
            value=m["total_messages"],
        )
    with col2:
        st.metric(
            label="🧑 Mensagens Humanas",
            value=m["human_messages"],
            help=f"~{m['human_tokens']:,} tokens",
        )
    with col3:
        st.metric(
            label="🤖 Respostas do AI",
            value=m["ai_messages"],
            help=f"~{m['ai_tokens']:,} tokens",
        )

    st.markdown("")

    # Detalhamento de tokens por papel
    if m["total_tokens"] > 0:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"🧑 **Tokens humanos:** `{m['human_tokens']:,}`  \n"
                f"🤖 **Tokens do AI:** `{m['ai_tokens']:,}`"
            )
        with col_b:
            # Mini gráfico de pizza via proporção textual
            human_pct = (m["human_tokens"] / total * 100) if total > 0 else 0
            ai_pct    = 100 - human_pct
            st.markdown(
                f"📊 Human: **{human_pct:.1f}%**  \n"
                f"📊 AI: **{ai_pct:.1f}%**"
            )


def _render_rag_tab(m: dict) -> None:
    """
    Aba 📡 RAG — exibe métricas de recuperação de documentos.

    Mostra:
        - Chunks recuperados, fontes únicas, tokens de contexto
        - Tamanho médio dos chunks
        - Tabela detalhada com metadados de cada chunk
    """
    st.markdown("##### Recuperação RAG — Última Query")

    if m["chunks_retrieved"] == 0:
        st.info("Nenhum chunk recuperado ainda. Envie uma mensagem para ver as métricas.", icon="ℹ️")
        return

    # Métricas principais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Chunks Recuperados", m["chunks_retrieved"])
    with col2:
        st.metric("📁 Fontes Únicas", len(m["unique_sources"]))
    with col3:
        st.metric("🔤 Tokens de Contexto", f"{m['total_context_tokens']:,}")

    col4, col5 = st.columns(2)
    with col4:
        st.metric("📏 Média por Chunk", f"{m['avg_chunk_tokens']} tok")
    with col5:
        st.metric("❓ Tokens da Query", f"{m['query_tokens']:,}")

    # Fontes únicas
    if m["unique_sources"]:
        st.markdown("**📂 Arquivos referenciados:**")
        for src in m["unique_sources"]:
            st.markdown(f"- `{src}`")

    # Tabela detalhada dos chunks
    st.markdown("**🔎 Detalhes dos chunks recuperados:**")
    for chunk in m["chunks_detail"]:
        with st.expander(
            f"Chunk {chunk['index'] + 1} — `{chunk['source']}` · p.{chunk['page']} · {chunk['tokens']} tokens",
            expanded=False,
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Arquivo:** `{chunk['source']}`")
                st.markdown(f"**Página:** {chunk['page']}")
            with col_b:
                st.markdown(f"**Tokens:** `{chunk['tokens']}`")
                st.markdown(f"**Caracteres:** `{chunk['char_count']}`")
            st.caption(f"**Prévia:** {chunk['preview']}")


def _render_model_tab(m: dict) -> None:
    """
    Aba 🤖 Modelo — exibe metadados e configuração do LLM ativo.

    Mostra:
        - Provedor e modelo selecionado
        - Temperatura configurada
        - Janela de contexto estimada do provedor
    """
    st.markdown("##### Configuração do Modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Provedor:** {m['provider_icon']} {m['provider_label']}")
        st.markdown(f"**Modelo:** `{m['model']}`")

    with col2:
        st.markdown(f"**Temperatura:** `{m['temperature']}`")
        st.markdown(f"**Janela de contexto:** {m['context_window']}")

    st.divider()

    # Interpretação visual da temperatura
    temp = m["temperature"]
    if temp <= 0.2:
        temp_desc = "🧊 Muito determinístico — respostas consistentes e previsíveis."
    elif temp <= 0.5:
        temp_desc = "⚖️ Balanceado — boa combinação de precisão e naturalidade."
    elif temp <= 0.8:
        temp_desc = "🌤 Criativo — respostas variadas com alguma imprevisibilidade."
    else:
        temp_desc = "🔥 Alta criatividade — respostas diversas, pode afetar precisão."

    st.info(temp_desc, icon="🌡️")


def _render_performance_tab(latency: float) -> None:
    """
    Aba ⏱ Desempenho — exibe latência da última resposta.

    Mostra:
        - Tempo de resposta da última query
        - Classificação qualitativa da latência
        - Histórico das últimas latências (session_state)
    """
    st.markdown("##### Desempenho da Última Resposta")

    if latency <= 0:
        st.info("Aguardando a primeira resposta para medir latência.", icon="⏳")
        return

    # Classificação qualitativa
    if latency < 2:
        quality = ("🟢 Excelente", "Resposta ultra-rápida.")
    elif latency < 5:
        quality = ("🟡 Boa", "Latência dentro do esperado.")
    elif latency < 15:
        quality = ("🟠 Moderada", "Pode haver sobrecarga no servidor.")
    else:
        quality = ("🔴 Lenta", "Verifique sua conexão ou troque de provedor.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("⏱ Tempo de resposta", f"{latency}s")
    with col2:
        st.markdown(f"**Avaliação:** {quality[0]}  \n{quality[1]}")

    # Histórico de latências da sessão
    if "latency_history" in st.session_state and st.session_state.latency_history:
        st.markdown("**📈 Histórico de latências (sessão atual):**")
        history = st.session_state.latency_history

        # Formata como mini tabela
        cols = st.columns(min(len(history), 6))
        for i, lat in enumerate(history[-6:]):  # últimas 6
            with cols[i]:
                st.metric(f"Q{len(history) - len(history[-6:]) + i + 1}", f"{lat}s")

        avg = sum(history) / len(history)
        st.markdown(
            f"Média da sessão: **`{avg:.2f}s`** · "
            f"Mín: **`{min(history)}s`** · "
            f"Máx: **`{max(history)}s`**"
        )
