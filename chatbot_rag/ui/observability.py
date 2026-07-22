"""
=============================================================================
ui/observability.py — Painel de Observabilidade e Métricas
=============================================================================
Responsabilidade:
    Renderizar o painel completo de observabilidade com st.tabs.

    Abas:
        🧠 Contexto    — tokens acumulados, distribuição Human/AI
        📡 RAG         — chunks recuperados, fontes, tokens de contexto
        🎯 Intenção    — tipo classificado, método, confiança, latência
        🗃 Cache       — hit/miss, similaridade, tamanho, latência
        🔀 Reranker    — scores do cross-encoder, docs antes/depois
        🤖 Modelo      — provedor, modelo, temperatura
        ⏱ Desempenho  — latência total e histórico da sessão

    Recebe dicionários/objetos de métricas já calculados — zero lógica
    de negócio aqui.
=============================================================================
"""

import streamlit as st

# CONTEXT_REFERENCE_TOKENS foi removido — a janela de contexto agora
# vem dinamicamente de llm_metadata["context_window"] (int),
# populado por extract_llm_metadata() via MODEL_CATALOG em core/models.py.


# ---------------------------------------------------------------------------
# Helper: descrição + glossário por aba
# ---------------------------------------------------------------------------

def _tab_header(description: str, glossary: dict[str, str]) -> None:
    """
    Renderiza um cabeçalho informativo e glossário colapsável para cada aba.

    Args:
        description: Texto explicando o que a aba entrega.
        glossary:    Dict {termo: definição} exibido no expander.
    """
    st.info(description, icon="ℹ️")
    with st.expander("📖 Glossário desta aba", expanded=False):
        for term, definition in glossary.items():
            st.markdown(f"**`{term}`** — {definition}")


def render_observability_panel(
    history_metrics: dict,
    rag_metrics:     dict,
    llm_metadata:    dict,
    latency:         float,
    intent_result=None,
    cache_result=None,
    rerank_result=None,
) -> None:
    """
    Renderiza todas as abas de observabilidade.

    Args:
        history_metrics: core.metrics.compute_history_metrics()
        rag_metrics:     core.metrics.compute_rag_metrics()
        llm_metadata:    core.metrics.extract_llm_metadata()
        latency:         Latência total da última resposta (s).
        intent_result:   core.intent.IntentResult | None
        cache_result:    core.cache.CacheResult | None
        rerank_result:   core.reranker.RerankResult | None
    """
    (tab_ctx, tab_rag, tab_intent,
     tab_cache, tab_rerank, tab_model, tab_perf) = st.tabs([
        "🧠 Contexto",
        "📡 RAG",
        "🎯 Intenção",
        "🗃 Cache",
        "🔀 Reranker",
        "🤖 Modelo",
        "⏱ Desempenho",
    ])

    with tab_ctx:    _render_context_tab(history_metrics, llm_metadata)
    with tab_rag:    _render_rag_tab(rag_metrics)
    with tab_intent: _render_intent_tab(intent_result, doc_knowledge)
    with tab_cache:  _render_cache_tab(cache_result)
    with tab_rerank: _render_rerank_tab(rerank_result)
    with tab_model:  _render_model_tab(llm_metadata)
    with tab_perf:   _render_performance_tab(latency)


# ---------------------------------------------------------------------------
# Aba 1 — Contexto
# ---------------------------------------------------------------------------

def _render_context_tab(m: dict, llm_metadata: dict | None = None) -> None:
    """
    Tokens acumulados no histórico — barra de uso relativa à janela real do modelo.

    A janela de contexto exibida na barra de progresso é obtida
    dinamicamente de llm_metadata["context_window"], que vem do
    MODEL_CATALOG em core/models.py. Cada modelo tem seu valor exato.
    """
    # Janela de contexto real do modelo selecionado (int)
    ctx_window = (llm_metadata or {}).get("context_window", 4_096)
    ctx_label  = (llm_metadata or {}).get("context_window_label", f"{ctx_window:,} tokens")
    tier       = (llm_metadata or {}).get("tier", "")

    _tab_header(
        "Mostra o volume de tokens acumulados no histórico da sessão, "
        "distribuídos entre mensagens humanas e respostas do AI. "
        "A barra de progresso reflete o limite real do modelo selecionado.",
        {
            "Tokens acumulados":    "Total de tokens em todo o histórico de mensagens da sessão atual.",
            "Janela de contexto":   "Limite máximo de tokens do modelo selecionado (vem do MODEL_CATALOG).",
            "Barra de uso":         "Proporção dos tokens acumulados em relação à janela real do modelo.",
            "Human tokens":         "Tokens gerados pelas mensagens do usuário.",
            "AI tokens":            "Tokens gerados pelas respostas do assistente.",
            "free+paid":            "Modelo com tier gratuito (rate limited) e pago (por token).",
        }
    )
    st.markdown("##### Janela de Contexto Atual")

    total = m.get("total_tokens", 0)
    pct   = min(total / ctx_window, 1.0) if ctx_window > 0 else 0.0

    # Badge de tier ao lado da janela
    tier_badge = ""
    if tier == "free+paid":
        tier_badge = " · <span style='background:#d1fae5;color:#065f46;padding:1px 8px;border-radius:999px;font-size:0.72rem;font-weight:700;'>FREE tier disponível</span>"
    elif tier == "paid":
        tier_badge = " · <span style='background:#e0e7ff;color:#3730a3;padding:1px 8px;border-radius:999px;font-size:0.72rem;font-weight:700;'>PAID</span>"

    st.markdown(
        f"**Tokens acumulados:** `{total:,}` "
        f"<span style='font-size:0.78rem;color:var(--text-secondary);'>"
        f"({pct*100:.1f}% de {ctx_label})</span>"
        f"{tier_badge}",
        unsafe_allow_html=True,
    )

    # Cor da barra muda conforme o uso
    if pct >= 0.90:
        st.error(f"⚠️ Contexto quase cheio! ({pct*100:.1f}% usado)", icon="🔴")
    elif pct >= 0.70:
        st.warning(f"Contexto em {pct*100:.1f}% — considere iniciar nova sessão.", icon="🟡")

    st.progress(pct)
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("💬 Total Mensagens",  m.get("total_messages", 0))
    with c2: st.metric("🧑 Humanas",          m.get("human_messages", 0),
                       help=f"~{m.get('human_tokens',0):,} tokens")
    with c3: st.metric("🤖 Respostas AI",     m.get("ai_messages", 0),
                       help=f"~{m.get('ai_tokens',0):,} tokens")

    if total > 0:
        st.markdown("")
        ca, cb = st.columns(2)
        human_pct = m.get("human_tokens", 0) / total * 100
        with ca:
            st.markdown(
                f"🧑 **Tokens humanos:** `{m.get('human_tokens',0):,}`  \n"
                f"🤖 **Tokens AI:** `{m.get('ai_tokens',0):,}`"
            )
        with cb:
            st.markdown(
                f"📊 Human: **{human_pct:.1f}%**  \n"
                f"📊 AI: **{100-human_pct:.1f}%**"
            )


# ---------------------------------------------------------------------------
# Aba 2 — RAG
# ---------------------------------------------------------------------------

def _render_rag_tab(m: dict) -> None:
    """Métricas de recuperação de documentos da última query RAG."""
    _tab_header(
        "Exibe as métricas de recuperação de documentos da última query processada "
        "pelo pipeline RAG. Permite avaliar a qualidade e relevância dos chunks "
        "recuperados do índice vetorial FAISS.",
        {
            "Chunks recuperados":   "Quantidade de trechos de texto retornados pelo retriever FAISS.",
            "Fontes únicas":        "Número de arquivos PDF distintos representados nos chunks recuperados.",
            "Tokens de contexto":   "Total de tokens dos chunks injetados no prompt do LLM.",
            "Média por chunk":      "Tamanho médio de cada chunk em tokens — reflete o chunk_size configurado.",
            "Tokens da query":      "Tokens da pergunta do usuário usada na busca vetorial.",
            "MMR":                  "Maximal Marginal Relevance — estratégia de busca que equilibra relevância e diversidade.",
        }
    )
    st.markdown("##### Recuperação RAG — Última Query")

    if not m or m.get("chunks_retrieved", 0) == 0:
        st.info("Nenhum chunk recuperado. Pode ser chitchat ou cache hit.", icon="ℹ️")
        return

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📦 Chunks Recuperados", m["chunks_retrieved"])
    with c2: st.metric("📁 Fontes Únicas",      len(m["unique_sources"]))
    with c3: st.metric("🔤 Tokens Contexto",    f"{m['total_context_tokens']:,}")

    c4, c5 = st.columns(2)
    with c4: st.metric("📏 Média por Chunk",    f"{m['avg_chunk_tokens']} tok")
    with c5: st.metric("❓ Tokens da Query",    f"{m['query_tokens']:,}")

    if m["unique_sources"]:
        st.markdown("**📂 Arquivos referenciados:**")
        for src in m["unique_sources"]:
            st.markdown(f"- `{src}`")

    st.markdown("**🔎 Detalhes dos chunks:**")
    for chunk in m.get("chunks_detail", []):
        with st.expander(
            f"Chunk {chunk['index']+1} — `{chunk['source']}` · "
            f"p.{chunk['page']} · {chunk['tokens']} tok",
            expanded=False,
        ):
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**Arquivo:** `{chunk['source']}`")
                st.markdown(f"**Página:** {chunk['page']}")
            with cb:
                st.markdown(f"**Tokens:** `{chunk['tokens']}`")
                st.markdown(f"**Chars:** `{chunk['char_count']}`")
            st.caption(f"**Prévia:** {chunk['preview']}")


# ---------------------------------------------------------------------------
# Aba 3 — Intenção
# ---------------------------------------------------------------------------

def _render_intent_tab(result, doc_knowledge: str = "") -> None:
    """Resultado do classificador de intenção (intent router)."""
    _tab_header(
        "Mostra como o sistema classificou a intenção da última mensagem "
        "antes de decidir qual pipeline executar. Evita acionar o RAG "
        "desnecessariamente para saudações e mensagens simples.",
        {
            "CHITCHAT":    "Saudação, agradecimento ou conversa genérica — respondido diretamente sem RAG.",
            "RAG_QUERY":   "Pergunta sobre o conteúdo dos documentos — aciona o pipeline RAG completo.",
            "FOLLOWUP":    "Pedido de esclarecimento da resposta anterior — reutiliza o contexto já recuperado.",
            "Método":      "'heuristic' = regex local (0ms, gratuito) · 'llm' = classificação via modelo.",
            "Confiança":   "100% para heurística (sempre certa em padrões conhecidos) · 90% para LLM.",
            "Latência":    "Tempo de classificação. Heurística ≈ 0ms · LLM varia conforme o provedor.",
        }
    )
    st.markdown("##### Classificação de Intenção")

    if result is None:
        st.info("Aguardando a primeira mensagem.", icon="🎯")
        return

    # Badge de intenção
    intent_styles = {
        "chitchat":  ("🗣 CHITCHAT",  "#d1fae5", "#065f46"),
        "rag_query": ("🔍 RAG QUERY", "#e0e7ff", "#3730a3"),
        "followup":  ("🔄 FOLLOWUP",  "#fef3c7", "#92400e"),
    }
    label, bg, fg = intent_styles.get(
        result.intent.value,
        (result.intent.value.upper(), "#f3f4f6", "#374151")
    )

    st.markdown(
        f"<div style='display:inline-block;padding:6px 16px;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:700;font-size:0.9rem;"
        f"margin-bottom:1rem;'>{label}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("⚙️ Método",      result.method)
    with c2:
        st.metric("🎯 Confiança",   f"{result.confidence*100:.0f}%")
    with c3:
        st.metric("⏱ Latência",    f"{result.latency_ms:.1f} ms")

    # Interpretação
    interpretations = {
        "chitchat":  "💬 Resposta direta via LLM — pipeline RAG **não executado**. Economia de tokens.",
        "rag_query": "🔍 Pipeline RAG completo executado (cache → retriever → reranker → LLM).",
        "followup":  "🔄 Reutilizou o contexto da query anterior — **sem nova busca vetorial**.",
    }
    interp = interpretations.get(result.intent.value, "")
    if interp:
        st.info(interp, icon="ℹ️")

    if doc_knowledge:
        with st.expander("📄 Contexto injetado no classificador"):
            st.markdown(result.system_prompt)

    # Debug: output bruto do LLM classificador
    if result.raw_llm_output:
        with st.expander("🔬 Output bruto do classificador LLM", expanded=False):
            st.code(result.raw_llm_output)


# ---------------------------------------------------------------------------
# Aba 4 — Cache
# ---------------------------------------------------------------------------

def _render_cache_tab(result) -> None:
    """Métricas do semantic cache — hit/miss, similaridade, tamanho."""
    _tab_header(
        "Exibe o resultado da consulta ao cache semântico. "
        "Quando ocorre um HIT, o LLM e o pipeline RAG não são invocados — "
        "economia direta de tokens e latência.",
        {
            "Cache HIT":      "Query semanticamente similar encontrada no cache — resposta retornada sem LLM.",
            "Cache MISS":     "Nenhuma query similar suficiente no cache — pipeline RAG executado normalmente.",
            "Similaridade":   "Score de cosseno entre a query atual e a mais próxima no cache (0–1).",
            "Threshold":      "Similaridade mínima para considerar HIT. Padrão: 0.92.",
            "Entradas":       "Número de respostas armazenadas no cache da sessão atual.",
            "Query casada":   "Texto da query original que gerou a resposta retornada pelo cache.",
        }
    )
    st.markdown("##### Semantic Cache")

    if result is None:
        st.info(
            "Cache não consultado nesta query "
            "(pode ser chitchat ou followup).",
            icon="🗃"
        )
        return

    # Badge hit/miss
    if result.hit:
        st.markdown(
            "<div style='display:inline-block;padding:6px 16px;border-radius:999px;"
            "background:#d1fae5;color:#065f46;font-weight:700;font-size:0.9rem;"
            "margin-bottom:1rem;'>✅ CACHE HIT</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='display:inline-block;padding:6px 16px;border-radius:999px;"
            "background:#fee2e2;color:#991b1b;font-weight:700;font-size:0.9rem;"
            "margin-bottom:1rem;'>❌ CACHE MISS</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📊 Similaridade",  f"{result.similarity:.4f}")
    with c2: st.metric("🗄 Entradas",      result.cache_size)
    with c3: st.metric("⏱ Latência",      f"{result.latency_ms:.1f} ms")

    if result.hit and result.matched_query:
        st.markdown(f"**Query casada:** _{result.matched_query}_")
        st.success(
            f"Resposta retornada do cache com similaridade **{result.similarity:.1%}** "
            f"— LLM e RAG não foram invocados.",
            icon="⚡"
        )

    if not result.hit:
        threshold = 0.92  # mesmo valor configurado no app.py
        st.markdown(
            f"Similaridade máxima encontrada: `{result.similarity:.4f}` "
            f"(threshold: `{threshold}`) — abaixo do mínimo para hit."
        )


# ---------------------------------------------------------------------------
# Aba 5 — Reranker
# ---------------------------------------------------------------------------

def _render_rerank_tab(result) -> None:
    """Scores do cross-encoder e comparação antes/depois do reranking."""
    _tab_header(
        "Exibe os scores do cross-encoder BGE-Reranker para cada chunk "
        "recuperado pelo FAISS. O reranker avalia a relevância real de "
        "cada chunk em relação à query, descartando os menos relevantes "
        "antes de enviar ao LLM — reduz tokens e melhora a qualidade.",
        {
            "Cross-encoder":    "Modelo que avalia (query, chunk) juntos — mais preciso que bi-encoder do FAISS.",
            "Top Score":        "Score do chunk mais relevante segundo o cross-encoder.",
            "Score delta":      "Diferença entre o melhor e pior score — alta dispersão indica boa separação.",
            "Docs entrada":     "Quantidade de chunks recebidos do FAISS antes do reranking.",
            "Docs saída":       "Quantidade de chunks enviados ao LLM após filtro do reranker.",
            "Top-K":            "Seleciona sempre os k melhores chunks independente do score.",
            "Threshold":        "Retorna todos os chunks com score acima do mínimo configurado.",
            "Adaptativo":       "Combina Top-K e Threshold — retorna até k chunks, desde que acima do mínimo.",
        }
    )
    st.markdown("##### Reranker Cross-Encoder")

    if result is None:
        st.info(
            "Reranker não executado nesta query "
            "(pode ser chitchat, followup ou cache hit).",
            icon="🔀"
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📥 Docs Entrada",  result.docs_before)
    with c2: st.metric("📤 Docs Saída",    result.docs_after)
    with c3: st.metric("🏆 Top Score",     f"{result.top_score:.4f}")
    with c4: st.metric("⏱ Latência",      f"{result.latency_ms:.1f} ms")

    st.markdown(f"**Modelo:** `{result.model_name}`")
    st.markdown(
        f"**Dispersão de scores** (top − worst): `{result.score_delta:.4f}` "
        f"— valores altos indicam boa separação de relevância."
    )

    if result.scores:
        st.markdown("**📊 Scores por chunk (ordem de relevância):**")
        for i, (doc, score) in enumerate(zip(result.docs, result.scores)):
            import os
            source = os.path.basename(doc.metadata.get("source", "?"))
            page   = doc.metadata.get("page", "?")

            # Barra de progresso normalizada pelo top score
            bar_val = score / result.top_score if result.top_score > 0 else 0

            col_rank, col_bar, col_meta = st.columns([0.5, 2, 2])
            with col_rank:
                st.markdown(f"**#{i+1}**")
            with col_bar:
                st.progress(max(0.0, min(bar_val, 1.0)))
                st.caption(f"score: `{score:.4f}`")
            with col_meta:
                st.caption(f"`{source}` · p.{page}")


# ---------------------------------------------------------------------------
# Aba 6 — Modelo
# ---------------------------------------------------------------------------

def _render_model_tab(m: dict) -> None:
    """Metadados e configuração do LLM ativo."""
    _tab_header(
        "Exibe os metadados do modelo de linguagem atualmente ativo "
        "e como a temperatura afeta o comportamento das respostas.",
        {
            "Provedor":          "Serviço de API que hospeda o modelo (OpenAI, Groq, HuggingFace Hub).",
            "Modelo":            "Versão específica do LLM selecionado na sidebar.",
            "Temperatura":       "Controla aleatoriedade: 0 = determinístico · 1 = máxima variação.",
            "Janela de contexto":"Limite máximo de tokens (input + output) que o modelo suporta.",
        }
    )
    st.markdown("##### Configuração do Modelo")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Provedor:** {m.get('provider_icon','')} {m.get('provider_label','')}")
        st.markdown(f"**Modelo:** `{m.get('model','')}`")
        st.markdown(f"**Tier:** `{m.get('tier','')}`")
    with c2:
        st.markdown(f"**Temperatura:** `{m.get('temperature','')}`")
        st.markdown(f"**Janela de contexto:** `{m.get('context_window_label', m.get('context_window',''))}`")
        p_in  = m.get('price_input',  0) * 1000   # /1K → /1M
        p_out = m.get('price_output', 0) * 1000
        st.markdown(f"**Preço:** `${p_in:.4f}` in · `${p_out:.4f}` out / 1M tokens")

    st.divider()

    temp = m.get("temperature", 0.1)
    if temp <= 0.2:   desc = "🧊 Muito determinístico — respostas consistentes e previsíveis."
    elif temp <= 0.5: desc = "⚖️ Balanceado — precisão e naturalidade."
    elif temp <= 0.8: desc = "🌤 Criativo — respostas variadas."
    else:             desc = "🔥 Alta criatividade — pode afetar precisão."
    st.info(desc, icon="🌡️")


# ---------------------------------------------------------------------------
# Aba 7 — Desempenho
# ---------------------------------------------------------------------------

def _render_performance_tab(latency: float) -> None:
    """Latência total da última resposta e histórico da sessão."""
    _tab_header(
        "Mede o tempo total de resposta do pipeline, do recebimento da "
        "query até a entrega da resposta final. Inclui histórico de "
        "latências da sessão para identificar degradação de performance.",
        {
            "Latência total":   "Tempo completo: intent + cache + RAG + reranker + LLM + analytics.",
            "Excelente":        "< 2s — típico de cache hit ou modelo Groq (inferência ultra-rápida).",
            "Boa":              "2–5s — pipeline RAG completo com modelo rápido.",
            "Moderada":         "5–15s — pipeline com modelo maior ou sobrecarga do servidor.",
            "Lenta":            "> 15s — verifique conexão ou troque para provedor mais rápido.",
        }
    )
    st.markdown("##### Desempenho da Última Resposta")

    if latency <= 0:
        st.info("Aguardando a primeira resposta.", icon="⏳")
        return

    if latency < 2:    quality = ("🟢 Excelente", "Resposta ultra-rápida.")
    elif latency < 5:  quality = ("🟡 Boa",       "Latência dentro do esperado.")
    elif latency < 15: quality = ("🟠 Moderada",  "Pode haver sobrecarga.")
    else:              quality = ("🔴 Lenta",     "Verifique conexão ou provedor.")

    c1, c2 = st.columns([1, 2])
    with c1: st.metric("⏱ Tempo total", f"{latency}s")
    with c2: st.markdown(f"**Avaliação:** {quality[0]}  \n{quality[1]}")

    history = st.session_state.get("latency_history", [])
    if history:
        st.markdown("**📈 Histórico de latências (sessão):**")
        last6 = history[-6:]
        cols  = st.columns(len(last6))
        for i, lat in enumerate(last6):
            with cols[i]:
                idx = len(history) - len(last6) + i + 1
                st.metric(f"Q{idx}", f"{lat}s")

        avg = sum(history) / len(history)
        st.markdown(
            f"Média: **`{avg:.2f}s`** · "
            f"Mín: **`{min(history)}s`** · "
            f"Máx: **`{max(history)}s`**"
        )


# ===========================================================================
# Atualização da assinatura pública — adiciona analytics e clustering
# ===========================================================================

_original_render = render_observability_panel

def render_observability_panel(
    history_metrics:  dict,
    rag_metrics:      dict,
    llm_metadata:     dict,
    latency:          float,
    intent_result=None,
    doc_knowledge=None,
    cache_result=None,
    rerank_result=None,
    analytics_result=None,
    clustering=None,
) -> None:
    """
    Versão estendida do painel — inclui abas Analytics e Clusters.

    Novas abas adicionadas:
        📊 Analytics  — grounding score, faithfulness, análise textual, custo
        🗺 Clusters   — clusterização de intenções, outliers, gaps no corpus
    """
    (tab_ctx, tab_rag, tab_intent, tab_cache,
     tab_rerank, tab_model, tab_perf,
     tab_analytics, tab_clusters) = st.tabs([
        "🧠 Contexto",
        "📡 RAG",
        "🎯 Intenção",
        "🗃 Cache",
        "🔀 Reranker",
        "🤖 Modelo",
        "⏱ Desempenho",
        "📊 Analytics",
        "🗺 Clusters",
    ])

    with tab_ctx:       _render_context_tab(history_metrics, llm_metadata)
    with tab_rag:       _render_rag_tab(rag_metrics)
    with tab_intent:    _render_intent_tab(intent_result, doc_knowledge)
    with tab_cache:     _render_cache_tab(cache_result)
    with tab_rerank:    _render_rerank_tab(rerank_result)
    with tab_model:     _render_model_tab(llm_metadata)
    with tab_perf:      _render_performance_tab(latency)
    with tab_analytics: _render_analytics_tab(analytics_result)
    with tab_clusters:  _render_clusters_tab(clustering)


# ---------------------------------------------------------------------------
# Aba 8 — Analytics
# ---------------------------------------------------------------------------

def _render_analytics_tab(a) -> None:
    """
    Aba 📊 Analytics — grounding, faithfulness, análise textual e custo.
    """
    st.markdown("##### Analytics de Qualidade")

    if a is None:
        st.info("Analytics disponíveis após a primeira query RAG.", icon="📊")
        return

    # --- Grounding ---
    st.markdown("**🎯 Grounding & Confiança**")

    grounding_color = {"Alta 🟢": "#d1fae5", "Moderada 🟡": "#fef3c7",
                       "Baixa 🔴": "#fee2e2", "Sem contexto RAG": "#f3f4f6"}
    bg = grounding_color.get(a.grounding_label, "#f3f4f6")
    st.markdown(
        f"<div style='background:{bg};padding:10px 16px;border-radius:8px;"
        f"margin-bottom:12px;font-weight:600;'>"
        f"Grounding: {a.grounding_label} &nbsp;|&nbsp; "
        f"Score: <code>{a.grounding_score:.4f}</code> &nbsp;|&nbsp; "
        f"Sobreposição semântica: <code>{a.semantic_overlap:.4f}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🎯 Grounding Score",      f"{a.grounding_score:.3f}")
    with c2: st.metric("🧬 Overlap Semântico",    f"{a.semantic_overlap:.3f}")
    with c3: st.metric("⚠️ Risco Alucinação",     f"{a.hallucination_risk*100:.1f}%")

    # Sentenças não ancoradas
    if a.ungrounded_sentences:
        with st.expander(
            f"⚠️ {len(a.ungrounded_sentences)} sentença(s) não ancorada(s) no contexto RAG",
            expanded=False,
        ):
            st.caption("Estas sentenças têm baixa similaridade com os chunks recuperados "
                       "e podem representar extrapolação do modelo:")
            for i, sent in enumerate(a.ungrounded_sentences):
                st.markdown(f"**{i+1}.** _{sent}_")

    st.divider()

    # --- Análise textual ---
    st.markdown("**📝 Análise da Resposta**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📏 Palavras",          a.response_words)
    with c2: st.metric("🔢 Sentenças",         a.response_sentences)
    with c3: st.metric("📚 Diversidade Léxica", f"{a.lexical_diversity:.3f}")
    with c4: st.metric("📐 Média Sent.",        f"{a.avg_sentence_length:.1f} pal.")

    c5, c6 = st.columns(2)
    with c5: st.metric("🤔 Hedging Rate",      f"{a.hedging_rate:.3f}")
    with c6: st.metric("🏷 Tipo Resposta",     a.response_type)

    if a.hedging_terms_found:
        st.markdown(
            "**Termos de incerteza encontrados:** " +
            " · ".join(f"`{t}`" for t in a.hedging_terms_found)
        )

    st.divider()

    # --- Análise da query ---
    st.markdown("**❓ Análise da Query**")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🏷 Tipo",         a.query_type)
    with c2: st.metric("📊 Complexidade", a.query_complexity)
    with c3: st.metric("📏 Palavras",     a.query_words)

    flags = []
    if a.has_negation:  flags.append("🚫 Contém negação")
    if a.has_ambiguity: flags.append("❓ Contém ambiguidade (pronomes)")
    if flags:
        st.info(" · ".join(flags))

    st.divider()

    # --- Custo ---
    st.markdown("**💰 Estimativa de Custo**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📥 Tokens Input",  f"{a.input_tokens:,}")
    with c2: st.metric("📤 Tokens Output", f"{a.output_tokens:,}")
    with c3: st.metric("💵 Custo Est.",    f"${a.estimated_cost_usd:.6f}")
    with c4: st.metric("⚡ Economia Cache", f"${a.cache_savings_usd:.6f}")

    # Acumulado da sessão
    if "last_analytics" in st.session_state:
        total_cost = sum(
            getattr(st.session_state, f"analytics_cost_{i}", 0.0)
            for i in range(100)
        )


# ---------------------------------------------------------------------------
# Aba 9 — Clusters
# ---------------------------------------------------------------------------

def _render_clusters_tab(c) -> None:
    """
    Aba 🗺 Clusters — visualização da clusterização automática de intenções.
    """
    st.markdown("##### Clusterização Automática de Intenções")

    if c is None:
        st.info(
            "O clustering ativa automaticamente após 5 queries RAG na sessão.",
            icon="🗺"
        )
        return

    if c.error:
        st.warning(c.error, icon="⚠️")
        return

    # Resumo geral
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🗂 Clusters",       c.n_clusters)
    with c2: st.metric("📊 Queries Total",  c.total_queries)
    with c3: st.metric("✅ Cobertura",      f"{c.coverage*100:.0f}%")
    with c4: st.metric("⏱ Latência",       f"{c.latency_ms:.0f}ms")

    st.markdown("")

    # Clusters descobertos
    if c.clusters:
        st.markdown("**📂 Clusters descobertos:**")
        for cluster in c.clusters:
            coherence_icon = "🟢" if cluster.coherence > 0.7 else "🟡" if cluster.coherence > 0.4 else "🔴"
            with st.expander(
                f"**{cluster.label}** · {cluster.size} queries · "
                f"Coesão: {coherence_icon} {cluster.coherence:.3f}",
                expanded=cluster.size >= 3,
            ):
                st.markdown(
                    "**Termos principais:** " +
                    " · ".join(f"`{t}`" for t in cluster.top_terms)
                )
                st.markdown("**Queries:**")
                for q in cluster.queries:
                    st.markdown(f"- _{q}_")

    # Outliers — gaps no corpus
    if c.outliers:
        st.divider()
        st.markdown(
            f"**🔍 {len(c.outliers)} query(ies) sem cluster "
            f"— possíveis gaps no corpus:**"
        )
        st.caption(
            "Queries que não pertencem a nenhum grupo podem indicar tópicos "
            "não cobertos pelos documentos indexados."
        )
        for q in c.outliers:
            st.markdown(
                f"<div style='background:#fef9c3;padding:6px 12px;"
                f"border-radius:6px;margin:4px 0;border-left:3px solid #eab308;"
                f"font-size:0.84rem;'>⚠️ {q}</div>",
                unsafe_allow_html=True,
            )
