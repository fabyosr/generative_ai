"""
=============================================================================
app.py — Ponto de Entrada e Orquestrador Principal
=============================================================================
Fluxo completo por mensagem:

    [query]
       │
       ▼
    PipelineTrace.start()          ← ui/pipeline_trace.py  (thinking box)
       │
       ▼
    classify_intent()              ← core/intent.py
       │
       ├── CHITCHAT ──────────────► _process_chitchat() → resposta direta
       ├── FOLLOWUP ──────────────► _process_followup() → reutiliza contexto
       └── RAG_QUERY
               │
               ▼
           cache.lookup()          ← core/cache.py
               │
               ├── HIT ──────────► resposta cacheada (0 LLM calls)
               └── MISS
                       │
                       ▼
                   rag_chain.invoke()     ← core/rag.py
                       │
                       ▼
                   reranker.rerank()      ← core/reranker.py
                       │
                       ▼
                   cache.store()
       │
       ▼
    compute_analytics()            ← core/analytics.py
       │
       ▼
    PipelineTrace.finish()
       │
       ▼
    render_ai_response()           ← ui/chat.py
=============================================================================
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

# --- Core ---
from core.secrets    import load_api_keys, get_available_providers
from core.models     import get_model
from core.rag        import build_retriever, build_rag_chain, EMBEDDING_MODEL
from core.intent     import classify_intent, IntentType
from core.cache      import SemanticCache
from core.reranker   import Reranker
from core.analytics  import compute_analytics
from core.clustering import cluster_queries
from core.metrics    import (
    LatencyTimer,
    compute_history_metrics,
    compute_rag_metrics,
    extract_llm_metadata,
)

# --- UI ---
from ui.sidebar        import render_sidebar
from ui.chat           import render_chat_history, render_user_message, render_ai_response
from ui.observability  import render_observability_panel
from ui.pipeline_trace import (
    PipelineTrace,
    intent_detail,
    cache_detail,
    rerank_detail,
    rag_retrieval_detail,
)


# =============================================================================
# Configuração da página
# =============================================================================

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --text-secondary:    #6b7280;
        --surface-secondary: #f3f4f6;
        --accent:            #6366f1;
        --accent-light:      #e0e7ff;
    }
    .block-container { padding-top: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-size: 0.84rem; }
    div[data-testid="stPopover"] button { font-size: 0.78rem; }
    div[data-testid="stMetric"] label  { font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def _init_session_state() -> None:
    """
    Inicializa todas as chaves do session_state.
    Idempotente — seguro chamar a cada ciclo de renderização.
    """
    defaults = {
        "chat_history": [
            AIMessage(content="Olá! Sou seu assistente de documentos 📚 "
                              "Envie um PDF na barra lateral e faça sua pergunta.")
        ],
        "indexed_files":         None,
        "retriever":             None,
        "embeddings":            None,
        "semantic_cache":        None,
        "reranker":              None,
        "last_context_docs":     [],
        # Métricas
        "last_rag_metrics":      {},
        "last_history_metrics":  {},
        "last_llm_metadata":     {},
        "last_analytics":        None,
        "last_latency":          0.0,
        "last_intent_result":    None,
        "last_cache_result":     None,
        "last_rerank_result":    None,
        "latency_history":       [],
        # Clustering: acumula apenas RAG queries (não chitchat)
        "rag_query_history":     [],
        "last_clustering":       None,
        # UI
        "show_pipeline_trace":   True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Inicialização de objetos pesados (uma vez por sessão)
# =============================================================================

def _init_heavy_objects() -> None:
    """
    Instancia modelos pesados uma única vez por sessão.

    O HuggingFaceEmbeddings (BGE-M3) é compartilhado entre:
    - FAISS retriever (build_retriever)
    - SemanticCache (lookup/store)
    - analytics.compute_analytics (grounding/faithfulness)
    - clustering.cluster_queries (embeddings das queries)
    """
    if st.session_state.embeddings is None:
        with st.spinner("⚙️ Carregando modelo de embeddings…"):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )

    if st.session_state.semantic_cache is None:
        st.session_state.semantic_cache = SemanticCache(
            embeddings_model=st.session_state.embeddings,
            threshold=0.92,
        )

    if st.session_state.reranker is None:
        st.session_state.reranker = Reranker(top_k=3)


# =============================================================================
# Indexação
# =============================================================================

def _maybe_reindex(uploads: list) -> None:
    """
    Reconstrói FAISS apenas quando os arquivos mudaram.
    Invalida cache semântico ao reindexar.
    """
    current_names = [f.name for f in uploads]
    if st.session_state.indexed_files != current_names:
        placeholder = st.empty()
        with st.spinner("⚙️ Indexando documentos…"):
            st.session_state.retriever     = build_retriever(
                uploads,
                embeddings=st.session_state.embeddings,
            )
            st.session_state.indexed_files = current_names
            st.session_state.semantic_cache.clear()
        st.toast(f"✅ {len(uploads)} documento(s) indexado(s)!", icon="📥")


# =============================================================================
# Handlers por tipo de intenção
# =============================================================================

def _process_chitchat(query: str, llm, trace: PipelineTrace) -> tuple[str, list]:
    """Resposta direta sem RAG — saudações e conversas genéricas."""
    def _run():
        messages = [
            SystemMessage(content="Você é um assistente simpático. "
                                  "Responda de forma breve e amigável em português."),
            *st.session_state.chat_history[-6:],
            HumanMessage(content=query),
        ]
        return llm.invoke(messages)

    response = trace.run_step(
        label     = "Gerando resposta direta (sem RAG)",
        icon      = "💬",
        fn        = _run,
        detail_fn = lambda r: f"{len(r.content.split())} palavras geradas",
    )
    return response.content, []


def _process_followup(query: str, llm, trace: PipelineTrace) -> tuple[str, list]:
    """Elabora resposta anterior reutilizando o último contexto RAG."""
    last_docs = st.session_state.last_context_docs
    context   = "\n\n".join(d.page_content for d in last_docs) if last_docs else ""

    trace.add_info(
        "♻️",
        "Reutilizando contexto anterior",
        f"{len(last_docs)} chunks da query anterior · sem nova busca vetorial",
    )

    def _run():
        messages = [
            SystemMessage(content=(
                "Você é um assistente prestativo. Elabore sua resposta anterior "
                "com base no contexto abaixo. Responda em português.\n\n"
                f"Contexto: {context}"
            )),
            *st.session_state.chat_history[-6:],
            HumanMessage(content=query),
        ]
        return llm.invoke(messages)

    response = trace.run_step(
        label     = "Elaborando com contexto anterior",
        icon      = "🔄",
        fn        = _run,
        detail_fn = lambda r: f"{len(r.content.split())} palavras geradas",
    )
    return response.content, last_docs


def _process_rag(
    query:  str,
    llm,
    config: dict,
    trace:  PipelineTrace,
) -> tuple[str, list]:
    """
    Pipeline RAG completo: cache → retrieval → reranker → LLM → store.
    Cada etapa é rastreada e exibida no PipelineTrace.
    """
    cache    = st.session_state.semantic_cache
    reranker = st.session_state.reranker

    # --- 1. Semantic Cache ---
    cache_result = trace.run_step(
        label     = "Verificando cache semântico",
        icon      = "🗃",
        fn        = lambda: cache.lookup(query),
        detail_fn = cache_detail,
    )
    st.session_state.last_cache_result = cache_result

    if cache_result.hit:
        trace.add_info("⚡", "Resposta recuperada do cache",
                       "LLM e RAG não foram invocados — economia máxima de tokens")
        return cache_result.answer, cache_result.sources or []

    # --- 2. RAG Chain ---
    rag_chain = build_rag_chain(llm, st.session_state.retriever)

    rag_result = trace.run_step(
        label     = "Recuperando documentos (FAISS MMR)",
        icon      = "🔍",
        fn        = lambda: rag_chain.invoke({
            "input":        query,
            "chat_history": st.session_state.chat_history,
        }),
        detail_fn = lambda r: rag_retrieval_detail(r.get("context", [])),
    )

    answer       = rag_result["answer"]
    context_docs = rag_result.get("context", [])

    # --- 3. Reranker ---
    rerank_result = trace.run_step(
        label     = "Rerankeando chunks (cross-encoder)",
        icon      = "🔀",
        fn        = lambda: reranker.rerank(query, context_docs),
        detail_fn = rerank_detail,
    )
    st.session_state.last_rerank_result = rerank_result
    final_docs = rerank_result.docs if rerank_result.docs else context_docs

    # --- 4. Store no cache ---
    trace.run_step(
        label     = "Armazenando no cache semântico",
        icon      = "💾",
        fn        = lambda: cache.store(query, answer, final_docs),
        detail_fn = lambda _: f"Cache: {cache.size} entradas · threshold: {cache.threshold}",
    )

    return answer, final_docs


# =============================================================================
# Processamento principal
# =============================================================================

def _process_query(query: str, config: dict) -> None:
    """
    Orquestra o pipeline completo para uma mensagem do usuário.

    Fluxo:
        1. Renderiza mensagem humana
        2. Instancia LLM
        3. Abre PipelineTrace (thinking box)
        4. Classifica intenção → despacha para handler correto
        5. Finaliza trace
        6. Calcula analytics e métricas
        7. Executa clustering (se ≥ 5 RAG queries)
        8. Renderiza resposta
    """
    render_user_message(query)

    llm = get_model(
        provider    = config["provider"],
        model       = config["model"],
        temperature = config["temperature"],
    )

    timer = LatencyTimer()
    timer.start()

    # -----------------------------------------------------------------------
    # Pipeline Trace (thinking box)
    # -----------------------------------------------------------------------
    with st.chat_message("assistant", avatar="⚙️"):
        trace = PipelineTrace()

        trace.add_info(
            "📨", "Query recebida",
            f'<code>"{query[:80]}{"…" if len(query) > 80 else ""}"</code> · '
            f"{len(query.split())} palavras · "
            f"Modelo: <code>{config['model']}</code>",
        )
        trace.add_divider()

        # --- Intenção ---
        intent_result = trace.run_step(
            label     = "Classificando intenção",
            icon      = "🎯",
            fn        = lambda: classify_intent(query, st.session_state.chat_history, llm),
            detail_fn = intent_detail,
        )
        st.session_state.last_intent_result  = intent_result
        st.session_state.last_cache_result   = None
        st.session_state.last_rerank_result  = None

        trace.add_divider()

        # --- Despacha por intenção ---
        if intent_result.intent == IntentType.CHITCHAT:
            answer, context_docs = _process_chitchat(query, llm, trace)

        elif intent_result.intent == IntentType.FOLLOWUP:
            answer, context_docs = _process_followup(query, llm, trace)

        else:  # RAG_QUERY
            answer, context_docs = _process_rag(query, llm, config, trace)
            # Acumula query RAG para clustering
            st.session_state.rag_query_history.append(query)

        # --- Analytics ---
        trace.add_divider()
        analytics = trace.run_step(
            label     = "Calculando analytics de qualidade",
            icon      = "📊",
            fn        = lambda: compute_analytics(
                query          = query,
                answer         = answer,
                context_docs   = context_docs,
                provider       = config["provider"],
                model          = config["model"],
                rerank_result  = st.session_state.last_rerank_result,
                cache_result   = st.session_state.last_cache_result,
                embeddings     = st.session_state.embeddings,
                history_tokens = st.session_state.last_history_metrics.get("total_tokens", 0),
            ),
            detail_fn = lambda a: (
                f"Grounding: <strong>{a.grounding_label}</strong> "
                f"({a.grounding_score:.3f}) · "
                f"Risco alucinação: {a.hallucination_risk*100:.0f}% · "
                f"Hedging: {a.hedging_rate:.2f} · "
                f"Custo est.: ${a.estimated_cost_usd:.6f}"
            ),
        )

        # --- Clustering (se ≥ 5 RAG queries) ---
        rag_queries = st.session_state.rag_query_history
        if len(rag_queries) >= 5:
            clustering = trace.run_step(
                label     = f"Clusterizando intenções ({len(rag_queries)} queries)",
                icon      = "🗺",
                fn        = lambda: cluster_queries(
                    queries    = rag_queries,
                    embeddings = st.session_state.embeddings,
                    llm        = llm,
                ),
                detail_fn = lambda c: (
                    f"{c.n_clusters} clusters · "
                    f"Cobertura: {c.coverage*100:.0f}% · "
                    f"{len(c.outliers)} outlier(s) · "
                    f"{c.latency_ms:.0f}ms"
                ) if not c.error else f"⚠️ {c.error}",
            )
            st.session_state.last_clustering = clustering
        else:
            remaining = 5 - len(rag_queries)
            trace.add_info(
                "🗺", "Clustering de intenções",
                f"Aguardando mais {remaining} query(ies) RAG para ativar",
            )

        # --- Finaliza trace ---
        cache_status = ""
        if st.session_state.last_cache_result:
            cache_status = "Cache HIT ⚡" if st.session_state.last_cache_result.hit else "Cache MISS"

        latency = timer.stop()
        trace.finish(
            summary=(
                f"{intent_result.intent.value.upper()} · "
                f"{cache_status or 'sem cache'} · "
                f"Grounding: {analytics.grounding_score:.2f} · "
                f"${analytics.estimated_cost_usd:.6f}"
            )
        )

    # -----------------------------------------------------------------------
    # Atualiza estado
    # -----------------------------------------------------------------------
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.last_context_docs = context_docs
    st.session_state.last_analytics    = analytics

    # Métricas
    history_metrics = compute_history_metrics(st.session_state.chat_history, config["provider"])
    rag_metrics     = compute_rag_metrics(context_docs, query, config["provider"])
    llm_metadata    = extract_llm_metadata(config["provider"], config["model"], config["temperature"])

    st.session_state.last_history_metrics = history_metrics
    st.session_state.last_rag_metrics     = rag_metrics
    st.session_state.last_llm_metadata    = llm_metadata
    st.session_state.last_latency         = latency
    st.session_state.latency_history.append(latency)

    # Resposta final
    render_ai_response(answer, latency, context_docs)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Ciclo principal de renderização do Streamlit."""
    _init_session_state()

    secrets_status      = load_api_keys()
    available_providers = get_available_providers(secrets_status)

    _init_heavy_objects()

    config = render_sidebar(available_providers)

    if not config["uploads"]:
        st.markdown("""
        <div style='display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:4rem 2rem;text-align:center;color:#6b7280;'>
            <span style='font-size:3.5rem;'>📂</span>
            <h3 style='margin:0.8rem 0 0.4rem 0;color:#374151;'>Nenhum documento carregado</h3>
            <p style='max-width:380px;line-height:1.6;'>
                Envie um ou mais arquivos <strong>PDF</strong> pela barra lateral.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    _maybe_reindex(config["uploads"])

    st.markdown("""
    <div style='margin-bottom:1.2rem;'>
        <h2 style='margin:0;font-size:1.5rem;font-weight:700;color:#1f2937;'>📚 RAG Document Chat</h2>
        <p style='margin:0.2rem 0 0 0;font-size:0.85rem;color:#6b7280;'>
            Converse com seus documentos usando IA com recuperação semântica.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Toggle do pipeline trace na sidebar
    with st.sidebar:
        st.divider()
        st.session_state.show_pipeline_trace = st.toggle(
            "🧠 Mostrar processamento",
            value=st.session_state.show_pipeline_trace,
            help="Exibe o trace do pipeline (thinking box) no chat.",
        )

    tab_chat, tab_obs = st.tabs(["💬 Chat", "🔍 Observabilidade"])

    with tab_chat:
        render_chat_history(st.session_state.chat_history)
        user_query = st.chat_input("Digite sua pergunta sobre os documentos…")
        if user_query:
            _process_query(user_query, config)

    with tab_obs:
        if st.session_state.last_llm_metadata:
            render_observability_panel(
                history_metrics  = st.session_state.last_history_metrics,
                rag_metrics      = st.session_state.last_rag_metrics,
                llm_metadata     = st.session_state.last_llm_metadata,
                latency          = st.session_state.last_latency,
                intent_result    = st.session_state.last_intent_result,
                cache_result     = st.session_state.last_cache_result,
                rerank_result    = st.session_state.last_rerank_result,
                analytics_result = st.session_state.last_analytics,
                clustering       = st.session_state.last_clustering,
            )
        else:
            st.info("As métricas aparecerão aqui após a primeira pergunta.", icon="📊")


if __name__ == "__main__":
    main()
