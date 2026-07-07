"""
=============================================================================
app.py — Ponto de Entrada e Orquestrador Principal
=============================================================================
Responsabilidade:
    Coordenar todos os módulos da aplicação sem conter lógica de negócio.

    Fluxo principal por mensagem do usuário:

        [mensagem]
            │
            ▼
        classify_intent()          ← core/intent.py
            │
            ├── CHITCHAT  ──────── resposta direta via LLM (sem RAG)
            │
            ├── FOLLOWUP  ──────── RAG com último contexto cacheado
            │
            └── RAG_QUERY
                    │
                    ▼
                semantic_cache.lookup()   ← core/cache.py
                    │
                    ├── HIT  ────── retorna resposta cacheada
                    │
                    └── MISS
                            │
                            ▼
                        rag_chain.invoke()        ← core/rag.py
                            │
                            ▼
                        reranker.rerank()         ← core/reranker.py
                            │
                            ▼
                        semantic_cache.store()
                            │
                            ▼
                        render_ai_response()

    Este arquivo NÃO deve conter:
        - Lógica de embeddings, chunking ou FAISS
        - Instanciação direta de LLMs
        - Cálculos de tokens ou métricas
        - HTML/CSS customizado (isso fica em ui/)
=============================================================================
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

# --- Módulos de negócio ---
from core.secrets  import load_api_keys, get_available_providers
from core.models   import get_model
from core.rag      import build_retriever, build_rag_chain, EMBEDDING_MODEL
from core.intent   import classify_intent, IntentType
from core.cache    import SemanticCache
from core.reranker import Reranker
from core.metrics  import (
    LatencyTimer,
    compute_history_metrics,
    compute_rag_metrics,
    extract_llm_metadata,
)

# --- Módulos de UI ---
from ui.sidebar       import render_sidebar
from ui.chat          import render_chat_history, render_user_message, render_ai_response
from ui.observability import render_observability_panel


# =============================================================================
# Configuração global da página
# =============================================================================

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --text-secondary:    #6b7280;
            --surface-secondary: #f3f4f6;
            --accent:            #6366f1;
            --accent-light:      #e0e7ff;
        }
        .block-container { padding-top: 1.8rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            font-size: 0.84rem;
        }
        div[data-testid="stPopover"] button { font-size: 0.78rem; }
        div[data-testid="stMetric"] label  { font-size: 0.78rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Session State
# =============================================================================

def _init_session_state() -> None:
    """
    Inicializa todas as chaves do session_state com valores padrão.

    Inclui as novas chaves para intent, cache e reranker.
    Chamada no início de cada ciclo de renderização — idempotente.
    """
    defaults = {
        # Chat
        "chat_history": [
            AIMessage(content="Olá! Sou seu assistente de documentos 📚 "
                              "Envie um PDF na barra lateral e faça sua pergunta.")
        ],
        "indexed_files": None,
        "retriever":     None,

        # Objetos reutilizáveis entre queries (instanciados uma vez)
        "embeddings":    None,   # HuggingFaceEmbeddings compartilhado
        "semantic_cache": None,  # SemanticCache (por sessão)
        "reranker":      None,   # Reranker (lazy load do modelo)

        # Último contexto RAG (para FOLLOWUP sem nova busca)
        "last_context_docs": [],

        # Métricas da última interação
        "last_rag_metrics":     {},
        "last_history_metrics": {},
        "last_llm_metadata":    {},
        "last_latency":         0.0,
        "last_intent_result":   None,
        "last_cache_result":    None,
        "last_rerank_result":   None,
        "latency_history":      [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Inicialização de objetos pesados (uma vez por sessão)
# =============================================================================

def _init_heavy_objects() -> None:
    """
    Instancia objetos que carregam modelos pesados apenas uma vez por sessão.

    - HuggingFaceEmbeddings: compartilhado entre retriever e SemanticCache
      para evitar carregar o BGE-M3 duas vezes na memória.
    - SemanticCache: depende do embeddings_model.
    - Reranker: lazy load interno — o modelo cross-encoder só carrega
      na primeira chamada a rerank().
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
    Reconstrói o índice FAISS apenas quando os arquivos mudaram.

    Ao reindexar, também limpa o SemanticCache — respostas antigas
    podem não ser válidas para o novo conjunto de documentos.

    Args:
        uploads: Lista de UploadedFile da sidebar.
    """
    current_names = [f.name for f in uploads]

    if st.session_state.indexed_files != current_names:
        with st.spinner("⚙️ Indexando documentos…"):
            st.session_state.retriever     = build_retriever(
                uploads,
                embeddings=st.session_state.embeddings,
            )
            st.session_state.indexed_files = current_names
            # Invalida cache: novo conjunto de docs = novas respostas
            st.session_state.semantic_cache.clear()

        st.toast(f"✅ {len(uploads)} documento(s) indexado(s)!", icon="📥")


# =============================================================================
# Handlers de processamento por tipo de intenção
# =============================================================================

def _process_chitchat(query: str, llm) -> tuple[str, list]:
    """
    Responde diretamente ao LLM sem acionar o pipeline RAG.

    Usado para saudações e mensagens que não requerem busca em documentos.
    Inclui o histórico para manter coerência conversacional.

    Args:
        query: Texto do usuário.
        llm:   Instância de LLM.

    Returns:
        tuple: (answer: str, context_docs: list vazia)
    """
    from langchain_core.messages import SystemMessage

    messages = [
        SystemMessage(
            content="Você é um assistente simpático para conversa com documentos. "
                    "Responda de forma breve e amigável em português."
        ),
        *st.session_state.chat_history[-6:],
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    return response.content, []


def _process_followup(query: str, llm) -> tuple[str, list]:
    """
    Responde usando o último contexto RAG sem nova busca vetorial.

    Economiza uma busca FAISS e uma chamada ao history_aware_retriever
    ao reutilizar os chunks da query anterior.

    Args:
        query: Pedido de esclarecimento do usuário.
        llm:   Instância de LLM.

    Returns:
        tuple: (answer: str, context_docs: list com chunks anteriores)
    """
    from langchain_core.messages import SystemMessage

    last_docs = st.session_state.last_context_docs
    context   = "\n\n".join(d.page_content for d in last_docs) if last_docs else ""

    messages = [
        SystemMessage(
            content=(
                "Você é um assistente prestativo. Use o contexto abaixo para "
                "elaborar sua resposta anterior. Responda em português.\n\n"
                f"Contexto: {context}"
            )
        ),
        *st.session_state.chat_history[-6:],
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    return response.content, last_docs


def _process_rag(query: str, llm, config: dict) -> tuple[str, list]:
    """
    Executa o pipeline RAG completo com cache e reranker.

    Fluxo:
        1. Lookup no semantic cache
        2. Se miss: invoca rag_chain → reranker → store no cache
        3. Retorna (answer, context_docs)

    Args:
        query:  Texto do usuário.
        llm:    Instância de LLM.
        config: Configurações da sidebar.

    Returns:
        tuple: (answer: str, context_docs: list de Document)
    """
    cache    = st.session_state.semantic_cache
    reranker = st.session_state.reranker

    # --- 1. Lookup no cache ---
    cache_result = cache.lookup(query)
    st.session_state.last_cache_result = cache_result

    if cache_result.hit:
        return cache_result.answer, cache_result.sources or []

    # --- 2. Pipeline RAG ---
    rag_chain = build_rag_chain(llm, st.session_state.retriever)

    result       = rag_chain.invoke({
        "input":        query,
        "chat_history": st.session_state.chat_history,
    })
    answer       = result["answer"]
    context_docs = result.get("context", [])

    # --- 3. Reranking ---
    rerank_result = reranker.rerank(query, context_docs)
    st.session_state.last_rerank_result = rerank_result
    # Usa os docs rerankeados como contexto final
    final_docs = rerank_result.docs if rerank_result.docs else context_docs

    # --- 4. Armazena no cache ---
    cache.store(query, answer, final_docs)

    return answer, final_docs


# =============================================================================
# Processamento principal de uma query
# =============================================================================

def _process_query(query: str, config: dict) -> None:
    """
    Orquestra o processamento completo de uma mensagem do usuário.

    Etapas:
        1. Renderiza mensagem humana
        2. Classifica intenção (intent router)
        3. Despacha para o handler correto (chitchat / followup / rag)
        4. Mede latência
        5. Calcula e persiste todas as métricas
        6. Renderiza resposta

    Args:
        query:  Texto digitado pelo usuário.
        config: Configurações retornadas pela sidebar.
    """
    render_user_message(query)

    llm = get_model(
        provider    = config["provider"],
        model       = config["model"],
        temperature = config["temperature"],
    )

    timer = LatencyTimer()
    timer.start()

    with st.spinner("🤔 Processando…"):

        # --- Classificação de intenção ---
        intent_result = classify_intent(
            query,
            st.session_state.chat_history,
            llm,
        )
        st.session_state.last_intent_result  = intent_result
        st.session_state.last_cache_result   = None
        st.session_state.last_rerank_result  = None

        # --- Despacha por intenção ---
        if intent_result.intent == IntentType.CHITCHAT:
            answer, context_docs = _process_chitchat(query, llm)

        elif intent_result.intent == IntentType.FOLLOWUP:
            answer, context_docs = _process_followup(query, llm)

        else:  # RAG_QUERY
            answer, context_docs = _process_rag(query, llm, config)

    latency = timer.stop()

    # --- Atualiza histórico ---
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.last_context_docs = context_docs

    # --- Calcula métricas ---
    history_metrics = compute_history_metrics(
        st.session_state.chat_history, config["provider"]
    )
    rag_metrics = compute_rag_metrics(context_docs, query, config["provider"])
    llm_metadata = extract_llm_metadata(
        config["provider"], config["model"], config["temperature"]
    )

    # --- Persiste métricas ---
    st.session_state.last_history_metrics = history_metrics
    st.session_state.last_rag_metrics     = rag_metrics
    st.session_state.last_llm_metadata    = llm_metadata
    st.session_state.last_latency         = latency
    st.session_state.latency_history.append(latency)

    # --- Renderiza resposta ---
    render_ai_response(answer, latency, context_docs)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """
    Ciclo principal de renderização do Streamlit.

    Ordem:
        1. Inicializa session_state
        2. Carrega chaves de API
        3. Inicializa objetos pesados (embeddings, cache, reranker)
        4. Renderiza sidebar
        5. Gate de upload
        6. Reindexação condicional
        7. Tabs: Chat | Observabilidade
    """
    _init_session_state()

    secrets_status      = load_api_keys()
    available_providers = get_available_providers(secrets_status)

    _init_heavy_objects()

    config = render_sidebar(available_providers)

    # --- Gate: exige upload ---
    if not config["uploads"]:
        st.markdown(
            """
            <div style='display:flex;flex-direction:column;align-items:center;
                        justify-content:center;padding:4rem 2rem;text-align:center;
                        color:#6b7280;'>
                <span style='font-size:3.5rem;'>📂</span>
                <h3 style='margin:0.8rem 0 0.4rem 0;color:#374151;'>
                    Nenhum documento carregado
                </h3>
                <p style='max-width:380px;line-height:1.6;'>
                    Envie um ou mais arquivos <strong>PDF</strong> pela barra lateral
                    para começar a conversar com seus documentos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    _maybe_reindex(config["uploads"])

    # --- Cabeçalho ---
    st.markdown(
        """
        <div style='margin-bottom:1.2rem;'>
            <h2 style='margin:0;font-size:1.5rem;font-weight:700;color:#1f2937;'>
                📚 RAG Document Chat
            </h2>
            <p style='margin:0.2rem 0 0 0;font-size:0.85rem;color:#6b7280;'>
                Converse com seus documentos usando IA com recuperação semântica.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tabs principais ---
    tab_chat, tab_obs = st.tabs(["💬 Chat", "🔍 Observabilidade"])

    with tab_chat:
        render_chat_history(st.session_state.chat_history)
        user_query = st.chat_input("Digite sua pergunta sobre os documentos…")
        if user_query:
            _process_query(user_query, config)

    with tab_obs:
        if st.session_state.last_llm_metadata:
            render_observability_panel(
                history_metrics = st.session_state.last_history_metrics,
                rag_metrics     = st.session_state.last_rag_metrics,
                llm_metadata    = st.session_state.last_llm_metadata,
                latency         = st.session_state.last_latency,
                intent_result   = st.session_state.last_intent_result,
                cache_result    = st.session_state.last_cache_result,
                rerank_result   = st.session_state.last_rerank_result,
            )
        else:
            st.info("As métricas aparecerão aqui após a primeira pergunta.", icon="📊")


if __name__ == "__main__":
    main()
