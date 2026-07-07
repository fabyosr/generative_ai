"""
=============================================================================
app.py — Ponto de Entrada e Orquestrador Principal
=============================================================================
Responsabilidade:
    Coordenar todos os módulos da aplicação sem conter lógica de negócio.

    Fluxo principal por ciclo Streamlit:
        1. Renderizar sidebar → obter configurações do usuário
        2. Gerenciar session_state (histórico, retriever, métricas)
        3. Exibir o conteúdo principal via st.tabs (Chat | Observabilidade)
        4. Processar input do usuário:
               a. Indexar documentos se necessário (build_retriever)
               b. Instanciar LLM (get_model)
               c. Construir e invocar RAG chain (build_rag_chain)
               d. Calcular métricas (core/metrics)
               e. Renderizar resposta + painel de observabilidade

    Este arquivo NÃO deve conter:
        - Lógica de embeddings, chunking ou FAISS
        - Instanciação direta de LLMs
        - Cálculos de tokens ou métricas
        - HTML/CSS customizado (isso fica em ui/)
=============================================================================
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# --- Módulos de negócio (sem dependência Streamlit) ---
from core.secrets import load_api_keys, get_available_providers
from core.models import get_model
from core.rag import build_retriever, build_rag_chain
from core.metrics import (
    LatencyTimer,
    compute_history_metrics,
    compute_rag_metrics,
    extract_llm_metadata,
)

# --- Módulos de UI (apenas renderização) ---
from ui.sidebar import render_sidebar
from ui.chat import render_chat_history, render_user_message, render_ai_response
from ui.observability import render_observability_panel


# =============================================================================
# Configuração global da página Streamlit
# =============================================================================

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS global — variáveis de tema e ajustes visuais mínimos
st.markdown(
    """
    <style>
        /* Variáveis de cor reutilizadas nos componentes ui/ */
        :root {
            --text-secondary:    #6b7280;
            --surface-secondary: #f3f4f6;
            --accent:            #6366f1;
            --accent-light:      #e0e7ff;
        }

        /* Remove padding excessivo do topo */
        .block-container { padding-top: 1.8rem; }

        /* Tabs com visual mais discreto */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            font-size: 0.84rem;
        }

        /* Popovers de fonte */
        div[data-testid="stPopover"] button {
            font-size: 0.78rem;
        }

        /* Métricas menores no painel de observabilidade */
        div[data-testid="stMetric"] label {
            font-size: 0.78rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Inicialização do Session State
# =============================================================================

def _init_session_state() -> None:
    """
    Garante que todas as chaves do session_state existam antes do uso.

    Chamada uma única vez no início de cada ciclo de renderização.
    Evita KeyError e define valores padrão explícitos.
    """
    defaults = {
        # Histórico de mensagens do chat (AIMessage / HumanMessage)
        "chat_history": [
            AIMessage(content="Olá! Sou seu assistente de documentos 📚 "
                              "Envie um PDF na barra lateral e faça sua pergunta.")
        ],
        # Lista de arquivos atualmente indexados (para detectar mudanças)
        "indexed_files": None,
        # Retriever FAISS ativo
        "retriever": None,
        # Métricas da última resposta (dict)
        "last_rag_metrics": {},
        "last_history_metrics": {},
        "last_llm_metadata": {},
        "last_latency": 0.0,
        # Histórico de latências para o gráfico na aba Desempenho
        "latency_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Lógica de indexação — detecta novos uploads e reconstrói o retriever
# =============================================================================

def _maybe_reindex(uploads: list) -> None:
    """
    Reconstrói o índice FAISS apenas quando os arquivos enviados mudaram.

    Compara os nomes dos arquivos atuais com os da última indexação.
    Se houver diferença, chama build_retriever() e atualiza o session_state.

    Args:
        uploads: Lista de UploadedFile retornada pela sidebar.
    """
    current_names = [f.name for f in uploads]

    if st.session_state.indexed_files != current_names:
        with st.spinner("⚙️ Indexando documentos… isso pode levar alguns segundos."):
            st.session_state.retriever    = build_retriever(uploads)
            st.session_state.indexed_files = current_names
        st.toast(f"✅ {len(uploads)} documento(s) indexado(s)!", icon="📥")


# =============================================================================
# Processamento de uma query do usuário
# =============================================================================

def _process_query(query: str, config: dict) -> None:
    """
    Executa o pipeline completo para uma query do usuário:
        1. Adiciona mensagem humana ao histórico
        2. Renderiza mensagem do usuário
        3. Instancia LLM com configurações da sidebar
        4. Constrói e invoca RAG chain
        5. Mede latência
        6. Calcula e armazena métricas
        7. Renderiza resposta

    Args:
        query:  Texto digitado pelo usuário.
        config: Dicionário de configurações retornado por render_sidebar().
    """
    # --- Adiciona ao histórico ---
    st.session_state.chat_history.append(HumanMessage(content=query))
    render_user_message(query)

    # --- Instancia modelo e chain ---
    llm = get_model(
        provider=config["provider"],
        model=config["model"],
        temperature=config["temperature"],
    )
    rag_chain = build_rag_chain(llm, st.session_state.retriever)

    # --- Invoca chain com cronômetro ---
    timer = LatencyTimer()
    timer.start()

    with st.spinner("🤔 Gerando resposta…"):
        result = rag_chain.invoke({
            "input":        query,
            "chat_history": st.session_state.chat_history,
        })

    latency = timer.stop()

    st.markdown(f'st.session_state.retriever: {st.session_state.retriever}')
    st.markdown(f'resultado rag: {result}')

    # --- Extrai resposta e contexto ---
    answer       = result["answer"]
    context_docs = result.get("context", [])

    # --- Calcula métricas ---
    st.session_state.chat_history.append(AIMessage(content=answer))

    history_metrics = compute_history_metrics(
        st.session_state.chat_history, config["provider"]
    )
    rag_metrics = compute_rag_metrics(context_docs, query, config["provider"])
    llm_metadata = extract_llm_metadata(
        config["provider"], config["model"], config["temperature"]
    )

    # --- Persiste métricas no session_state ---
    st.session_state.last_history_metrics = history_metrics
    st.session_state.last_rag_metrics     = rag_metrics
    st.session_state.last_llm_metadata    = llm_metadata
    st.session_state.last_latency         = latency
    st.session_state.latency_history.append(latency)

    # --- Renderiza resposta ---
    render_ai_response(answer, latency, context_docs)


# =============================================================================
# Entry point — ciclo principal de renderização
# =============================================================================

def main() -> None:
    """
    Função principal do app. Chamada a cada ciclo de renderização do Streamlit.

    Estrutura:
        1. Inicializa session_state
        2. Renderiza sidebar e obtém config
        3. Exibe gate de "envie um arquivo"
        4. Detecta novos uploads e reindexar se necessário
        5. Renderiza layout em duas colunas: cabeçalho + tabs
        6. Tab Chat: histórico + input
        7. Tab Observabilidade: painel de métricas
    """
    _init_session_state()

    # --- Carrega chaves de API do st.secrets → os.environ ---
    secrets_status    = load_api_keys()
    available_providers = get_available_providers(secrets_status)

    # --- Sidebar ---
    config = render_sidebar(available_providers)

    # --- Gate: exige pelo menos um arquivo ---
    if not config["uploads"]:
        st.markdown(
            """
            <div style='
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; padding: 4rem 2rem; text-align:center;
                color: #6b7280;
            '>
                <span style='font-size:3.5rem;'>📂</span>
                <h3 style='margin:0.8rem 0 0.4rem 0; color:#374151;'>
                    Nenhum documento carregado
                </h3>
                <p style='max-width:380px; line-height:1.6;'>
                    Envie um ou mais arquivos <strong>PDF</strong> pela barra lateral
                    para começar a conversar com seus documentos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # --- Reindexar se necessário ---
    _maybe_reindex(config["uploads"])

    # --- Cabeçalho ---
    st.markdown(
        """
        <div style='margin-bottom:1.2rem;'>
            <h2 style='margin:0; font-size:1.5rem; font-weight:700; color:#1f2937;'>
                📚 RAG Document Chat
            </h2>
            <p style='margin:0.2rem 0 0 0; font-size:0.85rem; color:#6b7280;'>
                Converse com seus documentos usando IA com recuperação semântica.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tabs principais ---
    tab_chat, tab_obs = st.tabs(["💬 Chat", "🔍 Observabilidade"])

    with tab_chat:
        # Renderiza histórico existente
        render_chat_history(st.session_state.chat_history)

        # Input do usuário
        user_query = st.chat_input("Digite sua pergunta sobre os documentos…")

        if user_query:
            _process_query(user_query, config)

    with tab_obs:
        # Exibe métricas da última interação (ou estado vazio)
        if st.session_state.last_llm_metadata:
            render_observability_panel(
                history_metrics = st.session_state.last_history_metrics,
                rag_metrics     = st.session_state.last_rag_metrics,
                llm_metadata    = st.session_state.last_llm_metadata,
                latency         = st.session_state.last_latency,
            )
        else:
            st.info(
                "As métricas aparecerão aqui após a primeira pergunta.",
                icon="📊",
            )


# =============================================================================
# Execução
# =============================================================================

if __name__ == "__main__":
    main()
