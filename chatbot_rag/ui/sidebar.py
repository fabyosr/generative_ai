"""
=============================================================================
ui/sidebar.py — Barra Lateral: Documentos, Modelo, Reranker e UI
=============================================================================
Responsabilidade:
    Renderizar todos os controles da sidebar e retornar configurações
    estruturadas para o app.py. Zero lógica de negócio aqui.

Seções:
    1. Documentos     — upload de PDFs
    2. Modelo         — provedor, modelo, temperatura
    3. Reranker       — método (Top-K / Threshold / Adaptativo), max-k, score mínimo
    4. Interface      — toggle de pipeline trace
    5. Rodapé
=============================================================================
"""

import streamlit as st
from core.models import AVAILABLE_MODELS, PROVIDER_LABELS


def render_sidebar(available_providers: list | None = None) -> dict:
    """
    Renderiza a sidebar e retorna as configurações selecionadas.

    Args:
        available_providers: Provedores com chave de API configurada.
                             None = exibe todos (modo dev).

    Returns:
        dict com chaves:
            uploads, provider, model, temperature,
            reranker_method, reranker_max_k, reranker_min_score,
            show_pipeline_trace
    """
    providers = available_providers if available_providers else list(PROVIDER_LABELS.keys())

    with st.sidebar:

        # ── Cabeçalho ───────────────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center;padding:0.5rem 0 1rem 0;'>"
            "<span style='font-size:2.4rem;'>📚</span>"
            "<p style='margin:0;font-size:0.75rem;color:#6b7280;"
            "letter-spacing:0.08em;text-transform:uppercase;'>"
            "RAG · Document Chat</p></div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 1. Documentos ───────────────────────────────────────────────────
        st.markdown("#### 📂 Documentos")
        uploads = st.file_uploader(
            label="Enviar PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Envie um ou mais arquivos PDF para indexação no FAISS.",
        )
        if uploads:
            st.success(f"{len(uploads)} arquivo(s) carregado(s)", icon="✅")
        else:
            st.info("Nenhum arquivo enviado.", icon="ℹ️")

        st.divider()

        # ── 2. Modelo ───────────────────────────────────────────────────────
        st.markdown("#### 🤖 Modelo")

        provider = st.selectbox(
            "Provedor",
            options=providers,
            format_func=lambda k: PROVIDER_LABELS.get(k, k),
            help="Apenas provedores com chave de API configurada no secrets.toml.",
        )

        model = st.selectbox(
            "Modelo",
            options=AVAILABLE_MODELS[provider],
            help="Modelos disponíveis para o provedor selecionado.",
        )

        temperature = st.slider(
            "Temperatura",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="0 = determinístico · 1 = mais criativo.",
        )

        st.divider()

        # ── 3. Classificador de Intenção ────────────────────────────────────
        st.markdown("#### 🎯 Classificador de Intenção")
        st.caption(
            "Modelo dedicado para classificar cada mensagem antes de acionar o RAG. "
            "Use um modelo leve para economizar tokens do modelo principal."
        )

        classifier_provider = st.selectbox(
            "Provedor do classificador",
            options=["hf_serverless", "hf_hub", "openai", "groq"],
            format_func=lambda k: PROVIDER_LABELS.get(k, k),
            index=0,   # hf_serverless como padrão (gratuito)
            help=(
                "HuggingFace Serverless é recomendado — gratuito e suficiente "
                "para classificação de intenção."
            ),
            key="classifier_provider_select",
        )

        from core.models import CLASSIFIER_MODELS
        classifier_model_default = CLASSIFIER_MODELS.get(
            classifier_provider, "Qwen/Qwen2.5-3B-Instruct"
        )
        st.markdown(
            f"Modelo: `{classifier_model_default}`",
            help="Modelo selecionado automaticamente para o provedor escolhido.",
        )
        st.markdown("#### 🔀 Reranker")
        st.caption(
            "Controla como o cross-encoder seleciona os chunks "
            "mais relevantes após a busca FAISS."
        )

        reranker_method = st.selectbox(
            "Método de seleção",
            options=["top_k", "threshold", "adaptive"],
            format_func=lambda m: {
                "top_k":     "🔢 Top-K fixo — retorna sempre k chunks",
                "threshold": "📏 Threshold — retorna chunks acima do score mínimo",
                "adaptive":  "⚡ Adaptativo — Top-K com score mínimo (combinado)",
            }[m],
            help=(
                "Top-K: sempre retorna k chunks independente do score.\n"
                "Threshold: retorna todos acima do score mínimo.\n"
                "Adaptativo: combina os dois — respeita k máximo E score mínimo."
            ),
        )

        reranker_max_k = st.slider(
            "Max chunks (k)",
            min_value=1, max_value=8, value=3, step=1,
            help=(
                "Número máximo de chunks enviados ao LLM.\n"
                "Mais chunks = mais contexto, mais tokens consumidos."
            ),
        )

        # Score mínimo só é relevante em threshold e adaptive
        reranker_min_score = 0.0
        if reranker_method in ("threshold", "adaptive"):
            reranker_min_score = st.slider(
                "Score mínimo do reranker",
                min_value=0.0, max_value=1.0, value=0.30, step=0.05,
                help=(
                    "Chunks com score abaixo deste valor são descartados.\n"
                    "0.30 é um bom ponto de partida para a maioria dos casos.\n"
                    "Aumente para respostas mais precisas (menos contexto).\n"
                    "Diminua se o reranker descartar chunks relevantes."
                ),
            )

        st.divider()

        # ── 4. Interface ────────────────────────────────────────────────────
        st.markdown("#### 🖥 Interface")

        show_trace = st.toggle(
            "🧠 Mostrar processamento",
            value=st.session_state.get("show_pipeline_trace", True),
            help="Exibe a caixa de pensamento do agente durante o processamento.",
        )
        st.session_state.show_pipeline_trace = show_trace

        st.divider()

        # ── 5. Rodapé ───────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.71rem;color:#9ca3af;"
            "text-align:center;line-height:1.7;'>"
            "LangChain · FAISS · BGE-M3 · BGE-Reranker<br>"
            "<span style='opacity:0.5;'>v3.0</span></div>",
            unsafe_allow_html=True,
        )

    return {
        "uploads":              uploads,
        "provider":             provider,
        "model":                model,
        "temperature":          temperature,
        "classifier_provider":  classifier_provider,
        "reranker_method":      reranker_method,
        "reranker_max_k":       reranker_max_k,
        "reranker_min_score":   reranker_min_score,
        "show_pipeline_trace":  show_trace,
    }
