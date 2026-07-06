"""
=============================================================================
ui/sidebar.py — Componentes da Barra Lateral (Sidebar)
=============================================================================
Responsabilidade:
    Renderizar todos os controles da sidebar do Streamlit:
      - Upload de arquivos PDF
      - Seleção de provedor LLM
      - Seleção de modelo dentro do provedor
      - Ajuste de temperatura
      - Informações de configuração ativa

    Retorna as seleções do usuário de forma estruturada para o app.py,
    sem executar nenhuma lógica de negócio.
=============================================================================
"""

import streamlit as st

from core.models import AVAILABLE_MODELS, PROVIDER_LABELS


def render_sidebar(available_providers: list | None = None) -> dict:
    """
    Renderiza a sidebar completa e retorna as configurações selecionadas.

    Controles exibidos:
        1. Upload de PDFs (múltiplos arquivos)
        2. Seletor de provedor LLM (filtrado pelos que têm chave configurada)
        3. Seletor de modelo do provedor escolhido
        4. Slider de temperatura
        5. Rodapé informativo

    Args:
        available_providers: Lista de provedores com chave de API configurada.
                             Se None ou vazia, exibe todos (modo dev/local).

    Returns:
        dict: {
            "uploads":     list[UploadedFile] | None,
            "provider":    str,
            "model":       str,
            "temperature": float,
        }
    """
    # Se nenhum provedor disponível foi informado, usa todos (fallback dev)
    providers = available_providers if available_providers else list(PROVIDER_LABELS.keys())
    with st.sidebar:
        # --- Logo / cabeçalho da sidebar ---
        st.markdown(
            """
            <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
                <span style='font-size:2.4rem;'>📚</span>
                <p style='margin:0; font-size:0.75rem;
                          color: var(--text-secondary);
                          letter-spacing: 0.08em; text-transform: uppercase;'>
                    RAG · Document Chat
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # --- 1. Upload de documentos ---
        st.markdown("#### 📂 Documentos")
        uploads = st.file_uploader(
            label="Enviar PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Envie um ou mais arquivos PDF para indexação.",
        )

        if uploads:
            st.success(f"{len(uploads)} arquivo(s) carregado(s)", icon="✅")
        else:
            st.info("Nenhum arquivo enviado ainda.", icon="ℹ️")

        st.divider()

        # --- 2. Configuração do modelo ---
        st.markdown("#### 🤖 Modelo")

        provider = st.selectbox(
            "Provedor",
            options=providers,
            format_func=lambda k: PROVIDER_LABELS.get(k, k),
            help="Exibe apenas provedores com chave de API configurada.",
        )

        model = st.selectbox(
            "Modelo",
            options=AVAILABLE_MODELS[provider],
            help="Modelos disponíveis para o provedor selecionado.",
        )

        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help=(
                "Controla a criatividade das respostas. "
                "0 = determinístico, 1 = mais criativo."
            ),
        )

        st.divider()

        # --- 3. Rodapé ---
        st.markdown(
            """
            <div style='font-size:0.72rem; color: var(--text-secondary);
                        text-align:center; line-height:1.6;'>
                Powered by LangChain · FAISS · BGE-M3<br>
                <span style='opacity:0.5;'>v2.0</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "uploads":     uploads,
        "provider":    provider,
        "model":       model,
        "temperature": temperature,
    }
