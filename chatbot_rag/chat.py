"""
=============================================================================
ui/chat.py — Renderização do Chat e Exibição de Fontes
=============================================================================
Responsabilidade:
    Funções puras de renderização do chat no Streamlit:
      - Exibir histórico de mensagens anteriores
      - Renderizar nova resposta do assistente em streaming simulado
      - Exibir popovers com as fontes (chunks) recuperadas pelo RAG
      - Exibir badge de latência ao lado da resposta

    Toda lógica de negócio (geração, métricas) fica fora deste módulo.
=============================================================================
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Avatares dos participantes do chat
# ---------------------------------------------------------------------------

AVATAR_AI    = "🤖"
AVATAR_HUMAN = "🧑"


def render_chat_history(chat_history: list) -> None:
    """
    Renderiza todas as mensagens do histórico de conversa.

    Itera sobre chat_history e exibe cada mensagem com o avatar correto.
    Chamada uma vez por ciclo de renderização do Streamlit, antes do input.

    Args:
        chat_history: Lista de AIMessage e HumanMessage do session_state.
    """
    for message in chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar=AVATAR_AI):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("human", avatar=AVATAR_HUMAN):
                st.markdown(message.content)


def render_user_message(query: str) -> None:
    """
    Renderiza a mensagem recém-enviada pelo usuário.

    Chamada imediatamente após o usuário submeter uma mensagem,
    antes de aguardar a resposta do LLM.

    Args:
        query: Texto digitado pelo usuário.
    """
    with st.chat_message("human", avatar=AVATAR_HUMAN):
        st.markdown(query)


def render_ai_response(answer: str, latency: float, context_docs: list) -> None:
    """
    Renderiza a resposta do assistente com latência e fontes.

    Exibe:
        - Resposta em markdown
        - Badge com tempo de resposta
        - Popovers para cada chunk recuperado (fonte + página + trecho)

    Args:
        answer:       Texto da resposta gerada pelo LLM.
        latency:      Tempo de geração em segundos.
        context_docs: Lista de Document retornados pelo RAG.
    """
    with st.chat_message("assistant", avatar=AVATAR_AI):
        st.markdown(answer)

        # --- Badge de latência ---
        st.markdown(
            f"<span style='"
            f"font-size:0.72rem; color:var(--text-secondary); "
            f"background:var(--surface-secondary); "
            f"padding:2px 8px; border-radius:999px;'>"
            f"⏱ {latency}s</span>",
            unsafe_allow_html=True,
        )

        # --- Fontes recuperadas ---
        if context_docs:
            st.markdown(
                "<p style='font-size:0.78rem; margin-top:0.8rem; "
                "color:var(--text-secondary);'>📎 Fontes utilizadas:</p>",
                unsafe_allow_html=True,
            )

            cols = st.columns(min(len(context_docs), 3))
            for idx, doc in enumerate(context_docs):
                import os
                source = doc.metadata.get("source", "desconhecido")
                file   = os.path.basename(source)
                page   = doc.metadata.get("page", "?")

                with cols[idx % 3]:
                    with st.popover(f"📄 {file} · p.{page}", use_container_width=True):
                        st.markdown(
                            f"**Arquivo:** `{file}`  \n"
                            f"**Página:** {page}"
                        )
                        st.divider()
                        st.caption(doc.page_content)
