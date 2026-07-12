"""
=============================================================================
ui/chat.py — Renderização do Chat e Exibição de Fontes
=============================================================================
Responsabilidade:
    Funções puras de renderização do chat no Streamlit:
      - Exibir histórico em container scrollável com altura fixa
      - Renderizar nova resposta do assistente
      - Exibir popovers com fontes RAG e badge de latência

Solução para input fixo:
    O st.chat_input é posicionado FORA das tabs em app.py, e o histórico
    é renderizado dentro de um st.container(height=...) scrollável.
    Isso ancora o input sempre abaixo do container, sem depender do
    conteúdo dinâmico do histórico.
=============================================================================
"""

import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

AVATAR_AI    = "🤖"
AVATAR_HUMAN = "🧑"

# Altura do container de histórico em pixels.
# O st.chat_input fica naturalmente ancorado abaixo deste container.
CHAT_CONTAINER_HEIGHT = 520


def render_chat_history(chat_history: list) -> None:
    """
    Renderiza o histórico de mensagens dentro de um container scrollável.

    O uso de st.container(height=CHAT_CONTAINER_HEIGHT) fixa a altura
    da área de mensagens, fazendo o st.chat_input se posicionar sempre
    imediatamente abaixo — comportamento consistente independente do
    volume de mensagens no histórico.

    Args:
        chat_history: Lista de AIMessage e HumanMessage do session_state.
    """
    with st.container(height=CHAT_CONTAINER_HEIGHT, border=False):
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

    Chamada fora do container de histórico (no fluxo de _process_query)
    para que apareça após o container scrollável, abaixo do input.

    Args:
        query: Texto digitado pelo usuário.
    """
    with st.chat_message("human", avatar=AVATAR_HUMAN):
        st.markdown(query)


def render_ai_response(answer: str, latency: float, context_docs: list) -> None:
    """
    Renderiza a resposta final do assistente com badge de latência e fontes.

    Exibe:
        - Resposta em markdown (já limpa de blocos <think> pelo think_parser)
        - Badge com tempo de resposta
        - Popovers para cada chunk RAG recuperado

    Args:
        answer:       Texto limpo da resposta (sem <think>).
        latency:      Tempo total de geração em segundos.
        context_docs: Lista de Document retornados pelo RAG.
    """
    with st.chat_message("assistant", avatar=AVATAR_AI):
        st.markdown(answer)

        # Badge de latência
        st.markdown(
            f"<span style='font-size:0.71rem;color:#6b7280;"
            f"background:#f3f4f6;padding:2px 9px;border-radius:999px;"
            f"display:inline-block;margin-top:4px;'>"
            f"⏱ {latency}s</span>",
            unsafe_allow_html=True,
        )

        # Fontes RAG
        if context_docs:
            st.markdown(
                "<p style='font-size:0.77rem;margin-top:10px;"
                "color:#6b7280;'>📎 Fontes utilizadas:</p>",
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(context_docs), 3))
            for idx, doc in enumerate(context_docs):
                source = doc.metadata.get("source", "desconhecido")
                file   = os.path.basename(source)
                page   = doc.metadata.get("page", "?")
                with cols[idx % 3]:
                    with st.popover(f"📄 {file} · p.{page}", use_container_width=True):
                        st.markdown(f"**Arquivo:** `{file}`  \n**Página:** {page}")
                        st.divider()
                        st.caption(doc.page_content)
