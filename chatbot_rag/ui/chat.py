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

    Exibe o pipeline trace (HTML estático) entre cada par Human/AI,
    re-renderizando os traces salvos no session_state. Isso garante
    que os traces persistam entre re-execuções do Streamlit sem
    precisar re-executar o pipeline.

    Args:
        chat_history: Lista de AIMessage e HumanMessage do session_state.
    """
    traces     = st.session_state.get("chat_history_traces", [])
    show_trace = st.session_state.get("show_pipeline_trace", True)

    # Injeta o CSS do trace uma vez (necessário para os traces históricos)
    if traces and show_trace:
        from ui.pipeline_trace import THINKING_BOX_CSS
        if "thinking_css_injected" not in st.session_state:
            st.markdown(THINKING_BOX_CSS, unsafe_allow_html=True)
            st.session_state.thinking_css_injected = True

    with st.container(height=CHAT_CONTAINER_HEIGHT, border=False):
        # Índice de traces: cada par (HumanMessage, AIMessage) consome 1 trace
        trace_idx = 0

        i = 0
        while i < len(chat_history):
            msg = chat_history[i]

            if isinstance(msg, HumanMessage):
                with st.chat_message("human", avatar=AVATAR_HUMAN):
                    st.markdown(msg.content)

                # Pipeline trace entre human e AI (se disponível e habilitado)
                if show_trace and trace_idx < len(traces):
                    st.markdown(traces[trace_idx], unsafe_allow_html=True)
                    trace_idx += 1

                # Próxima mensagem é a resposta AI correspondente
                i += 1
                if i < len(chat_history) and isinstance(chat_history[i], AIMessage):
                    with st.chat_message("assistant", avatar=AVATAR_AI):
                        st.markdown(chat_history[i].content)
                    i += 1

            elif isinstance(msg, AIMessage):
                # Primeira mensagem de boas-vindas (sem human antes)
                with st.chat_message("assistant", avatar=AVATAR_AI):
                    st.markdown(msg.content)
                i += 1

            else:
                i += 1


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
