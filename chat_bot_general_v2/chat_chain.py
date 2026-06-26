"""
chat_chain.py
=============
Constrói o prompt e retorna o gerador de streaming.
Desacoplado da UI — recebe dados puros e devolve um gerador.
"""

from typing import Generator, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_factory import get_llm


def build_stream(
    user_query: str,
    chat_history: list[BaseMessage],
    provider: str,
    temperature: float,
    system_prompt: str,
) -> Generator[Any, None, None]:
    """
    Cria o chain prompt → llm e retorna o gerador de chunks do stream.

    Parâmetros
    ----------
    user_query   : texto enviado pelo usuário nesta rodada
    chat_history : lista de BaseMessage com o histórico completo
    provider     : chave do provedor LLM
    temperature  : criatividade
    system_prompt: instrução de personalidade do sistema
    """
    llm = get_llm(provider, temperature)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    chain = prompt | llm  # sem StrOutputParser → preserva metadados nos chunks

    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
    })
