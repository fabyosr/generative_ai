"""
llm_factory.py
==============
Fábrica de modelos LLM.
Cada provedor é instanciado aqui; a UI e as métricas não precisam saber
qual SDK está sendo usado.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm(provider: str, temperature: float) -> BaseChatModel:
    """
    Retorna um chat model LangChain pronto para uso.

    Parâmetros
    ----------
    provider    : chave do provedor ('openai' | 'ollama' | 'hf_endpoint')
    temperature : criatividade do modelo [0.0 – 1.0]

    Retorna
    -------
    BaseChatModel com stream_usage habilitado onde disponível.
    """
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            stream_usage=True,
        )

    if provider == "ollama":
        return ChatOllama(
            model="phi-3",
            temperature=temperature,
            stream_usage=True,
        )

    if provider == "hf_endpoint":
        endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=temperature,
            return_full_text=False,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Provedor desconhecido: '{provider}'")
