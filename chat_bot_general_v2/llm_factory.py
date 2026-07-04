"""
llm_factory.py
==============
Fábrica de modelos LLM.

Para adicionar um novo modelo:
  1. Adicione a chave em LLM_PROVIDERS (config.py)
  2. Adicione um bloco `if provider == "nova_chave"` aqui
  Nenhum outro arquivo precisa mudar.

Provedores disponíveis:
  - openai       : ChatOpenAI (GPT-4o-mini) — requer OPENAI_API_KEY
  - hf_endpoint  : ChatHuggingFace (Llama 3 8B) — requer HUGGINGFACEHUB_API_TOKEN
  - ollama       : ChatOllama (Phi-3) — requer servidor Ollama local (localhost:11434)
                   ⚠️  Não disponível no Streamlit Cloud
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm(provider: str, temperature: float, **kwargs) -> BaseChatModel:
    """
    Retorna um chat model LangChain pronto para uso.

    Parâmetros
    ----------
    provider    : chave do provedor ('openai' | 'ollama' | 'hf_endpoint')
    temperature : criatividade do modelo [0.0 – 1.0]
    **kwargs    : parâmetros extras repassados ao modelo (ex: api_key)
    """
    if provider == "openai":
        # api_key pode vir do kwargs (input da UI) ou do ambiente
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        return ChatOpenAI(
            model       = "gpt-4o-mini",
            temperature = temperature,
            api_key     = api_key or None,
            stream_usage= True,
        )

    if provider == "hf_endpoint":
        endpoint = HuggingFaceEndpoint(
            repo_id         = "meta-llama/Meta-Llama-3-8B-Instruct",
            temperature     = temperature,
            return_full_text= False,
            max_new_tokens  = 512,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "ollama":
        # Ollama só funciona com servidor local — não disponível no Streamlit Cloud
        return ChatOllama(
            model       = "phi-3",
            temperature = temperature,
            stream_usage= True,
        )

    if provider == "hf_llama31":
        endpoint = HuggingFaceEndpoint(
            repo_id         = "meta-llama/Llama-3.1-8B-Instruct",
            temperature     = temperature,
            return_full_text= False,
            max_new_tokens  = 1024,   # 3.1 suporta respostas mais longas
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "hf_mistral":
        return ChatOpenAI(
            model           = "mistralai/Mistral-7B-Instruct-v0.3",
            base_url        = "https://router.huggingface.co/v1",
            api_key         = os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""),
            temperature     = temperature,
            stream_usage    = True,
        )

    if provider == "hf_qwen":
        return ChatOpenAI(
            model           = "Qwen/Qwen2.5-7B-Instruct",
            base_url        = "https://router.huggingface.co/v1",
            api_key         = os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""),
            temperature     = temperature,
            stream_usage    = True,
        )

    if provider == "hf_gemma2":
        return ChatOpenAI(
            model           = "google/gemma-2-2b-it",
            base_url        = "https://router.huggingface.co/v1",
            api_key         = os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""),
            temperature     = temperature,
            stream_usage    = True,
        )

    if provider == "hf_phi35":
        endpoint = HuggingFaceEndpoint(
            repo_id         = "microsoft/Phi-3.5-mini-instruct",
            temperature     = temperature,
            return_full_text= False,
            max_new_tokens  = 512,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "hf_sabia":
        return HuggingFaceEndpoint(
            repo_id="maritaca-ai/sabia-7b",
            task="text-generation",
            huggingfacehub_api_token=sec_token,
            temperature=0.3,
            model_kwargs={ "max_length": 512}
        )

    raise ValueError(f"Provedor desconhecido: '{provider}'")


def is_provider_available(provider: str) -> tuple[bool, str]:
    """
    Verifica se o provedor está disponível no ambiente atual.

    Retorna
    -------
    (disponível, motivo) — motivo é string vazia quando disponível
    """
    if provider == "openai":
        has_key = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("STREAMLIT_OPENAI_KEY")
        )
        # Chave pode ser inserida via UI — considera disponível para não bloquear
        return True, ""

    if provider == "hf_endpoint":
        has_token = bool(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
        if not has_token:
            return False, "Token HUGGINGFACEHUB_API_TOKEN não encontrado."
        return True, ""

    if provider == "ollama":
        # Testa conexão com servidor Ollama local
        try:
            import httpx
            r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if r.status_code == 200:
                return True, ""
            return False, "Servidor Ollama respondeu com erro."
        except Exception:
            return False, (
                "Ollama não disponível (requer servidor local em localhost:11434). "
                "Não funciona no Streamlit Cloud."
            )

    return False, f"Provedor desconhecido: '{provider}'"
