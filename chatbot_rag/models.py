"""
=============================================================================
core/models.py — Fábrica de Modelos LLM
=============================================================================
Responsabilidade:
    Instanciar e retornar objetos LLM de diferentes provedores com base
    na seleção do usuário. Completamente desacoplado do Streamlit.

Provedores suportados:
    - HuggingFace Hub  (hf_hub)
    - OpenAI           (openai)
    - Groq             (groq)

Padrão de projeto:
    Factory Method — a função `get_model` é o ponto único de entrada,
    delegando para funções especializadas conforme o provedor.
=============================================================================
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# Constantes — catálogo de modelos disponíveis por provedor
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "hf_hub": [
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    "groq": [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
}

# Rótulos amigáveis exibidos na sidebar
PROVIDER_LABELS = {
    "hf_hub": "🤗 HuggingFace Hub",
    "openai": "🟢 OpenAI",
    "groq":   "⚡ Groq",
}


# ---------------------------------------------------------------------------
# Funções privadas de instanciação por provedor
# ---------------------------------------------------------------------------

def _model_hf_hub(model: str, temperature: float) -> ChatHuggingFace:
    """
    Instancia um modelo do HuggingFace Hub.

    Envolve HuggingFaceEndpoint em ChatHuggingFace para padronizar
    a interface de chat (mensagens estruturadas) com os demais provedores.

    Args:
        model:       ID do repositório no HuggingFace Hub.
        temperature: Criatividade da geração (0.0 = determinístico).

    Returns:
        ChatHuggingFace: LLM com interface de chat padronizada.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=1024,
        task="text-generation",
    )
    return ChatHuggingFace(llm=endpoint)


def _model_openai(model: str, temperature: float) -> ChatOpenAI:
    """
    Instancia um modelo da OpenAI.

    A chave de API é lida automaticamente via st.secrets["OPENAI_API_KEY"]
    ou variável de ambiente OPENAI_API_KEY.

    Args:
        model:       Nome do modelo (ex: "gpt-4o-mini").
        temperature: Criatividade da geração.

    Returns:
        ChatOpenAI: LLM com interface de chat.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
    )


def _model_groq(model: str, temperature: float) -> ChatGroq:
    """
    Instancia um modelo hospedado na Groq (inferência ultra-rápida).

    Args:
        model:       ID do modelo Groq (ex: "llama3-70b-8192").
        temperature: Criatividade da geração.

    Returns:
        ChatGroq: LLM com interface de chat.
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Ponto único de entrada — Factory Method
# ---------------------------------------------------------------------------

def get_model(provider: str, model: str, temperature: float = 0.1):
    """
    Fábrica de LLMs: instancia e retorna o modelo correto dado um provedor.

    Args:
        provider:    Chave do provedor ("hf_hub" | "openai" | "groq").
        model:       Nome/ID do modelo dentro do provedor.
        temperature: Temperatura de geração (padrão 0.1).

    Returns:
        BaseChatModel: Instância de LLM compatível com LangChain.

    Raises:
        ValueError: Se o provedor informado não for suportado.
    """
    factory = {
        "hf_hub": _model_hf_hub,
        "openai": _model_openai,
        "groq":   _model_groq,
    }

    if provider not in factory:
        raise ValueError(
            f"Provedor '{provider}' não suportado. "
            f"Escolha entre: {list(factory.keys())}"
        )

    return factory[provider](model, temperature)
