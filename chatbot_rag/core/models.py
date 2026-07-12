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

from core.secrets import get_api_key


# ---------------------------------------------------------------------------
# Constantes — catálogo de modelos disponíveis por provedor
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "hf_hub": [
        # Modelos validados com HuggingFace Inference Providers (provider="auto")
        # O parâmetro provider="auto" seleciona automaticamente o melhor
        # provider disponível para cada modelo (Together, Nebius, Novita, etc.)
        "deepseek-ai/DeepSeek-R1-0528",
        "meta-llama/Llama-3.1-8B-Instruct",   # era Meta-Llama-3-8B (descontinuado)
        "mistralai/Mistral-7B-Instruct-v0.3",  # era v0.2 (descontinuado)
        "Qwen/Qwen2.5-72B-Instruct",
        "microsoft/Phi-4",                      # era Phi-3-mini (descontinuado)
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
    Instancia um modelo do HuggingFace Hub via Inference Providers.

    O parâmetro `provider="auto"` é obrigatório desde a migração do HF
    para o sistema de Inference Providers (2025). Sem ele, a API retorna
    'model_not_supported' mesmo para modelos válidos, porque o endpoint
    legado (Serverless Inference API) foi descontinuado para a maioria
    dos modelos.

    Com `provider="auto"`, o HF seleciona automaticamente o melhor
    provider disponível para o modelo (Together AI, Nebius, Novita, etc.),
    conforme a configuração em hf.co/settings/inference-providers.

    O token é passado como `huggingfacehub_api_token` (nome correto do
    parâmetro na versão atual de langchain-huggingface).

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
        provider="auto",                          # ← obrigatório no HF atual
        huggingfacehub_api_token=get_api_key("HUGGINGFACEHUB_API_TOKEN"),
    )
    return ChatHuggingFace(llm=endpoint)


def _model_openai(model: str, temperature: float) -> ChatOpenAI:
    """
    Instancia um modelo da OpenAI.

    A chave OPENAI_API_KEY é passada explicitamente via get_api_key()
    para garantir que o valor correto do st.secrets seja usado,
    independente do estado de os.environ no ciclo atual do Streamlit.

    Args:
        model:       Nome do modelo (ex: "gpt-4o-mini").
        temperature: Criatividade da geração.

    Returns:
        ChatOpenAI: LLM com interface de chat.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=get_api_key("OPENAI_API_KEY"),
    )


def _model_groq(model: str, temperature: float) -> ChatGroq:
    """
    Instancia um modelo hospedado na Groq (inferência ultra-rápida).

    A chave GROQ_API_KEY é passada explicitamente via get_api_key(),
    lendo diretamente do st.secrets a cada instanciação. Isso resolve
    o AuthenticationError 401 causado por race conditions entre ciclos
    de re-execução do Streamlit, onde os.environ pode ainda não ter
    recebido a chave quando o ChatGroq é instanciado.

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
        groq_api_key=get_api_key("GROQ_API_KEY"),
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
