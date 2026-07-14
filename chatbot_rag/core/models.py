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
from langchain_community.llms import HuggingFaceHub

from core.secrets import get_api_key


# ---------------------------------------------------------------------------
# Constantes — catálogo de modelos disponíveis por provedor
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "hf_hub": [
        # Via Inference Providers (provider="auto") — crédito mensal
        "deepseek-ai/DeepSeek-R1-0528",       # 671B MoE, CoT explícito (<think>)
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    "hf_serverless": [
        # Via Serverless Inference API — rate limit por hora, sem crédito mensal
        # Modelos ≤10B validados como chat-compatible no Serverless (julho/2026)
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-3B-Instruct",           # mais leve — bom para o classificador
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    "groq": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
}

# Modelos disponíveis para o classificador de intenção (leves, sem custo alto)
# Independentes do modelo principal selecionado pelo usuário
CLASSIFIER_MODELS = {
    "hf_serverless": "Qwen/Qwen2.5-3B-Instruct",      # padrão — leve e gratuito
    "hf_hub":        "Qwen/Qwen2.5-7B-Instruct",
    "openai":        "gpt-4o-mini",                     # mais barato da OpenAI
    "groq":          "openai/gpt-oss-20b",              # mais rápido do Groq
}

PROVIDER_LABELS = {
    "hf_hub":        "🤗 HuggingFace Providers",
    "hf_serverless": "🆓 HuggingFace Serverless",
    "openai":        "🟢 OpenAI",
    "groq":          "⚡ Groq",
}


# ---------------------------------------------------------------------------
# Catálogo de modelos — janela de contexto e preços por token (USD/1K tokens)
# ---------------------------------------------------------------------------
# Fonte preços OpenAI: https://openai.com/pricing  (julho/2026)
# Fonte preços Groq:   https://groq.com/pricing    (julho/2026)
# Fonte HF Hub: inferência via Inference Providers — cobrança variável
# por provider (Together, Nebius, Novita); usamos estimativa conservadora.
#
# Formato: (input_per_1k, output_per_1k) em USD
# Janela de contexto em tokens (número inteiro)
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, dict] = {

    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-4o-mini": {
        "context_window": 128_000,
        "price":          (0.000150, 0.000600),
        "tier":           "paid",
    },
    "gpt-4o": {
        "context_window": 128_000,
        "price":          (0.002500, 0.010000),
        "tier":           "paid",
    },
    "gpt-3.5-turbo": {
        "context_window": 16_385,
        "price":          (0.000500, 0.001500),
        "tier":           "paid",
    },

    # ── Groq (produção confirmada — julho/2026) ───────────────────────────
    "openai/gpt-oss-120b": {
        "context_window": 131_072,
        "price":          (0.000150, 0.000600),
        "tier":           "free+paid",
    },
    "openai/gpt-oss-20b": {
        "context_window": 131_072,
        "price":          (0.000075, 0.000300),
        "tier":           "free+paid",
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "context_window": 131_072,
        "price":          (0.000110, 0.000340),
        "tier":           "free+paid",
    },

    # ── HuggingFace Serverless ────────────────────────────────────────────
    # Rate limit por hora (não crédito mensal). Modelos ≤10B parâmetros.
    # Preços: gratuito até o rate limit; PRO $9/mês aumenta os limites.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "context_window": 131_072,
        "price":          (0.0, 0.0),           # gratuito no free tier
        "tier":           "free",
    },
    "google/gemma-2-9b-it": {
        "context_window": 8_192,
        "price":          (0.0, 0.0),
        "tier":           "free",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "context_window": 32_768,
        "price":          (0.0, 0.0),
        "tier":           "free",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "context_window": 32_768,
        "price":          (0.0, 0.0),
        "tier":           "free",
    },
    "deepseek-ai/DeepSeek-R1-0528": {
        "context_window": 163_840,
        "price":          (0.000300, 0.000300),
        "tier":           "paid",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "context_window": 131_072,
        "price":          (0.000050, 0.000080),
        "tier":           "paid",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "context_window": 131_072,
        "price":          (0.000290, 0.000390),
        "tier":           "paid",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "context_window": 131_072,
        "price":          (0.000070, 0.000100),
        "tier":           "paid",
    },
}

# Fallbacks quando o modelo não estiver no catálogo
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai":        {"context_window": 16_385,  "price": (0.000500, 0.001500), "tier": "paid"},
    "groq":          {"context_window": 131_072, "price": (0.000075, 0.000300), "tier": "free+paid"},
    "hf_hub":        {"context_window": 32_768,  "price": (0.000100, 0.000100), "tier": "paid"},
    "hf_serverless": {"context_window": 8_192,   "price": (0.0,       0.0),     "tier": "free"},
}


def get_model_info(provider: str, model: str) -> dict:
    """
    Retorna as informações do catálogo para um modelo específico.

    Busca pelo nome exato do modelo no MODEL_CATALOG.
    Se não encontrar, usa o fallback do provedor.

    Args:
        provider: Provedor LLM ("openai" | "groq" | "hf_hub").
        model:    Nome/ID do modelo.

    Returns:
        dict com chaves: context_window (int), price (tuple), tier (str).
    """
    return MODEL_CATALOG.get(model) or _PROVIDER_DEFAULTS.get(provider) or {
        "context_window": 4_096,
        "price":          (0.000100, 0.000100),
        "tier":           "unknown",
    }


# ---------------------------------------------------------------------------
# Funções privadas de instanciação por provedor
# ---------------------------------------------------------------------------

def _validate_key(key_name: str, provider_label: str, url: str) -> str:
    """
    Valida e retorna uma chave de API, lançando ValueError descritivo se ausente.

    Args:
        key_name:       Nome da chave no secrets.toml (ex: "GROQ_API_KEY").
        provider_label: Nome amigável do provedor para a mensagem de erro.
        url:            URL para obter a chave.

    Returns:
        str: Valor da chave.

    Raises:
        ValueError: Se a chave não estiver configurada.
    """
    key = get_api_key(key_name)
    if not key:
        raise ValueError(
            f"{key_name} não encontrada. "
            f"Configure em .streamlit/secrets.toml:\n\n"
            f"  {key_name} = \"...\"\n\n"
            f"Obtenha em: {url}"
        )
    return key


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
    token = _validate_key(
        "HUGGINGFACEHUB_API_TOKEN",
        "HuggingFace Hub",
        "https://huggingface.co/settings/tokens",
    )
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=1024,
        task="text-generation",
        provider="auto",
        huggingfacehub_api_token=token,
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
        api_key=_validate_key(
            "OPENAI_API_KEY",
            "OpenAI",
            "https://platform.openai.com/api-keys",
        ),
    )


def _model_groq(model: str, temperature: float) -> ChatGroq:
    """
    Instancia um modelo hospedado na Groq (inferência ultra-rápida).

    A chave é lida diretamente do st.secrets via get_api_key() e passada
    explicitamente no construtor como `api_key` — nome correto do parâmetro
    no SDK groq>=0.9 / langchain-groq>=0.2.

    Por que não usar `groq_api_key`:
        O parâmetro `groq_api_key` foi renomeado para `api_key` nas versões
        recentes do langchain-groq. Usar o nome antigo resulta em o SDK
        ignorar o valor passado e tentar ler de os.environ, onde a chave
        pode não estar disponível neste ciclo do Streamlit → 401.

    Por que não depender de os.environ:
        O Streamlit re-executa o script inteiro a cada interação. A janela
        entre load_api_keys() injetar em os.environ e o ChatGroq ser
        instanciado pode ser zero em alguns ciclos, causando 401 intermitente.
        Passar explicitamente elimina essa race condition.

    Validação:
        Lança ValueError descritivo se a chave não estiver configurada,
        evitando o 401 genérico do SDK que não informa onde configurar.

    Args:
        model:       ID do modelo Groq (ex: "llama3-70b-8192").
        temperature: Criatividade da geração.

    Returns:
        ChatGroq: LLM com interface de chat.

    Raises:
        ValueError: Se GROQ_API_KEY não estiver configurada no secrets.toml.
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_retries=2,
        api_key=_validate_key(
            "GROQ_API_KEY",
            "Groq",
            "https://console.groq.com/keys",
        ),
    )


# ---------------------------------------------------------------------------
# Ponto único de entrada — Factory Method
# ---------------------------------------------------------------------------

def _model_hf_serverless(model: str, temperature: float) -> ChatHuggingFace:
    """
    Instancia um modelo via HuggingFace Serverless Inference API.

    Diferença em relação ao hf_hub (Inference Providers):
        - hf_hub:        usa provider="auto" → roteia para Together/Nebius/etc.
                         consome crédito mensal esgotável.
        - hf_serverless: usa api-inference.huggingface.co diretamente →
                         rate limit por hora (não crédito mensal).
                         Gratuito no free tier para modelos ≤10B parâmetros.

    Usa HuggingFaceEndpoint sem provider="auto" e apontando para o
    endpoint serverless, que é o comportamento default antes do HF
    introduzir os Inference Providers.

    Args:
        model:       ID do modelo (deve ser ≤10B e chat-compatible).
        temperature: Temperatura de geração.

    Returns:
        ChatHuggingFace: LLM com interface de chat.
    """
    token = _validate_key(
        "HUGGINGFACEHUB_API_TOKEN",
        "HuggingFace Serverless",
        "https://huggingface.co/settings/tokens",
    )
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=1024,
        task="text-generation",
        # Sem provider="auto" → usa Serverless Inference API diretamente
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=endpoint)


def get_model(provider: str, model: str, temperature: float = 0.1):
    """
    Fábrica de LLMs: instancia e retorna o modelo correto dado um provedor.

    Args:
        provider:    "hf_hub" | "hf_serverless" | "openai" | "groq"
        model:       Nome/ID do modelo dentro do provedor.
        temperature: Temperatura de geração (padrão 0.1).

    Returns:
        BaseChatModel: Instância de LLM compatível com LangChain.

    Raises:
        ValueError: Se o provedor informado não for suportado.
    """
    factory = {
        "hf_hub":        _model_hf_hub,
        "hf_serverless": _model_hf_serverless,
        "openai":        _model_openai,
        "groq":          _model_groq,
    }

    if provider not in factory:
        raise ValueError(
            f"Provedor '{provider}' não suportado. "
            f"Escolha entre: {list(factory.keys())}"
        )

    return factory[provider](model, temperature)


def get_classifier_model(main_provider: str, classifier_provider: str):
    """
    Instancia o modelo dedicado para o classificador de intenção.

    O classificador usa um modelo independente do modelo principal,
    geralmente mais leve e barato, para não consumir tokens do modelo
    principal em cada classificação.

    Hierarquia de seleção:
        1. Usa o provider selecionado pelo usuário na sidebar para o classificador
        2. Seleciona o modelo padrão do CLASSIFIER_MODELS para esse provider
        3. Se o provider do classificador não tiver chave configurada,
           faz fallback para hf_serverless (gratuito)

    Args:
        main_provider:       Provedor do modelo principal (para fallback).
        classifier_provider: Provedor selecionado para o classificador.

    Returns:
        BaseChatModel: LLM leve para classificação de intenção.
    """
    model = CLASSIFIER_MODELS.get(
        classifier_provider,
        CLASSIFIER_MODELS.get(main_provider, "Qwen/Qwen2.5-3B-Instruct")
    )

    try:
        return get_model(classifier_provider, model, temperature=0.0)
    except ValueError:
        # Fallback para serverless gratuito se o provider falhar
        return get_model("hf_serverless", "Qwen/Qwen2.5-3B-Instruct", temperature=0.0)
