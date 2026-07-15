"""
=============================================================================
core/secrets.py — Injeção de Chaves de API via st.secrets
=============================================================================
Responsabilidade:
    Carregar as chaves de API do st.secrets e injetá-las como variáveis
    de ambiente, garantindo que os SDKs dos provedores (OpenAI, Groq,
    HuggingFace) as encontrem automaticamente.

Por que esta abordagem:
    - st.secrets funciona tanto localmente (.streamlit/secrets.toml)
      quanto no Streamlit Cloud (painel de configuração do app).
    - Os SDKs das libs (openai, groq, huggingface_hub) leem de os.environ,
      então injetar lá centraliza o gerenciamento em um único lugar.
    - Fallback silencioso: se uma chave não existir no secrets, não quebra —
      apenas o provedor correspondente ficará indisponível.

Chamada em:
    app.py — antes de qualquer instanciação de modelo.
=============================================================================
"""

import os
import streamlit as st


# Mapeamento: nome lógico → chave no secrets.toml
_SECRETS_MAP = {
    "OPENAI_API_KEY":           "OPENAI_API_KEY",
    "GROQ_API_KEY":             "GROQ_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN": "HUGGINGFACEHUB_API_TOKEN",
}


def load_api_keys() -> dict[str, bool]:
    """
    Lê st.secrets e injeta as chaves de API em os.environ.

    Por que injetar em os.environ E retornar os valores:
        - os.environ garante compatibilidade com SDKs que leem env vars
          diretamente (ex: huggingface_hub).
        - O dict de valores retornado por get_api_key() permite passar
          a chave explicitamente para ChatGroq/ChatOpenAI, evitando
          race conditions entre ciclos de re-execução do Streamlit.

    Returns:
        dict[str, bool]: Status de carregamento de cada chave.
    """
    status = {}

    for secret_key, env_var in _SECRETS_MAP.items():
        try:
            value = st.secrets[secret_key]
            # Injeta no ambiente (para SDKs que leem os.environ)
            os.environ[env_var] = value
            status[secret_key] = True
        except (KeyError, FileNotFoundError):
            status[secret_key] = False

    return status


def get_api_key(key_name: str) -> str | None:
    """
    Retorna o valor de uma chave de API lendo diretamente do st.secrets.

    Preferível a os.environ para passar explicitamente aos construtores
    dos SDKs (ChatGroq, ChatOpenAI), eliminando dependência de timing
    da injeção no ambiente.

    Args:
        key_name: Nome da chave (ex: "GROQ_API_KEY").

    Returns:
        str | None: Valor da chave, ou None se não configurada.
    """
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        # Fallback para os.environ (ex: variáveis definidas no sistema)
        return os.environ.get(key_name)


def get_available_providers(status: dict[str, bool]) -> list[str]:
    """
    Retorna lista de provedores disponíveis com base nas chaves carregadas.

    Útil para filtrar o seletor de provedor na sidebar, exibindo apenas
    os que têm chave configurada.

    Args:
        status: Retorno de load_api_keys().

    Returns:
        list[str]: Provedores disponíveis (ex: ["openai", "groq"]).
    """
    provider_requirements = {
        "openai":        "OPENAI_API_KEY",
        "groq":          "GROQ_API_KEY",
        "hf_hub":        "HUGGINGFACEHUB_API_TOKEN",
        "hf_serverless": "HUGGINGFACEHUB_API_TOKEN",  # mesma chave do hf_hub
    }

    return [
        provider
        for provider, key in provider_requirements.items()
        if status.get(key, False)
    ]
