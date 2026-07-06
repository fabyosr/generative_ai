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


# Mapeamento: chave no secrets.toml → variável de ambiente esperada pelo SDK
_SECRETS_MAP = {
    "OPENAI_API_KEY":          "OPENAI_API_KEY",
    "GROQ_API_KEY":            "GROQ_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN": "HUGGINGFACEHUB_API_TOKEN",
}


def load_api_keys() -> dict[str, bool]:
    """
    Lê st.secrets e injeta as chaves de API em os.environ.

    Para cada chave mapeada em _SECRETS_MAP:
        - Se presente no st.secrets → injeta em os.environ
        - Se ausente → ignora silenciosamente (sem exceção)

    Returns:
        dict[str, bool]: Status de carregamento de cada chave.
            Ex: {"OPENAI_API_KEY": True, "GROQ_API_KEY": False, ...}
    """
    status = {}

    for secret_key, env_var in _SECRETS_MAP.items():
        try:
            value = st.secrets[secret_key]
            os.environ[env_var] = value
            status[secret_key] = True
        except (KeyError, FileNotFoundError):
            # Chave não configurada — provedor correspondente indisponível
            status[secret_key] = False

    return status


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
        "openai": "OPENAI_API_KEY",
        "groq":   "GROQ_API_KEY",
        "hf_hub": "HUGGINGFACEHUB_API_TOKEN",
    }

    return [
        provider
        for provider, key in provider_requirements.items()
        if status.get(key, False)
    ]
