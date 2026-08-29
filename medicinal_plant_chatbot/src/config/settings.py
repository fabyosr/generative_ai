"""
Configuração de runtime: segredos e parâmetros que variam por ambiente.

Resolução híbrida de segredos, compatível com três cenários sem exigir
decisão antecipada de plataforma de deploy:
    1. Desenvolvimento local: .env (via python-dotenv) -> os.environ
    2. Streamlit Cloud: st.secrets (checado primeiro)
    3. Google Cloud (Cloud Run, App Engine etc.): variáveis de ambiente
       nativas da plataforma -> os.environ (fallback natural)

Nenhum segredo real deve ser commitado. Ver `.env.example` para o
conjunto de variáveis esperadas e `.gitignore` para a exclusão de `.env`.

Constantes de domínio (lista de plantas, limiares padrão) NÃO ficam
aqui — ver `config/constants.py`. Este arquivo é só para o que varia
por ambiente/é sensível.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # no-op silencioso se .env não existir (ex.: em produção)


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Resolve um segredo em cascata: st.secrets (Streamlit Cloud)
    primeiro, depois variável de ambiente (.env local, GCP, ou qualquer
    outra plataforma que injete variáveis de ambiente nativamente).
    """
    try:
        import streamlit as st  # import local: evita dependência obrigatória
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError):
        pass  # streamlit ausente ou sem secrets.toml configurado
    return os.environ.get(key, default)


# --- Segredos / chaves de API ---
TAVILY_API_KEY: str | None = _get_secret("TAVILY_API_KEY")

# LLM: múltiplos provedores podem estar configurados simultaneamente —
# tools/llm.py::criar_llm_client decide qual usar, com base no provedor
# selecionado (padrão em LLM_PROVIDER, ou escolhido na sidebar em
# runtime). Nem toda chave precisa estar preenchida; só a do(s)
# provedor(es) que você pretende usar.
#
# Padrão: Groq (tier gratuito, já implementado e testado em tools/llm.py).
# HuggingFace foi cogitado mas não implementado ainda — endpoint
# compatível com OpenAI não verificado neste projeto.
LLM_PROVIDER: str = _get_secret("LLM_PROVIDER", default="groq")
ANTHROPIC_API_KEY: str | None = _get_secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = _get_secret("OPENAI_API_KEY")
GROQ_API_KEY: str | None = _get_secret("GROQ_API_KEY")
XAI_API_KEY: str | None = _get_secret("XAI_API_KEY")

# --- Observabilidade: LangSmith (rastro técnico completo, para depuração) ---
#
# Como já usamos LangGraph, ativar isso não exige nenhuma mudança de
# código no grafo — toda chamada de nó, ferramenta e LLM é rastreada
# automaticamente pelo LangSmith assim que estas variáveis existem em
# os.environ (é o próprio SDK do LangSmith/LangChain que as lê, não
# nosso código).
#
# IMPORTANTE: o SDK do LangSmith lê os.environ diretamente — não passa
# por _get_secret(). Isso funciona sozinho com .env local (python-dotenv
# já popula os.environ), mas NÃO funcionaria no Streamlit Cloud
# (st.secrets não vaza para os.environ automaticamente) sem a ponte
# explícita abaixo.
#
# Esta é uma camada COMPLEMENTAR à aba de observabilidade da interface
# (ver observability/logger.py::registrar_evento) — LangSmith é para
# depuração técnica; a aba do app é uma visão simplificada para o
# usuário final, que normalmente não tem acesso/conhecimento do
# LangSmith.
LANGSMITH_TRACING: str | None = _get_secret("LANGSMITH_TRACING", default="true")
LANGSMITH_API_KEY: str | None = _get_secret("LANGSMITH_API_KEY")
LANGSMITH_PROJECT: str | None = _get_secret(
    "LANGSMITH_PROJECT", default="plantas-medicinais-agent"
)

if LANGSMITH_API_KEY:
    # Ponte explícita: garante que o SDK do LangSmith enxergue estas
    # variáveis mesmo quando a origem foi st.secrets, não .env.
    os.environ.setdefault("LANGSMITH_TRACING", LANGSMITH_TRACING or "true")
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT or "default")

_CHAVES_POR_PROVEDOR: dict[str, str | None] = {
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
    "groq": GROQ_API_KEY,
    "xai": XAI_API_KEY,
}

# --- Caminhos de modelos / recursos locais ---
DUAL_ENCODER_MODEL_PATH: str | None = _get_secret("DUAL_ENCODER_MODEL_PATH")
RAG_BASE_CURADA_PATH: str = _get_secret(
    "RAG_BASE_CURADA_PATH", default="data/rag/plantas_curadas.md"
)
KOKORO_MODEL_PATH: str | None = _get_secret("KOKORO_MODEL_PATH")


def validar_configuracao_obrigatoria() -> None:
    """Levanta erro explícito e claro se alguma variável obrigatória
    não estiver configurada, em vez de falhar de forma obscura no meio
    de uma chamada a uma ferramenta.

    Só exige a chave do provedor de LLM_PROVIDER (não todas as chaves
    de LLM cadastradas) — os demais provedores ficam disponíveis para
    seleção em runtime (sidebar), sem serem obrigatórios de antemão.

    TODO: chamar esta função na inicialização de `app.py`.
    """
    faltantes = []

    if not _CHAVES_POR_PROVEDOR.get(LLM_PROVIDER):
        faltantes.append(
            f"chave para o provedor padrão LLM_PROVIDER='{LLM_PROVIDER}' "
            f"(configure {LLM_PROVIDER.upper()}_API_KEY)"
        )
    if not TAVILY_API_KEY:
        faltantes.append("TAVILY_API_KEY")
    if faltantes:
        raise ValueError(
            f"Variáveis de configuração obrigatórias não encontradas: "
            f"{', '.join(faltantes)}. Configure via .env (local), "
            f"st.secrets (Streamlit Cloud) ou variáveis de ambiente (GCP)."
        )
