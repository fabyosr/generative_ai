"""
config.py
=========
Centraliza todas as constantes, personalidades e parâmetros de custo.
Altere aqui sem tocar em lógica ou UI.
"""

# ---------------------------------------------------------------------------
# Personalidades disponíveis para o assistente
# ---------------------------------------------------------------------------
PERSONALITIES: dict[str, str] = {
    "Técnico 💻": (
        "You are a highly technical assistant. "
        "CRITICAL: Reason in English, answer in Brazilian Portuguese."
    ),
    "Comercial 💼": (
        "You are a business-oriented assistant. "
        "CRITICAL: Reason in English, answer in Brazilian Portuguese."
    ),
    "Descontraído ☕": (
        "You are a friendly companion. "
        "CRITICAL: Reason in English, answer in Brazilian Portuguese."
    ),
}

# ---------------------------------------------------------------------------
# Mapeamento de provedor → label legível
# ---------------------------------------------------------------------------
LLM_PROVIDERS: dict[str, str] = {
    "openai":       "OpenAI (GPT-4o-mini)",
    "ollama":       "Ollama (Phi-3)",
    "hf_endpoint":  "HuggingFace (Llama 3)",
}

# ---------------------------------------------------------------------------
# Preços de referência (USD por 1 M de tokens) — ajuste conforme tabela atual
# ---------------------------------------------------------------------------
PRICING = {
    "input_per_million":  0.15,   # gpt-4o-mini input
    "output_per_million": 0.60,   # gpt-4o-mini output
}

# ---------------------------------------------------------------------------
# Valores iniciais do dicionário de métricas de resposta
# ---------------------------------------------------------------------------
DEFAULT_METRICS: dict = {
    "last_input_tokens":  0,
    "last_output_tokens": 0,
    "latency":            0.0,
    "tokens_per_sec":     0.0,
    "finish_reason":      "N/A",
    "system_fingerprint": "N/A",
    "reasoning_tokens":   0,
}
