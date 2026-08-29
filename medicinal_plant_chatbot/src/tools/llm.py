"""
Client de LLM, com suporte a múltiplos provedores atrás do mesmo
contrato LLMClient (core/use_cases.py::LLMClient).

Dois adapters cobrem toda a lista de provedores considerada no projeto:

- AnthropicLLMClient: Claude, via SDK oficial da Anthropic — API com
  formato próprio, não compatível com a OpenAI.
- OpenAICompatibleLLMClient: cobre OpenAI, Groq e xAI — todos expõem uma
  API compatível com o SDK da OpenAI, mudando apenas base_url e a
  chave (confirmado nas respectivas documentações oficiais: Groq em
  https://api.groq.com/openai/v1, xAI em https://api.x.ai/v1). Evita
  reimplementar a mesma lógica de chamada três vezes.

HuggingFace Inference Providers também expõe um endpoint compatível,
mas não foi verificado/testado neste projeto — se for necessário,
seguiria o mesmo padrão de OpenAICompatibleLLMClient.

DESIGN — chave e modelo são parâmetros explícitos de construção, nunca
lidos deste módulo diretamente do ambiente:
  1. Permite usar modelos DIFERENTES para papéis diferentes do pipeline
     (ex.: um modelo mais rápido/barato para os classificadores —
     extrator de intenção, guardrails — e um mais capaz para a resposta
     final) sem nenhuma mudança de código, já que core/use_cases.py já
     injeta LLMClient por chamada, não globalmente.
  2. Permite que a chave venha de origens diferentes por provedor —
     .env/st.secrets (padrão) OU digitada pelo usuário na sidebar em
     runtime (para não expor uma chave paga sua a uso por terceiros, se
     o app for publicado). Essa decisão é de app.py — este módulo só
     recebe a string já resolvida.
"""

from __future__ import annotations

import anthropic
from openai import OpenAI

from core.models import RespostaLLM


class AnthropicLLMClient:
    """Client para modelos Claude via SDK oficial da Anthropic."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        if not api_key:
            raise ValueError("AnthropicLLMClient requer uma api_key não vazia.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def gerar(self, prompt: str, **kwargs: object) -> RespostaLLM:
        max_tokens = kwargs.pop("max_tokens", 1024)
        resposta = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return RespostaLLM(
            texto=resposta.content[0].text,
            tokens_entrada=getattr(resposta.usage, "input_tokens", None),
            tokens_saida=getattr(resposta.usage, "output_tokens", None),
            modelo=self._model,
        )


class OpenAICompatibleLLMClient:
    """Client para qualquer provedor com API compatível com a OpenAI.

    Cobre OpenAI (base_url padrão do SDK), Groq
    (https://api.groq.com/openai/v1) e xAI (https://api.x.ai/v1).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAICompatibleLLMClient requer uma api_key não vazia."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def gerar(self, prompt: str, **kwargs: object) -> RespostaLLM:
        resposta = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        uso = resposta.usage
        return RespostaLLM(
            texto=resposta.choices[0].message.content,
            tokens_entrada=getattr(uso, "prompt_tokens", None) if uso else None,
            tokens_saida=getattr(uso, "completion_tokens", None) if uso else None,
            modelo=self._model,
        )


# base_url conhecidos, para conveniência de quem monta o client (app.py).
# "openai": None -> usa o padrão do SDK oficial da OpenAI.
BASE_URLS_CONHECIDOS: dict[str, str | None] = {
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
}


def criar_llm_client(
    provedor: str,
    api_key: str,
    model: str,
) -> AnthropicLLMClient | OpenAICompatibleLLMClient:
    """Factory que resolve o adapter certo a partir do nome do provedor.

    Args:
        provedor: "anthropic", "openai", "groq" ou "xai".
        api_key: chave já resolvida (de .env/st.secrets ou digitada na
            sidebar) — este factory não decide a origem da chave.
        model: identificador do modelo específico do provedor.

    Levanta ValueError para provedor não suportado — falha explícita,
    não um fallback silencioso para outro provedor.
    """
    if provedor == "anthropic":
        return AnthropicLLMClient(api_key=api_key, model=model)

    if provedor in BASE_URLS_CONHECIDOS:
        return OpenAICompatibleLLMClient(
            api_key=api_key, model=model, base_url=BASE_URLS_CONHECIDOS[provedor]
        )

    raise ValueError(
        f"Provedor '{provedor}' não suportado. Opções: 'anthropic', "
        f"{', '.join(repr(p) for p in BASE_URLS_CONHECIDOS)}."
    )
