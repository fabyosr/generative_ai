"""
Guardrail de escopo: verifica se a mensagem está dentro do domínio de
plantas medicinais.

Mantido como camada SEPARADA e POSTERIOR no pipeline (não embutida na
extração de intenção) por decisão explícita de projeto, para fins
acadêmicos de demonstrar guardrails como componente independente,
testável e auditável isoladamente — mesmo que `tipo_mensagem` já tenha
vindo como "fora_de_escopo" da extração de intenção, este guardrail
reavalia do zero (defesa em profundidade: um erro isolado de
classificação na extração não é o único ponto de falha do sistema).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.models import RespostaLLM
from core.use_cases import parsear_json_llm
from prompts.loader import carregar_prompt


class LLMClient(Protocol):
    """Mesmo contrato de core/use_cases.py::LLMClient — repetido aqui
    (não importado de lá) para manter guardrails/ sem depender de
    core/use_cases.py além do estritamente necessário (parsear_json_llm)."""

    def gerar(self, prompt: str, **kwargs: object) -> RespostaLLM: ...


@dataclass
class ResultadoGuardrailEscopo:
    dentro_do_escopo: bool
    resposta_llm: RespostaLLM
    """Metadados de uso reais da chamada (tokens, modelo) — para quem
    for logar observabilidade (ver agents/graph.py)."""
    mensagem_redirecionamento: str | None = None


def verificar_escopo(
    mensagem_usuario: str,
    llm: LLMClient,
    historico_resumido: str = "",
) -> ResultadoGuardrailEscopo:
    """Avalia se a mensagem do usuário pertence ao domínio de plantas
    medicinais.

    Quando `dentro_do_escopo=False`, `mensagem_redirecionamento` sempre
    vem preenchido (contrato do prompt guardrail_escopo.md) — nunca um
    bloqueio silencioso, conforme princípio de fallback seguro do
    projeto.

    Levanta ValueError se a resposta do LLM não tiver o formato
    esperado, incluindo o caso de "fora do escopo sem mensagem de
    redirecionamento" — falha explícita é preferível a uma aprovação
    silenciosa por padrão, o lado errado para errar em um guardrail de
    segurança.
    """
    prompt = (
        carregar_prompt("guardrail_escopo")
        .replace("{{mensagem_usuario}}", mensagem_usuario)
        .replace("{{historico_resumido}}", historico_resumido or "(nenhum)")
    )

    resposta_llm = llm.gerar(prompt, temperature=0)
    dados = parsear_json_llm(resposta_llm.texto)

    if "dentro_do_escopo" not in dados:
        raise ValueError(
            f"Resposta do guardrail de escopo sem campo "
            f"'dentro_do_escopo': {dados!r}"
        )

    dentro_do_escopo = bool(dados["dentro_do_escopo"])
    mensagem_redirecionamento = dados.get("mensagem_redirecionamento")

    if not dentro_do_escopo and not mensagem_redirecionamento:
        raise ValueError(
            "Guardrail de escopo classificou a mensagem como fora do "
            "escopo, mas não forneceu mensagem_redirecionamento — "
            "viola o contrato de prompts/guardrail_escopo.md."
        )

    return ResultadoGuardrailEscopo(
        dentro_do_escopo=dentro_do_escopo,
        resposta_llm=resposta_llm,
        mensagem_redirecionamento=(
            mensagem_redirecionamento if not dentro_do_escopo else None
        ),
    )
