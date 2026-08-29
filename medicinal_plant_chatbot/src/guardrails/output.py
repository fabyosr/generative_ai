"""
Guardrails de saída: validação e filtragem do texto gerado pelo LLM
antes de ser exibido, narrado (TTS) ou usado para montar o envelope de
resposta final.

DESIGN: uma única função (`avaliar_saida`) faz UMA chamada de LLM que
cobre tanto a aprovação de segurança (diagnóstico, prescrição,
vazamento de prompt) quanto o score de groundedness — não duas
chamadas separadas. O stub original deste módulo tinha duas funções
públicas (`validar_saida`, `verificar_groundedness`); unificadas aqui
porque prompts/guardrail_saida.md já foi desenhado para responder as
duas coisas em uma única chamada — manter duas funções públicas
independentes dobraria custo/latência sempre que ambas fossem
necessárias, o caso comum.

Deve rodar DEPOIS de core/use_cases.py::injetar_aviso_confianca (o
aviso já injetado também passa pela validação) e ANTES do texto ser
enviado ao TTS ou exibido na interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.models import RespostaLLM, TrechoRecuperado
from core.use_cases import formatar_trechos_para_prompt, parsear_json_llm
from prompts.loader import carregar_prompt


class LLMClient(Protocol):
    def gerar(self, prompt: str, **kwargs: object) -> RespostaLLM: ...


@dataclass
class ResultadoAvaliacaoSaida:
    aprovado: bool
    """Cobre estritamente violações de SEGURANÇA (diagnóstico,
    prescrição, vazamento de prompt) — julgamento binário, separado do
    groundedness_score (ver prompts/guardrail_saida.md)."""
    motivo_bloqueio: str | None
    groundedness_score: float
    """Sinal numérico contínuo. A decisão de qual limiar de
    groundedness bloqueia uma resposta é tomada pelo código chamador
    (não por este módulo) — mesmo padrão dos limiares de confiança do
    dual-encoder e do RAG: parâmetro do sistema, não julgamento do LLM
    a cada chamada."""
    resposta_llm: RespostaLLM
    """Metadados de uso reais da chamada (tokens, modelo) — para quem
    for logar observabilidade (ver agents/graph.py)."""
    trechos_nao_fundamentados: list[str] = field(default_factory=list)


def avaliar_saida(
    texto_resposta: str,
    fontes: list[TrechoRecuperado],
    llm: LLMClient,
    plantas_em_foco: str = "nenhuma",
) -> ResultadoAvaliacaoSaida:
    """Faz UMA chamada de LLM cobrindo aprovação de segurança e score
    de groundedness — ver docstring do módulo.

    Levanta ValueError se a resposta do LLM não tiver o formato
    esperado — falha explícita, não uma aprovação silenciosa por
    padrão (o lado errado para errar em um guardrail de segurança).
    """
    prompt = (
        carregar_prompt("guardrail_saida")
        .replace("{{texto_resposta_gerado}}", texto_resposta)
        .replace("{{trechos_recuperados}}", formatar_trechos_para_prompt(fontes))
        .replace("{{plantas_em_foco}}", plantas_em_foco)
    )

    resposta_llm = llm.gerar(prompt, temperature=0)
    dados = parsear_json_llm(resposta_llm.texto)

    campos_obrigatorios = {"aprovado", "groundedness_score"}
    faltantes = campos_obrigatorios - set(dados)
    if faltantes:
        raise ValueError(
            f"Resposta do guardrail de saída incompleta, faltando "
            f"campos {faltantes}: {dados!r}"
        )

    return ResultadoAvaliacaoSaida(
        aprovado=bool(dados["aprovado"]),
        motivo_bloqueio=dados.get("motivo_bloqueio"),
        groundedness_score=float(dados["groundedness_score"]),
        resposta_llm=resposta_llm,
        trechos_nao_fundamentados=list(dados.get("trechos_nao_fundamentados", [])),
    )
