"""
Lógica pura de app.py, extraída para um módulo próprio sem NENHUMA
dependência pesada (sem streamlit, sem torch, sem kokoro) — permite
testar isoladamente sem precisar stubar os módulos vendorizados
(tools/vendor/botanical_search.py, tools/vendor/voice_engine.py) que
app.py acaba arrastando via tools/dual_encoder.py e tools/tts.py.

Mantido separado de app.py por essa razão prática de testabilidade,
não por camada arquitetural formal.
"""

from __future__ import annotations


def calcular_historico_resumido(mensagens: list[dict], n_ultimas: int = 3) -> str:
    """Simplificação de MVP: concatena as últimas N trocas em vez de um
    resumo real gerado por LLM — suficiente para dar contexto ao
    extrator de intenção e ao guardrail de escopo, sem o custo de mais
    uma chamada de LLM só para resumir.
    """
    recentes = mensagens[-(n_ultimas * 2):]
    if not recentes:
        return ""
    return "\n".join(f"{m['role']}: {m['content']}" for m in recentes)


def calcular_novo_turno_topico(
    planta_anterior: str | None,
    planta_atual: str | None,
    turno_topico_atual: int,
) -> tuple[int, str | None]:
    """Decide o próximo `numero_turno_topico` (ver
    core/use_cases.py::calcular_estagio_conversa) comparando a planta
    em foco desta resposta com a da resposta anterior.

    Heurística simples de MVP: se a planta principal mudou, é um novo
    tópico (reseta para 1); se é a mesma (ou nenhuma planta em nenhum
    dos dois casos), incrementa. Não distingue múltiplas plantas na
    mesma resposta — usa apenas a primeira, documentado como
    simplificação conhecida.
    """
    if planta_atual != planta_anterior:
        return 1, planta_atual
    return turno_topico_atual + 1, planta_atual


def extrair_planta_principal(envelope) -> str | None:
    """Primeira planta identificada na resposta, ou None — usado por
    `calcular_novo_turno_topico` para detectar troca de assunto.

    Aceita qualquer objeto com atributo `.imagens` (lista de itens com
    `.presente` e `.planta_identificada`) — tipicamente um
    EnvelopeResposta, mas não importa o tipo diretamente, para manter
    este módulo livre de dependência de core/models.py.
    """
    for imagem in envelope.imagens:
        if imagem.presente and imagem.planta_identificada:
            return imagem.planta_identificada
    return None
