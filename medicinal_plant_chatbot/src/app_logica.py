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

import pandas as pd

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

def construir_tabela_observabilidade(
    mensagens: list[dict],
    eventos: list,
    precos_por_modelo: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Agrega eventos (granularidade por etapa) e mensagens (conteúdo)
    em uma linha por turno, para exibição tabular.

    Só gera linha para turnos com eventos associados — a mensagem de
    boas-vindas inicial (sem numero_turno_sessao) é ignorada.
    """
    eventos_por_turno: dict[int, list] = {}
    for evento in eventos:
        turno = getattr(evento, "numero_turno_sessao", None)
        if turno is None:
            continue
        eventos_por_turno.setdefault(turno, []).append(evento)

    pares: list[tuple[str, str]] = []
    pendente_usuario: str | None = None
    for msg in mensagens:
        if msg["role"] == "user":
            pendente_usuario = msg["content"]
        elif msg["role"] == "assistant" and pendente_usuario is not None:
            pares.append((pendente_usuario, msg["content"]))
            pendente_usuario = None

    linhas = []
    for i, turno in enumerate(sorted(eventos_por_turno)):
        tokens_entrada = tokens_saida = 0
        latencia_total_ms = 0.0
        modelo = None
        guardrail_escopo_bloqueou = guardrail_saida_bloqueou = False
        clareza_mensagem = None

        for ev in eventos_por_turno[turno]:
            latencia_total_ms += ev.latencia_ms or 0.0
            extra = ev.metadados_extra or {}
            tokens_entrada += extra.get("tokens_entrada") or 0
            tokens_saida += extra.get("tokens_saida") or 0
            modelo = extra.get("modelo") or modelo
            if ev.etapa == "verificar_escopo" and extra.get("dentro_do_escopo") is False:
                guardrail_escopo_bloqueou = True
            if ev.etapa == "avaliar_saida" and extra.get("aprovado") is False:
                guardrail_saida_bloqueou = True
            if ev.etapa == "extrair_intencao":
                clareza_mensagem = extra.get("clareza_mensagem")

        preco_in, preco_out = precos_por_modelo.get(modelo, (None, None))
        custo_usd = (
            (tokens_entrada / 1_000_000) * preco_in + (tokens_saida / 1_000_000) * preco_out
            if preco_in is not None else None
        )
        prompt_usuario, resposta_llm = pares[i] if i < len(pares) else ("", "")

        linhas.append({
            "turno": turno,
            "prompt_usuario": prompt_usuario,
            "resposta_llm": resposta_llm,
            "modelo": modelo,
            "clareza_mensagem": clareza_mensagem,
            "escopo_bloqueou": guardrail_escopo_bloqueou,
            "saida_bloqueou": guardrail_saida_bloqueou,
            "tokens_entrada": tokens_entrada,
            "tokens_saida": tokens_saida,
            "tokens_totais": tokens_entrada + tokens_saida,
            "latencia_s": round(latencia_total_ms / 1000, 2),
            "custo_usd": round(custo_usd, 6) if custo_usd is not None else None,
        })

    return pd.DataFrame(linhas)