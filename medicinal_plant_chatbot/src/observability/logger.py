"""
Logging estruturado do pipeline, usado pela aba de observabilidade da
interface web.

Cada nó relevante do grafo (agents/graph.py) deve logar, no mínimo:
    - ferramenta acionada (dual-encoder, RAG, Wikipedia, Tavily, TTS);
    - score e limiar_usado (quando aplicável) — logados juntos, nunca
      separados, para preservar rastreabilidade (o limiar é ajustável
      pelo usuário e pode mudar entre turnos);
    - latência da etapa;
    - tokens consumidos (quando a etapa envolve LLM);
    - erros, quando ocorrerem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EventoObservabilidade:
    """Representa um evento estruturado de uma etapa do pipeline."""

    etapa: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sessao_id: str | None = None
    numero_turno_sessao: int | None = None
    ferramenta_acionada: str | None = None
    score: float | None = None
    limiar_usado: float | None = None
    latencia_ms: float | None = None
    tokens_consumidos: int | None = None
    erro: str | None = None
    metadados_extra: dict = field(default_factory=dict)


def get_logger(nome: str) -> logging.Logger:
    """Retorna um logger configurado para o módulo chamador.

    TODO: configurar formatter estruturado (JSON) e handler apropriado
    para o ambiente (stdout local, integração com serviço de observabilidade
    em produção, se aplicável).
    """
    logger = logging.getLogger(nome)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def registrar_evento(
    logger: logging.Logger,
    evento: EventoObservabilidade,
    historico: list[EventoObservabilidade] | None = None,
) -> None:
    """Registra um EventoObservabilidade de forma estruturada.

    Duas coisas acontecem aqui, servindo às duas camadas de
    observabilidade do projeto:

    1. Sempre grava um log estruturado via `logging` — é o rastro
       técnico de baixo nível (stdout/arquivo, conforme configuração do
       logger). Complementar ao LangSmith (que rastreia automaticamente
       as chamadas do LangGraph via variáveis de ambiente, sem precisar
       desta função) — este log cobre metadados específicos do domínio
       (score, limiar, planta) que o LangSmith não tem como saber.

    2. Se `historico` for fornecido — tipicamente uma lista viva em
       `st.session_state`, mantida por `app.py` — o evento também é
       anexado a ela. Essa lista é a fonte de dados da aba de
       observabilidade da interface: uma visão simplificada, pensada
       para o usuário final, que normalmente não tem acesso nem
       conhecimento do LangSmith.

    Erros (evento.erro preenchido) são logados como warning, não como
    error — a ocorrência de um erro tratado (ex.: fallback acionado) faz
    parte do funcionamento normal do sistema, não é uma falha do
    logger em si.
    """
    partes = [
        f"sessao_id={evento.sessao_id}",
        f"turno={evento.numero_turno_sessao}",
        f"etapa={evento.etapa}",
        f"ferramenta={evento.ferramenta_acionada}",
        f"score={evento.score}",
        f"limiar_usado={evento.limiar_usado}",
        f"latencia_ms={evento.latencia_ms}",
        f"tokens_consumidos={evento.tokens_consumidos}",
    ]
    if evento.erro:
        partes.append(f"erro={evento.erro}")
    if evento.metadados_extra:
        partes.append(f"metadados_extra={evento.metadados_extra}")

    mensagem = " ".join(partes)

    if evento.erro:
        logger.warning(mensagem)
    else:
        logger.info(mensagem)

    if historico is not None:
        historico.append(evento)
