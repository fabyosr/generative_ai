"""
Busca externa de fallback: Wikipedia e Tavily.

Único ponto do projeto com uma interface (Protocol) formal para múltiplas
implementações — Wikipedia e Tavily de fato compartilham o mesmo contrato
de uso (fallback hierárquico) e são chamadas de forma intercambiável pelo
grafo. Ver ADR-003 (arquitetura modular simplificada) para o racional de
por que as demais integrações NÃO têm essa abstração.

Design: ambos os serviços recebem o texto de consulta exatamente como
passado pelo chamador — não enriquecem nem reescrevem a query
internamente (ex.: não acrescentam "planta medicinal" automaticamente).
Essa decisão pertence à camada de orquestração
(core/use_cases.py::recuperar_conhecimento_fallback), não a este adapter,
para manter tools/ como camada burra, sem lógica de negócio escondida.

Ambos falham de forma "suave" (retornam lista vazia) em caso de erro de
rede ou resultado não encontrado, registrando um aviso via observability
— diferente de tools/rag.py, que falha alto para as 6 plantas canônicas.
A assimetria é intencional: ausência de resultado externo é uma condição
normal e esperada de um fallback sobre fontes não controladas; ausência
de conteúdo para uma das 6 plantas curadas indicaria um dado incompleto,
não uma variação normal.
"""

from __future__ import annotations

from typing import Protocol

import wikipedia
from tavily import TavilyClient

from core.models import OrigemInformacao, TrechoRecuperado
from observability.logger import get_logger

_logger = get_logger(__name__)


class SearchClient(Protocol):
    """Contrato comum para qualquer fonte de busca de fallback."""

    def buscar(self, consulta: str) -> list[TrechoRecuperado]: ...


class WikipediaSearchService:
    """Implementação de busca via Wikipedia (fallback de 1ª ordem)."""

    def __init__(self, idioma: str = "pt") -> None:
        self._idioma = idioma

    def buscar(self, consulta: str) -> list[TrechoRecuperado]:
        """Busca na Wikipedia e retorna o resumo da página mais relevante,
        com citação da URL de origem.

        Retorna lista vazia (não levanta exceção) em caso de página não
        encontrada, ambiguidade sem opções, ou falha de rede — este é um
        fallback sobre fonte externa não controlada, e a ausência de
        resultado deve permitir que o pipeline siga para Tavily, não
        interromper o fluxo.
        """
        wikipedia.set_lang(self._idioma)

        try:
            titulos = wikipedia.search(consulta, results=1)
        except Exception as e:
            _logger.warning("Falha ao buscar no Wikipedia para %r: %s", consulta, e)
            return []

        if not titulos:
            return []

        try:
            pagina = wikipedia.page(titulos[0], auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if not e.options:
                return []
            try:
                pagina = wikipedia.page(e.options[0], auto_suggest=False)
            except Exception as e2:
                _logger.warning(
                    "Falha ao resolver desambiguação do Wikipedia para %r: %s",
                    consulta,
                    e2,
                )
                return []
        except wikipedia.exceptions.PageError:
            return []
        except Exception as e:
            _logger.warning(
                "Falha ao carregar página do Wikipedia para %r: %s", consulta, e
            )
            return []

        if not pagina.summary:
            return []

        return [
            TrechoRecuperado(
                texto=pagina.summary,
                origem=OrigemInformacao.WIKIPEDIA,
                fonte_citacao=pagina.url,
                score_relevancia=None,
            )
        ]


class TavilySearchService:
    """Implementação de busca via Tavily (fallback de 2ª ordem,
    acionado quando Wikipedia não retorna conteúdo suficiente)."""

    def __init__(self, api_key: str, max_resultados: int = 3) -> None:
        if not api_key:
            raise ValueError(
                "TavilySearchService requer uma api_key não vazia "
                "(ver config/settings.py e a variável TAVILY_API_KEY)."
            )
        self._client = TavilyClient(api_key=api_key)
        self._max_resultados = max_resultados

    def buscar(self, consulta: str) -> list[TrechoRecuperado]:
        """Busca via API do Tavily e retorna trechos com citação da URL
        de origem.

        Retorna lista vazia (não levanta exceção) em caso de falha de
        rede ou de API — mesma justificativa de WikipediaSearchService:
        este é o último elo do fallback, e uma falha aqui deve resultar
        em "sem conteúdo externo disponível", tratado pela camada de
        guardrails de saída, não em uma exceção não tratada.
        """
        try:
            resposta = self._client.search(
                query=consulta,
                max_results=self._max_resultados,
                search_depth="basic",
            )
        except Exception as e:
            _logger.warning("Falha ao buscar no Tavily para %r: %s", consulta, e)
            return []

        resultados = resposta.get("results", []) if resposta else []

        return [
            TrechoRecuperado(
                texto=r["content"],
                origem=OrigemInformacao.TAVILY,
                fonte_citacao=r.get("url", ""),
                score_relevancia=r.get("score"),
            )
            for r in resultados
            if r.get("content")
        ]
