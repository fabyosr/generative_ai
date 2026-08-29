"""
RAG restrito às 6 plantas curadas manualmente (ADR-002).

DECISÃO ARQUITETURAL: como o roteamento até o RAG já é 100%
determinístico (identificar_planta resolve a identidade da planta ANTES
de qualquer chamada aqui — seja por nome direto ou por busca de atributo
via dual-encoder), este módulo NÃO faz busca semântica/vetorial. Vira
uma consulta estruturada por chave: nome da planta -> lista de trechos
curados. Não há embeddings, não há score de relevância, não há vector
store — decisão consciente dado o design de roteamento, não uma
limitação de escopo. Ver comentário em config/constants.py sobre a
remoção de LIMIAR_RAG_PADRAO.

FORMATO ESPERADO DO ARQUIVO CURADO (Markdown, um único arquivo):

    # Nome Popular Da Planta

    ## Nome da seção (livre — "Uso tradicional", "Contraindicações" etc.)
    Texto da seção...

    ## Outra seção
    Texto...

    # Próxima Planta
    ...

Cada cabeçalho de nível 1 (`# `) inicia uma planta — o texto do
cabeçalho é comparado (via `normalizar_nome`) contra
`Planta.nome_popular` das 6 plantas canônicas. Cada cabeçalho de nível 2
(`## `) dentro de uma planta vira UM TrechoRecuperado independente,
preservando a estratégia de chunking semântico por seção definida
anteriormente. O nome da seção não é validado contra uma lista fixa —
qualquer texto após `## ` é aceito como rótulo da seção.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.models import OrigemInformacao, Planta, TrechoRecuperado
from core.use_cases import normalizar_nome

_PADRAO_PLANTA = re.compile(r"(?m)^# ")
_PADRAO_SECAO = re.compile(r"(?m)^## ")

# Captura o diretório onde o rag.py está (prj/src/tools)
CURRENT_DIR = Path(__file__).resolve().parent


class RAGService:
    """Consulta estruturada sobre a base curada das 6 plantas.

    Implementa implicitamente o Protocol `RAGClient` de
    `core/use_cases.py` (duck typing).
    """

    def __init__(self, base_curada_path: str | Path) -> None:
        """
        Args:
            base_curada_path: caminho do arquivo Markdown curado (ver
                config/settings.py::RAG_BASE_CURADA_PATH).
        """
        self._base_curada_path = CURRENT_DIR.parent/base_curada_path
        self._chunks_por_planta: dict[str, list[TrechoRecuperado]] | None = None

    def carregar(self) -> None:
        """Lê e parseia o arquivo Markdown curado, populando o índice
        em memória. Chamado uma vez, na inicialização da aplicação (ver
        app.py) — não em cada consulta.

        Levanta FileNotFoundError se o arquivo não existir, com
        mensagem indicando o caminho esperado (configurável via
        RAG_BASE_CURADA_PATH no .env/st.secrets).
        """
        if not self._base_curada_path.exists():
            raise FileNotFoundError(
                f"Base curada do RAG não encontrada em "
                f"'{self._base_curada_path}'. Configure "
                f"RAG_BASE_CURADA_PATH ou coloque o arquivo Markdown "
                f"curado nesse caminho."
            )
        texto = self._base_curada_path.read_text(encoding="utf-8")
        self._chunks_por_planta = self._parsear(texto)

    def _parsear(self, texto: str) -> dict[str, list[TrechoRecuperado]]:
        resultado: dict[str, list[TrechoRecuperado]] = {}

        blocos_planta = _PADRAO_PLANTA.split(texto)
        for bloco in blocos_planta:
            bloco = bloco.strip()
            if not bloco:
                continue  # texto antes do primeiro "# ", se houver

            primeira_linha, _, corpo = bloco.partition("\n")
            nome_planta = primeira_linha.strip()
            if not nome_planta:
                continue

            chave = normalizar_nome(nome_planta)
            chunks = self._parsear_secoes(corpo, nome_planta)
            resultado[chave] = chunks

        return resultado

    def _parsear_secoes(
        self, corpo: str, nome_planta: str
    ) -> list[TrechoRecuperado]:
        chunks: list[TrechoRecuperado] = []
        blocos_secao = _PADRAO_SECAO.split(corpo)

        for bloco in blocos_secao:
            bloco = bloco.strip()
            if not bloco:
                continue  # texto solto antes da primeira "## ", se houver

            primeira_linha, _, texto_secao = bloco.partition("\n")
            nome_secao = primeira_linha.strip()
            texto_secao = texto_secao.strip()

            if not nome_secao or not texto_secao:
                continue

            chunks.append(
                TrechoRecuperado(
                    texto=texto_secao,
                    origem=OrigemInformacao.RAG,
                    fonte_citacao=f"Base curada — {nome_planta} — {nome_secao}",
                    score_relevancia=None,
                )
            )

        return chunks

    def consultar(self, planta: Planta) -> list[TrechoRecuperado]:
        """Retorna todos os trechos curados da planta já identificada.

        Não faz busca nem ranqueamento — a identidade já foi resolvida
        antes desta chamada (ver docstring do módulo). Levanta
        ValueError se a planta não tiver seção correspondente no
        arquivo curado — para as 6 plantas canônicas isso indicaria um
        arquivo incompleto, um erro de dado a corrigir, não um caso a
        silenciar com uma lista vazia.
        """
        if self._chunks_por_planta is None:
            raise RuntimeError(
                "RAGService.carregar() precisa ser chamado antes de consultar()."
            )

        chave = normalizar_nome(planta.nome_popular)
        chunks = self._chunks_por_planta.get(chave)

        if chunks is None:
            raise ValueError(
                f"Nenhuma seção encontrada para '{planta.nome_popular}' em "
                f"'{self._base_curada_path}'. Verifique se o arquivo contém "
                f"o cabeçalho '# {planta.nome_popular}' (comparação é "
                f"normalizada — acentos/caixa/hífen não importam, mas o "
                f"nome em si precisa corresponder)."
            )

        return chunks
