"""
Adapter entre BotanicalSearchEngine (código já treinado, ver
tools/vendor/botanical_search.py) e o contrato DualEncoderClient
esperado por core/use_cases.py.

O conteúdo de botanical_search.py NÃO é alterado — este arquivo só
adapta sua API (BotanicalSearchEngine.search_by_text) ao contrato usado
pelo resto do projeto (ResultadoIdentificacao), com duas responsabilidades
extras: mapear o nome de planta retornado para nossa entidade Planta
canônica, e falhar de forma clara se esse mapeamento não bater.

REAPROVEITADO PARA AMBOS OS TIPOS DE SOLICITAÇÃO, mas com papéis
diferentes (ver docs/adr): para "busca_por_atributo", o score é
epistemicamente relevante (mede confiança de identificação a partir de
uma descrição) e alimenta `verificar_confianca`. Para "planta_nomeada",
a identidade já foi resolvida deterministicamente por `identificar_planta`
antes desta chamada — o dual-encoder aqui serve SÓ para buscar uma
imagem representativa; o score retornado NÃO deve ser usado para gerar
aviso de baixa confiança (verificação empírica mostrou score baixo,
~0.4, mesmo para nome canônico — não é confiável como sinal de
confiança de identidade, seja por modelo em treino ou por descompasso
entre nome nu e o texto usado para construir os protótipos).

IMPORTANTE (cache): instanciar DualEncoderService carrega BERT +
ResNet-50 do zero (BotanicalSearchEngine não tem cache interno, ao
contrário de tools/vendor/voice_engine.py). Manter uma única instância
viva entre reruns do Streamlit é responsabilidade de quem compõe a
aplicação (app.py, via st.cache_resource) — este adapter não importa
Streamlit, para manter tools/ desacoplado do framework de UI.
"""

from __future__ import annotations

from core.models import Planta, ResultadoIdentificacao
from core.use_cases import normalizar_nome
from tools.vendor.botanical_search import BotanicalSearchEngine


class DualEncoderService:
    """Client de acesso ao modelo dual-encoder treinado
    (BotanicalSearchEngine, ver tools/vendor/botanical_search.py).

    Implementa implicitamente o Protocol `DualEncoderClient` definido em
    `core/use_cases.py` (duck typing — sem necessidade de herança formal).
    """

    def __init__(
        self,
        plantas_conhecidas: tuple[Planta, ...],
        auto_download: bool = True,
    ) -> None:
        """
        Args:
            plantas_conhecidas: lista canônica das plantas (ver
                config/constants.py::PLANTAS_CONHECIDAS), usada para
                mapear o nome retornado pelo motor de busca às nossas
                entidades Planta.
            auto_download: repassado a BotanicalSearchEngine — baixa
                pesos/protótipos do Google Drive na primeira execução,
                se ainda não estiverem em disco.
        """
        self._engine = BotanicalSearchEngine(auto_download=auto_download)
        self._plantas_por_nome_normalizado = {
            normalizar_nome(p.nome_popular): p for p in plantas_conhecidas
        }
        self._plantas_por_nome_normalizado.update(
            {normalizar_nome(p.nome_cientifico): p for p in plantas_conhecidas}
        )

    def buscar(self, texto: str) -> ResultadoIdentificacao:
        """Codifica `texto`, compara contra os protótipos de imagem das
        6 plantas e retorna a planta mais similar + score.

        Não aplica nenhum limiar — retorna sempre o melhor candidato,
        mesmo com score baixo. A decisão de exibir aviso de baixa
        confiança é feita em camada superior (ver core/use_cases.py).

        Levanta ValueError se o nome retornado por BotanicalSearchEngine
        não corresponder a nenhuma planta canônica — isso indicaria uma
        divergência entre metadata.csv e config/constants.py que precisa
        ser corrigida, não silenciada.
        """
        resultados = self._engine.search_by_text(texto, top_k=1)
        if not resultados:
            raise RuntimeError(
                "BotanicalSearchEngine não retornou nenhum resultado para a "
                f"consulta '{texto}'."
            )

        melhor = resultados[0]
        planta = self._mapear_planta(melhor["plant_name"])

        return ResultadoIdentificacao(
            planta=planta,
            score_similaridade=float(melhor["similarity"]),
            imagem_url_ou_ref=self._engine.get_image_path(melhor["image_filename"]),
        )

    def _mapear_planta(self, plant_name: str) -> Planta:
        chave = normalizar_nome(plant_name)
        planta = self._plantas_por_nome_normalizado.get(chave)
        if planta is None:
            raise ValueError(
                f"O dual-encoder retornou o nome '{plant_name}', que não "
                f"corresponde a nenhuma planta em "
                f"config/constants.py::PLANTAS_CONHECIDAS. Verifique se "
                f"metadata.csv usa a mesma convenção de nomes (popular ou "
                f"científico) cadastrada nas constantes."
            )
        return planta
