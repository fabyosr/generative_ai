"""
Modelos de domínio do projeto.

Contém as entidades e value objects que representam os conceitos centrais
do sistema, independentemente de como são obtidos (dual-encoder, RAG,
busca web) ou apresentados (Streamlit, API).

Não deve importar nada de `tools/`, `agents/`, `guardrails/` ou `web/` —
este módulo é o núcleo e não conhece a infraestrutura que o alimenta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClasseTerapeutica(str, Enum):
    """As 4 classes terapêuticas cobertas pelo modelo dual-encoder.

    Distribuição real das 6 plantas entre as classes (ver
    config/constants.py): 3 em CARDIOVASCULAR, 1 em cada uma das
    demais. Não assumir distribuição uniforme em nenhum cálculo
    (ex.: amostragem para avaliação, ver Fase 6).
    """

    CARDIOVASCULAR = "cardiovascular"
    ANTI_INFLAMATORIA = "anti_inflamatoria"
    DIGESTIVA = "digestiva"
    CALMANTE = "calmante"


class TipoMensagem(str, Enum):
    """Classificação da mensagem do usuário como um todo.

    Corresponde ao campo `tipo_mensagem` do schema de saída de
    `prompts/extrator_intencao.md`. Ver ADR-002 para o racional da
    extração determinística de uma única chamada.
    """

    CONSULTA_DOMINIO = "consulta_dominio"   # ao menos 1 solicitação (planta ou atributo)
    PERGUNTA_GERAL = "pergunta_geral"         # relacionada ao domínio, sem solicitação específica
    FORA_DE_ESCOPO = "fora_de_escopo"         # não relacionada a plantas medicinais


class TipoSolicitacao(str, Enum):
    """Classificação de UMA solicitação dentro da lista `solicitacoes`
    extraída da mensagem — uma mensagem pode conter várias, cada uma
    com seu próprio tipo (ex.: "camomila e algo pra gases" tem uma
    solicitação de cada tipo).
    """

    BUSCA_POR_ATRIBUTO = "busca_por_atributo"   # ex.: "algo calmante" -> dual-encoder
    PLANTA_NOMEADA = "planta_nomeada"             # ex.: "camomila" -> checagem contra as 6


class OrigemInformacao(str, Enum):
    """De onde veio o conteúdo textual usado para compor a resposta."""

    RAG = "rag"
    WIKIPEDIA = "wikipedia"
    TAVILY = "tavily"


@dataclass(frozen=True)
class Planta:
    """Representa uma das 6 plantas cobertas pelo dual-encoder e pelo RAG.

    Esta entidade corresponde à lista canônica definida em
    `config/constants.py` — não deve ser instanciada livremente a partir
    de texto sem passar pelo matching determinístico (ver
    `core/use_cases.py::identificar_planta`).
    """

    nome_popular: str
    nome_cientifico: str
    classe_terapeutica: ClasseTerapeutica
    sinonimos: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResultadoIdentificacao:
    """Saída da consulta ao dual-encoder: planta candidata + score."""

    planta: Planta
    score_similaridade: float
    imagem_url_ou_ref: str


@dataclass(frozen=True)
class TrechoRecuperado:
    """Um trecho de conteúdo recuperado (RAG, Wikipedia ou Tavily)."""

    texto: str
    origem: OrigemInformacao
    fonte_citacao: str
    score_relevancia: float | None = None


@dataclass(frozen=True)
class Solicitacao:
    """Um item da lista `solicitacoes` extraída pelo prompt de
    extração de intenção — uma mensagem pode gerar várias.

    `valor` é o texto bruto extraído (nome de planta ou descrição de
    atributo), ainda não normalizado nem validado contra a base
    canônica — isso acontece na etapa seguinte, determinística
    (`core/use_cases.py::identificar_planta`, aplicada por item).
    """

    tipo: TipoSolicitacao
    valor: str


@dataclass
class ResultadoSolicitacao:
    """Resultado do processamento de UMA solicitação: identificação
    (via dual-encoder, para ambos os tipos) e o conteúdo textual
    recuperado (RAG ou fallback).

    Tanto solicitações "planta_nomeada" quanto "busca_por_atributo"
    passam pelo dual-encoder para obter uma imagem — mas o SCORE só tem
    significado epistêmico para "busca_por_atributo" (onde a identidade
    é genuinamente inferida). Para "planta_nomeada", a identidade já foi
    resolvida de forma determinística por `identificar_planta` antes
    desta etapa; a chamada ao dual-encoder serve apenas para buscar uma
    imagem representativa, não para confirmar nada. Verificação empírica
    mostrou que o score para nome canônico não é necessariamente alto
    (~0.4 observado antes de calibração/possível descompasso de
    distribuição entre nome nu e texto de treino) — por isso
    `verificar_confianca` e o aviso de baixa confiança devem ser
    aplicados SOMENTE a resultados de "busca_por_atributo", nunca a
    "planta_nomeada" (ver docs/adr).
    """

    solicitacao: Solicitacao
    resultado_identificacao: ResultadoIdentificacao | None
    planta_identificada: Planta | None
    trechos: list[TrechoRecuperado]
    abaixo_do_limiar: bool = False
    limiar_usado: float | None = None
    """`abaixo_do_limiar`/`limiar_usado` só têm significado quando
    `solicitacao.tipo == TipoSolicitacao.BUSCA_POR_ATRIBUTO` (ver
    docstring da classe e ADR-004). Para "planta_nomeada", permanecem
    nos valores padrão (False/None) — não são calculados nem usados."""


@dataclass(frozen=True)
class RespostaLLM:
    """Resposta de uma chamada de LLM, incluindo metadados de uso reais
    (não estimados) — vindos diretamente da API do provedor.

    `LLMClient.gerar()` retorna isto, não uma string simples, para que
    tokens de entrada/saída fiquem disponíveis para observabilidade
    (aba de observabilidade da interface e/ou LangSmith) sem precisar
    estimar com um tokenizador à parte.

    Vive em core/models.py (não em tools/llm.py) pelo mesmo motivo de
    ResultadoIdentificacao/TrechoRecuperado: é um contrato compartilhado
    entre core/ (que define o Protocol LLMClient) e tools/ (que o
    implementa) — tools/ já importa de core/models.py normalmente,
    nunca o contrário.

    `tokens_entrada`/`tokens_saida` podem ser None se o provedor não
    retornar essa informação (não deveria acontecer com os provedores
    já integrados, mas o código que consome isto deve tratar a
    ausência, não presumir presença).
    """

    texto: str
    tokens_entrada: int | None
    tokens_saida: int | None
    modelo: str


@dataclass
class ImagemResposta:
    """Componente de imagem do envelope de resposta.

    `abaixo_do_limiar` e `limiar_usado` são registrados por turno,
    nunca globalmente — o limiar é ajustável pelo usuário via sidebar,
    então cada resposta deve carregar o valor vigente no momento em
    que foi gerada (ver decisão de rastreabilidade em docs/adr).
    """

    presente: bool
    url_ou_ref: str | None
    score_similaridade: float | None
    planta_identificada: str | None
    abaixo_do_limiar: bool
    limiar_usado: float


@dataclass
class AudioResposta:
    """Componente de áudio (TTS) do envelope de resposta.

    O áudio é gerado a partir do `texto_resposta` já filtrado por
    guardrails e já com o aviso de baixa confiança injetado — nunca a
    partir de texto bruto do LLM. Não existe conteúdo de áudio
    independente do texto.

    `ref` carrega os bytes de um WAV em memória (não um caminho de
    arquivo) — consistente com tools/vendor/voice_engine.py::synthesize,
    que já retorna bytes prontos para uso direto (ex.: st.audio(bytes)
    no Streamlit), sem necessidade de persistir em disco.
    """

    presente: bool
    ref: bytes | None


@dataclass
class FonteCitada:
    origem: OrigemInformacao
    citacao: str


@dataclass
class EnvelopeResposta:
    """Contrato único de saída do sistema: texto + imagens + áudio + metadados.

    `imagens` e `avisos_confianca` são LISTAS — uma mensagem pode gerar
    múltiplas solicitações (ex.: "camomila e algo pra gases"), cada uma
    com sua própria consulta ao dual-encoder e seu próprio score de
    confiança. `texto_resposta` e `audio`, por outro lado, permanecem
    ÚNICOS: a composição final sintetiza todas as solicitações em UMA
    resposta coerente (uma chamada a resposta_final.md), não uma
    resposta por solicitação — e o áudio narra esse texto único.

    Este é o objeto que atravessa guardrails de saída, é renderizado na
    interface e serve de insumo para o TTS. Qualquer novo canal de saída
    futuro deve consumir este envelope, não recriar sua própria lógica.
    """

    texto_resposta: str
    imagens: list[ImagemResposta]
    audio: AudioResposta
    fontes: list[FonteCitada]
    avisos_confianca: list[str] = field(default_factory=list)
