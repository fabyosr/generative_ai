"""
Casos de uso da aplicação.

Cada função aqui corresponde a um nó (ou a lógica interna de um nó) do
grafo LangGraph definido em `agents/graph.py`. O grafo é responsável por
orquestrar a *ordem* de execução; a lógica de negócio em si vive aqui,
para que possa ser testada isoladamente sem subir o LangGraph.

Convenção: cada caso de uso recebe suas dependências explicitamente por
parâmetro (injeção simples, sem framework de DI) — isso mantém as funções
testáveis com mocks simples de `unittest.mock`.

Nenhuma função aqui deve conter texto de prompt embutido — prompts vêm
de `prompts/loader.py`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Protocol

from config.constants import (
    LIMITE_TURNOS_DESENVOLVIMENTO,
    LIMITE_TURNOS_FORCADO,
    MAX_SOLICITACOES_POR_MENSAGEM,
    PLANTAS_CONHECIDAS,
)
from core.models import (
    AudioResposta,
    EnvelopeResposta,
    FonteCitada,
    ImagemResposta,
    Planta,
    RespostaLLM,
    ResultadoIdentificacao,
    ResultadoSolicitacao,
    Solicitacao,
    TipoMensagem,
    TipoSolicitacao,
    TrechoRecuperado,
)
from prompts.loader import carregar_prompt


class LLMClient(Protocol):
    """Contrato mínimo que qualquer provedor de LLM precisa cumprir
    para ser usado pelos casos de uso abaixo.

    Retorna RespostaLLM (não uma string simples) para que tokens de
    entrada/saída reais — vindos da própria API, não estimados — fiquem
    disponíveis para observabilidade (ver core/models.py::RespostaLLM).
    """

    def gerar(self, prompt: str, **kwargs: object) -> RespostaLLM: ...


def parsear_json_llm(texto: str) -> dict:
    """Extrai e parseia JSON da resposta de um LLM, tolerando cercas de
    bloco de código (```json ... ```) que o modelo às vezes adiciona
    mesmo quando instruído a não fazer isso.

    Pública (sem underscore) — reaproveitada por `guardrails/*.py`
    quando implementados, mesmo padrão de `normalizar_nome`.

    Levanta ValueError com a resposta bruta incluída na mensagem, para
    facilitar depuração, se o parsing falhar.
    """
    texto_limpo = texto.strip()
    if texto_limpo.startswith("```"):
        texto_limpo = re.sub(r"^```(?:json)?\s*", "", texto_limpo)
        texto_limpo = re.sub(r"\s*```\s*$", "", texto_limpo)
    try:
        return json.loads(texto_limpo)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Não foi possível parsear JSON da resposta do LLM: {texto!r}"
        ) from e


def extrair_intencao(
    mensagem_usuario: str,
    llm: LLMClient,
    historico_resumido: str = "",
) -> tuple[TipoMensagem, list[Solicitacao], bool, RespostaLLM]:
    """Classifica a mensagem e extrai as solicitações nela contidas.

    Retorna (tipo_mensagem, lista_de_solicitacoes, sinal_encerramento,
    resposta_llm) — o quarto elemento carrega os metadados de uso reais
    da chamada (tokens, modelo), para quem for logar observabilidade
    (ver agents/graph.py).

    Uma mensagem pode conter MÚLTIPLAS solicitações (ex.: "camomila e
    algo pra gases" -> duas solicitações, uma de cada tipo). O matching
    final de cada nome de planta contra a lista canônica das 6 plantas
    (config/constants.py) NÃO acontece aqui — é feito por item, de forma
    determinística, por `identificar_planta`.

    Usa saída estruturada do LLM (ex.: JSON/function calling), não um loop
    ReAct — ver ADR-002 para o racional dessa escolha.

    Itens malformados na lista `solicitacoes` retornada pelo LLM são
    ignorados individualmente (não derrubam a extração inteira) — um
    único item ruim não deveria impedir o processamento dos demais.
    """
    lista_plantas = "\n".join(
        f"- {p.nome_popular} ({p.nome_cientifico})" for p in PLANTAS_CONHECIDAS
    )
    prompt = (
        carregar_prompt("extrator_intencao")
        .replace("{{mensagem_usuario}}", mensagem_usuario)
        .replace("{{plantas_conhecidas_lista}}", lista_plantas)
        .replace("{{historico_resumido}}", historico_resumido or "(nenhum)")
    )

    resposta_llm = llm.gerar(prompt, temperature=0)
    dados = parsear_json_llm(resposta_llm.texto)

    try:
        tipo_mensagem = TipoMensagem(dados["tipo_mensagem"])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Resposta do extrator de intenção com 'tipo_mensagem' "
            f"ausente ou inválido: {dados!r}"
        ) from e

    solicitacoes_brutas = dados.get("solicitacoes", [])
    solicitacoes: list[Solicitacao] = []
    for item in solicitacoes_brutas[:MAX_SOLICITACOES_POR_MENSAGEM]:
        try:
            tipo = TipoSolicitacao(item["tipo"])
            valor = str(item["valor"]).strip()
        except (KeyError, ValueError, TypeError):
            continue  # item malformado — ignora, não derruba a extração
        if valor:
            solicitacoes.append(Solicitacao(tipo=tipo, valor=valor))

    sinal_encerramento = bool(dados.get("sinal_encerramento", False))

    return tipo_mensagem, solicitacoes, sinal_encerramento, resposta_llm


def normalizar_nome(texto: str) -> str:
    """Normaliza um nome para comparação: minúsculas, sem acentos,
    hífen e espaço tratados como equivalentes, espaços múltiplos
    colapsados.

    Existe para que "sete-sangrias", "Sete Sangrias" e "sete   sangrias"
    sejam reconhecidos como o mesmo nome, sem depender de listar cada
    variação manualmente em `sinonimos`.

    Reaproveitada fora deste módulo por `tools/dual_encoder.py`, para
    mapear o nome retornado pelo dual-encoder aos nomes canônicos —
    por isso é pública (sem underscore), não uma função interna.
    """
    texto_sem_hifen = texto.strip().lower().replace("-", " ")
    texto_sem_acento = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", texto_sem_hifen)
        if not unicodedata.combining(caractere)
    )
    return " ".join(texto_sem_acento.split())


def identificar_planta(
    nome_extraido: str | None,
    plantas_conhecidas: tuple[Planta, ...],
) -> Planta | None:
    """Faz o matching determinístico entre o texto extraído e a lista
    canônica das 6 plantas (nome popular, científico ou sinônimo).

    Retorna None se o nome extraído não corresponder a nenhuma delas —
    nesse caso, o roteamento deve seguir para fallback (Wikipedia/Tavily),
    nunca consultar o RAG (decisão de escopo do projeto, ver ADR-002).

    A comparação é feita via `normalizar_nome`, não por igualdade direta
    de string — cobre variação de caixa, acentuação e hífen/espaço sem
    exigir que `sinonimos` liste cada grafia possível.
    """
    if not nome_extraido:
        return None

    nome_normalizado = normalizar_nome(nome_extraido)

    for planta in plantas_conhecidas:
        candidatos = (planta.nome_popular, planta.nome_cientifico, *planta.sinonimos)
        if nome_normalizado in {normalizar_nome(c) for c in candidatos}:
            return planta

    return None


def buscar_por_atributo(
    descricao_atributo: str,
    dual_encoder: "DualEncoderClient",
) -> ResultadoIdentificacao:
    """Aciona o dual-encoder para uma solicitação do tipo
    busca_por_atributo (ex.: "algo calmante"), retornando a planta
    candidata e o score.

    O score retornado AQUI é epistemicamente relevante — alimenta
    `verificar_confianca` normalmente (ver ADR-004; contraste com
    `buscar_imagem_para_planta`, abaixo).

    `DualEncoderClient` é definido em `tools/dual_encoder.py` — importado
    via type hint em string para não criar dependência direta deste
    módulo em `tools/` no nível de import (mantém o core desacoplado).
    """
    return dual_encoder.buscar(descricao_atributo)


def buscar_imagem_para_planta(
    planta: Planta,
    dual_encoder: "DualEncoderClient",
) -> ResultadoIdentificacao:
    """Aciona o dual-encoder para obter uma imagem representativa de
    uma planta JÁ IDENTIFICADA (solicitação do tipo planta_nomeada).

    O score retornado aqui NÃO deve alimentar `verificar_confianca` nem
    gerar aviso de baixa confiança — a identidade já foi resolvida
    deterministicamente por `identificar_planta` antes desta chamada.
    Esta função existe só para obter a imagem (ver ADR-004 para o
    racional completo, incluindo a verificação empírica que motivou essa
    separação: score ~0.4 mesmo para nome canônico, não confiável como
    sinal de confiança de identidade).
    """
    return dual_encoder.buscar(planta.nome_popular)


def verificar_confianca(
    score: float,
    limiar: float,
) -> bool:
    """Retorna True se o score estiver ABAIXO do limiar (baixa confiança).

    Função pura e determinística — não deve conter chamada a LLM.

    Aplicável APENAS a resultados de `buscar_por_atributo` — nunca a
    resultados de `buscar_imagem_para_planta` (ver ADR-004).
    """
    return score < limiar


def recuperar_conhecimento_rag(
    planta: Planta,
    rag_client: "RAGClient",
) -> list[TrechoRecuperado]:
    """Consulta a base RAG (restrita às 6 plantas) para a planta identificada.

    Só deve ser chamada quando `identificar_planta` já confirmou que a
    planta pertence à base canônica — o RAG não é consultado para plantas
    fora dela (essa decisão é tomada antes, não por fallback reativo).
    """
    return rag_client.consultar(planta)


def recuperar_conhecimento_fallback(
    consulta: str,
    wikipedia_client: "SearchClient",
    tavily_client: "SearchClient",
) -> list[TrechoRecuperado]:
    """Fallback hierárquico: tenta Wikipedia primeiro, depois Tavily.

    Acionado quando a planta mencionada não está entre as 6 cobertas
    pelo RAG, ou quando a intenção é uma pergunta geral do domínio.

    Ambos os SearchClient já falham suave (lista vazia, não exceção) em
    caso de erro de rede ou ausência de resultado — ver tools/search.py.
    """
    trechos = wikipedia_client.buscar(consulta)
    if trechos:
        return trechos
    return tavily_client.buscar(consulta)


TEMPLATE_AVISO_BAIXA_CONFIANCA = (
    "Atenção: a correspondência entre a sua descrição e esta planta tem "
    "confiança baixa — considere como indicativo, não conclusivo."
)
"""Template determinístico do aviso de baixa confiança — nunca gerado
livremente pelo LLM (ver docstring de injetar_aviso_confianca)."""

TEMPLATE_FALLBACK_SAIDA_REPROVADA = (
    "Não foi possível gerar uma resposta segura para esta pergunta. "
    "Por favor, reformule sua pergunta ou consulte um profissional de "
    "saúde qualificado."
)
"""Template determinístico usado quando guardrails/output.py::avaliar_saida
reprova o texto gerado — nunca se expõe ao usuário o texto reprovado
nem o motivo técnico do bloqueio (evita vazar detalhes de mecanismo de
detecção, mesmo princípio de segurança infantil aplicado aqui a
segurança de saúde: não se narra a mecânica da detecção)."""


def calcular_estagio_conversa(
    numero_turno_sessao: int,
    numero_turno_topico: int,
    sinal_encerramento: bool,
    limite_desenvolvimento: int = LIMITE_TURNOS_DESENVOLVIMENTO,
    limite_forcado: int = LIMITE_TURNOS_FORCADO,
) -> str:
    """Calcula {{estagio_conversa}} (prompts/resposta_final.md) de forma
    determinística — nunca deixado a cargo do LLM decidir sozinho quando
    "fechar" a conversa (ver explicação de estratégia registrada na
    conversa que originou este projeto).

    `numero_turno_sessao` e `numero_turno_topico` são contadores mantidos
    fora do grafo (por app.py, em st.session_state — persistem entre
    invocações separadas do grafo, ao contrário do resto do estado, que
    vive só durante uma única invocação). `numero_turno_topico` reseta
    quando o usuário muda de planta/assunto; `numero_turno_sessao` não
    reseta nunca dentro da mesma sessão.

    Regras (nesta ordem):
        1. Primeiro turno da sessão -> "abertura" (independente de tudo mais).
        2. Turno do tópico atingiu o teto rígido -> "fechamento_forcado"
           (ignora sinal_encerramento — é uma rede de segurança contra
           loop, não uma sugestão).
        3. Sinal de encerramento OU turno do tópico atingiu o limite
           normal -> "fechamento".
        4. Caso contrário -> "desenvolvimento".
    """
    if numero_turno_sessao <= 1:
        return "abertura"
    if numero_turno_topico >= limite_forcado:
        return "fechamento_forcado"
    if sinal_encerramento or numero_turno_topico >= limite_desenvolvimento:
        return "fechamento"
    return "desenvolvimento"


def formatar_trechos_para_prompt(fontes: list[TrechoRecuperado]) -> str:
    """Formata uma lista de TrechoRecuperado como texto para inserção em
    {{trechos_recuperados}} (resposta_final.md e guardrail_saida.md) —
    função compartilhada entre core/use_cases.py e guardrails/output.py,
    para não duplicar a mesma lógica de formatação em dois lugares.
    """
    if not fontes:
        return "(nenhum trecho fornecido)"
    linhas = [
        f"{i}. [{trecho.origem.value}] {trecho.fonte_citacao}: {trecho.texto}"
        for i, trecho in enumerate(fontes, start=1)
    ]
    return "\n".join(linhas)


def formatar_plantas_em_foco(resultados_solicitacoes: list[ResultadoSolicitacao]) -> str:
    """Formata os nomes das plantas identificadas em
    `resultados_solicitacoes` para inserção em {{plantas_em_foco}}
    (resposta_final.md e guardrail_saida.md). Retorna "nenhuma" se
    nenhuma planta foi identificada em nenhuma solicitação.
    """
    nomes = [
        resultado.planta_identificada.nome_popular
        for resultado in resultados_solicitacoes
        if resultado.planta_identificada is not None
    ]
    return ", ".join(nomes) if nomes else "nenhuma"


def injetar_aviso_confianca(
    texto_resposta: str,
    abaixo_do_limiar: bool,
) -> str:
    """Insere o aviso de baixa confiança no texto usando um template
    determinístico (não gerado livremente pelo LLM).

    Como o template é uma string FIXA (nunca varia), a etapa de
    avaliação (Fase 6) pode excluí-lo do texto por correspondência
    exata de substring (ver `remover_aviso_confianca`), sem precisar de
    marcadores/delimitadores visíveis no texto — que apareceriam de
    forma estranha na tela e seriam narrados literalmente pelo TTS.

    Este mesmo texto (já com o aviso, se aplicável) é o que será: (a)
    exibido na tela, e (b) enviado ao TTS — não existe versão separada
    do aviso para áudio.
    """
    if not abaixo_do_limiar:
        return texto_resposta
    return f"{texto_resposta.rstrip()} {TEMPLATE_AVISO_BAIXA_CONFIANCA}"


def remover_aviso_confianca(texto_resposta: str) -> str:
    """Remove o aviso de baixa confiança do texto, se presente.

    Uso: avaliação de groundedness/faithfulness (Fase 6) — o aviso não
    é uma afirmação sobre a planta, não deve contar como conteúdo a
    fundamentar contra as fontes recuperadas.
    """
    return texto_resposta.replace(f" {TEMPLATE_AVISO_BAIXA_CONFIANCA}", "").rstrip()


def montar_resposta(
    texto_resposta: str,
    resultados_solicitacoes: list[ResultadoSolicitacao],
    audio_ref: bytes | None,
) -> EnvelopeResposta:
    """Monta o EnvelopeResposta final a partir das peças produzidas
    pelas etapas anteriores do grafo.

    Pré-condições sobre `texto_resposta`: já deve ter passado por (a)
    composição final (resposta_final.md), (b) injeção de aviso de
    confiança por solicitação abaixo do limiar
    (`injetar_aviso_confianca`), e (c) guardrails de saída.

    Constrói uma `ImagemResposta` para cada `ResultadoSolicitacao` que
    tiver `resultado_identificacao` preenchido (nem toda solicitação
    necessariamente passa pelo dual-encoder — ex.: uma solicitação cuja
    planta não está entre as 6 não gera imagem). `fontes` agrega os
    `TrechoRecuperado` de todas as solicitações, na ordem em que
    aparecem — sem deduplicação (deixado como possível refinamento, não
    necessário para o MVP com 6 plantas distintas).
    """
    imagens: list[ImagemResposta] = []
    avisos_confianca: list[str] = []
    fontes: list[FonteCitada] = []

    for resultado in resultados_solicitacoes:
        if resultado.resultado_identificacao is not None:
            ri = resultado.resultado_identificacao
            imagens.append(
                ImagemResposta(
                    presente=True,
                    url_ou_ref=ri.imagem_url_ou_ref,
                    score_similaridade=ri.score_similaridade,
                    planta_identificada=ri.planta.nome_popular,
                    abaixo_do_limiar=resultado.abaixo_do_limiar,
                    limiar_usado=resultado.limiar_usado
                    if resultado.limiar_usado is not None
                    else 0.0,
                )
            )
            if resultado.abaixo_do_limiar:
                avisos_confianca.append(TEMPLATE_AVISO_BAIXA_CONFIANCA)

        for trecho in resultado.trechos:
            fontes.append(
                FonteCitada(origem=trecho.origem, citacao=trecho.fonte_citacao)
            )

    return EnvelopeResposta(
        texto_resposta=texto_resposta,
        imagens=imagens,
        audio=AudioResposta(presente=audio_ref is not None, ref=audio_ref),
        fontes=fontes,
        avisos_confianca=avisos_confianca,
    )


# Protocols abaixo evitam import direto de `tools/` no `core/`, preservando
# o core livre de dependência de infraestrutura, sem precisar de uma
# camada de "ports" formal separada.

class DualEncoderClient(Protocol):
    def buscar(self, texto: str) -> ResultadoIdentificacao: ...


class RAGClient(Protocol):
    def consultar(self, planta: Planta) -> list[TrechoRecuperado]: ...


class SearchClient(Protocol):
    def buscar(self, consulta: str) -> list[TrechoRecuperado]: ...
