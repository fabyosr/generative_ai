"""
Grafo de orquestração do agente, construído com LangGraph.

Este módulo é infraestrutura/framework — conhece LangGraph, mas contém
o MÍNIMO de lógica de negócio possível: cada nó chama uma função de
`core/use_cases.py` (ou `guardrails/*.py`) e só adapta entrada/saída ao
formato de estado do grafo.

DEPENDÊNCIAS (LLM, dual-encoder, RAG, busca, TTS) são injetadas via
`construir_grafo(dependencias)` — os nós são fechamentos (closures)
definidos dentro dessa função, capturando `dependencias`. Isso evita
tanto variáveis globais quanto a maquinária de `config`/`configurable`
do LangGraph, mantendo o mesmo princípio de injeção explícita por
parâmetro usado em todo o `core/use_cases.py`.

FAN-OUT SOBRE SOLICITAÇÕES MÚLTIPLAS: processado por um laço Python
determinístico dentro de um único nó (`processar_solicitacoes`), não
pelo mecanismo de `Send`/map-reduce do LangGraph. Dado o teto de 5
solicitações por mensagem (config/constants.py), paralelismo real não
traria benefício que justifique a complexidade adicional — mesmo
critério de "evitar complexidade sem benefício demonstrável" aplicado
em todo o projeto.

Fluxo (ver docs/architecture para o diagrama completo):

    entrada do usuário
        -> validar_entrada (determinístico) --[inválida]--> resposta_entrada_invalida -> FIM
        -> extrair_intencao (1 chamada LLM, saída estruturada)
        -> verificar_escopo (1 chamada LLM, camada separada — ver ADR)
             --[fora de escopo]--> resposta_fora_de_escopo -> sintetizar_audio -> montar_resposta_final -> FIM
        -> processar_solicitacoes (laço determinístico: identificar_planta,
           dual-encoder, RAG ou fallback, por item)
        -> compor_resposta (1 chamada LLM, síntese única de todas as solicitações)
        -> injetar_aviso (determinístico)
        -> avaliar_saida (1 chamada LLM — aprovação + groundedness juntos)
        -> sintetizar_audio (TTS, se habilitado)
        -> montar_resposta_final (monta o EnvelopeResposta)
        -> FIM
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from config.constants import LIMIAR_DUAL_ENCODER_PADRAO, PLANTAS_CONHECIDAS
from core.models import AudioResposta, EnvelopeResposta, ResultadoSolicitacao, TipoSolicitacao
from core.use_cases import (
    TEMPLATE_FALLBACK_SAIDA_REPROVADA,
    buscar_imagem_para_planta,
    buscar_por_atributo,
    calcular_estagio_conversa,
    extrair_intencao,
    formatar_plantas_em_foco,
    formatar_trechos_para_prompt,
    identificar_planta,
    injetar_aviso_confianca,
    montar_resposta,
    recuperar_conhecimento_fallback,
    recuperar_conhecimento_rag,
    verificar_confianca,
)
from guardrails.escopo import verificar_escopo
from guardrails.input import validar_entrada
from guardrails.output import avaliar_saida
from observability.logger import EventoObservabilidade, get_logger, registrar_evento
from prompts.loader import carregar_prompt

_logger = get_logger(__name__)


def _somar_tokens(resposta_llm) -> int | None:
    """Soma tokens_entrada + tokens_saida de uma RespostaLLM, tratando
    ausência (None) com segurança — nem todo provedor garante retornar
    os dois valores (ver core/models.py::RespostaLLM)."""
    if resposta_llm.tokens_entrada is None and resposta_llm.tokens_saida is None:
        return None
    return (resposta_llm.tokens_entrada or 0) + (resposta_llm.tokens_saida or 0)


@dataclass
class Dependencias:
    """Serviços concretos injetados no grafo — ver docstring do módulo.

    `tts` é opcional: se None, `sintetizar_audio` não faz nada, mesmo
    que `audio_habilitado=True` no estado (permite rodar o grafo sem
    TTS configurado, ex.: em testes).
    """

    llm: object  # LLMClient (core/use_cases.py)
    dual_encoder: object  # DualEncoderClient
    rag: object  # RAGClient
    wikipedia: object  # SearchClient
    tavily: object  # SearchClient
    tts: object | None = None  # TTSService (tools/tts.py)


class EstadoAgente(TypedDict, total=False):
    """Estado compartilhado entre os nós do grafo, para UMA invocação
    (uma mensagem do usuário) — não confundir com estado de sessão
    (histórico entre mensagens), que vive fora do grafo, em app.py.
    """

    # --- Entrada (fornecida por app.py a cada invocação) ---
    mensagem_usuario: str
    historico_resumido: str
    numero_turno_sessao: int
    numero_turno_topico: int
    limiar_dual_encoder: float
    audio_habilitado: bool
    voz_selecionada: str | None

    # --- Populado por validar_entrada ---
    entrada_valida: bool
    motivo_bloqueio_entrada: str | None

    # --- Populado por extrair_intencao ---
    tipo_mensagem: object  # TipoMensagem
    solicitacoes: list  # list[Solicitacao]
    sinal_encerramento: bool
    estagio_conversa: str

    # --- Populado por verificar_escopo ---
    dentro_do_escopo: bool
    mensagem_redirecionamento: str | None

    # --- Populado por processar_solicitacoes ---
    resultados_solicitacoes: list[ResultadoSolicitacao]

    # --- Populado por compor_resposta / injetar_aviso / avaliar_saida ---
    texto_resposta: str | None
    groundedness_score: float | None

    # --- Populado por sintetizar_audio ---
    audio_bytes: bytes | None

    # --- Saída final ---
    envelope_resposta: EnvelopeResposta | None
    erro: str | None

    # --- Observabilidade (lista viva, mantida por app.py em
    #     st.session_state — mutada in-place a cada evento, não
    #     substituída; alimenta a aba de observabilidade da interface) ---
    historico_observabilidade: list


def construir_grafo(dependencias: Dependencias):
    """Monta e compila o StateGraph do LangGraph.

    Retorna o grafo compilado, pronto para ser invocado por `app.py`
    via `.invoke(estado_inicial)`.
    """

    # -----------------------------------------------------------------
    # Nós
    # -----------------------------------------------------------------

    def no_validar_entrada(estado: EstadoAgente) -> dict:
        resultado = validar_entrada(estado["mensagem_usuario"])
        return {
            "entrada_valida": resultado.valida,
            "motivo_bloqueio_entrada": resultado.motivo_bloqueio,
        }

    def no_extrair_intencao(estado: EstadoAgente) -> dict:
        inicio = time.perf_counter()
        tipo_mensagem, solicitacoes, sinal_encerramento, resposta_llm = extrair_intencao(
            estado["mensagem_usuario"],
            dependencias.llm,
            estado.get("historico_resumido", ""),
        )
        latencia_ms = (time.perf_counter() - inicio) * 1000
        estagio = calcular_estagio_conversa(
            numero_turno_sessao=estado.get("numero_turno_sessao", 1),
            numero_turno_topico=estado.get("numero_turno_topico", 1),
            sinal_encerramento=sinal_encerramento,
        )

        registrar_evento(
            _logger,
            EventoObservabilidade(
                etapa="extrair_intencao",
                ferramenta_acionada="LLM",
                latencia_ms=latencia_ms,
                tokens_consumidos=_somar_tokens(resposta_llm),
                metadados_extra={
                    "tipo_mensagem": tipo_mensagem.value,
                    "numero_solicitacoes": len(solicitacoes),
                    "estagio_conversa": estagio,
                    "modelo": resposta_llm.modelo,
                    "tokens_entrada": resposta_llm.tokens_entrada,
                    "tokens_saida": resposta_llm.tokens_saida,
                },
            ),
            historico=estado.get("historico_observabilidade"),
        )

        return {
            "tipo_mensagem": tipo_mensagem,
            "solicitacoes": solicitacoes,
            "sinal_encerramento": sinal_encerramento,
            "estagio_conversa": estagio,
        }

    def no_verificar_escopo(estado: EstadoAgente) -> dict:
        inicio = time.perf_counter()
        resultado = verificar_escopo(
            estado["mensagem_usuario"],
            dependencias.llm,
            estado.get("historico_resumido", ""),
        )
        registrar_evento(
            _logger,
            EventoObservabilidade(
                etapa="verificar_escopo",
                ferramenta_acionada="LLM",
                latencia_ms=(time.perf_counter() - inicio) * 1000,
                tokens_consumidos=_somar_tokens(resultado.resposta_llm),
                metadados_extra={
                    "dentro_do_escopo": resultado.dentro_do_escopo,
                    "modelo": resultado.resposta_llm.modelo,
                    "tokens_entrada": resultado.resposta_llm.tokens_entrada,
                    "tokens_saida": resultado.resposta_llm.tokens_saida,
                },
            ),
            historico=estado.get("historico_observabilidade"),
        )
        return {
            "dentro_do_escopo": resultado.dentro_do_escopo,
            "mensagem_redirecionamento": resultado.mensagem_redirecionamento,
        }

    def no_processar_solicitacoes(estado: EstadoAgente) -> dict:
        limiar = estado.get("limiar_dual_encoder", LIMIAR_DUAL_ENCODER_PADRAO)
        historico_obs = estado.get("historico_observabilidade")
        resultados: list[ResultadoSolicitacao] = []

        for solicitacao in estado.get("solicitacoes", []):
            inicio = time.perf_counter()

            if solicitacao.tipo == TipoSolicitacao.PLANTA_NOMEADA:
                planta = identificar_planta(solicitacao.valor, PLANTAS_CONHECIDAS)

                if planta is not None:
                    # Dentro das 6 -> RAG + imagem (sem checagem de
                    # confiança — ver ADR-004: identidade já resolvida).
                    resultado_id = buscar_imagem_para_planta(
                        planta, dependencias.dual_encoder
                    )
                    trechos = recuperar_conhecimento_rag(planta, dependencias.rag)
                    resultados.append(
                        ResultadoSolicitacao(
                            solicitacao=solicitacao,
                            resultado_identificacao=resultado_id,
                            planta_identificada=planta,
                            trechos=trechos,
                        )
                    )
                    registrar_evento(
                        _logger,
                        EventoObservabilidade(
                            etapa="processar_solicitacao",
                            ferramenta_acionada="RAG",
                            latencia_ms=(time.perf_counter() - inicio) * 1000,
                            metadados_extra={
                                "tipo": "planta_nomeada",
                                "planta": planta.nome_popular,
                                # DEBUG TEMPORÁRIO — remover após
                                # diagnóstico de volume de tokens do RAG.
                                "volume_por_secao": [
                                    {
                                        "secao": t.fonte_citacao,
                                        "caracteres": len(t.texto),
                                        "palavras": len(t.texto.split()),
                                    }
                                    for t in trechos
                                ],
                                "total_caracteres_trechos": sum(
                                    len(t.texto) for t in trechos
                                ),
                            },
                        ),
                        historico=historico_obs,
                    )
                else:
                    # Fora das 6 -> fallback, sem imagem (ADR-002).
                    trechos = recuperar_conhecimento_fallback(
                        solicitacao.valor, dependencias.wikipedia, dependencias.tavily
                    )
                    resultados.append(
                        ResultadoSolicitacao(
                            solicitacao=solicitacao,
                            resultado_identificacao=None,
                            planta_identificada=None,
                            trechos=trechos,
                        )
                    )
                    registrar_evento(
                        _logger,
                        EventoObservabilidade(
                            etapa="processar_solicitacao",
                            ferramenta_acionada="fallback_wikipedia_tavily",
                            latencia_ms=(time.perf_counter() - inicio) * 1000,
                            metadados_extra={
                                "tipo": "planta_nomeada",
                                "planta_fora_das_6": solicitacao.valor,
                                "trechos_encontrados": len(trechos),
                            },
                        ),
                        historico=historico_obs,
                    )

            elif solicitacao.tipo == TipoSolicitacao.BUSCA_POR_ATRIBUTO:
                # O dual-encoder sempre resolve para uma das 6 plantas
                # (garantido pelo mapeamento em tools/dual_encoder.py),
                # então RAG (não fallback) é sempre a fonte aqui —
                # possivelmente com aviso de baixa confiança anexado.
                resultado_id = buscar_por_atributo(
                    solicitacao.valor, dependencias.dual_encoder
                )
                abaixo = verificar_confianca(resultado_id.score_similaridade, limiar)
                trechos = recuperar_conhecimento_rag(resultado_id.planta, dependencias.rag)
                resultados.append(
                    ResultadoSolicitacao(
                        solicitacao=solicitacao,
                        resultado_identificacao=resultado_id,
                        planta_identificada=resultado_id.planta,
                        trechos=trechos,
                        abaixo_do_limiar=abaixo,
                        limiar_usado=limiar,
                    )
                )
                registrar_evento(
                    _logger,
                    EventoObservabilidade(
                        etapa="processar_solicitacao",
                        ferramenta_acionada="dual_encoder",
                        score=resultado_id.score_similaridade,
                        limiar_usado=limiar,
                        latencia_ms=(time.perf_counter() - inicio) * 1000,
                        metadados_extra={
                            "tipo": "busca_por_atributo",
                            "planta_identificada": resultado_id.planta.nome_popular,
                            "abaixo_do_limiar": abaixo,# DEBUG TEMPORÁRIO — mesmo propósito do outro branch.
                            "volume_por_secao": [
                                {
                                    "secao": t.fonte_citacao,
                                    "caracteres": len(t.texto),
                                    "palavras": len(t.texto.split()),
                                }
                                for t in trechos
                            ],
                            "total_caracteres_trechos": sum(len(t.texto) for t in trechos),
                        },
                    ),
                    historico=historico_obs,
                )

        return {"resultados_solicitacoes": resultados}

    def no_compor_resposta(estado: EstadoAgente) -> dict:
        resultados = estado.get("resultados_solicitacoes", [])
        todos_trechos = [t for r in resultados for t in r.trechos]

        prompt = (
            carregar_prompt("resposta_final")
            .replace("{{mensagem_usuario}}", estado["mensagem_usuario"])
            .replace("{{trechos_recuperados}}", formatar_trechos_para_prompt(todos_trechos))
            .replace("{{plantas_em_foco}}", formatar_plantas_em_foco(resultados))
            .replace("{{estagio_conversa}}", estado.get("estagio_conversa", "desenvolvimento"))
            .replace("{{historico_resumido}}", estado.get("historico_resumido", "") or "(nenhum)")
        )

        inicio = time.perf_counter()
        resposta_llm = dependencias.llm.gerar(prompt)
        registrar_evento(
            _logger,
            EventoObservabilidade(
                etapa="compor_resposta",
                ferramenta_acionada="LLM",
                latencia_ms=(time.perf_counter() - inicio) * 1000,
                tokens_consumidos=_somar_tokens(resposta_llm),
                metadados_extra={
                    "modelo": resposta_llm.modelo,
                    "tokens_entrada": resposta_llm.tokens_entrada,
                    "tokens_saida": resposta_llm.tokens_saida,"caracteres_trechos_recuperados": len(
                        formatar_trechos_para_prompt(todos_trechos)
                        ), # DEBUG TEMPORÁRIO — ver o prompt real enviado ao LLM.
                    "tamanho_prompt_caracteres": len(prompt),
                    "prompt_enviado": prompt,
                },
            ),
            historico=estado.get("historico_observabilidade"),
        )

        return {"texto_resposta": resposta_llm.texto}

    def no_injetar_aviso(estado: EstadoAgente) -> dict:
        resultados = estado.get("resultados_solicitacoes", [])
        # Simplificação de MVP: um único aviso genérico se QUALQUER
        # solicitação de busca_por_atributo estiver abaixo do limiar —
        # não distingue qual planta especificamente, caso haja mais de
        # uma solicitação nessa condição na mesma mensagem (documentado
        # como limitação conhecida, não um bug).
        algum_abaixo_do_limiar = any(r.abaixo_do_limiar for r in resultados)
        texto = injetar_aviso_confianca(estado["texto_resposta"], algum_abaixo_do_limiar)
        return {"texto_resposta": texto}

    def no_avaliar_saida(estado: EstadoAgente) -> dict:
        resultados = estado.get("resultados_solicitacoes", [])
        todos_trechos = [t for r in resultados for t in r.trechos]

        inicio = time.perf_counter()
        avaliacao = avaliar_saida(
            estado["texto_resposta"],
            todos_trechos,
            dependencias.llm,
            formatar_plantas_em_foco(resultados),
        )
        registrar_evento(
            _logger,
            EventoObservabilidade(
                etapa="avaliar_saida",
                ferramenta_acionada="LLM",
                score=avaliacao.groundedness_score,
                latencia_ms=(time.perf_counter() - inicio) * 1000,
                tokens_consumidos=_somar_tokens(avaliacao.resposta_llm),
                erro=avaliacao.motivo_bloqueio if not avaliacao.aprovado else None,
                metadados_extra={
                    "aprovado": avaliacao.aprovado,
                    "modelo": avaliacao.resposta_llm.modelo,
                    "tokens_entrada": avaliacao.resposta_llm.tokens_entrada,
                    "tokens_saida": avaliacao.resposta_llm.tokens_saida,
                },
            ),
            historico=estado.get("historico_observabilidade"),
        )

        texto_final = (
            estado["texto_resposta"] if avaliacao.aprovado else TEMPLATE_FALLBACK_SAIDA_REPROVADA
        )
        return {
            "texto_resposta": texto_final,
            "groundedness_score": avaliacao.groundedness_score,
        }

    def no_sintetizar_audio(estado: EstadoAgente) -> dict:
        if not estado.get("audio_habilitado") or dependencias.tts is None:
            return {"audio_bytes": None}
        audio = dependencias.tts.sintetizar(
            estado["texto_resposta"], voz=estado.get("voz_selecionada")
        )
        return {"audio_bytes": audio}

    def no_montar_resposta_final(estado: EstadoAgente) -> dict:
        envelope = montar_resposta(
            texto_resposta=estado["texto_resposta"],
            resultados_solicitacoes=estado.get("resultados_solicitacoes", []),
            audio_ref=estado.get("audio_bytes"),
        )
        return {"envelope_resposta": envelope}

    def no_resposta_entrada_invalida(estado: EstadoAgente) -> dict:
        # Simplificação de MVP: entrada inválida não gera áudio nem
        # passa pelos guardrails de saída — é um erro de formato, não
        # uma resposta de conteúdo a validar.
        envelope = EnvelopeResposta(
            texto_resposta=estado.get("motivo_bloqueio_entrada") or "Mensagem inválida.",
            imagens=[],
            audio=AudioResposta(presente=False, ref=None),
            fontes=[],
        )
        return {
            "envelope_resposta": envelope,
            "erro": estado.get("motivo_bloqueio_entrada"),
        }

    def no_resposta_fora_de_escopo(estado: EstadoAgente) -> dict:
        return {"texto_resposta": estado["mensagem_redirecionamento"]}

    # -----------------------------------------------------------------
    # Arestas condicionais
    # -----------------------------------------------------------------

    def rotear_apos_validacao(estado: EstadoAgente) -> str:
        return "extrair_intencao" if estado.get("entrada_valida") else "resposta_entrada_invalida"

    def rotear_apos_escopo(estado: EstadoAgente) -> str:
        return (
            "processar_solicitacoes"
            if estado.get("dentro_do_escopo")
            else "resposta_fora_de_escopo"
        )

    # -----------------------------------------------------------------
    # Montagem do grafo
    # -----------------------------------------------------------------

    grafo = StateGraph(EstadoAgente)

    grafo.add_node("validar_entrada", no_validar_entrada)
    grafo.add_node("extrair_intencao", no_extrair_intencao)
    grafo.add_node("verificar_escopo", no_verificar_escopo)
    grafo.add_node("processar_solicitacoes", no_processar_solicitacoes)
    grafo.add_node("compor_resposta", no_compor_resposta)
    grafo.add_node("injetar_aviso", no_injetar_aviso)
    grafo.add_node("avaliar_saida", no_avaliar_saida)
    grafo.add_node("sintetizar_audio", no_sintetizar_audio)
    grafo.add_node("montar_resposta_final", no_montar_resposta_final)
    grafo.add_node("resposta_entrada_invalida", no_resposta_entrada_invalida)
    grafo.add_node("resposta_fora_de_escopo", no_resposta_fora_de_escopo)

    grafo.set_entry_point("validar_entrada")

    grafo.add_conditional_edges("validar_entrada", rotear_apos_validacao)
    grafo.add_edge("extrair_intencao", "verificar_escopo")
    grafo.add_conditional_edges("verificar_escopo", rotear_apos_escopo)

    grafo.add_edge("processar_solicitacoes", "compor_resposta")
    grafo.add_edge("compor_resposta", "injetar_aviso")
    grafo.add_edge("injetar_aviso", "avaliar_saida")
    grafo.add_edge("avaliar_saida", "sintetizar_audio")

    # Fora de escopo pula extração de solicitações/geração/avaliação —
    # o texto já é o redirecionamento determinístico do guardrail, mas
    # ainda passa por síntese de áudio (é narrado, como qualquer outra
    # resposta — decisão de consistência de UX).
    grafo.add_edge("resposta_fora_de_escopo", "sintetizar_audio")

    grafo.add_edge("sintetizar_audio", "montar_resposta_final")
    grafo.add_edge("montar_resposta_final", END)
    grafo.add_edge("resposta_entrada_invalida", END)

    return grafo.compile()
