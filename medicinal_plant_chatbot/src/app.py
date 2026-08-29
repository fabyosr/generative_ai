"""
Ponto de entrada da aplicação web (Streamlit).

CORREÇÃO DE PATH (leia antes de mexer no resto): quando o Python roda
este arquivo (via `streamlit run`, de qualquer diretório), ele coloca o
diretório DESTE ARQUIVO (src/) no início do sys.path — não a raiz do
projeto, mesmo que o comando tenha sido chamado da raiz. Isso quebra
todo import do tipo `from config import ...` usado no resto do
projeto (não existe um pacote "src" visível de dentro de "src/"). As
duas linhas abaixo, ANTES de qualquer import interno, inserem a raiz do
projeto (pai deste arquivo) no sys.path, resolvendo isso de forma
independente de onde/como o comando é chamado.

Executar com (de qualquer diretório): streamlit run src/app.py
"""

from __future__ import annotations

import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agents.graph import Dependencias, construir_grafo
from app_logica import (
    calcular_historico_resumido,
    calcular_novo_turno_topico,
    extrair_planta_principal,
)
from config import constants, settings
from observability.logger import get_logger
from tools.dual_encoder import DualEncoderService
from tools.llm import criar_llm_client
from tools.rag import RAGService
from tools.search import TavilySearchService, WikipediaSearchService
from tools.tts import KokoroTTSService

st.set_page_config(page_title="Plantas Medicinais", page_icon="🌿", layout="wide")

_logger = get_logger(__name__)

PROVEDORES_DISPONIVEIS = ["groq", "openai", "anthropic", "xai"]
MODELOS_PADRAO_POR_PROVEDOR = {
    "groq":"openai/gpt-oss-20b", #"groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "xai": "grok-2-latest",
}


# ---------------------------------------------------------------------------
# Recursos pesados — cacheados, carregados uma única vez por processo
# (ver risco sinalizado ao integrar botanical_search.py: sem isso, o
# Streamlit recarregaria BERT + ResNet-50 do zero a cada mensagem).
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Carregando modelo de identificação de plantas...")
def _carregar_dual_encoder() -> DualEncoderService:
    return DualEncoderService(
        plantas_conhecidas=constants.PLANTAS_CONHECIDAS, auto_download=True
    )


@st.cache_resource(show_spinner="Carregando base de conhecimento curada...")
def _carregar_rag() -> RAGService:
    servico = RAGService(settings.RAG_BASE_CURADA_PATH)
    servico.carregar()
    return servico


@st.cache_resource(show_spinner="Carregando modelo de voz...")
def _carregar_tts() -> KokoroTTSService:
    return KokoroTTSService()


def _montar_dependencias(provedor: str, api_key: str, model: str) -> Dependencias:
    llm = criar_llm_client(provedor, api_key=api_key, model=model)
    return Dependencias(
        llm=llm,
        dual_encoder=_carregar_dual_encoder(),
        rag=_carregar_rag(),
        wikipedia=WikipediaSearchService(),
        tavily=TavilySearchService(api_key=settings.TAVILY_API_KEY),
        tts=_carregar_tts(),
    )


# ---------------------------------------------------------------------------
# Estado de sessão
# ---------------------------------------------------------------------------


def _inicializar_estado_sessao() -> None:
    padroes = {
        "mensagens": [],
        "numero_turno_sessao": 0,
        "numero_turno_topico": 0,
        "planta_topico_atual": None,
        "eventos_observabilidade": [],
    }
    for chave, valor in padroes.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _renderizar_sidebar() -> dict:
    st.sidebar.header("Configurações")

    indice_padrao = (
        PROVEDORES_DISPONIVEIS.index(settings.LLM_PROVIDER)
        if settings.LLM_PROVIDER in PROVEDORES_DISPONIVEIS
        else 0
    )
    provedor = st.sidebar.selectbox(
        "Provedor de LLM", PROVEDORES_DISPONIVEIS, index=indice_padrao
    )

    chave_do_servidor = {
        "groq": settings.GROQ_API_KEY,
        "openai": settings.OPENAI_API_KEY,
        "anthropic": settings.ANTHROPIC_API_KEY,
        "xai": settings.XAI_API_KEY,
    }.get(provedor)

    chave_digitada = st.sidebar.text_input(
        f"Chave de API ({provedor})",
        type="password",
        help=(
            "Deixe em branco para usar a chave configurada no servidor "
            "(se houver). Digite sua própria chave para usar seus "
            "próprios créditos, sem depender da chave do servidor — "
            "útil se este app estiver publicamente acessível."
        ),
    )
    api_key = chave_digitada or chave_do_servidor or ""

    if not chave_digitada and chave_do_servidor:
        st.sidebar.caption("✅ Usando chave configurada no servidor.")
    elif chave_digitada:
        st.sidebar.caption("✅ Usando sua chave (não é persistida em nenhum lugar).")
    else:
        st.sidebar.caption("⚠️ Nenhuma chave disponível para este provedor.")

    model = st.sidebar.text_input(
        "Modelo", value=MODELOS_PADRAO_POR_PROVEDOR.get(provedor, "")
    )

    st.sidebar.divider()

    limiar_dual_encoder = st.sidebar.slider(
        "Limiar de confiança (identificação por descrição)",
        min_value=0.0,
        max_value=1.0,
        value=constants.LIMIAR_DUAL_ENCODER_PADRAO,
        step=0.05,
        help=(
            "Abaixo deste valor, a resposta inclui um aviso de baixa "
            "confiança. Só se aplica a buscas por descrição/sintoma — "
            "não a plantas nomeadas diretamente (ver ADR-004)."
        ),
    )

    st.sidebar.divider()

    audio_habilitado = st.sidebar.toggle("Narrar respostas em áudio", value=False)
    voz_selecionada = None
    if audio_habilitado:
        try:
            vozes = _carregar_tts().vozes_disponiveis()
            nome_escolhido = st.sidebar.selectbox("Voz", list(vozes.keys()))
            voz_selecionada = vozes[nome_escolhido]
        except Exception as e:
            st.sidebar.warning(f"Não foi possível carregar as vozes: {e}")
            audio_habilitado = False

    st.sidebar.divider()
    st.sidebar.caption(
        f"Sessão: turno {st.session_state.numero_turno_sessao} · "
        f"tópico: turno {st.session_state.numero_turno_topico}"
    )

    return {
        "provedor": provedor,
        "api_key": api_key,
        "model": model,
        "limiar_dual_encoder": limiar_dual_encoder,
        "audio_habilitado": audio_habilitado,
        "voz_selecionada": voz_selecionada,
    }


# ---------------------------------------------------------------------------
# Renderização de uma resposta (reaproveitada no histórico e na mensagem nova)
# ---------------------------------------------------------------------------


def _renderizar_envelope(envelope) -> None:
    st.write(envelope.texto_resposta)
    for imagem in envelope.imagens:
        if not imagem.presente:
            continue
        legenda = imagem.planta_identificada or "Planta identificada"
        if imagem.abaixo_do_limiar:
            legenda += " ⚠️ baixa confiança"
        st.image(imagem.url_ou_ref, caption=legenda, width=250)
    if envelope.audio.presente and envelope.audio.ref:
        st.audio(envelope.audio.ref, , format="audio/wav", key=f"audio_turno_{st.session_state.numero_turno_sessao}", autoplay=True)


# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------


def _renderizar_aba_chat(config: dict) -> None:
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            if msg.get("envelope"):
                _renderizar_envelope(msg["envelope"])
            else:
                st.write(msg["content"])

    prompt = st.chat_input("Pergunte sobre plantas medicinais...")
    if not prompt:
        return

    if not config["api_key"]:
        st.error(
            "Nenhuma chave de API disponível para o provedor selecionado. "
            "Configure no servidor (.env) ou digite sua própria chave na "
            "barra lateral."
        )
        return

    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.numero_turno_sessao += 1

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                dependencias = _montar_dependencias(
                    config["provedor"], config["api_key"], config["model"]
                )
                grafo = construir_grafo(dependencias)
                estado_inicial = {
                    "mensagem_usuario": prompt,
                    "historico_resumido": calcular_historico_resumido(
                        st.session_state.mensagens
                    ),
                    "numero_turno_sessao": st.session_state.numero_turno_sessao,
                    "numero_turno_topico": st.session_state.numero_turno_topico or 1,
                    "limiar_dual_encoder": config["limiar_dual_encoder"],
                    "audio_habilitado": config["audio_habilitado"],
                    "voz_selecionada": config["voz_selecionada"],
                    "historico_observabilidade": st.session_state.eventos_observabilidade,
                }
                resultado = grafo.invoke(estado_inicial)
                envelope = resultado["envelope_resposta"]
            except Exception as e:
                _logger.exception("Erro ao processar mensagem do usuário")
                st.error(f"Ocorreu um erro ao processar sua mensagem: {e}")
                return

        _renderizar_envelope(envelope)

    st.session_state.mensagens.append(
        {"role": "assistant", "content": envelope.texto_resposta, "envelope": envelope}
    )

    planta_atual = extrair_planta_principal(envelope)
    novo_turno_topico, nova_planta = calcular_novo_turno_topico(
        st.session_state.planta_topico_atual,
        planta_atual,
        st.session_state.numero_turno_topico,
    )
    st.session_state.numero_turno_topico = novo_turno_topico
    st.session_state.planta_topico_atual = nova_planta


def _renderizar_aba_observabilidade() -> None:
    eventos = st.session_state.eventos_observabilidade
    if not eventos:
        st.info("Nenhum evento registrado ainda — envie uma mensagem no chat.")
        return

    st.caption(
        "Visão simplificada de observabilidade, pensada para o usuário "
        "final. Para rastro técnico completo (tokens por chamada, spans "
        "aninhados), consulte o LangSmith, se configurado no .env."
    )

    total_tokens = sum(e.tokens_consumidos or 0 for e in eventos)
    col1, col2, col3 = st.columns(3)
    col1.metric("Eventos nesta sessão", len(eventos))
    col2.metric("Tokens consumidos (soma)", total_tokens)
    col3.metric(
        "Latência total (ms)",
        round(sum(e.latencia_ms or 0 for e in eventos)),
    )

    for evento in reversed(eventos[-30:]):
        titulo = f"{evento.etapa} — {evento.ferramenta_acionada or ''}"
        with st.expander(titulo):
            if evento.score is not None:
                st.write(f"Score: {evento.score:.3f}")
            if evento.limiar_usado is not None:
                st.write(f"Limiar usado: {evento.limiar_usado:.3f}")
            if evento.latencia_ms is not None:
                st.write(f"Latência: {evento.latencia_ms:.0f} ms")
            if evento.tokens_consumidos is not None:
                st.write(f"Tokens consumidos: {evento.tokens_consumidos}")
            if evento.erro:
                st.warning(f"Erro: {evento.erro}")
            if evento.metadados_extra:
                st.json(evento.metadados_extra)


def _renderizar_aba_fontes() -> None:
    mensagens_assistente = [
        m for m in st.session_state.mensagens if m["role"] == "assistant"
    ]
    if not mensagens_assistente:
        st.info("Nenhuma fonte recuperada ainda.")
        return

    envelope = mensagens_assistente[-1].get("envelope")
    if not envelope or not envelope.fontes:
        st.info("A última resposta não usou nenhuma fonte externa.")
        return

    st.caption("Fontes usadas na última resposta:")
    for fonte in envelope.fontes:
        st.markdown(f"**[{fonte.origem.value}]** {fonte.citacao}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🌿 Assistente de Plantas Medicinais")
    st.caption(
        "Informação educacional sobre 6 plantas medicinais — não "
        "substitui orientação de um profissional de saúde."
    )

    try:
        settings.validar_configuracao_obrigatoria()
    except ValueError as e:
        st.warning(
            f"Configuração incompleta no servidor: {e}. Você ainda pode "
            f"preencher a chave de API na barra lateral para testar."
        )

    _inicializar_estado_sessao()
    config = _renderizar_sidebar()

    aba_chat, aba_observabilidade, aba_fontes = st.tabs(
        ["💬 Chat", "📊 Observabilidade", "📚 Fontes"]
    )
    with aba_chat:
        _renderizar_aba_chat(config)
    with aba_observabilidade:
        _renderizar_aba_observabilidade()
    with aba_fontes:
        _renderizar_aba_fontes()


if __name__ == "__main__":
    main()
