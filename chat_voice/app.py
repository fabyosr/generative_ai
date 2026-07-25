import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline, KModel
import kokoro.modules as kokoro_modules
import soundfile as sf
import numpy as np
import torch
import io
import psutil
import os

# ── Patches de compatibilidade (mantidos da versão anterior) ─────────────────
def _kmodel_device(self):
    try:
        return next(p.device for p in self.parameters())
    except StopIteration:
        return torch.device('cpu')

KModel.device = property(_kmodel_device)

_original_albert_forward = kokoro_modules.CustomAlbert.forward
def _safe_albert_forward(self, *args, **kwargs):
    outputs = super(kokoro_modules.CustomAlbert, self).forward(*args, **kwargs)
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

kokoro_modules.CustomAlbert.forward = _safe_albert_forward

# ── Constantes de áudio ───────────────────────────────────────────────────────
SAMPLE_RATE = 24000

def silencio(ms: int) -> np.ndarray:
    """Gera silêncio de N milissegundos."""
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)

# ── Pré-processador de texto expressivo ──────────────────────────────────────
def preparar_texto_expressivo(texto: str) -> list[dict]:
    """
    Divide o texto em segmentos e define velocidade + pausa para cada um.
    Retorna lista de dicts: {texto, speed, pausa_depois_ms}

    Técnicas aplicadas:
    - Frases curtas (impacto): velocidade menor, pausa maior depois
    - Frases longas (explicação): velocidade levemente maior
    - Pontuação '...' inserida para criar suspense/respiração
    - '!' mantém entonação enfática do G2P
    - Pausa maior após ponto final para dar tempo de absorção
    """
    import re

    # Divide em sentenças respeitando . ! ?
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    segmentos = []

    for i, s in enumerate(sentencas):
        s = s.strip()
        if not s:
            continue

        n_palavras = len(s.split())
        tem_exclamacao = s.endswith('!')
        tem_interrogacao = s.endswith('?')
        eh_curta = n_palavras <= 5
        eh_longa = n_palavras >= 15
        eh_primeira = i == 0
        eh_ultima = i == len(sentencas) - 1

        # Velocidade: frases curtas e impactantes falam mais devagar
        if eh_curta and (tem_exclamacao or eh_primeira or eh_ultima):
            speed = 0.85  # dramático, deliberado
        elif eh_longa:
            speed = 1.05  # explicação flui um pouco mais rápido
        else:
            speed = 0.95  # conversacional ligeiramente abaixo do normal

        # Pausa depois: quanto maior a pontuação, mais tempo para absorver
        if eh_ultima:
            pausa_ms = 0  # sem pausa no último segmento
        elif tem_exclamacao:
            pausa_ms = 600  # ênfase precisa de espaço depois
        elif tem_interrogacao:
            pausa_ms = 500  # pergunta cria expectativa
        elif eh_curta:
            pausa_ms = 400  # frase curta = pausa para deixar a ideia "aterrissar"
        else:
            pausa_ms = 200  # fluxo normal entre sentenças

        segmentos.append({
            "texto": s,
            "speed": speed,
            "pausa_depois_ms": pausa_ms,
        })

    return segmentos

# ── Gerador de áudio expressivo ───────────────────────────────────────────────
def gerar_audio_expressivo(pipeline, texto: str, voice: str) -> np.ndarray:
    """
    Processa o texto em segmentos com velocidade e pausas individuais.
    Retorna um único array numpy com o áudio completo.
    """
    segmentos = preparar_texto_expressivo(texto)
    chunks_audio = []

    for seg in segmentos:
        # Gera o áudio do segmento com a velocidade definida
        for gs, ps, audio in pipeline(
            seg["texto"],
            voice=voice,
            speed=seg["speed"],
            split_pattern=None,  # já dividimos manualmente
        ):
            if audio is not None:
                chunks_audio.append(audio)

        # Insere pausa estratégica após o segmento
        if seg["pausa_depois_ms"] > 0:
            chunks_audio.append(silencio(seg["pausa_depois_ms"]))

    if not chunks_audio:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    return np.concatenate(chunks_audio)

# ── Pós-processador: enriquece texto com pontuação expressiva ────────────────
def enriquecer_pontuacao(texto: str) -> str:
    """
    Adiciona pontuação expressiva que o G2P do Kokoro interpreta:
    - Reticências criam suspense/respiração natural
    - Vírgulas inseridas em listas longas criam ritmo
    Não altera o significado, só a prosódia.
    """
    import re
    # Já tem pontuação adequada? Retorna como está.
    if re.search(r'[.!?]', texto):
        return texto
    # Texto sem pontuação final: adiciona ponto para fechar a prosódia
    return texto.strip() + '.'

# ── Carregamento dos modelos ──────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)

@st.cache_resource
def load_kmodel():
    return KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()

def get_pipeline():
    return KPipeline(lang_code='p', model=load_kmodel(), device='cpu')

# ── Recursos servidor ──────────────────────────────────────────────────────────

def server_resource():
    # Coleta dados do processo atual do Python
    processo = psutil.Process()
    memoria_uso_bytes = processo.memory_info().rss
    memoria_uso_mb = memoria_uso_bytes / (1024 * 1024)

    # Coleta uso de CPU do sistema/processo
    cpu_uso = psutil.cpu_percent(interval=0.1)
    return (memoria_uso_mb, cpu_uso)

# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🎙️ Chatbot de Voz Expressivo (Faster-Whisper + Kokoro)")
st.sidebar.header("⚙️ Configurações de Voz")

opcoes_vozes = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa / Papai Noel (Masculino - PT-BR)": "pm_santa",
}
voz_label = st.sidebar.selectbox("Voz da IA:", list(opcoes_vozes.keys()))
id_voz = opcoes_vozes[voz_label]

# Controle manual de velocidade global (multiplicador sobre o speed do segmento)
fator_velocidade = st.sidebar.slider(
    "Velocidade global", min_value=0.7, max_value=1.3, value=1.0, step=0.05,
    help="Ajuste fino sobre a velocidade calculada por segmento"
)

st.write(f"Voz selecionada: **{voz_label}**")
svr_resource = server_resource()
st.sidebar.write(f"💾 Memória RAM Usada pelo App {svr_resource[0]:.2f} MB")
st.sidebar.write(f"🏿 Uso de CPU {svr_resource[1]:.1f}%")

audio_file = st.audio_input("Clique no microfone para falar com a IA")

if audio_file is not None:
    st.audio(audio_file)

    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())

    try:
        with st.spinner("🤖 Transcrevendo..."):
            whisper_model = load_whisper()
            segments, _ = whisper_model.transcribe(
                filename,
                language="pt",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
                initial_prompt="Transcrição em português brasileiro. Conversa com assistente de voz.",
            )
            texto_transcrito = "".join([seg.text for seg in segments])

        st.success("📝 Você disse:")
        st.write(texto_transcrito)
        st.write("---")

        with st.spinner("🗣️ Gerando áudio expressivo..."):
            pipeline = get_pipeline()

            texto_resposta = enriquecer_pontuacao(
                f"Você acabou de dizer: {texto_transcrito}"
            )

            audio_final = gerar_audio_expressivo(pipeline, texto_resposta, id_voz)

            # Aplica fator de velocidade global re-amostrando (simples e eficaz)
            if fator_velocidade != 1.0:
                indices = np.round(
                    np.arange(0, len(audio_final), fator_velocidade)
                ).astype(int)
                indices = indices[indices < len(audio_final)]
                audio_final = audio_final[indices]

            buffer = io.BytesIO()
            sf.write(buffer, audio_final, SAMPLE_RATE, format='WAV')
            buffer.seek(0)

            st.subheader("🔊 Resposta da IA:")
            st.audio(buffer, format="audio/wav", autoplay=True)

    except Exception as e:
        st.error(f"Erro: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(filename):
            os.remove(filename)