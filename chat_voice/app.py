import streamlit as st
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np
import io
import os
import psutil

# Versão Código completo com ONNX + quantização

SAMPLE_RATE = 24000

# ── Utilitários expressivos (mantidos da versão anterior) ─────────────────────
def silencio(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)

def preparar_segmentos(texto: str) -> list[dict]:
    import re
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    segmentos = []
    for i, s in enumerate(sentencas):
        s = s.strip()
        if not s:
            continue
        n = len(s.split())
        tem_exclamacao = s.endswith('!')
        tem_interrogacao = s.endswith('?')
        eh_curta = n <= 5
        eh_ultima = i == len(sentencas) - 1

        speed = 0.85 if (eh_curta and tem_exclamacao) else 1.05 if n >= 15 else 0.95
        pausa_ms = 0 if eh_ultima else 600 if tem_exclamacao else 500 if tem_interrogacao else 400 if eh_curta else 200

        segmentos.append({"texto": s, "speed": speed, "pausa_ms": pausa_ms})
    return segmentos

# ── Carregamento dos modelos ──────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)

@st.cache_resource
def load_kokoro():
    """
    Baixa e carrega o modelo ONNX quantizado (q8f16 = 86 MB).
    Sem PyTorch, sem transformers, sem patches.
    """
    return Kokoro.from_pretrained(
        "onnx-community/Kokoro-82M-v1.0-ONNX",
        dtype="q8f16",   # opções: fp32, fp16, q8, q8f16, q4, q4f16
    )

# ── Recursos servidor ──────────────────────────────────────────────────────────

def server_resource():
    # Coleta dados do processo atual do Python
    processo = psutil.Process()
    memoria_uso_bytes = processo.memory_info().rss
    memoria_uso_mb = memoria_uso_bytes / (1024 * 1024)

    # Coleta uso de CPU do sistema/processo
    cpu_uso = psutil.cpu_percent(interval=0.1)
    
    # Get complete system memory statistics
    memory = psutil.virtual_memory()

    # Convert bytes to Gigabytes (GB) for easy reading
    gb = 1024 ** 3

    st.sidebar.write(f"💾 Memória RAM Usada {memoria_uso_mb:.2f} MB")
    st.sidebar.write(f"🏿 Uso de CPU {cpu_uso:.1f}%")


    st.sidebar.write(f"Total RAM:       {memory.total:.2f} GB")
    st.sidebar.write(f"Total RAM:       {memory.total / gb:.2f} GB")
    st.sidebar.write(f"Available RAM:   {memory.available / gb:.2f} GB")
    st.sidebar.write(f"Used RAM:        {memory.used / gb:.2f} GB")
    st.sidebar.write(f"Free RAM:        {memory.free / gb:.2f} GB")
    st.sidebar.write(f"RAM Usage:       {memory.percent}%")


# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🎙️ Chatbot de Voz (Faster-Whisper + Kokoro ONNX)")
st.sidebar.header("⚙️ Configurações")

opcoes_vozes = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa (Masculino - PT-BR)": "pm_santa",
}
voz_label = st.sidebar.selectbox("Voz:", list(opcoes_vozes.keys()))
id_voz = opcoes_vozes[voz_label]

st.write(f"Voz: **{voz_label}**")
server_resource()

audio_file = st.audio_input("🎤 Clique para falar")

if audio_file is not None:
    st.audio(audio_file)

    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())

    try:
        # --- TRANSCRIÇÃO ---
        with st.spinner("🤖 Transcrevendo..."):
            whisper = load_whisper()
            segments, _ = whisper.transcribe(
                filename,
                language="pt",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
                initial_prompt="Transcrição em português brasileiro.",
            )
            texto = "".join([s.text for s in segments])

        st.success("📝 Você disse:")
        st.write(texto)
        st.write("---")

        # --- SÍNTESE ONNX EXPRESSIVA ---
        with st.spinner("🗣️ Gerando áudio..."):
            kokoro = load_kokoro()
            resposta = f"Você disse: {texto}."
            segmentos = preparar_segmentos(resposta)

            chunks = []
            for seg in segmentos:
                # kokoro_onnx retorna (samples, sample_rate)
                audio_seg, sr = kokoro.create(
                    seg["texto"],
                    voice=id_voz,
                    speed=seg["speed"],
                    lang="pt-br",
                )
                chunks.append(audio_seg)
                if seg["pausa_ms"] > 0:
                    chunks.append(silencio(seg["pausa_ms"]))

            audio_final = np.concatenate(chunks) if chunks else np.zeros(sr)

            buffer = io.BytesIO()
            sf.write(buffer, audio_final, SAMPLE_RATE, format="WAV")
            buffer.seek(0)

            st.subheader("🔊 Resposta:")
            st.audio(buffer, format="audio/wav", autoplay=True)

    except Exception as e:
        st.error(f"Erro: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(filename):
            os.remove(filename)