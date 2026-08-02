import streamlit as st
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import numpy as np
import io
import os

SAMPLE_RATE = 24000
REPO_ID     = "onnx-community/Kokoro-82M-v1.0-ONNX"

# Vozes PT-BR disponíveis no repositório
VOZES_PTBR = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa (Masculino - PT-BR)": "pm_santa",
}

# ── Utilitários expressivos ───────────────────────────────────────────────────
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
        eh_ultima = i == len(sentencas) - 1
        speed    = 0.85 if (n <= 5 and s.endswith('!')) else 1.05 if n >= 15 else 0.95
        pausa_ms = (0   if eh_ultima       else
                    600 if s.endswith('!') else
                    500 if s.endswith('?') else
                    400 if n <= 5          else 200)
        segmentos.append({"texto": s, "speed": speed, "pausa_ms": pausa_ms})
    return segmentos

# ── Carregamento dos modelos ──────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)

@st.cache_resource
def load_kokoro():
    """
    1. Baixa o modelo ONNX quantizado (q8f16 = 86 MB) do HuggingFace
    2. Baixa cada voz PT-BR como .bin individual (~512 KB cada)
    3. Consolida os .bin num único .npz que o kokoro-onnx consegue ler
    4. Inicializa Kokoro(model_path, voices_npz_path)
    """

    # 1. Modelo quantizado
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="onnx/model_q8f16.onnx",   # 86 MB — melhor custo-benefício
    )

    # 2. Baixar cada .bin de voz PT-BR
    vozes_arrays = {}
    for nome_voz in VOZES_PTBR.values():
        bin_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"voices/{nome_voz}.bin",
        )
        # Cada .bin é uma matriz float32 de shape (N, 1, 256)
        # N = número de tamanhos de sequência suportados (até 510)
        arr = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 1, 256)
        vozes_arrays[nome_voz] = arr

    # 3. Consolidar num .npz para que np.load() retorne objeto indexável por string
    npz_path = "/tmp/kokoro_voices_ptbr.npz"
    np.savez(npz_path, **vozes_arrays)

    # 4. Inicializar com a API real do kokoro-onnx
    return Kokoro(model_path=model_path, voices_path=npz_path)

# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🎙️ Chatbot de Voz (Faster-Whisper + Kokoro ONNX q8f16)")
st.sidebar.header("⚙️ Configurações")

voz_label = st.sidebar.selectbox("Voz:", list(VOZES_PTBR.keys()))
id_voz    = VOZES_PTBR[voz_label]
st.write(f"Voz: **{voz_label}**")

audio_file = st.audio_input("🎤 Clique para falar")

if audio_file is not None:
    st.audio(audio_file)

    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())

    try:
        # --- TRANSCRIÇÃO ---
        with st.spinner("🤖 Transcrevendo..."):
            whisper  = load_whisper()
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
            kokoro   = load_kokoro()
            resposta = f"Você disse: {texto}."
            chunks   = []

            for seg in preparar_segmentos(resposta):
                audio_seg, _ = kokoro.create(
                    seg["texto"],
                    voice=id_voz,
                    speed=seg["speed"],
                    lang="pt-br",
                )
                chunks.append(audio_seg)
                if seg["pausa_ms"] > 0:
                    chunks.append(silencio(seg["pausa_ms"]))

            audio_final = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE)

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