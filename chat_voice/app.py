import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline
import soundfile as sf
import io
import os

# 1. Carregamento do Whisper
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. Pipeline do Kokoro — inicializado uma única vez e cacheado
#    KPipeline cuida internamente do KModel; não precisamos instanciar KModel à mão.
@st.cache_resource
def load_kokoro_pipeline():
    return KPipeline(lang_code='p')   # 'p' = Português Brasileiro

# --- INTERFACE ---
st.title("🎙️ Chatbot de Voz Otimizado (Faster-Whisper + Kokoro)")
st.sidebar.header("⚙️ Configurações de Voz")

opcoes_vozes = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa / Papai Noel (Masculino - PT-BR)": "pm_santa"
}
voz_selecionada_label = st.sidebar.selectbox("Escolha a voz da IA:", list(opcoes_vozes.keys()))
id_da_voz = opcoes_vozes[voz_selecionada_label]
st.write(f"A IA responderá usando a voz: **{voz_selecionada_label}**")

audio_file = st.audio_input("Clique no microfone para falar com a IA")

if audio_file is not None:
    st.audio(audio_file)

    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())

    with st.spinner("🤖 Transcrevendo o que você disse..."):
        try:
            whisper_model = load_whisper()
            segments, info = whisper_model.transcribe(filename, beam_size=5, language="pt")
            texto_transcrito = "".join([segment.text for segment in segments])

            st.success("📝 Você disse:")
            st.write(texto_transcrito)
            st.write("---")

            with st.spinner("🗣️ Kokoro gerando resposta em áudio..."):
                pipeline = load_kokoro_pipeline()
                texto_resposta = f"Você acabou de dizer: {texto_transcrito}"

                generator = pipeline(
                    texto_resposta,
                    voice=id_da_voz,
                    speed=1.0,
                    split_pattern=r'\n+'
                )

                for gs, ps, audio in generator:
                    buffer = io.BytesIO()
                    sf.write(buffer, audio, 24000, format='WAV')
                    buffer.seek(0)

                    st.subheader("🔊 Resposta da IA (Autoplay):")
                    st.audio(buffer, format="audio/wav", autoplay=True)

        except Exception as e:
            st.error(f"Ocorreu um erro no processamento: {e}")

        finally:
            if os.path.exists(filename):
                os.remove(filename)