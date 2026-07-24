import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline, KModel
import soundfile as sf
import torch
import io
import os

def _kmodel_device(self):
    """
    Substitui a property device original que faz self.bert.device.
    Usa os parâmetros do próprio KModel -- mais robusto após deserialização
    pelo cache do Streamlit, onde self.bert pode não ter parâmetros acessíveis.
    """
    try:
        return next(p.device for p in self.parameters())
    except StopIteration:
        return torch.device('cpu')

# Aplica o patch NA CLASSE antes de qualquer instância ser criada
KModel.device = property(_kmodel_device)

@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

@st.cache_resource
def load_kmodel():
    """
    Cacheia APENAS o KModel (os pesos pesados).
    O patch acima garante que .device sempre funciona,
    mesmo após o pickle/unpickle do @st.cache_resource.
    """
    return KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()

def get_pipeline():
    """
    KPipeline é leve (só G2P). Criado a cada chamada usando
    o KModel já cacheado -- evita qualquer problema de serialização
    de objetos compostos.
    """
    kmodel = load_kmodel()
    return KPipeline(lang_code='p', model=kmodel, device='cpu')

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

    try:
        with st.spinner("🤖 Transcrevendo..."):
            whisper_model = load_whisper()
            segments, _ = whisper_model.transcribe(filename, beam_size=5, language="pt")
            texto_transcrito = "".join([seg.text for seg in segments])

        st.success("📝 Você disse:")
        st.write(texto_transcrito)
        st.write("---")

        with st.spinner("🗣️ Gerando áudio..."):
            pipeline = get_pipeline()
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
                st.subheader("🔊 Resposta da IA:")
                st.audio(buffer, format="audio/wav", autoplay=True)

    except Exception as e:
        st.error(f"Erro: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(filename):
            os.remove(filename)