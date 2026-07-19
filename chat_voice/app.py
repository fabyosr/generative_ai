import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline, KModel
import soundfile as sf
import torch
import io
import os

@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

@st.cache_resource
def load_kokoro_pipeline():
    """
    Inicializa o KModel e KPipeline de forma explícita e segura.
    O erro real (falha de download HF, falta de bert, etc.) vai aparecer aqui,
    não mascarado pelo except genérico da UI.
    """
    # Força CPU antes de qualquer operação
    # map_location='cpu' no torch.load (feito internamente pelo KModel) garante
    # que os pesos não tentam ir para GPU inexistente
    kmodel = KModel(repo_id='hexgrad/Kokoro-82M').to(torch.device('cpu')).eval()

    # device='cpu' passado ao KPipeline evita que ele tente detectar CUDA
    pipeline = KPipeline(lang_code='p', model=kmodel, device='cpu')
    return pipeline

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

# Carrega o pipeline na inicialização do app (fora do fluxo de áudio)
# Erros de download/rede aparecem aqui como st.error visível, não mascarados
try:
    kokoro_pipeline = load_kokoro_pipeline()
except Exception as e:
    st.error(f"❌ Falha ao carregar o modelo Kokoro: {e}")
    st.info("Verifique se o Streamlit Community Cloud tem acesso ao HuggingFace Hub.")
    st.stop()

audio_file = st.audio_input("Clique no microfone para falar com a IA")

if audio_file is not None:
    st.audio(audio_file)

    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())

    try:
        # --- TRANSCRIÇÃO ---
        with st.spinner("🤖 Transcrevendo..."):
            whisper_model = load_whisper()
            segments, _ = whisper_model.transcribe(filename, beam_size=5, language="pt")
            texto_transcrito = "".join([seg.text for seg in segments])

        st.success("📝 Você disse:")
        st.write(texto_transcrito)
        st.write("---")

        # --- SÍNTESE DE VOZ ---
        with st.spinner("🗣️ Gerando áudio..."):
            texto_resposta = f"Você acabou de dizer: {texto_transcrito}"
            generator = kokoro_pipeline(
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
        # Agora o erro real aparece — não mais mascarado
        st.error(f"Erro no processamento: {e}")
        import traceback
        st.code(traceback.format_exc())  # mostra o traceback completo para diagnóstico

    finally:
        if os.path.exists(filename):
            os.remove(filename)