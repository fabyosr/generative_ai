import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline
import torch  # Importamos o PyTorch para gerenciar o hardware corretamente
import soundfile as sf
import io
import os

# 1. Mantemos o cache para o Whisper porque ele funciona perfeitamente com o Streamlit
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. NOVA ABORDAGEM: Inicializador explícito sem cache polimórfico
def get_kokoro_pipeline():
    # Forçamos o PyTorch a rodar em CPU antes de criar o pipeline
    device = "cpu"
    # Inicializamos o pipeline informando o idioma Português ('p')
    # O Kokoro carrega os pesos em memória de forma leve na CPU
    pipeline = KPipeline(lang_code='p', device=device)
    return pipeline

st.title("🎙️ Chatbot de Voz Otimizado (Faster-Whisper + Kokoro)")

# --- CONFIGURAÇÃO DE VOZES NA BARRA LATERAL ---
st.sidebar.header("⚙️ Configurações de Voz")

opcoes_vozes = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa / Papai Noel (Masculino - PT-BR)": "pm_santa"
}

voz_selecionada_label = st.sidebar.selectbox("Escolha a voz da IA:", list(opcoes_vozes.keys()))
id_da_voz = opcoes_vozes[voz_selecionada_label]

st.write(f"A IA responderá usando a voz: **{voz_selecionada_label}**")

# --- COMPONENTE DE ÁUDIO NATIVO ---
audio_file = st.audio_input("Clique no microfone para falar com a IA")

if audio_file is not None:
    st.audio(audio_file)
    
    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())
        
    # --- PROCESSO 1: TRANSCRIÇÃO (STT) ---
    with st.spinner("🤖 Transcrevendo o que você disse..."):
        try:
            whisper_model = load_whisper()
            segments, info = whisper_model.transcribe(filename, beam_size=5, language="pt")
            texto_transcrito = "".join([segment.text for segment in segments])
            
            st.success("📝 Você disse:")
            st.write(texto_transcrito)
            
            st.write("---")
            
            # --- PROCESSO 2: SÍNTESE DE VOZ REALISTA (TTS) ---
            with st.spinner("🗣️ Kokoro gerando resposta realista em áudio..."):
                # Chamamos a função sem o cache do Streamlit para evitar o erro do "device"
                pipeline = get_kokoro_pipeline()
                
                texto_resposta = f"Você acabou de dizer: {texto_transcrito}"
                
                # O Kokoro processa o áudio em formato de gerador (generator)
                generator = pipeline(texto_resposta, voice=id_da_voz, speed=1.0, split_pattern=r'\n+')
                
                # Coleta e une os pedaços de áudio gerados pela IA
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
