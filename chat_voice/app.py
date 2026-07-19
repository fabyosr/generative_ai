import streamlit as st
from faster_whisper import WhisperModel
from gtts import gTTS
import io
import os

# Configuração e cache do modelo de IA para a CPU da nuvem
@st.cache_resource
def load_whisper():
    # O modelo 'tiny' é o ideal para a nuvem gratuita do Streamlit (consome pouca memória)
    return WhisperModel("tiny", device="cpu", compute_type="int8")

st.title("🎙️ Aplicativo de Voz Completo na Nuvem")
st.write("Grave sua voz diretamente pelo navegador:")

# 1. Componente NATIVO de áudio do Streamlit (Substitui o st_audio_recorder)
audio_file = st.audio_input("Clique no microfone para gravar")

if audio_file is not None:
    # Mostra o player do que foi gravado
    st.audio(audio_file)
    
    # Salva o arquivo temporariamente para a IA ler
    filename = "temp_audio.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())
        
    # 2. Transcrição (Speech-to-Text) com Faster-Whisper
    with st.spinner("🤖 Transcrevendo seu áudio..."):
        try:
            model = load_whisper()
            segments, info = model.transcribe(filename, beam_size=5, language="pt")
            texto_transcrito = "".join([segment.text for segment in segments])
            
            st.success("📝 Texto Detectado:")
            st.write(texto_transcrito)
            
            # 3. Exemplo de Resposta (Text-to-Speech) com gTTS
            # Aqui fazemos a IA "repetir" o que você disse, mas você pode mudar o texto
            st.write("---")
            st.subheader("🗣️ Resposta em Áudio da IA:")
            
            texto_resposta = f"Você acabou de dizer: {texto_transcrito}"
            tts = gTTS(text=texto_resposta, lang="pt", tld="com.br")
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            st.audio(audio_buffer, format="audio/mp3", autoplay=True)
            
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            
        finally:
            # Garante que apaga o arquivo temporário da nuvem
            if os.path.exists(filename):
                os.remove(filename)
