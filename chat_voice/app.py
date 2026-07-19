import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline, KModel
import soundfile as sf
import io
import os

# --- PASSO A PASSO DA RESOLUÇÃO ---

# 1. Carregamento do Whisper (Funciona bem isolado com cache)
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. RESOLUÇÃO COMPLETA DO BUG DO DEVICE:
# Criamos o modelo puro fora do pipeline e injetamos o atributo em nível de classe/objeto.
@st.cache_resource
def get_clean_kokoro_model():
    # Instancia o modelo puro do Kokoro
    model = KModel()
    
    # Injetamos explicitamente a propriedade "device" diretamente na estrutura base 
    # do objeto para saciar a busca interna exigida pelas threads do Streamlit
    object.__setattr__(model, 'device', 'cpu') 
    
    # Colocamos o modelo em modo de avaliação (padrão PyTorch para inferência)
    model.eval()
    return model

# 3. Construção do Pipeline passando o modelo blindado acima
def get_kokoro_pipeline():
    # Buscamos nossa instância limpa do modelo da CPU
    core_model = get_clean_kokoro_model()
    
    # Inicializamos o pipeline do idioma ('p' para Português) injetando o modelo customizado
    return KPipeline(lang_code='p', model=core_model)


# --- INTERFACE VISUAL DO STREAMLIT ---

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

# Componente Nativo do Streamlit para Microfone
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
                pipeline = get_kokoro_pipeline()
                
                texto_resposta = f"Você acabou de dizer: {texto_transcrito}"
                
                # Executa a geração usando a nossa estrutura blindada
                generator = pipeline(texto_resposta, voice=id_da_voz, speed=1.0, split_pattern=r'\n+')
                
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
