import streamlit as st
from st_audio_recorder import st_audio_recorder
import os

st.title("🎙️ Enviar Áudio para o Python")
st.write("Clique no botão abaixo para gravar sua voz:")

# Cria o componente visual de gravação de áudio
audio_bytes = st_audio_recorder()

# Verifica se o usuário gravou algo e terminou a gravação
if audio_bytes is not None:
    st.audio(audio_bytes, format="audio/wav")
    
    # Define o nome do arquivo onde o áudio será salvo
    filename = "audio_gravado.wav"
    
    # Salva o arquivo de áudio no servidor onde o Python está rodando
    with open(filename, "wb") as f:
        f.write(audio_bytes)
        
    st.success(f"✅ Áudio processado pelo Python e salvo como '{filename}'!")
    
    # --- Seu código Python entra aqui ---
    # Exemplo: Você pode ler o tamanho do arquivo ou mandar para uma API
    file_size = os.path.getsize(filename) / 1024
    st.info(f"O Python leu o arquivo. Tamanho: {file_size:.2f} KB")
