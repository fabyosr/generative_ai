import streamlit as st
from faster_whisper import WhisperModel
from kokoro import KPipeline, KModel
import kokoro.modules as kokoro_modules
import soundfile as sf
import torch
import io
import os

# ── PATCH 1: device robusto ──────────────────────────────────────────────────
# KModel.device original faz self.bert.device → pode falhar se bert não tiver
# parâmetros acessíveis após deserialização pelo cache do Streamlit.
def _kmodel_device(self):
    try:
        return next(p.device for p in self.parameters())
    except StopIteration:
        return torch.device('cpu')

KModel.device = property(_kmodel_device)

# ── PATCH 2: CustomAlbert compatível com qualquer versão do transformers ─────
# transformers < 4.0 e algumas builds retornam tupla em vez de objeto com
# .last_hidden_state. Este patch trata os dois casos.
_original_albert_forward = kokoro_modules.CustomAlbert.forward

def _safe_albert_forward(self, *args, **kwargs):
    outputs = super(kokoro_modules.CustomAlbert, self).forward(*args, **kwargs)
    # transformers >= 4.0: retorna BaseModelOutputWithPooling (tem .last_hidden_state)
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    # transformers < 4.0 ou return_dict=False: retorna tupla (hidden_state, pooled, ...)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

kokoro_modules.CustomAlbert.forward = _safe_albert_forward

# ── Carregamento dos modelos ──────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return WhisperModel(
        "small",          # ← principal melhoria: tiny → small
        device="cpu",
        compute_type="int8",
        cpu_threads=2,    # Streamlit Community tem 2 vCPUs
    )

@st.cache_resource
def load_kmodel():
    return KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()

def get_pipeline():
    kmodel = load_kmodel()
    return KPipeline(lang_code='p', model=kmodel, device='cpu')

# ── Interface ─────────────────────────────────────────────────────────────────
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
            segments, info = whisper_model.transcribe(
                filename,
                language="pt",
                beam_size=5,
                best_of=5,
                temperature=0.0,                  # decodificação determinística
                condition_on_previous_text=False, # evita propagação de erros em turnos curtos
                vad_filter=True,                  # remove silêncio e ruído
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # silêncios > 500ms são cortados
                    threshold=0.5,                # sensibilidade do VAD (0-1)
                ),
                initial_prompt=(
                    "Transcrição em português brasileiro. "
                    "Conversa com assistente de voz."
                ),
            )
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