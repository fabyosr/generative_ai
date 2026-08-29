import streamlit as st
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as rt
import soundfile as sf
import numpy as np
import io
import os
import psutil

# Patch pontual: model_quantized.onnx exportação antiga
# espera speed como float32, mas o branch 'input_ids' do kokoro-onnx
# envia int32 — sobrescrevemos só o _create_audio para corrigir o dtype
import types

SAMPLE_RATE = 24000
REPO_ID     = "onnx-community/Kokoro-82M-v1.0-ONNX"

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
        n         = len(s.split())
        eh_ultima = i == len(sentencas) - 1
        speed     = 0.85 if (n <= 5 and s.endswith('!')) else 1.05 if n >= 15 else 0.95
        pausa_ms  = (0   if eh_ultima       else
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
    from kokoro_onnx import Kokoro
    from kokoro_onnx.session import create_session

    model_path = hf_hub_download(repo_id=REPO_ID, filename="onnx/model_quantized.onnx")

    # Baixar vozes PT-BR e consolidar num .npz
    vozes_arrays = {}
    for nome_voz in VOZES_PTBR.values():
        bin_path = hf_hub_download(repo_id=REPO_ID, filename=f"voices/{nome_voz}.bin")
        vozes_arrays[nome_voz] = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 1, 256)

    # Salvar no diretório do próprio app — gravável no Streamlit Community
    # e persistente dentro da mesma sessão do worker (ao contrário do /tmp)
    voices_npz_path = os.path.join(os.path.dirname(__file__), "voices_ptbr.npz")
    np.savez(voices_npz_path, **vozes_arrays)

    # from_session chama _setup corretamente -- inicializa _tokens_input,
    # _input_dtypes, _stops, _spaces e todos os atributos necessários
    opts = rt.SessionOptions()
    opts.intra_op_num_threads     = 2
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC

    session = create_session(model_path)
    return Kokoro.from_session(session, voices_path=voices_npz_path)
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
st.title("🎙️ Chatbot de Voz (Faster-Whisper + Kokoro ONNX q8f16)")
st.sidebar.header("⚙️ Configurações")

voz_label = st.sidebar.selectbox("Voz:", list(VOZES_PTBR.keys()))
id_voz    = VOZES_PTBR[voz_label]
st.write(f"Voz: **{voz_label}**")
server_resource()

audio_file = st.audio_input("🎤 Clique para falar")

if audio_file is not None:
    st.audio(audio_file)

    filename = "temp_input.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())

    try:
        # ── Diagnóstico do áudio recebido ─────────────────────────────────
        audio_np, sr_in = sf.read(filename)
        duracao  = len(audio_np) / sr_in
        amplitude = float(np.abs(audio_np).max())
        st.caption(
            f"Áudio recebido: {duracao:.1f}s | "
            f"Sample rate: {sr_in} Hz | "
            f"Amplitude máx: {amplitude:.4f}"
        )
        if amplitude < 0.01:
            st.warning("⚠️ Áudio muito silencioso — verifique o microfone.")

        # ── Transcrição ───────────────────────────────────────────────────
        with st.spinner("🤖 Transcrevendo..."):
            whisper = load_whisper()
            segments, info = whisper.transcribe(
                filename,
                language="pt",
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=(
                    "Transcrição de fala em português do Brasil. "
                    "O usuário está conversando com um assistente de voz."
                ),
                # VAD DESLIGADO — st.audio_input já entrega áudio limpo de microfone.
                # O VAD estava eliminando toda a fala antes do Whisper processar.
                vad_filter=False,
            )
            # Forçar avaliação do generator (segments é lazy no faster-whisper)
            lista_segmentos = list(segments)
            texto = "".join([s.text for s in lista_segmentos])

        # Diagnóstico da transcrição
        st.caption(
            f"Idioma detectado: {info.language} "
            f"(confiança: {info.language_probability:.0%}) | "
            f"Segmentos: {len(lista_segmentos)}"
        )

        if not texto.strip():
            st.warning(
                "⚠️ Nenhum texto transcrito. Possíveis causas:\n"
                "- Fala muito curta ou baixa\n"
                f"- Idioma detectado como '{info.language}' em vez de 'pt'\n"
                "- `no_speech_threshold` muito alto (atual: 0.6)"
            )
        else:
            st.success("📝 Você disse:")
            st.write(texto)
            st.write("---")

            # ── Síntese de voz ────────────────────────────────────────────
            with st.spinner("🗣️ Gerando áudio..."):
                kokoro_instance = load_kokoro()
                kokoro_instance._create_audio = types.MethodType(kokoro_instance)
                resposta = f"Você disse: {texto}."
                chunks   = []
                st.write('load kokoro')
                print('load kokoro')

                for seg in preparar_segmentos(resposta):
                    audio_seg, _ = kokoro_instance.create(
                        seg["texto"],
                        voice=id_voz,
                        speed=seg["speed"],
                        lang="pt-br",
                        )
                    chunks.append(audio_seg)
                    if seg["pausa_ms"] > 0:
                        chunks.append(audio_seg.squeeze())  # (1, N) → (N,)
                        # chunks.append(silencio(seg["pausa_ms"]))

                print('passou x')

                #audio_final = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE)
                audio_final = np.concatenate([c.squeeze() for c in chunks]) if chunks else np.zeros(SAMPLE_RATE)
                audio_final = np.squeeze(audio_final)          # garante shape (N,) em qualquer caso
                audio_final = audio_final.astype(np.float32)   # garante dtype correto
                print('grou audio_final')

                buffer = io.BytesIO()
                sf.write(buffer, audio_final, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                buffer.seek(0)
                print('buffer.seek')

                st.subheader("🔊 Resposta:")
                st.audio(buffer, format="audio/wav", autoplay=True)

    except Exception as e:
        st.error(f"Erro: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(filename):
            os.remove(filename)