from faster_whisper import WhisperModel
from kokoro import KPipeline, KModel
import kokoro.modules as kokoro_modules
import soundfile as sf
import numpy as np
import torch
import io
import psutil
import os

# ── Patches de compatibilidade (mantidos da versão anterior) ─────────────────
def _kmodel_device(self):
    try:
        return next(p.device for p in self.parameters())
    except StopIteration:
        return torch.device('cpu')

KModel.device = property(_kmodel_device)

_original_albert_forward = kokoro_modules.CustomAlbert.forward
def _safe_albert_forward(self, *args, **kwargs):
    outputs = super(kokoro_modules.CustomAlbert, self).forward(*args, **kwargs)
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

kokoro_modules.CustomAlbert.forward = _safe_albert_forward

# ── Constantes de áudio ───────────────────────────────────────────────────────
SAMPLE_RATE = 24000

def silencio(ms: int) -> np.ndarray:
    """Gera silêncio de N milissegundos."""
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)

# ── Pré-processador de texto expressivo ──────────────────────────────────────
def preparar_texto_expressivo(texto: str) -> list[dict]:
    """
    Divide o texto em segmentos e define velocidade + pausa para cada um.
    Retorna lista de dicts: {texto, speed, pausa_depois_ms}

    Técnicas aplicadas:
    - Frases curtas (impacto): velocidade menor, pausa maior depois
    - Frases longas (explicação): velocidade levemente maior
    - Pontuação '...' inserida para criar suspense/respiração
    - '!' mantém entonação enfática do G2P
    - Pausa maior após ponto final para dar tempo de absorção
    """
    import re

    # Divide em sentenças respeitando . ! ?
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    segmentos = []

    for i, s in enumerate(sentencas):
        s = s.strip()
        if not s:
            continue

        n_palavras = len(s.split())
        tem_exclamacao = s.endswith('!')
        tem_interrogacao = s.endswith('?')
        eh_curta = n_palavras <= 5
        eh_longa = n_palavras >= 15
        eh_primeira = i == 0
        eh_ultima = i == len(sentencas) - 1

        # Velocidade: frases curtas e impactantes falam mais devagar
        if eh_curta and (tem_exclamacao or eh_primeira or eh_ultima):
            speed = 0.85  # dramático, deliberado
        elif eh_longa:
            speed = 1.05  # explicação flui um pouco mais rápido
        else:
            speed = 0.95  # conversacional ligeiramente abaixo do normal

        # Pausa depois: quanto maior a pontuação, mais tempo para absorver
        if eh_ultima:
            pausa_ms = 0  # sem pausa no último segmento
        elif tem_exclamacao:
            pausa_ms = 600  # ênfase precisa de espaço depois
        elif tem_interrogacao:
            pausa_ms = 500  # pergunta cria expectativa
        elif eh_curta:
            pausa_ms = 400  # frase curta = pausa para deixar a ideia "aterrissar"
        else:
            pausa_ms = 200  # fluxo normal entre sentenças

        segmentos.append({
            "texto": s,
            "speed": speed,
            "pausa_depois_ms": pausa_ms,
        })

    return segmentos

# ── Gerador de áudio expressivo ───────────────────────────────────────────────
def gerar_audio_expressivo(pipeline, texto: str, voice: str) -> np.ndarray:
    """
    Processa o texto em segmentos com velocidade e pausas individuais.
    Retorna um único array numpy com o áudio completo.
    """
    segmentos = preparar_texto_expressivo(texto)
    chunks_audio = []

    for seg in segmentos:
        # Gera o áudio do segmento com a velocidade definida
        for gs, ps, audio in pipeline(
            seg["texto"],
            voice=voice,
            speed=seg["speed"],
            split_pattern=None,  # já dividimos manualmente
        ):
            if audio is not None:
                chunks_audio.append(audio)

        # Insere pausa estratégica após o segmento
        if seg["pausa_depois_ms"] > 0:
            chunks_audio.append(silencio(seg["pausa_depois_ms"]))

    if not chunks_audio:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    return np.concatenate(chunks_audio)

# ── Pós-processador: enriquece texto com pontuação expressiva ────────────────
def enriquecer_pontuacao(texto: str) -> str:
    """
    Adiciona pontuação expressiva que o G2P do Kokoro interpreta:
    - Reticências criam suspense/respiração natural
    - Vírgulas inseridas em listas longas criam ritmo
    Não altera o significado, só a prosódia.
    """
    import re
    # Já tem pontuação adequada? Retorna como está.
    if re.search(r'[.!?]', texto):
        return texto
    # Texto sem pontuação final: adiciona ponto para fechar a prosódia
    return texto.strip() + '.'

# ── Carregamento dos modelos ──────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)

@st.cache_resource
def load_kmodel():
    return KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()

def get_pipeline():
    return KPipeline(lang_code='p', model=load_kmodel(), device='cpu')

opcoes_vozes = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa / Papai Noel (Masculino - PT-BR)": "pm_santa",
}

def tts(texto_resposta, id_voz = "pf_dora"):
	try:
	    pipeline = get_pipeline()
	    audio_final = gerar_audio_expressivo(pipeline, texto_resposta, id_voz)
	    buffer = io.BytesIO()
	    sf.write(buffer, audio_final, SAMPLE_RATE, format='WAV')
	    buffer.seek(0)
	    return buffer
	except Exception as e:
		return -1
	    # st.error(f"Erro: {e}")
	    # import traceback
	    # st.code(traceback.format_exc())
