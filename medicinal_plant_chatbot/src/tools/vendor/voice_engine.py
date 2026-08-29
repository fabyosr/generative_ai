"""
voice_engine.py
===============

Módulo genérico de Speech-to-Text (STT) e Text-to-Speech (TTS)
para uso em chatbots conversacionais.

Tecnologias:
- STT: faster-whisper (modelo "small", int8, CPU)
- TTS: Kokoro ONNX (modelo quantizado, vozes PT-BR)

Objetivo:
Extrair apenas a lógica de voz do app Streamlit original,
tornando-a reutilizável como funções puras em qualquer projeto Python.

Princípios deste módulo:
- Não altera o comportamento que já funciona no código original
- Modelos são carregados uma única vez (cache)
- Funções públicas simples e bem documentadas
- Código comentado de forma didática

Exemplo de uso básico:

from voice_engine import transcribe, synthesize, listar_vozes

# STT
resultado = transcribe(audio_bytes)          # ou caminho de arquivo
print(resultado["text"])

# TTS
wav_bytes = synthesize("Olá! Como posso te ajudar hoje?", voice="pf_dora")

"""

from __future__ import annotations

import io
import os
import re
import types
import tempfile
from functools import lru_cache
from typing import Union, Optional

import numpy as np
import soundfile as sf
import onnxruntime as rt
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from kokoro_onnx.session import create_session
from kokoro_onnx.tokenizer import Tokenizer


# =============================================================================
# CONSTANTES
# =============================================================================

SAMPLE_RATE = 24000
"""Taxa de amostragem padrão do Kokoro (24 kHz)."""

REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
"""Repositório Hugging Face que contém o modelo ONNX quantizado e as vozes."""

VOZES_PTBR = {
    "Dora (Feminina - PT-BR)": "pf_dora",
    "Alex (Masculino - PT-BR)": "pm_alex",
    "Santa (Masculino - PT-BR)": "pm_santa",
}
"""
Mapeamento amigável (nome legível → identificador interno do modelo).
Use o valor (ex: "pf_dora") ao chamar synthesize().
"""


# =============================================================================
# UTILITÁRIOS DE ÁUDIO E TEXTO
# =============================================================================

def silencio(ms: int) -> np.ndarray:
    """
    Gera um trecho de silêncio (zeros) com a duração solicitada.

    Por que existe:
    Permite inserir pausas naturais entre frases na síntese de voz,
    tornando a fala mais humana e menos "robótica".

    Args:
        ms: Duração do silêncio em milissegundos.

    Returns:
        Array numpy float32 com zeros (silêncio).
    """
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


def preparar_segmentos(texto: str) -> list[dict]:
    """
    Divide o texto em sentenças e calcula parâmetros expressivos
    (velocidade e pausa) para cada uma.

    Por que existe:
    O Kokoro gera áudio frase a frase. Esta função analisa o tamanho
    e a pontuação de cada sentença para:
    - Acelerar frases longas
    - Desacelerar frases curtas e exclamativas
    - Inserir pausas maiores após "!" e "?"

    Isso deixa a fala bem mais natural do que usar velocidade fixa.

    Args:
        texto: Texto completo que será sintetizado.

    Returns:
        Lista de dicionários no formato:
        [
            {
                "texto": "Frase aqui.",
                "speed": 0.95,
                "pausa_ms": 400
            },
            ...
        ]
    """
    # Divide o texto em sentenças respeitando pontuação final
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    segmentos = []

    for i, s in enumerate(sentencas):
        s = s.strip()
        if not s:
            continue

        n = len(s.split())                     # quantidade de palavras
        eh_ultima = i == len(sentencas) - 1    # última sentença não precisa de pausa

        # Lógica de velocidade (mesma do código original)
        if n <= 5 and s.endswith('!'):
            speed = 0.85
        elif n >= 15:
            speed = 1.05
        else:
            speed = 0.95

        # Lógica de pausa após a sentença
        if eh_ultima:
            pausa_ms = 0
        elif s.endswith('!'):
            pausa_ms = 600
        elif s.endswith('?'):
            pausa_ms = 500
        elif n <= 5:
            pausa_ms = 400
        else:
            pausa_ms = 200

        segmentos.append({
            "texto": s,
            "speed": speed,
            "pausa_ms": pausa_ms
        })

    return segmentos

# =============================================================================
# CARREGAMENTO DOS MODELOS (com cache)
# =============================================================================

@lru_cache(maxsize=1)
def load_whisper() -> WhisperModel:
    """
    Carrega o modelo Faster-Whisper uma única vez.

    Por que usar cache:
    O modelo "small" + int8 já ocupa uma boa quantidade de memória.
    Carregar toda vez que a função for chamada seria extremamente lento
    e consumiria muita RAM desnecessariamente.

    Configuração mantida do código original:
    - device="cpu"
    - compute_type="int8"
    - cpu_threads=2
    """
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=2
    )

@lru_cache(maxsize=1)
def load_kokoro() -> Kokoro:
    """
    Carrega o modelo Kokoro ONNX quantizado + vozes PT-BR uma única vez.

    Por que esta implementação é mais complexa:
    1. Baixa o modelo quantizado (int8 puro – ~92 MB)
    2. Baixa os arquivos .bin de cada voz
    3. Empacota as vozes em um arquivo .npz em memória
    4. Cria a sessão ONNX com otimizações básicas (importante em CPUs sem suporte a fp16)

    Retorna uma instância pronta para uso.
    """
    # 1. Baixa o modelo quantizado
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

# =============================================================================
# FUNÇÕES PÚBLICAS (API do módulo)
# =============================================================================

def transcribe(
    audio: Union[str, bytes, io.BytesIO],
    language: str = "pt",
    initial_prompt: Optional[str] = None,
) -> dict:
    """
    Converte áudio em texto (Speech-to-Text).

    Aceita três tipos de entrada:
    - Caminho de arquivo (str)
    - Bytes do áudio (bytes)
    - Objeto BytesIO

    Por que esta assinatura:
    Em chatbots reais o áudio quase sempre chega como bytes
    (gravado pelo navegador, WhatsApp, Telegram, etc.).
    Aceitar bytes evita ter que salvar arquivo temporário na maioria dos casos.

    Args:
        audio: Áudio de entrada (path, bytes ou BytesIO).
        language: Código do idioma (default "pt").
        initial_prompt: Texto de contexto para melhorar a transcrição
                        (opcional). Se None, usa o prompt padrão do código original.

    Returns:
        Dicionário com:
        {
            "text": "texto transcrito completo",
            "language": "pt",
            "language_probability": 0.97,
            "segments": [ ... lista de segmentos do Whisper ... ]
        }
    """
    # Define o prompt padrão (mesmo do código original)
    if initial_prompt is None:
        initial_prompt = (
            "Transcrição de fala em português do Brasil. "
            "O usuário está conversando com um assistente de voz."
        )

    # Se receber bytes ou BytesIO, grava em arquivo temporário
    # (faster-whisper trabalha melhor com path de arquivo)
    temp_path = None
    try:
        if isinstance(audio, (bytes, io.BytesIO)):
            # Cria arquivo temporário
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                if isinstance(audio, bytes):
                    tmp.write(audio)
                else:
                    tmp.write(audio.read())
                temp_path = tmp.name
            audio_path = temp_path
        else:
            audio_path = audio  # já é um path

        whisper = load_whisper()

        segments, info = whisper.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            vad_filter=False,  # mantido desligado (mesmo comportamento original)
        )

        # Força avaliação do generator (segments é lazy)
        lista_segmentos = list(segments)
        texto = "".join([s.text for s in lista_segmentos]).strip()

        return {
            "text": texto,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": lista_segmentos,
        }

    finally:
        # Limpa arquivo temporário se foi criado
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def synthesize(
    text: str,
    voice: str = "pf_dora",
    lang: str = "pt-br",
) -> bytes:
    """
    Converte texto em áudio (Text-to-Speech) e retorna os bytes de um arquivo WAV.

    Por que retornar bytes:
    - Fácil de enviar para o frontend (base64, streaming, etc.)
    - Fácil de salvar em arquivo
    - Não depende de Streamlit nem de nenhum framework

    A função aplica a mesma lógica expressiva do código original:
    - Divide o texto em sentenças
    - Ajusta velocidade e pausas conforme o conteúdo
    - Concatena tudo em um único áudio

    Args:
        text: Texto que será falado.
        voice: Identificador da voz ("pf_dora", "pm_alex" ou "pm_santa").
        lang: Idioma (default "pt-br").

    Returns:
        Bytes de um arquivo WAV (PCM 16-bit, 24 kHz).
    """
    if not text or not text.strip():
        # Retorna silêncio curto se o texto estiver vazio
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()

    kokoro = load_kokoro()

    # Gera os segmentos expressivos
    segmentos = preparar_segmentos(text)

    chunks = []
    for seg in segmentos:
        audio_seg, _ = kokoro.create(
            seg["texto"],
            voice=voice,
            speed=seg["speed"],
            lang=lang,
        )
        chunks.append(audio_seg)

        # Observação: no código original a linha de silêncio estava comentada.
        # Mantive o mesmo comportamento para não alterar o resultado.
        # Se quiser ativar as pausas, descomente a linha abaixo:
        # if seg["pausa_ms"] > 0:
        #     chunks.append(silencio(seg["pausa_ms"]))

    # Concatena todos os trechos
    if chunks:
        audio_final = np.concatenate([c.squeeze() for c in chunks])
    else:
        audio_final = np.zeros(SAMPLE_RATE, dtype=np.float32)

    audio_final = np.squeeze(audio_final).astype(np.float32)

    # Converte para WAV em memória
    buffer = io.BytesIO()
    sf.write(buffer, audio_final, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    return buffer.read()


# =============================================================================
# FUNÇÕES AUXILIARES DE CONVENIÊNCIA
# =============================================================================

def listar_vozes() -> dict:
    """
    Retorna o dicionário de vozes disponíveis.

    Útil para interfaces que precisam mostrar as opções ao usuário.
    """
    return VOZES_PTBR.copy()


def get_voice_id(nome_amigavel: str) -> str:
    """
    Converte o nome amigável da voz para o identificador interno.

    Exemplo:
        get_voice_id("Dora (Feminina - PT-BR)") → "pf_dora"
    """
    return VOZES_PTBR.get(nome_amigavel, "pf_dora")