"""
Adapter entre o módulo já funcional tools/vendor/voice_engine.py
(desenvolvido previamente pelo autor) e o contrato de TTS usado pelo
grafo de agentes.

O conteúdo de voice_engine.py NÃO é alterado — este arquivo só adapta
sua API (funções soltas) ao formato de serviço (classe com estado
mínimo) usado pelo resto do projeto.

Regra importante: este serviço deve SEMPRE receber o texto já processado
por `core/use_cases.py::injetar_aviso_confianca` e pelos guardrails de
saída — nunca texto bruto do LLM. O áudio é uma extensão da mesma saída
textual, não um canal de conteúdo independente (decisão de projeto).
"""

from __future__ import annotations

from tools.vendor.voice_engine import get_voice_id, listar_vozes, synthesize


class KokoroTTSService:
    """Client de síntese de voz via Kokoro ONNX (voice_engine.py)."""

    def __init__(self, voz_padrao: str = "pf_dora") -> None:
        """
        Args:
            voz_padrao: identificador interno de voz (ex.: "pf_dora"),
                usado quando nenhuma voz é explicitamente selecionada.
                Ver `vozes_disponiveis()` para os identificadores válidos.
        """
        self._voz_padrao = voz_padrao

    def sintetizar(self, texto: str, voz: str | None = None) -> bytes:
        """Gera áudio a partir do texto e retorna os bytes de um WAV.

        Args:
            texto: texto já filtrado por guardrails, incluindo o aviso
                de baixa confiança quando aplicável.
            voz: identificador de voz selecionado na interface (ex.:
                "pm_alex"); usa `voz_padrao` se None. Aceita também o
                nome amigável via `get_voice_id()`, se a interface
                preferir passar o nome exibido ao usuário.
        """
        return synthesize(texto, voice=voz or self._voz_padrao)

    def vozes_disponiveis(self) -> dict[str, str]:
        """Retorna {nome amigável: identificador interno}, para
        popular o seletor de voz na sidebar."""
        return listar_vozes()

    @staticmethod
    def resolver_id_voz(nome_amigavel: str) -> str:
        """Converte um nome amigável (ex.: "Dora (Feminina - PT-BR)")
        para o identificador interno (ex.: "pf_dora")."""
        return get_voice_id(nome_amigavel)
