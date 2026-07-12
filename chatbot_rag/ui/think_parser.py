"""
=============================================================================
core/think_parser.py — Parser de Chain-of-Thought Interno (<think>)
=============================================================================
Responsabilidade:
    Detectar, extrair e separar o bloco de raciocínio interno emitido por
    modelos que expõem chain-of-thought explícito via tags <think>...</think>.

Modelos que usam este padrão (2025/2026):
    - DeepSeek-R1 e variantes (DeepSeek-R1-0528, DeepSeek-R1-Distill-*)
    - QwQ-32B (Qwen)
    - Alguns fine-tunes de Llama-3 com CoT explícito

O que o parser faz:
    1. Detecta presença de <think>...</think> na resposta bruta do LLM
    2. Extrai o conteúdo do bloco think (raciocínio interno)
    3. Retorna a resposta limpa (texto após </think>)
    4. Calcula métricas do bloco think (tokens, linhas, tempo de raciocínio)

Por que separar:
    - O bloco <think> é raciocínio interno do modelo, não resposta ao usuário
    - Deve ser exibido no PipelineTrace (thinking box), não no chat
    - A resposta final ao usuário deve ser apenas o texto após </think>
    - Misturar os dois degrada a experiência e confunde o histórico

ThinkResult:
    - used_cot:       bool   — modelo usou chain-of-thought?
    - think_content:  str    — conteúdo do bloco <think> (pode ser "")
    - clean_answer:   str    — resposta sem o bloco think
    - think_tokens:   int    — estimativa de tokens do raciocínio
    - think_lines:    int    — linhas do bloco think
    - think_ratio:    float  — proporção think/total de tokens (0–1)
=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Padrões regex para detecção de blocos think
# ---------------------------------------------------------------------------

# Padrão principal: <think>...</think> (case-insensitive, multiline, greedy)
_THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.IGNORECASE | re.DOTALL,
)

# Padrão alternativo: alguns modelos usam [think]...[/think] ou <<think>>
_THINK_ALT_PATTERNS = [
    re.compile(r"\[think\](.*?)\[/think\]",   re.IGNORECASE | re.DOTALL),
    re.compile(r"<<think>>(.*?)<</think>>",    re.IGNORECASE | re.DOTALL),
    re.compile(r"<thinking>(.*?)</thinking>",  re.IGNORECASE | re.DOTALL),
]


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class ThinkResult:
    """
    Resultado do parsing de uma resposta com potencial bloco <think>.

    Attributes:
        used_cot:      True se o modelo emitiu um bloco think.
        think_content: Conteúdo extraído do bloco <think>.
                       String vazia se used_cot=False.
        clean_answer:  Resposta final sem o bloco think, stripped.
        think_tokens:  Estimativa de tokens do raciocínio interno.
        think_lines:   Número de linhas do bloco think.
        think_ratio:   Proporção think_tokens / total_tokens (0.0–1.0).
                       Indica quanto do output foi raciocínio vs resposta.
    """
    used_cot:      bool  = False
    think_content: str   = ""
    clean_answer:  str   = ""
    think_tokens:  int   = 0
    think_lines:   int   = 0
    think_ratio:   float = 0.0


# ---------------------------------------------------------------------------
# Função de parsing
# ---------------------------------------------------------------------------

def parse_think(raw_response: str) -> ThinkResult:
    """
    Extrai o bloco <think> de uma resposta bruta do LLM.

    Tenta o padrão principal (<think>...</think>) e os alternativos
    em sequência, parando no primeiro match encontrado.

    Comportamento quando não há bloco think:
        Retorna ThinkResult(used_cot=False, clean_answer=raw_response)
        sem modificar o texto original.

    Comportamento com múltiplos blocos think:
        Concatena todos os blocos em think_content (separados por newline),
        remove todos da resposta, retorna o texto restante como clean_answer.
        Raro, mas pode ocorrer em modelos que intercalam raciocínio e resposta.

    Args:
        raw_response: Texto completo retornado pelo LLM (AIMessage.content).

    Returns:
        ThinkResult com todos os campos preenchidos.
    """
    if not raw_response:
        return ThinkResult(clean_answer="")

    # --- Tenta padrão principal ---
    matches = _THINK_PATTERN.findall(raw_response)

    # --- Tenta padrões alternativos se não encontrou ---
    if not matches:
        for pattern in _THINK_ALT_PATTERNS:
            matches = pattern.findall(raw_response)
            if matches:
                # Remove usando este padrão alternativo
                clean = pattern.sub("", raw_response).strip()
                return _build_result(matches, clean, raw_response)

    if not matches:
        # Nenhum bloco think encontrado
        return ThinkResult(
            used_cot     = False,
            think_content = "",
            clean_answer  = raw_response.strip(),
        )

    # --- Remove todos os blocos think da resposta ---
    clean_answer = _THINK_PATTERN.sub("", raw_response).strip()

    return _build_result(matches, clean_answer, raw_response)


def _build_result(
    matches:      list[str],
    clean_answer: str,
    raw_response: str,
) -> ThinkResult:
    """
    Constrói o ThinkResult a partir dos matches extraídos.

    Args:
        matches:      Lista de conteúdos dos blocos think encontrados.
        clean_answer: Resposta sem os blocos think.
        raw_response: Resposta original completa (para calcular ratio).

    Returns:
        ThinkResult preenchido com métricas.
    """
    think_content = "\n\n---\n\n".join(m.strip() for m in matches)

    # Estimativa de tokens (heurística: 4 chars/token)
    think_tokens = max(1, len(think_content) // 4)
    total_tokens = max(1, len(raw_response)  // 4)
    think_ratio  = round(think_tokens / total_tokens, 4)
    think_lines  = think_content.count("\n") + 1 if think_content else 0

    return ThinkResult(
        used_cot      = True,
        think_content = think_content,
        clean_answer  = clean_answer,
        think_tokens  = think_tokens,
        think_lines   = think_lines,
        think_ratio   = think_ratio,
    )
