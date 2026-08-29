"""
Guardrails de entrada: validação da mensagem do usuário antes de
qualquer processamento pelo agente.

Cobre APENAS validações determinísticas — vazio, tamanho, caracteres de
controle. Detecção de prompt injection e moderação de conteúdo NÃO são
feitas aqui: já estão cobertas pelas seções de segurança embutidas em
prompts/extrator_intencao.md e prompts/guardrail_escopo.md, que são os
dois primeiros pontos de contato com o texto bruto do usuário. Uma
terceira chamada de LLM só para isso seria custo redundante sem ganho
real (decisão de arquitetura já registrada na conversa que originou
este módulo).

Não cobre a checagem de domínio (plantas medicinais vs. fora de escopo)
— essa é responsabilidade de `guardrails/escopo.py`, mantida separada
por decisão de projeto.

Executado ANTES de qualquer chamada de LLM — uma mensagem vazia ou
absurdamente longa é rejeitada sem gastar uma chamada de extração de
intenção com algo que uma checagem local já resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.constants import MAX_CARACTERES_MENSAGEM_USUARIO

# Caracteres de controle ASCII (exceto \n, \t, \r, que são normais em
# texto digitado/colado) — presença deles costuma indicar payload
# malformado ou tentativa de manipular o parsing downstream, não uma
# mensagem legítima de usuário.
_CARACTERES_DE_CONTROLE_PROIBIDOS = frozenset(
    chr(i) for i in range(0, 32) if chr(i) not in ("\n", "\t", "\r")
)


@dataclass
class ResultadoValidacaoEntrada:
    valida: bool
    motivo_bloqueio: str | None = None


def validar_entrada(mensagem_usuario: str) -> ResultadoValidacaoEntrada:
    """Aplica validações determinísticas de entrada.

    Retorna resultado com `valida=False` e `motivo_bloqueio` preenchido
    quando a mensagem deve ser rejeitada antes de chegar ao agente.
    """
    if not mensagem_usuario or not mensagem_usuario.strip():
        return ResultadoValidacaoEntrada(
            valida=False, motivo_bloqueio="Mensagem vazia."
        )

    if len(mensagem_usuario) > MAX_CARACTERES_MENSAGEM_USUARIO:
        return ResultadoValidacaoEntrada(
            valida=False,
            motivo_bloqueio=(
                f"Mensagem excede o limite de "
                f"{MAX_CARACTERES_MENSAGEM_USUARIO} caracteres "
                f"(recebido: {len(mensagem_usuario)})."
            ),
        )

    caracteres_invalidos = set(mensagem_usuario) & _CARACTERES_DE_CONTROLE_PROIBIDOS
    if caracteres_invalidos:
        return ResultadoValidacaoEntrada(
            valida=False,
            motivo_bloqueio="Mensagem contém caracteres de controle inválidos.",
        )

    return ResultadoValidacaoEntrada(valida=True)
