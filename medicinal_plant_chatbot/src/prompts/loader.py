"""
Carregador de prompts a partir de arquivos de texto (.md).

Prompts ficam fora do código Python propositalmente: eles são iterados
com muita mais frequência do que a lógica durante a fase de avaliação
(Fase 6), e mantê-los como arquivos versionáveis isoladamente facilita
o registro de qual versão de prompt foi usada em qual rodada de
experimento — critério de reprodutibilidade citado na estratégia de
avaliação do projeto.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent
_PADRAO_COMENTARIO_DOC = re.compile(r"\A<!--.*?-->\s*", re.DOTALL)

@lru_cache(maxsize=None)
def carregar_prompt(nome: str) -> str:
    """Carrega o conteúdo de um prompt pelo nome do arquivo (sem extensão).

    Ex.: carregar_prompt("extrator_intencao") lê
    `prompts/extrator_intencao.md`.

    Levanta FileNotFoundError com mensagem clara se o prompt não existir
    — falha explícita é preferível a um erro obscuro na chamada ao LLM.
    """
    caminho = _PROMPTS_DIR / f"{nome}.md"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Prompt '{nome}' não encontrado em {caminho}. "
            f"Verifique se o arquivo '{nome}.md' existe em src/prompts/."
        )
    texto = caminho.read_text(encoding="utf-8")
    # Remove o bloco de comentário de documentação do início do arquivo
    # ANTES de qualquer substituição de variável — evita que a sintaxe
    # de documentação ({{var}} usada só pra explicar, dentro de <!-- -->)
    # seja confundida com a sintaxe real de template pelo .replace()
    # dos chamadores.
    return _PADRAO_COMENTARIO_DOC.sub("", texto, count=1)