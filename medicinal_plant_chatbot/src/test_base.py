texto = open('src/data/rag/plantas_curadas.md', encoding='utf-8').read()
for padrao in ['Ã', 'â€', '\ufffd', 'Â ']:
    print(f'{padrao!r}: {texto.count(padrao)} ocorrências')

import tiktoken
enc = tiktoken.get_encoding('o200k_base')  # mesmo encoding do gpt-4o-mini

# Extrai só o bloco do Hibisco, igual rag.py faz
inicio = texto.find('# Hibisco')
fim = texto.find('\n# ', inicio + 1)
bloco = texto[inicio:fim if fim != -1 else None]

tokens = enc.encode(bloco)
print(f'Caracteres: {len(bloco)}')
print(f'Tokens (tiktoken): {len(tokens)}')
print(f'Razão: {len(bloco)/len(tokens):.2f} caracteres/token')


import re
from pathlib import Path

pasta = Path("src/prompts")
padrao = re.compile(r"\{\{(\w+)\}\}")

for arquivo in sorted(pasta.glob("*.md")):
    texto = arquivo.read_text(encoding="utf-8")
    contagem = {}
    for m in padrao.finditer(texto):
        contagem[m.group(1)] = contagem.get(m.group(1), 0) + 1
    print(arquivo.name)
    for var, n in sorted(contagem.items(), key=lambda x: -x[1]):
        marca = " ⚠️" if n > 1 else ""
        print(f"  {{{{{var}}}}}: {n}x{marca}")

# ===============================================================================
# print graph
# ===============================================================================

from agents.graph import Dependencias, construir_grafo
dependencias = Dependencias(llm=None, dual_encoder=None, rag=None, wikipedia=None, tavily=None)

grafo_compilado = construir_grafo(dependencias)
#print(grafo_compilado.get_graph().draw_mermaid())
print(grafo_compilado.get_graph().print_ascii())