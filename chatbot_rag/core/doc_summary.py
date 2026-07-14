"""
=============================================================================
core/doc_summary.py — Extração de Resumo dos Documentos para o Classificador
=============================================================================
Responsabilidade:
    Ao fazer upload de PDFs, extrair automaticamente os temas e assuntos
    principais dos documentos para injetar no system prompt do classificador
    de intenção. Isso permite que o classificador saiba QUAIS assuntos
    estão cobertos pela base de conhecimento, tornando a classificação
    contextualmente consciente.

Fluxo:
    PDFs carregados → build_retriever()
        ↓
    extract_doc_summary(docs, llm) → "resumo dos temas principais"
        ↓
    Armazenado em session_state.doc_knowledge_summary
        ↓
    classify_intent() recebe o resumo → prompt contextualizado

Estratégia de extração:
    1. Amostra até N chunks representativos dos documentos (não todos,
       para economizar tokens)
    2. Passa para um LLM leve extrair os temas principais em bullet points
    3. Resultado é um texto curto (~150 tokens) usado no system prompt
       do classificador

Sem LLM (fallback):
    Extrai heuristicamente os títulos e primeiras linhas de cada documento,
    gerando um resumo simples sem custo de tokens.
=============================================================================
"""

from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Número máximo de chunks amostrados para o resumo
# ---------------------------------------------------------------------------
MAX_SAMPLE_CHUNKS = 8    # equilibra qualidade vs custo de tokens
MAX_SUMMARY_TOKENS = 200 # limite do resumo gerado


# ---------------------------------------------------------------------------
# Extração heurística (sem LLM) — fallback ou uso direto
# ---------------------------------------------------------------------------

def _heuristic_summary(docs: list) -> str:
    """
    Gera um resumo heurístico dos documentos sem chamar nenhum LLM.

    Estratégia:
        - Extrai as primeiras linhas de cada documento (provável título/intro)
        - Identifica os termos mais frequentes (excluindo stopwords)
        - Retorna um texto curto descrevendo os temas principais

    Args:
        docs: Lista de Document do LangChain com page_content e metadata.

    Returns:
        str: Resumo heurístico dos temas dos documentos.
    """
    STOPWORDS = {
        "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "para", "com", "por", "que", "se", "uma", "um", "os", "as",
        "a", "o", "e", "é", "ao", "ou", "the", "of", "in", "and",
        "to", "a", "is", "for", "on", "with", "at", "by",
    }

    # Agrupa por arquivo fonte
    sources: dict[str, list[str]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "documento")
        import os
        source = os.path.basename(source).replace(".pdf", "")
        if source not in sources:
            sources[source] = []
        sources[source].append(doc.page_content[:300])

    # Extrai termos frequentes de todo o conteúdo
    all_text = " ".join(doc.page_content for doc in docs[:MAX_SAMPLE_CHUNKS])
    words    = re.findall(r"\b[a-záàâãéèêíóôõúüç]{4,}\b", all_text.lower())
    top_terms = [w for w, _ in Counter(words).most_common(20) if w not in STOPWORDS][:10]

    # Monta o resumo
    lines = ["Os documentos carregados abordam os seguintes temas:"]
    for source, snippets in list(sources.items())[:5]:
        first_line = snippets[0].split("\n")[0].strip()[:100] if snippets else ""
        if first_line:
            lines.append(f"• {source}: {first_line}")
        else:
            lines.append(f"• {source}")

    if top_terms:
        lines.append(f"\nTermos-chave identificados: {', '.join(top_terms)}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extração com LLM — qualidade superior
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = """Você é um assistente especializado em análise de documentos.
Analise os trechos fornecidos e responda SOMENTE com uma lista de bullet points
descrevendo os TEMAS PRINCIPAIS abordados nos documentos.

Formato esperado (máximo 8 bullets, cada um com até 15 palavras):
• [tema 1]
• [tema 2]
...

Não inclua introdução, explicação ou qualquer outro texto além dos bullets."""


def extract_doc_summary(docs: list, llm=None) -> str:
    """
    Extrai um resumo dos temas principais dos documentos carregados.

    Usado para contextualizar o classificador de intenção: com esse resumo,
    o classificador sabe exatamente quais assuntos estão na base de
    conhecimento e pode fazer perguntas como:
    "essa query é sobre os temas dos documentos ou é chitchat genérico?"

    Estratégia:
        - Se LLM disponível: amostra chunks e pede resumo estruturado
        - Se LLM=None (ou erro): usa heurística local gratuita

    Args:
        docs: Lista de Document retornados pelo PyPDFLoader.
        llm:  Instância de LLM (opcional). None = usa heurística.

    Returns:
        str: Resumo dos temas principais (~100–200 tokens).
    """
    if not docs:
        return "Nenhum documento carregado."

    # Sempre gera o heurístico como fallback
    heuristic = _heuristic_summary(docs)

    if llm is None:
        return heuristic

    # Amostra chunks representativos: primeiro, meio e último de cada doc
    import os
    sampled: list[str] = []
    grouped: dict[str, list] = {}

    for doc in docs:
        src = os.path.basename(doc.metadata.get("source", "doc"))
        grouped.setdefault(src, []).append(doc)

    for src, src_docs in grouped.items():
        n = len(src_docs)
        indices = {0, n // 2, n - 1}           # primeiro, meio, último
        for i in indices:
            chunk = src_docs[i].page_content[:400].strip()
            if chunk:
                sampled.append(f"[{src}]\n{chunk}")

        if len(sampled) >= MAX_SAMPLE_CHUNKS:
            break

    sample_text = "\n\n---\n\n".join(sampled[:MAX_SAMPLE_CHUNKS])

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=f"Trechos dos documentos:\n\n{sample_text}"),
        ])
        summary = response.content.strip()

        # Valida se o resultado parece um bullet list
        if "•" in summary or "-" in summary or summary.count("\n") >= 2:
            return summary
        return heuristic

    except Exception:
        return heuristic


# ---------------------------------------------------------------------------
# Formata o resumo para injeção no system prompt do classificador
# ---------------------------------------------------------------------------

def format_for_classifier(doc_summary: str) -> str:
    """
    Formata o resumo dos documentos para uso no system prompt do classificador.

    Retorna uma instrução clara sobre quais temas pertencem à base de
    conhecimento, orientando o classificador a só acionar RAG quando
    a query for relacionada a esses temas.

    Args:
        doc_summary: Saída de extract_doc_summary().

    Returns:
        str: Bloco de instrução para adicionar ao system prompt.
    """
    if not doc_summary or doc_summary == "Nenhum documento carregado.":
        return ""

    return (
        f"\n\nBASE DE CONHECIMENTO DISPONÍVEL:\n"
        f"{doc_summary}\n\n"
        f"REGRA IMPORTANTE: Classifique como RAG_QUERY APENAS se a pergunta "
        f"estiver relacionada aos temas acima. Se for sobre outro assunto "
        f"completamente diferente, classifique como CHITCHAT."
    )
