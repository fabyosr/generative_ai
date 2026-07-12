"""
=============================================================================
core/analytics.py — Analytics de Qualidade e Negócio
=============================================================================
Responsabilidade:
    Extrair métricas de qualidade cognitiva e de negócio a partir dos
    artefatos já disponíveis no pipeline: query, answer, context_docs,
    rerank_scores, chat_history, provider.

Todas as métricas são calculadas localmente — sem chamadas adicionais
ao LLM — usando embeddings, NLP clássico e heurísticas calibradas.

Grupos de métricas:

    1. Grounding Score
       Quão bem a resposta está ancorada no contexto RAG.
       Combina: similaridade semântica sentença↔chunk + reranker score.

    2. Faithfulness Proxy (sem LLM-as-a-Judge)
       Proporção de sentenças da resposta ancoradas no contexto.
       Sentença não ancorada → risco de alucinação.

    3. Análise Textual da Resposta
       Legibilidade, comprimento, complexidade lexical, hedging rate,
       tipo de resposta (factual / explicativo / procedural).

    4. Análise da Query
       Tipo de pergunta, complexidade, presença de negação/ambiguidade.

    5. Métricas de Negócio
       Custo estimado em tokens × preço, economia do cache,
       eficiência do retrieval.
=============================================================================
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Preços por token: delegados ao MODEL_CATALOG em core/models.py
# ---------------------------------------------------------------------------
# Não mantemos uma cópia local de preços aqui — a fonte única de verdade
# é MODEL_CATALOG / get_model_info() em core/models.py.
# A função _estimate_cost() abaixo consulta esse catálogo via get_model_info.

# Termos de incerteza/hedging em português e inglês
_HEDGING_TERMS = re.compile(
    r"\b(talvez|pode ser|possivelmente|provavelmente|acredito|imagino|"
    r"não tenho certeza|não sei ao certo|não estou certo|pode ser que|"
    r"é possível que|parece que|ao que parece|segundo|de acordo com|"
    r"não encontrei|não há informação|não consta|"
    r"perhaps|maybe|possibly|probably|i think|i believe|"
    r"i'm not sure|it seems|it appears)\b",
    re.IGNORECASE,
)

# Tipos de pergunta por padrão de abertura
_QUESTION_PATTERNS = {
    "factual":     re.compile(r"^(qual|quais|quando|onde|quem|o que é|what|who|when|where|which)\b", re.I),
    "explicativo": re.compile(r"^(como|por que|por quê|explique|explica|descreva|how|why|explain)\b", re.I),
    "procedural":  re.compile(r"^(como fazer|como posso|como eu|passo a passo|how to|how do)\b", re.I),
    "comparativo": re.compile(r"^(qual a diferença|compare|diferença entre|versus|vs\.?|compare)\b", re.I),
    "confirmação": re.compile(r"^(é verdade|isso é|está correto|confirma|é possível|can i|is it|does)\b", re.I),
}

# Padrões de ambiguidade (pronomes sem referente claro)
_AMBIGUITY_TERMS = re.compile(
    r"\b(isso|este|esse|esta|essa|ele|ela|eles|elas|aquilo|lá|aí|"
    r"this|that|it|they|there)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class AnalyticsResult:
    """
    Consolida todas as métricas analíticas de uma interação.

    Grupos de atributos:
        grounding_*:    Ancoragem da resposta no contexto RAG
        faithfulness_*: Proxy de fidelidade (sem LLM adicional)
        response_*:     Análise textual da resposta
        query_*:        Análise da pergunta do usuário
        cost_*:         Estimativas de custo em tokens/USD
    """
    # --- Grounding ---
    grounding_score:       float = 0.0   # 0–1: ancoragem geral
    grounding_label:       str   = ""    # "Alta" / "Moderada" / "Baixa" / "Sem contexto"
    semantic_overlap:      float = 0.0   # similaridade embedding answer↔context

    # --- Faithfulness proxy ---
    hallucination_risk:    float = 0.0   # 0–1: proporção de sentenças não ancoradas
    ungrounded_sentences:  list  = field(default_factory=list)
    total_sentences:       int   = 0

    # --- Análise da resposta ---
    response_tokens:       int   = 0
    response_words:        int   = 0
    response_sentences:    int   = 0
    hedging_rate:          float = 0.0   # 0–1: frequência de termos de incerteza
    hedging_terms_found:   list  = field(default_factory=list)
    response_type:         str   = ""    # "factual" / "explicativo" / etc.
    lexical_diversity:     float = 0.0   # type-token ratio
    avg_sentence_length:   float = 0.0   # palavras por sentença

    # --- Análise da query ---
    query_type:            str   = ""    # tipo de pergunta
    query_words:           int   = 0
    query_complexity:      str   = ""    # "simples" / "moderada" / "complexa"
    has_negation:          bool  = False
    has_ambiguity:         bool  = False

    # --- Custo ---
    input_tokens:          int   = 0
    output_tokens:         int   = 0
    estimated_cost_usd:    float = 0.0
    cache_savings_usd:     float = 0.0  # economia se foi cache hit

    # --- Meta ---
    latency_ms:            float = 0.0


# ---------------------------------------------------------------------------
# Funções auxiliares de NLP
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Divide texto em sentenças usando pontuação como delimitador.

    Heurística simples adequada para PT-BR sem dependência de spaCy/NLTK.

    Args:
        text: Texto a dividir.

    Returns:
        Lista de sentenças não-vazias.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _type_token_ratio(text: str) -> float:
    """
    Calcula a diversidade lexical (type-token ratio).

    TTR = tipos únicos / total de tokens.
    Valores altos → vocabulário diverso.
    Valores baixos → muita repetição (respostas genéricas).

    Args:
        text: Texto a analisar.

    Returns:
        float: TTR entre 0.0 e 1.0.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 3)


def _detect_question_type(query: str) -> str:
    """
    Classifica o tipo de pergunta por padrão de abertura.

    Args:
        query: Texto da pergunta do usuário.

    Returns:
        str: "factual" | "explicativo" | "procedural" |
             "comparativo" | "confirmação" | "aberto"
    """
    for qtype, pattern in _QUESTION_PATTERNS.items():
        if pattern.match(query.strip()):
            return qtype
    return "aberto"


def _count_tokens(text: str, provider: str) -> int:
    """
    Conta tokens usando tiktoken (OpenAI/Groq) ou heurística (HF).

    Args:
        text:     Texto a tokenizar.
        provider: Provedor LLM.

    Returns:
        int: Estimativa de tokens.
    """
    if provider in ("openai", "groq"):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
    return max(1, len(text) // 4)


def _estimate_cost(
    input_tokens:  int,
    output_tokens: int,
    provider:      str,
    model:         str,
) -> float:
    """
    Estima o custo em USD da interação consultando o MODEL_CATALOG.

    Delega a consulta de preços para get_model_info() em core/models.py,
    que é a fonte única de verdade para preços e janelas de contexto.
    Isso garante que qualquer atualização de preço feita em models.py
    reflete automaticamente aqui sem necessidade de editar dois arquivos.

    Args:
        input_tokens:  Tokens do prompt (query + contexto + histórico).
        output_tokens: Tokens da resposta gerada.
        provider:      Provedor LLM.
        model:         Nome do modelo.

    Returns:
        float: Custo estimado em USD (0.0 se preço não disponível).
    """
    from core.models import get_model_info

    model_info = get_model_info(provider, model)
    price_input, price_output = model_info["price"]

    cost = (input_tokens / 1_000 * price_input) + (output_tokens / 1_000 * price_output)
    return round(cost, 8)


# ---------------------------------------------------------------------------
# Cálculo de grounding via similaridade semântica
# ---------------------------------------------------------------------------

def _compute_semantic_overlap(
    answer:       str,
    context_docs: list,
    embeddings,
) -> tuple[float, float, list[str]]:
    """
    Calcula a sobreposição semântica entre a resposta e o contexto RAG.

    Para cada sentença da resposta, calcula a similaridade máxima
    com qualquer chunk do contexto. A média dessas similaridades
    máximas é o semantic overlap geral.

    Sentenças com similaridade abaixo de 0.50 são marcadas como
    "não ancoradas" → proxy de risco de alucinação.

    Args:
        answer:       Texto da resposta do LLM.
        context_docs: Chunks recuperados pelo RAG.
        embeddings:   Instância de HuggingFaceEmbeddings (compartilhada).

    Returns:
        tuple: (semantic_overlap, hallucination_risk, ungrounded_sentences)
    """
    if not context_docs or not embeddings:
        return 0.0, 0.0, []

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0, 0.0, []

    ANCHOR_THRESHOLD = 0.50

    # Gera embeddings dos chunks (concatenados para representação geral)
    chunk_texts = [doc.page_content for doc in context_docs]

    try:
        sent_embeddings  = np.array(embeddings.embed_documents(sentences),  dtype=np.float32)
        chunk_embeddings = np.array(embeddings.embed_documents(chunk_texts), dtype=np.float32)

        # Normaliza para similaridade de cosseno via produto escalar
        sent_norms  = np.linalg.norm(sent_embeddings,  axis=1, keepdims=True)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        sent_embeddings  = sent_embeddings  / np.maximum(sent_norms,  1e-9)
        chunk_embeddings = chunk_embeddings / np.maximum(chunk_norms, 1e-9)

        # Matriz de similaridade (n_sentences × n_chunks)
        sim_matrix  = sent_embeddings @ chunk_embeddings.T
        max_sims    = sim_matrix.max(axis=1)  # melhor chunk para cada sentença

        overlap     = float(np.mean(max_sims))
        ungrounded  = [
            sentences[i]
            for i, sim in enumerate(max_sims)
            if sim < ANCHOR_THRESHOLD
        ]
        halluc_risk = len(ungrounded) / len(sentences)

    except Exception:
        return 0.0, 0.0, []

    return round(overlap, 4), round(halluc_risk, 4), ungrounded


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------

def compute_analytics(
    query:         str,
    answer:        str,
    context_docs:  list,
    provider:      str,
    model:         str,
    rerank_result  = None,
    cache_result   = None,
    embeddings     = None,
    history_tokens: int = 0,
) -> AnalyticsResult:
    """
    Calcula todas as métricas analíticas de uma interação.

    Chamada uma vez por resposta, após o pipeline RAG completo.
    Todos os cálculos são locais — sem chamadas adicionais ao LLM.

    Args:
        query:          Texto da pergunta do usuário.
        answer:         Texto da resposta gerada pelo LLM.
        context_docs:   Chunks recuperados pelo RAG (pode ser vazio).
        provider:       Provedor LLM ("openai" | "groq" | "hf_hub").
        model:          Nome do modelo.
        rerank_result:  RerankResult do core/reranker.py (opcional).
        cache_result:   CacheResult do core/cache.py (opcional).
        embeddings:     HuggingFaceEmbeddings compartilhado (opcional).
        history_tokens: Tokens acumulados no histórico (para custo total).

    Returns:
        AnalyticsResult com todas as métricas preenchidas.
    """
    t0 = time.perf_counter()
    r  = AnalyticsResult()

    # -----------------------------------------------------------------------
    # 1. Análise da Query
    # -----------------------------------------------------------------------
    query_words       = re.findall(r"\b\w+\b", query)
    r.query_words     = len(query_words)
    r.query_type      = _detect_question_type(query)
    r.has_negation    = bool(re.search(r"\b(não|nao|nunca|jamais|sem|no|not|never|without)\b", query, re.I))
    r.has_ambiguity   = bool(_AMBIGUITY_TERMS.search(query))

    if r.query_words <= 5:
        r.query_complexity = "simples"
    elif r.query_words <= 15:
        r.query_complexity = "moderada"
    else:
        r.query_complexity = "complexa"

    # -----------------------------------------------------------------------
    # 2. Análise Textual da Resposta
    # -----------------------------------------------------------------------
    sentences              = _split_sentences(answer)
    words                  = re.findall(r"\b\w+\b", answer)
    hedging_matches        = _HEDGING_TERMS.findall(answer)

    r.response_tokens      = _count_tokens(answer, provider)
    r.response_words       = len(words)
    r.response_sentences   = len(sentences)
    r.hedging_rate         = round(len(hedging_matches) / max(len(sentences), 1), 3)
    r.hedging_terms_found  = list(set(h.lower() for h in hedging_matches))
    r.lexical_diversity    = _type_token_ratio(answer)
    r.avg_sentence_length  = round(len(words) / max(len(sentences), 1), 1)
    r.total_sentences      = len(sentences)

    # Tipo da resposta inferido pelo tamanho e estrutura
    if r.response_sentences <= 2:
        r.response_type = "concisa"
    elif "```" in answer or re.search(r"\d+\.", answer):
        r.response_type = "estruturada"
    elif r.response_words > 200:
        r.response_type = "elaborada"
    else:
        r.response_type = "padrão"

    # -----------------------------------------------------------------------
    # 3. Grounding e Faithfulness Proxy
    # -----------------------------------------------------------------------
    if context_docs and embeddings:
        overlap, halluc_risk, ungrounded = _compute_semantic_overlap(
            answer, context_docs, embeddings
        )
        r.semantic_overlap     = overlap
        r.hallucination_risk   = halluc_risk
        r.ungrounded_sentences = ungrounded

        # Grounding score composto
        reranker_score = 0.0
        if rerank_result and rerank_result.top_score:
            # Normaliza score do cross-encoder (pode ser negativo ou > 1)
            raw = rerank_result.top_score
            reranker_score = max(0.0, min(raw / max(abs(raw) + 1e-9, 1.0), 1.0))

        r.grounding_score = round(
            overlap           * 0.55 +
            reranker_score    * 0.30 +
            (1 - halluc_risk) * 0.15,
            4,
        )
    else:
        # Sem contexto RAG (chitchat / followup sem docs)
        r.grounding_score = 0.0

    # Label qualitativo do grounding
    if not context_docs:
        r.grounding_label = "Sem contexto RAG"
    elif r.grounding_score >= 0.75:
        r.grounding_label = "Alta 🟢"
    elif r.grounding_score >= 0.50:
        r.grounding_label = "Moderada 🟡"
    else:
        r.grounding_label = "Baixa 🔴"

    # -----------------------------------------------------------------------
    # 4. Estimativa de Custo
    # -----------------------------------------------------------------------
    # Tokens de input = query + contexto + histórico estimado
    context_text   = " ".join(d.page_content for d in context_docs)
    context_tokens = _count_tokens(context_text, provider)

    r.input_tokens  = _count_tokens(query, provider) + context_tokens + history_tokens
    r.output_tokens = r.response_tokens
    r.estimated_cost_usd = _estimate_cost(r.input_tokens, r.output_tokens, provider, model)

    # Economia do cache: se foi hit, o custo real foi zero (apenas embedding lookup)
    if cache_result and cache_result.hit:
        r.cache_savings_usd = r.estimated_cost_usd

    r.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return r
