"""
=============================================================================
core/reranker.py — Reranker Pós-Retrieval (Cross-Encoder)
=============================================================================
Responsabilidade:
    Reordenar e filtrar os chunks recuperados pelo FAISS usando um
    cross-encoder, que avalia a relevância de cada chunk em relação
    à query de forma muito mais precisa que o bi-encoder do retriever.

Diferença entre bi-encoder (FAISS) e cross-encoder (reranker):
    Bi-encoder:    Gera embeddings separados de query e chunk, compara
                   por cosseno. Rápido, mas aproximado.
    Cross-encoder: Recebe (query, chunk) juntos numa única passagem.
                   Muito mais preciso, roda sobre conjunto pequeno pré-filtrado.

Estratégia no pipeline:
    FAISS (fetch_k=20 candidatos) → cross-encoder → top_k melhores

Modelo padrão:
    BAAI/bge-reranker-v2-m3 — multilíngue, alta qualidade para PT-BR.

Métodos de seleção de chunks:
    top_k:     Sempre retorna os k melhores, independente do score.
    threshold: Retorna todos acima de min_score (sem limite de k).
    adaptive:  Combina — até k chunks, desde que acima de min_score.

Nota de design:
    top_k é passado diretamente no rerank() em vez de usar configure(),
    evitando mutação de estado e race conditions entre ciclos do Streamlit.
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Resultado do reranking com métricas
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """
    Encapsula os documentos rerankeados e as métricas do processo.

    Attributes:
        docs:        Documentos selecionados, ordenados por relevância.
        scores:      Score do cross-encoder por doc (mesma ordem que docs).
        docs_before: Quantidade de chunks recebidos do FAISS.
        docs_after:  Quantidade de chunks retornados ao LLM.
        top_score:   Score do chunk mais relevante.
        score_delta: Dispersão entre melhor e pior score — alta = boa separação.
        latency_ms:  Tempo de execução em milissegundos.
        model_name:  Nome do modelo cross-encoder utilizado.
    """
    docs:        list[Document]
    scores:      list[float]
    docs_before: int
    docs_after:  int
    top_score:   float = 0.0
    score_delta: float = 0.0
    latency_ms:  float = 0.0
    model_name:  str   = ""


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Reranker baseado em cross-encoder para reordenar chunks do RAG.

    O modelo é carregado apenas na primeira chamada (lazy loading),
    evitando overhead de inicialização quando o reranker não é usado.

    Args:
        model_name: ID do modelo cross-encoder no HuggingFace Hub.
        top_k:      Número padrão de chunks a retornar (pode ser sobrescrito
                    por chamada via parâmetro top_k no rerank()).
        device:     "cpu" | "cuda" | "mps".
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k:      int = 3,
        device:     str = "cpu",
    ):
        self._model_name = model_name
        self._top_k      = top_k
        self._device     = device
        self._model      = None  # carregado na primeira chamada a rerank()

    def _load_model(self) -> None:
        """
        Carrega o cross-encoder do HuggingFace Hub (lazy loading).

        Chamado automaticamente na primeira invocação de rerank().
        Lança ImportError descritivo se sentence-transformers não estiver
        instalado, em vez do genérico ModuleNotFoundError.
        """
        print('reranker................')
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device=self._device)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers é necessário para o reranker. "
                "Instale com: pip install sentence-transformers"
            ) from e

    def rerank(
        self,
        query:     str,
        docs:      list[Document],
        method:    str        = "top_k",
        min_score: float      = 0.0,
        top_k:     int | None = None,
    ) -> RerankResult:
        """
        Reordena os documentos por relevância em relação à query.

        O parâmetro top_k pode ser sobrescrito por chamada, evitando
        a necessidade de configure() e mutação de estado no objeto.
        Isso é especialmente importante no Streamlit, onde o mesmo
        objeto Reranker é reutilizado entre múltiplos ciclos de
        re-execução com configurações diferentes da sidebar.

        Fluxo:
            1. Lazy load do modelo (só na primeira chamada)
            2. Formata pares (query, chunk) para o cross-encoder
            3. Calcula scores de relevância par a par
            4. Ordena por score decrescente
            5. Aplica método de seleção (top_k / threshold / adaptive)
            6. Retorna RerankResult com docs e métricas

        Args:
            query:     Texto da query do usuário.
            docs:      Chunks candidatos vindos do FAISS.
            method:    "top_k" | "threshold" | "adaptive".
            min_score: Score mínimo (usado em threshold e adaptive).
            top_k:     Override do k máximo. None = usa self._top_k.

        Returns:
            RerankResult com docs selecionados, scores e métricas.
        """
        t0          = time.perf_counter()
        effective_k = top_k if top_k is not None else self._top_k

        # Sem documentos → retorna vazio sem chamar o modelo
        if not docs:
            return RerankResult(
                docs        = [],
                scores      = [],
                docs_before = 0,
                docs_after  = 0,
                latency_ms  = 0.0,
                model_name  = self._model_name,
            )

        # Lazy load do modelo
        if self._model is None:
            self._load_model()

        docs_before = len(docs)

        # --- Score de relevância par a par ---
        pairs      = [(query, doc.page_content) for doc in docs]
        raw_scores = self._model.predict(pairs).tolist()

        # --- Ordena por score decrescente ---
        scored_docs = sorted(
            zip(raw_scores, docs),
            key=lambda x: x[0],
            reverse=True,
        )

        # --- Aplica método de seleção ---
        if method == "threshold":
            # Todos acima do score mínimo, sem limite de k
            selected = [(s, d) for s, d in scored_docs if float(s) >= min_score]
            if not selected:
                selected = [scored_docs[0]]  # garante ao menos 1

        elif method == "adaptive":
            # Até effective_k, desde que acima de min_score
            selected = [
                (s, d) for s, d in scored_docs[:effective_k]
                if float(s) >= min_score
            ]
            if not selected:
                selected = [scored_docs[0]]  # garante ao menos 1

        else:  # top_k (padrão)
            selected = scored_docs[:effective_k]

        final_docs   = [doc   for _, doc   in selected]
        final_scores = [round(float(s), 4) for s, _ in selected]

        # --- Métricas de dispersão ---
        top_score   = final_scores[0]  if final_scores else 0.0
        worst_score = final_scores[-1] if final_scores else 0.0
        score_delta = round(top_score - worst_score, 4)

        print('RerankResult....')

        return RerankResult(
            docs        = final_docs,
            scores      = final_scores,
            docs_before = docs_before,
            docs_after  = len(final_docs),
            top_score   = round(top_score, 4),
            score_delta = score_delta,
            latency_ms  = round((time.perf_counter() - t0) * 1000, 1),
            model_name  = self._model_name,
        )

    @property
    def model_name(self) -> str:
        """Nome do modelo cross-encoder configurado."""
        return self._model_name

    @property
    def top_k(self) -> int:
        """Número padrão de chunks retornados (pode ser sobrescrito por chamada)."""
        return self._top_k
