"""
=============================================================================
core/reranker.py — Reranker Pós-Retrieval (Cross-Encoder)
=============================================================================
Responsabilidade:
    Reordenar e filtrar os chunks recuperados pelo FAISS usando um
    cross-encoder, que avalia a relevância de cada chunk em relação
    à query de forma muito mais precisa que o bi-encoder do retriever.

Diferença entre bi-encoder (FAISS) e cross-encoder (reranker):
    Bi-encoder:   Gera embeddings da query e dos chunks separadamente,
                  compara por cosseno. Rápido, mas aproximado — não vê
                  interações entre query e chunk.

    Cross-encoder: Recebe (query, chunk) juntos em uma única passagem.
                   Vê a interação direta entre os termos. Muito mais
                   preciso, mas mais lento — por isso roda sobre um
                   conjunto pequeno pré-filtrado pelo FAISS.

Estratégia no pipeline:
    FAISS retriever (fetch_k=10) → cross-encoder → top_k=3 melhores chunks

Modelo padrão:
    BAAI/bge-reranker-v2-m3 — mesmo fabricante do BGE-M3 usado nos
    embeddings. Excelente qualidade para português e multilíngue.
    Alternativa mais leve: cross-encoder/ms-marco-MiniLM-L-6-v2

Métricas coletadas por RerankResult:
    - scores_before: scores MMR originais do FAISS (se disponíveis)
    - scores_after:  scores do cross-encoder por chunk
    - docs_before:   quantidade de chunks antes do reranking
    - docs_after:    quantidade de chunks após filtro top_k
    - latency_ms:    tempo de reranking em ms
    - top_score:     score mais alto do cross-encoder
    - score_delta:   diferença entre maior e menor score (dispersão)
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Resultado do reranking com métricas
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """
    Encapsula os documentos rerankeados e as métricas do processo.

    Attributes:
        docs:         Lista de Document reordenada pelo cross-encoder.
        scores:       Score do cross-encoder para cada doc (mesma ordem).
        docs_before:  Quantidade de chunks recebidos pelo reranker.
        docs_after:   Quantidade de chunks retornados (top_k).
        top_score:    Maior score atribuído pelo cross-encoder.
        score_delta:  Dispersão: diferença entre maior e menor score.
        latency_ms:   Tempo de execução do reranking em ms.
        model_name:   Nome do modelo cross-encoder usado.
    """
    docs:        list[Document]
    scores:      list[float]
    docs_before: int
    docs_after:  int
    top_score:   float        = 0.0
    score_delta: float        = 0.0
    latency_ms:  float        = 0.0
    model_name:  str          = ""


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Reranker baseado em cross-encoder para reordenar chunks do RAG.

    Carrega o modelo cross-encoder uma única vez e o reutiliza em todas
    as chamadas, evitando overhead de carregamento por query.

    Args:
        model_name: ID do modelo cross-encoder no HuggingFace Hub.
        top_k:      Número de chunks a retornar após reranking.
        device:     "cpu" | "cuda" | "mps". Padrão: "cpu".
    """

    # Modelo padrão: BGE-Reranker-v2-M3 — multilíngue, alta qualidade
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str  = DEFAULT_MODEL,
        top_k:      int  = 3,
        device:     str  = "cpu",
    ):
        self._model_name = model_name
        self._top_k      = top_k
        self._device     = device
        self._model      = None   # lazy loading — carrega na primeira chamada

    def _load_model(self) -> None:
        """
        Carrega o cross-encoder do HuggingFace (lazy loading).

        Chamado automaticamente na primeira invocação de rerank().
        Evita carregar o modelo se o reranker não for utilizado.
        """
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
            )
        except ImportError as e:
            raise ImportError(
                "sentence-transformers é necessário para o reranker. "
                "Instale com: pip install sentence-transformers"
            ) from e

    def rerank(self, query: str, docs: list[Document]) -> RerankResult:
        """
        Reordena os documentos por relevância em relação à query.

        Etapas:
            1. Lazy load do modelo cross-encoder (apenas na primeira chamada)
            2. Formata pares (query, chunk_content) para o cross-encoder
            3. Calcula scores de relevância para cada par
            4. Ordena por score decrescente
            5. Retorna top_k documentos com métricas

        Args:
            query: Texto da query do usuário.
            docs:  Lista de Document retornados pelo FAISS retriever.

        Returns:
            RerankResult com docs reordenados, scores e métricas.
        """
        t0 = time.perf_counter()

        # Sem documentos → retorna vazio
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

        # --- Formata pares para o cross-encoder ---
        # O cross-encoder recebe (query, passage) e retorna um score
        # de relevância — quanto maior, mais relevante o chunk.
        pairs = [(query, doc.page_content) for doc in docs]

        # --- Calcula scores ---
        raw_scores = self._model.predict(pairs).tolist()

        # --- Ordena por score decrescente e seleciona top_k ---
        scored_docs = sorted(
            zip(raw_scores, docs),
            key=lambda x: x[0],
            reverse=True,
        )
        top_docs    = scored_docs[: self._top_k]
        final_docs  = [doc   for _, doc   in top_docs]
        final_scores = [round(float(score), 4) for score, _ in top_docs]

        # --- Métricas de dispersão ---
        top_score   = final_scores[0] if final_scores else 0.0
        worst_score = final_scores[-1] if final_scores else 0.0
        score_delta = round(top_score - worst_score, 4)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return RerankResult(
            docs        = final_docs,
            scores      = final_scores,
            docs_before = docs_before,
            docs_after  = len(final_docs),
            top_score   = round(top_score, 4),
            score_delta = score_delta,
            latency_ms  = latency_ms,
            model_name  = self._model_name,
        )

    @property
    def model_name(self) -> str:
        """Nome do modelo cross-encoder configurado."""
        return self._model_name

    @property
    def top_k(self) -> int:
        """Número de chunks retornados após reranking."""
        return self._top_k
