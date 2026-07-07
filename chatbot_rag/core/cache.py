"""
=============================================================================
core/cache.py — Semantic Cache
=============================================================================
Responsabilidade:
    Armazenar e recuperar respostas de queries semanticamente similares,
    evitando chamadas desnecessárias ao LLM e ao pipeline RAG.

Como funciona:
    - Cada query respondida tem seu embedding gerado e armazenado em FAISS.
    - Novas queries são comparadas por similaridade de cosseno com as
      queries cacheadas.
    - Se a similaridade ultrapassar o threshold configurado, a resposta
      cacheada é retornada sem invocar o LLM.

Escopo de sessão:
    O cache é mantido por instância (session_state do Streamlit), ou seja,
    é isolado por usuário e por conjunto de documentos. Isso evita que
    respostas de um documento sejam retornadas para perguntas sobre outro.

Métricas coletadas por CacheResult:
    - hit:          bool — se houve cache hit
    - similarity:   float — score de similaridade da query mais próxima
    - matched_query: str — query original que gerou o cache hit
    - cache_size:   int — total de entradas no cache
    - latency_ms:   float — tempo de lookup em ms

Invalidação:
    Chamar .clear() ao reindexar documentos (novos PDFs = novo contexto).
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Resultado do lookup no cache
# ---------------------------------------------------------------------------

@dataclass
class CacheResult:
    """
    Encapsula o resultado de uma consulta ao semantic cache.

    Attributes:
        hit:           True se encontrou resposta cacheada com similaridade
                       acima do threshold.
        answer:        Resposta cacheada (None se miss).
        sources:       Documentos de contexto da resposta cacheada (None se miss).
        similarity:    Score de similaridade 0.0–1.0 (0.0 se cache vazio).
        matched_query: Query original que gerou o hit (None se miss).
        cache_size:    Número de entradas no cache no momento do lookup.
        latency_ms:    Tempo de lookup em milissegundos.
    """
    hit:           bool
    answer:        str | None        = None
    sources:       list | None       = None
    similarity:    float             = 0.0
    matched_query: str | None        = None
    cache_size:    int               = 0
    latency_ms:    float             = 0.0


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Cache semântico baseado em similaridade de embeddings.

    Armazena pares (embedding_da_query → {answer, sources, query_text})
    e realiza lookup por similaridade de cosseno via numpy.

    Por que não usar FAISS diretamente aqui:
        O índice de documentos já usa FAISS. Para o cache, o volume de
        entradas é pequeno (dezenas a centenas por sessão), então numpy
        é suficiente, mais simples e sem dependência de índice externo.

    Args:
        embeddings_model: Instância de HuggingFaceEmbeddings (compartilhada
                          com o retriever para evitar carregar o modelo duas vezes).
        threshold:        Similaridade mínima para considerar cache hit.
                          Recomendado: 0.92–0.95. Padrão: 0.92.
    """

    def __init__(self, embeddings_model, threshold: float = 0.92):
        self._embeddings  = embeddings_model
        self._threshold   = threshold
        self._embeddings_matrix: np.ndarray | None = None  # (N, D)
        self._entries: list[dict] = []   # [{query, answer, sources}]

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def lookup(self, query: str) -> CacheResult:
        """
        Busca uma resposta cacheada semanticamente similar à query.

        Calcula o embedding da query e compara com todas as entradas
        via similaridade de cosseno. Retorna hit se o score máximo
        ultrapassar o threshold.

        Args:
            query: Texto da query do usuário.

        Returns:
            CacheResult com hit=True e resposta preenchida, ou hit=False.
        """
        t0 = time.perf_counter()

        # Cache vazio → miss imediato
        if not self._entries:
            return CacheResult(
                hit        = False,
                cache_size = 0,
                latency_ms = round((time.perf_counter() - t0) * 1000, 1),
            )

        # Gera embedding da query atual
        query_vec = self._embed(query)

        # Similaridade de cosseno com todas as entradas
        similarities = self._cosine_similarity(query_vec, self._embeddings_matrix)
        best_idx     = int(np.argmax(similarities))
        best_score   = float(similarities[best_idx])

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if best_score >= self._threshold:
            entry = self._entries[best_idx]
            return CacheResult(
                hit           = True,
                answer        = entry["answer"],
                sources       = entry["sources"],
                similarity    = round(best_score, 4),
                matched_query = entry["query"],
                cache_size    = len(self._entries),
                latency_ms    = latency_ms,
            )

        return CacheResult(
            hit        = False,
            similarity = round(best_score, 4),
            cache_size = len(self._entries),
            latency_ms = latency_ms,
        )

    def store(self, query: str, answer: str, sources: list) -> None:
        """
        Armazena uma nova entrada no cache.

        Gera o embedding da query e adiciona à matriz interna.
        Entradas duplicadas (mesma query exata) são ignoradas.

        Args:
            query:   Texto da query do usuário.
            answer:  Resposta gerada pelo LLM.
            sources: Lista de Document retornados pelo RAG.
        """
        # Evita duplicatas exatas
        if any(e["query"] == query for e in self._entries):
            return

        vec = self._embed(query)

        # Atualiza matriz de embeddings
        if self._embeddings_matrix is None:
            self._embeddings_matrix = vec.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, vec])

        self._entries.append({
            "query":   query,
            "answer":  answer,
            "sources": sources,
        })

    def clear(self) -> None:
        """
        Limpa todas as entradas do cache.

        Deve ser chamado ao reindexar documentos (novos PDFs podem
        gerar respostas diferentes para as mesmas perguntas).
        """
        self._embeddings_matrix = None
        self._entries           = []

    @property
    def size(self) -> int:
        """Número de entradas atualmente no cache."""
        return len(self._entries)

    @property
    def threshold(self) -> float:
        """Threshold de similaridade configurado."""
        return self._threshold

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """
        Gera o embedding normalizado de um texto.

        Normaliza para norma unitária para que o produto escalar
        seja equivalente à similaridade de cosseno.

        Args:
            text: Texto a ser embeddado.

        Returns:
            np.ndarray: Vetor 1D normalizado.
        """
        vec = np.array(self._embeddings.embed_query(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Calcula similaridade de cosseno entre a query e todas as entradas.

        Como os vetores já estão normalizados (norma = 1), o produto
        escalar é equivalente ao cosseno do ângulo entre eles.

        Args:
            query_vec: Vetor 1D da query (já normalizado).
            matrix:    Matriz (N, D) das entradas cacheadas (já normalizadas).

        Returns:
            np.ndarray: Array 1D de scores 0.0–1.0, um por entrada.
        """
        return matrix @ query_vec
