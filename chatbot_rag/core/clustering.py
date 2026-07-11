"""
=============================================================================
core/clustering.py — Clusterização Automática de Intenções
=============================================================================
Responsabilidade:
    Descobrir padrões temáticos nas queries da sessão usando clustering
    de densidade sobre embeddings, sem número de clusters predefinido.

Pipeline:
    queries[] → embeddings → UMAP (redução 2D) → HDBSCAN (clustering)
    → label automático por LLM → ClusteringResult

Por que UMAP + HDBSCAN (e não K-Means):
    - K-Means exige k predefinido e cria clusters esféricos — inadequado
      para texto cujos clusters têm formas arbitrárias.
    - HDBSCAN descobre o número de clusters automaticamente e identifica
      outliers (queries que não pertencem a nenhum grupo) — candidatas
      a gaps no corpus de documentos.
    - UMAP preserva estrutura local melhor que PCA/t-SNE e é mais
      rápido que t-SNE para conjuntos pequenos.

Requisitos mínimos:
    - Mínimo de 5 queries para executar clustering (abaixo disso,
      não há padrão estatisticamente significativo).

Saída (ClusteringResult):
    - clusters:      lista de ClusterInfo com label, queries, score médio
    - outliers:      queries sem cluster (gaps potenciais no corpus)
    - coordinates:   coordenadas 2D para visualização (x, y por query)
    - n_clusters:    número de clusters descobertos
    - coverage:      % de queries clusterizadas (vs outliers)
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Mínimo de queries para executar clustering
# ---------------------------------------------------------------------------
MIN_QUERIES_FOR_CLUSTERING = 5


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------

@dataclass
class ClusterInfo:
    """
    Representa um cluster de intenções descoberto.

    Attributes:
        id:           ID numérico do cluster (0, 1, 2, …).
        label:        Label gerado pelo LLM ou heurística
                      (ex: "consultar política de reembolso").
        queries:      Lista de queries que pertencem a este cluster.
        size:         Número de queries no cluster.
        coherence:    Coesão interna: similaridade média entre queries
                      e o centroide do cluster (0–1).
        top_terms:    Termos mais frequentes nas queries do cluster.
    """
    id:         int
    label:      str
    queries:    list[str]
    size:       int
    coherence:  float
    top_terms:  list[str] = field(default_factory=list)


@dataclass
class ClusteringResult:
    """
    Resultado completo da clusterização de uma sessão.

    Attributes:
        clusters:    Lista de ClusterInfo ordenada por tamanho (desc).
        outliers:    Queries sem cluster → candidatas a gaps no corpus.
        coordinates: Dict {query_text: (x, y)} para visualização 2D.
        n_clusters:  Número de clusters descobertos (excluindo outliers).
        coverage:    Proporção de queries clusterizadas (0–1).
        total_queries: Total de queries analisadas.
        latency_ms:  Tempo de execução do clustering em ms.
        error:       Mensagem de erro se o clustering falhou (ou None).
    """
    clusters:      list[ClusterInfo]      = field(default_factory=list)
    outliers:      list[str]              = field(default_factory=list)
    coordinates:   dict[str, tuple]       = field(default_factory=dict)
    n_clusters:    int                    = 0
    coverage:      float                  = 0.0
    total_queries: int                    = 0
    latency_ms:    float                  = 0.0
    error:         str | None             = None


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _extract_top_terms(queries: list[str], n: int = 5) -> list[str]:
    """
    Extrai os termos mais frequentes de um grupo de queries.

    Remove stopwords comuns em PT-BR e retorna as palavras
    mais frequentes como representação léxica do cluster.

    Args:
        queries: Lista de queries do cluster.
        n:       Número de termos a retornar.

    Returns:
        Lista de termos ordenados por frequência.
    """
    import re
    from collections import Counter

    STOPWORDS = {
        "a", "o", "e", "de", "do", "da", "em", "no", "na", "para",
        "com", "um", "uma", "os", "as", "que", "se", "por", "mais",
        "como", "qual", "quais", "quando", "onde", "quem", "the",
        "is", "of", "in", "to", "and", "a", "for", "what", "how",
    }

    all_words = []
    for q in queries:
        words = re.findall(r"\b\w{3,}\b", q.lower())
        all_words.extend(w for w in words if w not in STOPWORDS)

    return [term for term, _ in Counter(all_words).most_common(n)]


def _generate_cluster_label(queries: list[str], top_terms: list[str], llm=None) -> str:
    """
    Gera um label descritivo para o cluster.

    Estratégia em duas camadas:
        1. LLM (se disponível): gera label conciso de 3–6 palavras
        2. Heurística: concatena os 3 termos mais frequentes

    Args:
        queries:   Queries do cluster (amostra de até 5).
        top_terms: Termos mais frequentes extraídos das queries.
        llm:       Instância de LLM (opcional).

    Returns:
        str: Label descritivo do cluster.
    """
    if llm is not None:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            sample    = queries[:5]
            prompt    = (
                "Dado o conjunto de perguntas abaixo, gere um label conciso "
                "(3 a 6 palavras) que descreva a intenção comum delas. "
                "Responda APENAS com o label, sem pontuação extra.\n\n"
                "Perguntas:\n" + "\n".join(f"- {q}" for q in sample)
            )
            response = llm.invoke([
                SystemMessage(content="Você é um classificador de intenções de usuário."),
                HumanMessage(content=prompt),
            ])
            label = response.content.strip().strip("\"'").capitalize()
            if 3 <= len(label.split()) <= 10:
                return label
        except Exception:
            pass

    # Fallback heurístico
    if top_terms:
        return " · ".join(top_terms[:3]).capitalize()
    return "Cluster sem label"


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------

def cluster_queries(
    queries:    list[str],
    embeddings,
    llm=None,
    min_cluster_size: int = 2,
) -> ClusteringResult:
    """
    Executa o pipeline completo de clusterização sobre as queries da sessão.

    Pipeline:
        1. Verifica mínimo de queries
        2. Gera embeddings das queries (modelo compartilhado)
        3. Reduz dimensionalidade com UMAP (→ 2D)
        4. Clusteriza com HDBSCAN
        5. Identifica outliers (label = -1 no HDBSCAN)
        6. Gera labels para cada cluster via LLM ou heurística
        7. Calcula coesão interna por cluster
        8. Retorna ClusteringResult

    Args:
        queries:          Lista de queries RAG da sessão (já filtradas
                          — sem chitchat, apenas RAG_QUERY e FOLLOWUP).
        embeddings:       HuggingFaceEmbeddings compartilhado.
        llm:              Instância de LLM para geração de labels (opcional).
        min_cluster_size: Tamanho mínimo de um cluster válido.
                          Queries em grupos menores viram outliers.

    Returns:
        ClusteringResult com clusters, outliers, coordenadas e métricas.
    """
    t0 = time.perf_counter()

    # --- Validação de mínimo ---
    if len(queries) < MIN_QUERIES_FOR_CLUSTERING:
        return ClusteringResult(
            total_queries = len(queries),
            latency_ms    = 0.0,
            error         = (
                f"Mínimo de {MIN_QUERIES_FOR_CLUSTERING} queries necessário "
                f"para clustering. Atual: {len(queries)}."
            ),
        )

    # --- 1. Embeddings ---
    try:
        vectors = np.array(
            embeddings.embed_documents(queries), dtype=np.float32
        )
    except Exception as e:
        return ClusteringResult(
            total_queries = len(queries),
            error         = f"Erro ao gerar embeddings: {e}",
        )

    # --- 2. Redução dimensional com UMAP ---
    try:
        import umap
        n_neighbors = min(5, len(queries) - 1)
        reducer     = umap.UMAP(
            n_neighbors   = n_neighbors,
            n_components  = 2,
            metric        = "cosine",
            random_state  = 42,
            min_dist      = 0.1,
        )
        coords_2d = reducer.fit_transform(vectors)
    except ImportError:
        # Fallback: PCA simples se UMAP não estiver instalado
        try:
            from sklearn.decomposition import PCA
            pca       = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(vectors)
        except Exception as e:
            return ClusteringResult(
                total_queries = len(queries),
                error         = f"UMAP e PCA indisponíveis: {e}",
            )
    except Exception as e:
        return ClusteringResult(
            total_queries = len(queries),
            error         = f"Erro na redução dimensional: {e}",
        )

    # --- 3. Clustering com HDBSCAN ---
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            metric           = "euclidean",
            cluster_selection_method = "eom",
        )
        labels = clusterer.fit_predict(coords_2d)
    except ImportError:
        # Fallback: AgglomerativeClustering se HDBSCAN não disponível
        try:
            from sklearn.cluster import AgglomerativeClustering
            n_cls   = max(2, len(queries) // 3)
            ac      = AgglomerativeClustering(n_clusters=n_cls)
            labels  = ac.fit_predict(vectors)
        except Exception as e:
            return ClusteringResult(
                total_queries = len(queries),
                error         = f"HDBSCAN e AgglomerativeClustering indisponíveis: {e}",
            )
    except Exception as e:
        return ClusteringResult(
            total_queries = len(queries),
            error         = f"Erro no clustering: {e}",
        )

    # --- 4. Monta coordenadas 2D por query ---
    coordinates = {
        query: (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
        for i, query in enumerate(queries)
    }

    # --- 5. Separa outliers (label = -1) e clusters válidos ---
    unique_labels = set(labels)
    outlier_queries = [
        queries[i] for i, lbl in enumerate(labels) if lbl == -1
    ]

    # --- 6. Constrói ClusterInfo para cada cluster ---
    cluster_infos: list[ClusterInfo] = []
    valid_labels  = sorted(lbl for lbl in unique_labels if lbl != -1)

    for cluster_id in valid_labels:
        cluster_queries_list = [
            queries[i] for i, lbl in enumerate(labels) if lbl == cluster_id
        ]
        cluster_vectors = np.array([
            vectors[i] for i, lbl in enumerate(labels) if lbl == cluster_id
        ])

        # Coesão: similaridade média entre queries e centroide
        centroid  = cluster_vectors.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        norms     = np.linalg.norm(cluster_vectors, axis=1, keepdims=True)
        normed    = cluster_vectors / np.maximum(norms, 1e-9)
        coherence = float(np.mean(normed @ centroid))

        top_terms = _extract_top_terms(cluster_queries_list)
        label     = _generate_cluster_label(cluster_queries_list, top_terms, llm)

        cluster_infos.append(ClusterInfo(
            id        = int(cluster_id),
            label     = label,
            queries   = cluster_queries_list,
            size      = len(cluster_queries_list),
            coherence = round(coherence, 4),
            top_terms = top_terms,
        ))

    # Ordena por tamanho decrescente
    cluster_infos.sort(key=lambda c: c.size, reverse=True)

    n_clustered = len(queries) - len(outlier_queries)
    coverage    = round(n_clustered / len(queries), 4) if queries else 0.0

    return ClusteringResult(
        clusters      = cluster_infos,
        outliers      = outlier_queries,
        coordinates   = coordinates,
        n_clusters    = len(cluster_infos),
        coverage      = coverage,
        total_queries = len(queries),
        latency_ms    = round((time.perf_counter() - t0) * 1000, 1),
    )
