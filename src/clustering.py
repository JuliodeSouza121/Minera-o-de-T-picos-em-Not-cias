"""
Compara dois algoritmos de clusterização sobre os embeddings: K-Means e
HDBSCAN. Calcula o baseline obrigatório (silhouette + inspeção qualitativa)
ANTES de qualquer chamada ao LLM.

Decisões de design (ver Aula 4):
- Silhouette é calculado no espaço de embeddings original normalizado, não em
  uma projeção 2D (UMAP/t-SNE) -- projeção 2D mostra o desenho, não a
  estrutura real do espaço.
- Para o HDBSCAN, reduzimos com UMAP para um número intermediário de
  dimensões (nunca 2D) antes de clusterizar, porque HDBSCAN sofre com a
  maldição da dimensionalidade em ~768D direto.
- K-Means dá 0% de ruído por definição (aloca tudo) -- isso é lembrado
  explicitamente no relatório final para não comparar silhouettes de forma
  enganosa.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import umap

INPUT_DF_PATH = "data/processed/corpus_limpo.csv"
INPUT_EMB_PATH = "data/processed/embeddings.npy"
OUTPUT_RESULT_PATH = "data/processed/corpus_com_clusters.csv"
OUTPUT_METRICS_PATH = "outputs/metricas_baseline.json"

# --- parâmetros a justificar no DECISOES.md ---
K_RANGE = range(4, 15)
UMAP_N_COMPONENTS = 15  # nunca 2D para clusterizar de verdade
HDBSCAN_MIN_CLUSTER_SIZE = 25


def rodar_kmeans(embeddings: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Varre K_RANGE e escolhe o K com melhor silhouette."""
    melhor_k, melhor_score, melhores_labels = None, -1, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        print(f"  K-Means k={k}: silhouette={score:.4f}")
        if score > melhor_score:
            melhor_k, melhor_score, melhores_labels = k, score, labels
    return melhores_labels, melhor_k, melhor_score


def rodar_hdbscan(embeddings: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Reduz com UMAP (não para 2D) e roda HDBSCAN por densidade."""
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS, metric="cosine", random_state=42
    )
    embeddings_reduzidos = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, metric="euclidean"
    )
    labels = clusterer.fit_predict(embeddings_reduzidos)

    mascara_validos = labels != -1
    if mascara_validos.sum() > 1 and len(set(labels[mascara_validos])) > 1:
        score = silhouette_score(
            embeddings_reduzidos[mascara_validos],
            labels[mascara_validos],
            metric="euclidean",
        )
    else:
        score = float("nan")

    cobertura = mascara_validos.mean()
    return labels, score, cobertura


def calcular_pureza(labels: np.ndarray, categorias_reais: pd.Series) -> float:
    """
    Pureza: para cada cluster, qual fração pertence à categoria majoritária
    real. Usado SOMENTE como leitura crítica (a categoria não foi usada para
    treinar nada) -- não confundir com acurácia supervisionada.
    """
    df_temp = pd.DataFrame({"cluster": labels, "categoria": categorias_reais})
    df_temp = df_temp[df_temp["cluster"] != -1]  # ignora ruído do HDBSCAN

    total_correto = 0
    for cluster_id in df_temp["cluster"].unique():
        subset = df_temp[df_temp["cluster"] == cluster_id]
        contagem_majoritaria = subset["categoria"].value_counts().iloc[0]
        total_correto += contagem_majoritaria

    return total_correto / len(df_temp) if len(df_temp) > 0 else 0.0


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    df = pd.read_csv(INPUT_DF_PATH)
    embeddings = np.load(INPUT_EMB_PATH)

    print("=== K-Means ===")
    labels_km, melhor_k, score_km = rodar_kmeans(embeddings)
    pureza_km = calcular_pureza(labels_km, df["category"])
    print(f"Melhor K={melhor_k}, silhouette={score_km:.4f}, pureza={pureza_km:.4f}")
    print("Cobertura K-Means: 100% (por definição, sem ruído)\n")

    print("=== HDBSCAN ===")
    labels_hdb, score_hdb, cobertura_hdb = rodar_hdbscan(embeddings)
    pureza_hdb = calcular_pureza(labels_hdb, df["category"])
    n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    print(
        f"N clusters descobertos={n_clusters_hdb}, silhouette={score_hdb:.4f}, "
        f"cobertura={cobertura_hdb:.4f}, pureza={pureza_hdb:.4f}"
    )

    # --- ALERTA a registrar na análise crítica ---
    # Silhouette mede geometria, não sentido temático. Comparar K-Means e
    # HDBSCAN só é justo no mesmo espaço/métrica -- aqui K-Means foi calculado
    # no espaço original (cosine) e HDBSCAN no espaço reduzido por UMAP
    # (euclidean), então os números NÃO são diretamente comparáveis entre si.
    # Reportar isso explicitamente na análise de resultados.

    df["cluster_kmeans"] = labels_km
    df["cluster_hdbscan"] = labels_hdb
    df.to_csv(OUTPUT_RESULT_PATH, index=False)

    metricas = {
        "kmeans": {
            "k_escolhido": melhor_k,
            "silhouette": float(score_km),
            "pureza": float(pureza_km),
            "cobertura": 1.0,
            "metrica_distancia": "cosine",
        },
        "hdbscan": {
            "n_clusters_descobertos": int(n_clusters_hdb),
            "silhouette": float(score_hdb) if not np.isnan(score_hdb) else None,
            "pureza": float(pureza_hdb),
            "cobertura": float(cobertura_hdb),
            "metrica_distancia": "euclidean (apos UMAP)",
            "umap_n_components": UMAP_N_COMPONENTS,
        },
        "aviso": (
            "Silhouettes nao sao diretamente comparaveis entre os dois "
            "algoritmos: espacos/metricas diferentes. Pureza eh comparavel "
            "pois usa a mesma referencia externa (category)."
        ),
    }
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\nResultado salvo em {OUTPUT_RESULT_PATH}")
    print(f"Métricas baseline salvas em {OUTPUT_METRICS_PATH}")
