"""
Interface de demonstração

Rodar com: streamlit run app.py
(executar a partir da raiz do projeto, depois de já ter rodado todo o
pipeline em src/)
"""

import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mineração de Tópicos em Notícias", layout="wide")

CORPUS_PATH = "data/processed/corpus_com_clusters.csv"
ROTULOS_PATH = "outputs/clusters_resultado.csv"
METRICAS_PATH = "outputs/metricas_baseline.json"
CUSTOS_PATH = "outputs/estimativa_custos.json"

COLUNA_CLUSTER = "cluster_kmeans"  # manter igual ao usado em llm_labeling.py


@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv(CORPUS_PATH)
    except FileNotFoundError:
        return None, None, None, None

    try:
        rotulos = pd.read_csv(ROTULOS_PATH)
    except FileNotFoundError:
        rotulos = None

    try:
        with open(METRICAS_PATH, "r", encoding="utf-8") as f:
            metricas = json.load(f)
    except FileNotFoundError:
        metricas = None

    try:
        with open(CUSTOS_PATH, "r", encoding="utf-8") as f:
            custos = json.load(f)
    except FileNotFoundError:
        custos = None

    return df, rotulos, metricas, custos


df, rotulos, metricas, custos = carregar_dados()

st.title("Mineração de tópicos em notícias")
st.caption(
    "Demonstração do pipeline: embeddings → clusterização → rotulagem via LLM "
    "(Caminho A — Clusterização)"
)

if df is None:
    st.error(
        "Nenhum dado encontrado ainda. Rode o pipeline completo em `src/` "
        "antes de abrir esta interface:\n\n"
        "`python src/data_loading.py && python src/preprocessing.py && "
        "python src/embeddings.py && python src/clustering.py && "
        "python src/llm_labeling.py`"
    )
    st.stop()

# --- Visão geral / baseline ---
st.header("1. Baseline (antes do LLM)")
col1, col2, col3 = st.columns(3)

if metricas:
    col1.metric("K escolhido (K-Means)", metricas["kmeans"]["k_escolhido"])
    col2.metric("Silhouette (K-Means)", f"{metricas['kmeans']['silhouette']:.3f}")
    col3.metric("Pureza (K-Means)", f"{metricas['kmeans']['pureza']:.1%}")

    col1.metric(
        "Clusters descobertos (HDBSCAN)", metricas["hdbscan"]["n_clusters_descobertos"]
    )
    silhouette_hdb = metricas["hdbscan"]["silhouette"]
    col2.metric(
        "Silhouette (HDBSCAN)",
        f"{silhouette_hdb:.3f}" if silhouette_hdb is not None else "N/A",
    )
    col3.metric("Cobertura (HDBSCAN)", f"{metricas['hdbscan']['cobertura']:.1%}")

    st.caption(
        "⚠️ " + metricas.get(
            "aviso",
            "Silhouettes de algoritmos diferentes podem não ser diretamente comparáveis.",
        )
    )
else:
    st.warning("Métricas de baseline não encontradas (rode src/clustering.py).")

st.divider()

# --- Exploração dos clusters ---
st.header("2. Explorar clusters")

clusters_disponiveis = sorted(df[COLUNA_CLUSTER].unique())
cluster_selecionado = st.selectbox(
    "Selecione um cluster para inspecionar",
    clusters_disponiveis,
    format_func=lambda c: f"Cluster {c}" if c != -1 else "Ruído (-1)",
)

if rotulos is not None:
    info_rotulo = rotulos[rotulos["cluster_id"] == cluster_selecionado]
    if not info_rotulo.empty:
        linha = info_rotulo.iloc[0]
        st.subheader(f"📌 {linha['nome_topico']}")
        st.write(linha["resumo"])
        st.write("**Palavras-chave:**", ", ".join(eval(linha["palavras_chave"]))
                  if isinstance(linha["palavras_chave"], str) else linha["palavras_chave"])
        c1, c2 = st.columns(2)
        c1.metric("Confiança do LLM", f"{linha['nivel_confianca']:.0%}")
        c2.metric("Possível ruído?", "Sim" if linha["possivel_ruido"] else "Não")
    else:
        st.info("Este cluster ainda não foi rotulado pelo LLM.")
else:
    st.info("Rótulos do LLM não encontrados ainda (rode src/llm_labeling.py).")

st.write("**Notícias deste cluster (amostra):**")
amostra_cluster = df[df[COLUNA_CLUSTER] == cluster_selecionado].head(10)
st.dataframe(
    amostra_cluster[["title", "category", "date"]],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --- Distribuição geral ---
st.header("3. Distribuição dos clusters")
contagem = df[COLUNA_CLUSTER].value_counts().sort_index()
st.bar_chart(contagem)

st.divider()

# --- Custos do LLM ---
st.header("4. Custos estimados (LLM, projeção para 1.000 elementos)")
if custos:
    custos_df = pd.DataFrame(custos["custos_por_modelo"]).T
    st.dataframe(custos_df, use_container_width=True)
else:
    st.info("Estimativa de custos não encontrada (rode src/estimar_custos.py).")
