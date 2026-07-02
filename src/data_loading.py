"""
Carrega uma amostra do corpus de notícias em português (Hugging Face) e salva
em data/raw/. Rodar uma única vez (ou quando quiser trocar o período/tamanho
da amostra).
"""

import os
import pandas as pd
from datasets import load_dataset

# --- parâmetros que vão para o DECISOES.md ---
N_AMOSTRA = 1000
DATA_INICIO = "2020-01-01"
DATA_FIM = "2021-01-01"
SEED = 42
OUTPUT_PATH = "data/raw/corpus_noticias.csv"


def carregar_amostra() -> pd.DataFrame:
    print("Baixando dataset (pode demorar na primeira vez)...")
    ds = load_dataset("iara-project/news-articles-ptbr-dataset", split="train")
    df = ds.to_pandas()

    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    filtro = (df["date"] >= DATA_INICIO) & (df["date"] < DATA_FIM)
    df_periodo = df[filtro]

    print(f"Notícias disponíveis no período: {len(df_periodo)}")
    if len(df_periodo) < N_AMOSTRA:
        print("\n[DIAGNÓSTICO] Volume por ano no dataset completo:")
        print(df.groupby(df["date"].dt.year).size().to_string())
        raise ValueError(
            f"Período selecionado tem apenas {len(df_periodo)} notícias, "
            f"menos que N_AMOSTRA={N_AMOSTRA}. "
            "Ajuste DATA_INICIO/DATA_FIM ou N_AMOSTRA conforme o diagnóstico acima."
        )

    amostra = df_periodo.sample(N_AMOSTRA, random_state=SEED).reset_index(drop=True)

    # id estável para rastrear documentos ao longo do pipeline
    amostra["doc_id"] = amostra.index

    return amostra[["doc_id", "title", "text", "date", "category", "link"]]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = carregar_amostra()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Salvo em {OUTPUT_PATH} ({len(df)} documentos)")
    print("\nDistribuição por categoria (apenas para referência, não usar no treino):")
    print(df["category"].value_counts())
