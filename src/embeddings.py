"""
Gera embeddings densos para cada documento usando sentence-transformers.

Modelo escolhido: paraphrase-multilingual-mpnet-base-v2
- fine-tuned especificamente para similaridade semântica (STS), diferente de
  usar BERT cru com mean pooling (ver discussão de anisotropia na Aula 3)
- suporta português nativamente (multilingual)
- bom equilíbrio entre qualidade e custo computacional para um corpus de
  ~1.500 documentos

Salva os embeddings em formato .npy (não em CSV, para preservar precisão
numérica e evitar arquivos gigantes de texto).
"""

import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/processed/corpus_limpo.csv"
OUTPUT_EMBEDDINGS_PATH = "data/processed/embeddings.npy"
MODELO = "paraphrase-multilingual-mpnet-base-v2"


def gerar_embeddings(textos: list[str]) -> np.ndarray:
    print(f"Carregando modelo {MODELO}...")
    modelo = SentenceTransformer(MODELO)

    print(f"Gerando embeddings para {len(textos)} documentos...")
    embeddings = modelo.encode(
        textos,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalizado: necessário antes de
        # qualquer distância/cosseno (ver Aula 4) -- impede que vetores
        # "maiores" dominem a similaridade
    )
    return embeddings


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS_PATH), exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    embeddings = gerar_embeddings(df["texto_modelo"].tolist())
    np.save(OUTPUT_EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings salvos em {OUTPUT_EMBEDDINGS_PATH}, shape={embeddings.shape}")
