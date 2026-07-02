"""
Pré-processamento leve do texto antes de gerar embeddings.

Decisão importante (ver Aula 2 e Aula 3): sentence-transformers já foram
treinados sobre texto cru/natural, então pré-processamento agressivo
(remover stopwords, lematizar, remover pontuação) tende a ATRAPALHAR a
qualidade do embedding em vez de ajudar -- diferente do pipeline TF-IDF
clássico, que se beneficia dessa limpeza.

Aqui fazemos apenas limpeza estrutural (não linguística):
- concatenar título + texto (título primeiro, para mitigar truncamento de
  modelos com janela de contexto curta -- ver discussão da Aula 4 sobre
  BERTopic/LDA em textos longos)
- remover HTML/links residuais
- normalizar espaços em branco
- descartar textos vazios ou curtos demais para serem informativos
"""

import re
import os
import pandas as pd

INPUT_PATH = "data/raw/corpus_noticias.csv"
OUTPUT_PATH = "data/processed/corpus_limpo.csv"
MIN_CARACTERES = 50


def limpar_texto(texto: str) -> str:
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def preprocessar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("").apply(limpar_texto)
    df["text"] = df["text"].fillna("").apply(limpar_texto)

    # título primeiro: ajuda modelos de embedding com janela curta de tokens
    # a captar o assunto principal mesmo truncando o restante
    df["texto_modelo"] = (df["title"] + ". " + df["text"]).str.strip()

    antes = len(df)
    df = df[df["texto_modelo"].str.len() >= MIN_CARACTERES].reset_index(drop=True)
    print(f"Removidos {antes - len(df)} documentos vazios/curtos demais (< {MIN_CARACTERES} caracteres)")

    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    df_limpo = preprocessar(df)
    df_limpo.to_csv(OUTPUT_PATH, index=False)
    print(f"Salvo em {OUTPUT_PATH} ({len(df_limpo)} documentos)")
