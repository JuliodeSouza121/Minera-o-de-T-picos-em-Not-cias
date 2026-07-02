# Mineração de Tópicos em Notícias — Trabalho Integrador de Mineração de Textos

## Problema e cliente fictício

Um editor-chefe de redação recebe centenas de notícias publicadas por dia e precisa
identificar, de forma automática, quais são os principais temas/subtemas que estão
sendo cobertos em um determinado período, para apoiar o planejamento editorial
(quais pautas reforçar, quais temas estão saturados, quais estão emergindo).

Caminho escolhido: **A — Clusterização**.

## Corpus

`iara-project/news-articles-ptbr-dataset` (Hugging Face) — 352 mil notícias da
Folha de S.Paulo, com campos `title`, `text`, `date`, `category`. Usamos uma
amostra de ~1.500-2.000 notícias de um período específico.

A coluna `category` (19 categorias) **não é usada para treinar nada** — serve
apenas como referência externa para avaliar a pureza dos clusters encontrados
(ground truth de comparação, não rótulo de treino).

## Estrutura do repositório

```
.
├── README.md
├── DECISOES.md          # diário de decisões de implementação
├── requirements.txt
├── data/
│   ├── raw/              # amostra bruta baixada do HF
│   └── processed/        # após pré-processamento
├── src/                  # código de pipeline (não notebooks soltos)
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── clustering.py
│   ├── llm_labeling.py
│   └── schemas.py         # schemas Pydantic
├── prompts/              # prompts versionados (texto puro, não hardcoded no código)
│   └── cluster_labeling_v1.txt
├── notebooks/
│   └── pipeline_exploratorio.ipynb   # demo / exploração, não é o entregável de produção
└── outputs/
    ├── clusters_resultado.csv
    ├── metricas_baseline.json
    └── insights_finais.md
```

## Como rodar

```bash
pip install -r requirements.txt
python src/data_loading.py
python src/preprocessing.py
python src/embeddings.py
python src/clustering.py
python src/llm_labeling.py   # exige GEMINI_API_KEY no ambiente
python src/estimar_custos.py
```

### Interface de demonstração (bônus)

Depois de rodar o pipeline completo (gera os arquivos que o app lê):

```bash
streamlit run app.py
```

## Pipeline (visão geral)

```
Dados brutos (HF) → pré-processamento → embeddings (sentence-transformers)
  → redução (UMAP, opcional p/ HDBSCAN) → clusterização (K-Means + HDBSCAN)
  → baseline (silhouette + inspeção qualitativa) → c-TF-IDF (rótulo cru)
  → LLM com saída estruturada (Pydantic) → rótulo final + insights
```

## Métricas reportadas

- Silhouette score (K-Means e HDBSCAN), no espaço de embeddings original (não em
  projeção 2D).
- Pureza dos clusters em relação à `category` real (apenas para leitura crítica,
  não como otimização).
- Cobertura (proporção de documentos fora do ruído, no caso do HDBSCAN).
- Custos estimados de chamadas ao LLM para 1.000 elementos.

## Status

Esqueleto inicial gerado — ver `DECISOES.md` para escolhas em aberto.
