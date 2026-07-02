"""
Rotula cada cluster usando:
1. c-TF-IDF (palavras mais distintivas do cluster, sem LLM) -- baseline cru.
2. LLM com saída estruturada validada via Pydantic -- rótulo final enriquecido.

Tratamento de falha de validação: se o LLM devolver um JSON que não bate com
o schema (campo faltando, tipo errado, nome genérico), a falha é capturada,
registrada e o cluster pode ser reprocessado ou marcado para revisão manual
-- nunca aceitamos a saída crua sem validação.
"""

import json
import os
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pydantic import ValidationError
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from schemas import RotuloCluster

# --- cliente Gemini ---
# A API key deve vir de variável de ambiente, NUNCA hardcoded no código.
# No Colab: usar `from google.colab import userdata` + Secrets, ou
# `os.environ["GEMINI_API_KEY"] = "..."` numa célula que não vai pro Git.
# Pegue uma chave gratuita em https://aistudio.google.com/app/apikey
GEMINI_MODEL = "gemini-2.5-flash"  # ver DECISOES.md para a justificativa de escolha

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Defina a variável de ambiente GEMINI_API_KEY antes de rodar. "
                "Gere uma chave gratuita em https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client

INPUT_PATH = "data/processed/corpus_com_clusters.csv"
PROMPT_PATH = "prompts/cluster_labeling_v1.txt"
OUTPUT_PATH = "outputs/clusters_resultado.csv"
ERROS_PATH = "outputs/erros_validacao_llm.json"

COLUNA_CLUSTER = "cluster_kmeans"  # trocar para 'cluster_hdbscan' se preferir
N_EXEMPLOS_POR_CLUSTER = 4
N_PALAVRAS_CTFIDF = 10


def calcular_c_tfidf(df: pd.DataFrame, coluna_cluster: str) -> dict[int, list[str]]:
    """
    c-TF-IDF: trata cada cluster como um único documento agregado e calcula
    TF-IDF sobre esses "super-documentos" -- dá mais peso a palavras que
    distinguem aquele cluster dos demais (ver Aula 4, formulação do BERTopic).

    Nota de implementação: o vocabulário do CountVectorizer é ajustado sobre
    os documentos INDIVIDUAIS, não sobre os textos já agregados -- evita
    descartar palavras específicas de clusters pequenos por min_df alto.
    """
    df_validos = df[df[coluna_cluster] != -1]

    STOPWORDS_PT = [
        "de","do","da","dos","das","em","no","na","nos","nas","um","uma","uns",
        "umas","o","a","os","as","e","é","que","se","com","para","por","mais",
        "mas","ao","à","às","aos","ou","não","sua","seu","suas","seus","ele",
        "ela","eles","elas","isso","este","esta","esse","essa","isso","aqui",
        "já","foi","ser","tem","ter","são","como","sobre","também","quando",
        "muito","até","depois","ainda","entre","mesmo","então","porque","isso",
        "pelo","pela","pelos","pelas","num","numa","isso","este","aquele",
    ]

    vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, stop_words=STOPWORDS_PT
    )
    vectorizer.fit(df_validos["texto_modelo"])
    vocabulario = vectorizer.get_feature_names_out()

    palavras_por_cluster = {}
    contagem_total_por_palavra = None

    docs_agregados = {}
    for cluster_id in sorted(df_validos[coluna_cluster].unique()):
        textos_cluster = df_validos[df_validos[coluna_cluster] == cluster_id][
            "texto_modelo"
        ]
        docs_agregados[cluster_id] = " ".join(textos_cluster)

    matriz_agregada = vectorizer.transform(docs_agregados.values())
    contagem_total_por_palavra = matriz_agregada.sum(axis=0).A1

    import numpy as np

    media_palavras_por_cluster = matriz_agregada.sum(axis=1).mean()

    for i, cluster_id in enumerate(docs_agregados.keys()):
        freq_no_cluster = matriz_agregada[i].toarray().flatten()
        idf = np.log(1 + media_palavras_por_cluster / (contagem_total_por_palavra + 1))
        score = freq_no_cluster * idf
        top_indices = score.argsort()[::-1][:N_PALAVRAS_CTFIDF]
        palavras_por_cluster[int(cluster_id)] = [vocabulario[idx] for idx in top_indices]

    return palavras_por_cluster


def montar_exemplos(df: pd.DataFrame, coluna_cluster: str, cluster_id: int) -> str:
    subset = df[df[coluna_cluster] == cluster_id].head(N_EXEMPLOS_POR_CLUSTER)
    blocos = []
    for _, row in subset.iterrows():
        trecho = row["text"][:300]
        blocos.append(f"- Título: {row['title']}\n  Início: {trecho}...")
    return "\n".join(blocos)


def chamar_llm(prompt_preenchido: str) -> tuple[RotuloCluster | None, dict]:
    """
    Chama o Gemini pedindo saída estruturada diretamente no schema Pydantic
    (response_schema=RotuloCluster). A própria API força o formato antes de
    devolver, então o parsing manual de JSON deixa de ser necessário na
    maioria dos casos -- mas a validação Pydantic explícita é mantida como
    rede de segurança (e para registrar eventuais falhas na análise de erro).

    Retorna (rotulo_validado_ou_None, info_de_uso_de_tokens).
    """
    client = get_client()

    max_tentativas = 3
    for tentativa in range(1, max_tentativas + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_preenchido,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RotuloCluster,
                    temperature=0.2,  # baixa: queremos consistência, não criatividade
                ),
            )
            uso = {
                "tokens_entrada": response.usage_metadata.prompt_token_count,
                "tokens_saida": response.usage_metadata.candidates_token_count,
            }

            # response.parsed já tenta instanciar o Pydantic model; mesmo
            # assim revalidamos explicitamente para não confiar cegamente
            if response.parsed is not None:
                try:
                    rotulo = RotuloCluster.model_validate(
                        response.parsed.model_dump()
                    )
                    return rotulo, uso
                except ValidationError as e:
                    registrar_erro_validacao(
                        cluster_id=-1, resposta_crua=response.text, erro=str(e)
                    )
                    return None, uso
            else:
                # API respondeu mas não conseguiu encaixar no schema
                registrar_erro_validacao(
                    cluster_id=-1,
                    resposta_crua=response.text,
                    erro="response.parsed veio None (schema não encaixou)",
                )
                return None, uso

        except genai_errors.APIError as e:
            # erros transitórios (rate limit, 5xx) -- vale tentar de novo
            print(f"[AVISO] Tentativa {tentativa}/{max_tentativas} falhou: {e}")
            if tentativa == max_tentativas:
                registrar_erro_validacao(
                    cluster_id=-1, resposta_crua="", erro=f"APIError após retries: {e}"
                )
                return None, {"tokens_entrada": 0, "tokens_saida": 0}
            time.sleep(2 ** tentativa)  # backoff exponencial: 2s, 4s, 8s

    return None, {"tokens_entrada": 0, "tokens_saida": 0}


def rotular_cluster_com_llm(
    cluster_id: int, palavras_chave: list[str], exemplos: str, prompt_template: str
) -> tuple[RotuloCluster | None, dict]:
    prompt = prompt_template.format(
        cluster_id=cluster_id,
        palavras_chave=", ".join(palavras_chave),
        exemplos=exemplos,
    )
    rotulo, uso = chamar_llm(prompt)
    return rotulo, uso


def registrar_erro_validacao(cluster_id: int, resposta_crua: str, erro: str) -> None:
    """Acumula falhas de validação para a análise de erro estruturada do trabalho."""
    registro = {
        "cluster_id": cluster_id,
        "resposta_crua": resposta_crua,
        "erro": erro,
        "timestamp": time.time(),
    }
    erros_existentes = []
    if os.path.exists(ERROS_PATH):
        with open(ERROS_PATH, "r", encoding="utf-8") as f:
            erros_existentes = json.load(f)
    erros_existentes.append(registro)
    with open(ERROS_PATH, "w", encoding="utf-8") as f:
        json.dump(erros_existentes, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    print("Calculando c-TF-IDF (rótulo cru, sem LLM)...")
    palavras_por_cluster = calcular_c_tfidf(df, COLUNA_CLUSTER)
    for cluster_id, palavras in palavras_por_cluster.items():
        print(f"  Cluster {cluster_id}: {', '.join(palavras)}")

    print(f"\nRotulando clusters com {GEMINI_MODEL} (saída validada via Pydantic)...")
    resultados = []
    uso_total = {"tokens_entrada": 0, "tokens_saida": 0, "chamadas": 0}

    for cluster_id, palavras in palavras_por_cluster.items():
        exemplos = montar_exemplos(df, COLUNA_CLUSTER, cluster_id)
        rotulo, uso = rotular_cluster_com_llm(
            cluster_id, palavras, exemplos, prompt_template
        )
        uso_total["tokens_entrada"] += uso["tokens_entrada"]
        uso_total["tokens_saida"] += uso["tokens_saida"]
        uso_total["chamadas"] += 1

        if rotulo:
            resultados.append(rotulo.model_dump())
            print(f"  Cluster {cluster_id} -> '{rotulo.nome_topico}'")
        else:
            print(f"  Cluster {cluster_id} -> FALHA (ver outputs/erros_validacao_llm.json)")

    if resultados:
        pd.DataFrame(resultados).to_csv(OUTPUT_PATH, index=False)
        print(f"\nRótulos salvos em {OUTPUT_PATH}")

    n_clusters = len(palavras_por_cluster)
    uso_total["media_tokens_entrada_por_chamada"] = (
        uso_total["tokens_entrada"] / max(uso_total["chamadas"], 1)
    )
    uso_total["media_tokens_saida_por_chamada"] = (
        uso_total["tokens_saida"] / max(uso_total["chamadas"], 1)
    )
    with open("outputs/uso_tokens_real.json", "w", encoding="utf-8") as f:
        json.dump(uso_total, f, ensure_ascii=False, indent=2)

    print(f"\nUso real de tokens (base para estimativa de custo): {uso_total}")
    print("Rode src/estimar_custos.py para projetar o custo para 1000 elementos.")
