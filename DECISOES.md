# Diário de Decisões (DECISOES.md)

Registro de escolhas de implementação e o porquê. Atualizar conforme o trabalho avança.

## Corpus

- Escolhido `iara-project/news-articles-ptbr-dataset` em vez de consumidor.gov.br
  porque o texto livre de reclamações ("Relato do Consumidor") não está disponível
  em CSV de bulk fácil — exigiria scraping/busca manual no portal, incompatível
  com o prazo. O dataset de notícias já vem limpo, em parquet, com categoria real
  para validação cruzada da qualidade dos clusters.

## Amostragem

- Tamanho da amostra escolhido: 1.000 documentos, período: 2020.
  Motivo: volume suficiente sem custo computacional alto de embeddings.

## Pré-processamento

Neste trabalho foi adotado um pré-processamento leve, preservando ao máximo o texto original, pois o modelo de embeddings (sentence-transformers) foi treinado em linguagem natural e tende a produzir melhores representações sem alterações linguísticas agressivas.

As etapas aplicadas foram:

- Concatenação do título e do corpo da notícia: o campo title foi concatenado ao campo text, colocando o título primeiro. Essa estratégia aumenta a probabilidade de que o tema principal da notícia esteja presente no início do texto, reduzindo o impacto do truncamento em modelos com limite de contexto.
- Remoção de URLs: links iniciados por http ou www foram removidos, pois não agregam significado semântico relevante para a geração dos embeddings.
Remoção de tags HTML: quaisquer marcações HTML residuais foram eliminadas para manter apenas o conteúdo textual.
- Normalização dos espaços em branco: múltiplos espaços, quebras de linha e tabulações foram substituídos por um único espaço, deixando o texto consistente sem alterar seu conteúdo.
- Tratamento de valores ausentes: campos vazios (NaN) foram substituídos por strings vazias antes da concatenação, evitando erros durante o processamento.
- Remoção de documentos muito curtos: notícias com menos de 50 caracteres após o pré-processamento foram descartadas, pois normalmente não contêm informação suficiente para gerar embeddings representativos.

As seguintes técnicas não foram utilizadas:

- Conversão para letras minúsculas (lowercase);
- Remoção de pontuação;
- Remoção de stopwords;
- Lematização ou stemming;
- Tokenização manual.

Essas etapas são comuns em abordagens tradicionais baseadas em frequência de palavras, como TF-IDF, pois ajudam a reduzir a dimensionalidade e a padronizar o vocabulário. Entretanto, em modelos modernos de sentence-transformers, esse tipo de pré-processamento costuma ser desnecessário ou até prejudicial, já que o modelo foi treinado para compreender textos em sua forma natural.

## Modelo de embedding

Foi utilizado o modelo *paraphrase-multilingual-mpnet-base-v2*, da biblioteca Sentence-Transformers.

A escolha desse modelo foi motivada pelos seguintes fatores:

- Suporte ao português: o modelo é multilíngue e foi treinado para compreender diversos idiomas, incluindo o português, produzindo representações semânticas de boa qualidade sem necessidade de tradução dos textos.
- Otimizado para similaridade semântica: diferentemente de utilizar um modelo BERT "puro", esse modelo foi ajustado (fine-tuned) especificamente para tarefas de Semantic Textual Similarity (STS), gerando embeddings mais adequados para medir similaridade entre documentos e para técnicas de agrupamento como BERTopic.
- Boa relação entre qualidade e custo computacional: embora seja um modelo relativamente robusto, seu tempo de processamento e consumo de memória permanecem viáveis para um corpus com aproximadamente 1.500 documentos, oferecendo um excelente equilíbrio entre desempenho e eficiência.
- Dimensão dos vetores: cada documento é representado por um embedding de 768 dimensões, suficiente para capturar informações semânticas complexas sem gerar um custo computacional excessivo.

## Medida de similaridade

Foi utilizada a similaridade do cosseno (Cosine Similarity) para comparar os embeddings dos documentos.

Essa escolha é adequada porque os embeddings gerados pelo Sentence-Transformers são vetores densos que representam o significado semântico dos textos. Nesse contexto, a similaridade do cosseno mede o ângulo entre os vetores, indicando o quão semelhantes semanticamente dois documentos são, independentemente de sua magnitude.

Além disso, os embeddings foram gerados com o parâmetro *normalize_embeddings=True*, que realiza a normalização L2 dos vetores. Com essa normalização, a similaridade do cosseno torna-se uma métrica ainda mais apropriada, pois a comparação passa a depender exclusivamente da direção dos vetores, refletindo melhor a proximidade semântica entre os documentos.

Essa métrica é amplamente utilizada em tarefas de recuperação de informação, busca semântica, agrupamento de documentos e modelagem de tópicos baseada em embeddings, apresentando desempenho superior a medidas tradicionais de distância em espaços de alta dimensionalidade.

## Algoritmos de clusterização comparados

Foram comparados os algoritmos K-Means e HDBSCAN.

No K-Means, foram testados valores de K entre 4 e 14, sendo escolhido automaticamente o que apresentou o maior Silhouette Score utilizando a similaridade do cosseno. Foram utilizados random_state=42 e n_init=10 para garantir reprodutibilidade e maior estabilidade dos resultados.

No HDBSCAN, os embeddings foram previamente reduzidos para 15 dimensões com UMAP (metric="cosine"), diminuindo os efeitos da alta dimensionalidade. Em seguida, foi utilizado min_cluster_size=25 e metric="euclidean" para identificar automaticamente clusters e documentos considerados ruído.

A escolha desses algoritmos permite comparar uma abordagem baseada em centroides (K-Means), que atribui todos os documentos a um grupo, com uma abordagem baseada em densidade (HDBSCAN), que também é capaz de identificar outliers e descobrir automaticamente o número de clusters.

## Redução de dimensionalidade

Antes da execução do HDBSCAN, os embeddings foram reduzidos para 15 dimensões utilizando UMAP. Essa redução foi realizada para minimizar os efeitos da alta dimensionalidade dos embeddings (768 dimensões), melhorando o desempenho do algoritmo de clusterização. Optou-se por 15 dimensões, em vez de 2D, pois a projeção bidimensional é indicada apenas para visualização e pode distorcer a estrutura dos dados, prejudicando a qualidade dos clusters.

## Schema Pydantic

O schema de rotulagem foi composto pelos campos cluster_id, nome_topico, palavras_chave, resumo, nivel_confianca e possivel_ruido.

cluster_id (int): identifica o cluster rotulado.
nome_topico (str): fornece um nome curto e descritivo para facilitar a interpretação pelo usuário.
palavras_chave (list[str]): reúne os principais termos que justificam o rótulo atribuído ao cluster.
resumo (str): apresenta uma breve descrição do tema predominante no grupo.
nivel_confianca (float): representa a confiança do LLM (0 a 1) na coerência temática do cluster.
possivel_ruido (bool): indica se o cluster parece misturar temas diferentes, auxiliando na identificação de agrupamentos de baixa qualidade.

A utilização do Pydantic garante que todos os campos sejam validados quanto ao tipo e às restrições definidas, assegurando uma saída estruturada e consistente.

## Prompt do LLM

Você é um analista editorial ajudando a organizar notícias em temas.

Você receberá, para UM cluster de notícias:
- Uma lista de palavras-chave mais distintivas do cluster (extraídas via c-TF-IDF).
- De 3 a 5 trechos de exemplo (título + início do texto) de notícias representativas
  desse cluster (as mais próximas do centróide).

Sua tarefa é retornar um JSON com EXATAMENTE estes campos, sem nenhum texto antes
ou depois do JSON:

{{
  "cluster_id": <int, fornecido na entrada>,
  "nome_topico": "<nome curto, 3-6 palavras, em português>",
  "palavras_chave": ["<3 a 8 palavras que melhor resumem o cluster>"],
  "resumo": "<1 a 2 frases descrevendo o que une os documentos deste cluster>",
  "nivel_confianca": <float entre 0.0 e 1.0>,
  "possivel_ruido": <true ou false>
}}

Regras importantes:
- O nome do tópico deve ser específico (ex: "Reforma tributária no Congresso",
  não "Política" ou "Economia").
- NUNCA use "Geral", "Outros", "Diversos" ou termos igualmente vagos como nome
  do tópico.
- Se os exemplos de notícias parecerem tratar de assuntos muito diferentes entre
  si, defina possivel_ruido como true e justifique isso reduzindo o
  nivel_confianca (abaixo de 0.5).
- Baseie-se SOMENTE nas palavras-chave e nos exemplos fornecidos. Não invente
  fatos, datas ou nomes que não apareçam no material fornecido.
- A saída deve ser um JSON válido, sem comentários, sem markdown (` ``` `), sem
  texto explicativo adicional.

---
ENTRADA:

cluster_id: {cluster_id}

palavras_chave_ctfidf: {palavras_chave}

exemplos_representativos:
{exemplos}


## Cliente do LLM

- Usado o SDK `google-genai` (pacote atual recomendado pela Google), não o
  `google-generativeai` (legado). O `response_schema` aceita o modelo Pydantic
  diretamente, então a API já força a saída a bater com o schema antes de
  devolver -- reduz bastante a taxa de falha de validação, mas o
  `try/except ValidationError` foi mantido como rede de segurança (e fonte de
  dados para a análise de erro estruturada).
- Modelo usado na implementação: `gemini-2.5-flash` (rápido, barato, contexto
  de 1M tokens -- mais que suficiente para os prompts curtos de rotulagem de
  cluster). API key via variável de ambiente `GEMINI_API_KEY`, nunca
  hardcoded no código nem commitada no Git.
- Custos projetados para 1000 elementos calculados com preços oficiais de
  ai.google.dev (ver `outputs/estimativa_custos.json`), comparando
  gemini-2.5-flash, gemini-3-flash-preview e gemini-3.1-pro-preview conforme
  exigido na seção 7 do enunciado.



Durante os testes não foram observadas ocorrências frequentes de alucinação ou geração de JSON inválido, principalmente devido ao uso do response_schema da API Gemini juntamente com a validação via Pydantic.

Como medidas preventivas, foram adotados os seguintes ajustes:

- Rótulos genéricos: o prompt foi elaborado proibindo nomes vagos como "Geral", "Outros" e "Diversos", e o schema Pydantic também valida essa restrição.
- JSON inválido: a API foi configurada para retornar JSON estruturado (response_mime_type="application/json" e response_schema=RotuloCluster), reduzindo significativamente erros de formatação.
- Falhas de validação: foi implementado tratamento com try/except ValidationError, registrando automaticamente qualquer resposta incompatível com o schema em outputs/erros_validacao_llm.json.
- Erros da API: falhas temporárias (como rate limit ou erros 5xx) são tratadas por meio de até três tentativas com backoff exponencial.

Até os testes realizados, essas estratégias foram suficientes para garantir respostas consistentes e compatíveis com o formato esperado.
