"""
Schemas Pydantic para a saída estruturada do LLM.

Por que esses campos (registrar justificativa também em DECISOES.md):

- nome_topico: rótulo curto e legível, é o que o gestor vê primeiro.
- palavras_chave: lista curta, ancora o rótulo nas evidências (palavras que o
  c-TF-IDF já indicou como mais distintivas do cluster) -- evita que o LLM
  invente um tema desconectado do conteúdo real.
- resumo: 1-2 frases, para o gestor entender o cluster sem ler os documentos.
- nivel_confianca: float 0-1, autoavaliação do LLM sobre a coerência do cluster
  (clusters de ruído / muito heterogêneos devem receber confiança baixa).
- possivel_ruido: bool, sinaliza explicitamente se o LLM acha que o cluster
  parece misturar temas diferentes -- dado direto para a análise de erro.
"""

from pydantic import BaseModel, Field, field_validator


class RotuloCluster(BaseModel):
    """Saída estruturada esperada do LLM para um único cluster."""

    cluster_id: int = Field(..., description="ID do cluster sendo rotulado")
    nome_topico: str = Field(
        ...,
        min_length=3,
        max_length=60,
        description="Nome curto e descritivo do tópico, em português",
    )
    palavras_chave: list[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Palavras-chave que justificam o rótulo, extraídas do conteúdo",
    )
    resumo: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="Resumo de 1-2 frases do que caracteriza este cluster",
    )
    nivel_confianca: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiança do modelo de que o cluster é tematicamente coeso",
    )
    possivel_ruido: bool = Field(
        default=False,
        description="True se o cluster parece misturar temas muito diferentes",
    )

    @field_validator("nome_topico")
    @classmethod
    def nome_nao_generico(cls, v: str) -> str:
        genericos = {"geral", "outros", "diversos", "variado", "vários"}
        if v.strip().lower() in genericos:
            raise ValueError(
                f"Nome de tópico genérico demais: '{v}'. "
                "Seja mais específico com base nas palavras-chave."
            )
        return v.strip()


class LoteDeRotulos(BaseModel):
    """Caso se opte por rotular vários clusters em uma única chamada ao LLM."""

    rotulos: list[RotuloCluster]


# --- Exemplo de uso com tratamento de falha de validação ---
#
# from pydantic import ValidationError
#
# def parse_resposta_llm(texto_json: str) -> RotuloCluster | None:
#     try:
#         return RotuloCluster.model_validate_json(texto_json)
#     except ValidationError as e:
#         # Registrar a falha para a análise de erro estruturada do trabalho
#         # (categoria: "falha de validação Pydantic")
#         print(f"[ERRO DE VALIDACAO] {e}")
#         return None
