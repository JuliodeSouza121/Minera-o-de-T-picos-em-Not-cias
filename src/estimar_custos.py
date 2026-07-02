"""
Estimativa de custo para processar 1.000 elementos (seção 7 do enunciado),
usando os preços oficiais publicados em https://ai.google.dev/gemini-api/docs/pricing.

Importante: o enunciado pede a estimativa para gemini-2.5-flash,
gemini-3-flash-preview e gemini-3.1-pro-preview MESMO que outro modelo tenha
sido usado na implementação -- por isso este módulo é parametrizado por preço,
não está acoplado ao cliente real usado em llm_labeling.py.

Preços por 1M de tokens, verificados em ai.google.dev (consultar a página
oficial antes da entrega final, pois preços de modelos preview mudam com
frequência):
"""

PRECOS_POR_MILHAO_TOKENS = {
    # (entrada_usd, saida_usd) -- confirmar valores de preview antes de entregar,
    # pois mudam com frequência; os de gemini-2.5-flash são estáveis (GA).
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

N_ELEMENTOS_PROJETADO = 1000


def estimar_custo(
    tokens_entrada_por_chamada: float,
    tokens_saida_por_chamada: float,
    n_elementos: int = N_ELEMENTOS_PROJETADO,
) -> dict:
    """
    Projeta o custo total para processar n_elementos, assumindo uma chamada
    ao LLM por elemento (ajustar se o seu pipeline agrupar vários documentos
    por chamada, como ao rotular um cluster inteiro de uma vez).
    """
    resultados = {}
    for modelo, (preco_entrada, preco_saida) in PRECOS_POR_MILHAO_TOKENS.items():
        total_tokens_entrada = tokens_entrada_por_chamada * n_elementos
        total_tokens_saida = tokens_saida_por_chamada * n_elementos

        custo_entrada = (total_tokens_entrada / 1_000_000) * preco_entrada
        custo_saida = (total_tokens_saida / 1_000_000) * preco_saida
        custo_total = custo_entrada + custo_saida

        resultados[modelo] = {
            "custo_entrada_usd": round(custo_entrada, 4),
            "custo_saida_usd": round(custo_saida, 4),
            "custo_total_usd": round(custo_total, 4),
        }
    return resultados


if __name__ == "__main__":
    import json

    # Carrega o uso REAL medido em llm_labeling.py (rode esse script antes)
    try:
        with open("outputs/uso_tokens_real.json", "r", encoding="utf-8") as f:
            uso = json.load(f)
        media_entrada = uso["media_tokens_entrada_por_chamada"]
        media_saida = uso["media_tokens_saida_por_chamada"]
        print(
            f"Usando médias REAIS medidas: "
            f"{media_entrada:.0f} tokens entrada, {media_saida:.0f} tokens saída por chamada"
        )
    except FileNotFoundError:
        # valores de fallback só para teste, ANTES de rodar o pipeline de
        # verdade -- substituir pelos valores reais assim que disponíveis
        media_entrada, media_saida = 600, 150
        print(
            "[AVISO] outputs/uso_tokens_real.json não encontrado -- usando "
            f"valores de exemplo ({media_entrada} entrada / {media_saida} saída). "
            "Rode src/llm_labeling.py primeiro para ter números reais."
        )

    resultado = estimar_custo(media_entrada, media_saida)

    print(f"\nProjeção de custo para {N_ELEMENTOS_PROJETADO} elementos:\n")
    for modelo, custos in resultado.items():
        print(f"  {modelo}:")
        print(f"    entrada: US$ {custos['custo_entrada_usd']}")
        print(f"    saída:   US$ {custos['custo_saida_usd']}")
        print(f"    TOTAL:   US$ {custos['custo_total_usd']}")
        print()

    with open("outputs/estimativa_custos.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "premissas": {
                    "tokens_entrada_por_chamada": media_entrada,
                    "tokens_saida_por_chamada": media_saida,
                    "n_elementos": N_ELEMENTOS_PROJETADO,
                    "fonte_precos": "https://ai.google.dev/gemini-api/docs/pricing",
                },
                "custos_por_modelo": resultado,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Salvo em outputs/estimativa_custos.json")
