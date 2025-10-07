destinos = [
    {
        "nome": "Rio de Janeiro",
        "clima": "quente",
        "ambiente": "natureza",
        "preco": 3000,
    },
    {"nome": "Gramado", "clima": "frio", "ambiente": "urbano", "preco": 2500},
    {
        "nome": "Fernando de Noronha",
        "clima": "quente",
        "ambiente": "natureza",
        "preco": 5000,
    },
    {"nome": "São Paulo", "clima": "quente", "ambiente": "urbano", "preco": 2000},
    {"nome": "Bariloche", "clima": "frio", "ambiente": "natureza", "preco": 4000},
    {"nome": "Nova York", "clima": "frio", "ambiente": "urbano", "preco": 6000},
]


def obter_preferencia(pergunta, opcoes_validas):
    while True:
        resposta = input(pergunta).strip().lower()
        if resposta in opcoes_validas:
            return resposta
        else:
            print(
                f"Resposta inválida. Por favor, escolha entre: {', '.join(opcoes_validas)}."
            )


def recomendar_destino(clima, ambiente, orcamento):
    for destino in destinos:
        if (
            destino["clima"] == clima
            and destino["ambiente"] == ambiente
            and destino["preco"] <= orcamento
        ):
            return destino
    return None


def sistema_recomendacao():
    print("🌍 Bem-vindo ao sistema de recomendação de viagens!")

    clima = obter_preferencia(
        "Qual o clima de sua preferência, quente ou frio? ", ["quente", "frio"]
    )
    ambiente = obter_preferencia(
        "Você pretende ir para um lugar com natureza ou paisagens urbanas? ",
        ["natureza", "urbano"],
    )

    while True:
        try:
            orcamento = int(
                input("Qual é o seu orçamento disponível para a viagem? (em reais) ")
            )
            break
        except ValueError:
            print("Por favor, insira um valor numérico válido.")

    destino = recomendar_destino(clima, ambiente, orcamento)

    if destino:
        print(f"\n🎯 Destino recomendado: {destino['nome']}")
        print(
            f"Justificativa: Combina com seu gosto por clima {clima}, ambiente de {ambiente} e está dentro do seu orçamento."
        )
    else:
        print(
            "\n😕 Nenhum destino encontrado com essas preferências. Tente ajustar seu orçamento ou preferências."
        )


sistema_recomendacao()
