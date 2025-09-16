import numpy as np


class MonitorDeTemperatura:
    "Classe para monitoramento de temperaturas"

    analiseEstatistica = 0

    def __init__(self):
        self.listaTemperaturas = []

    def AdicionarTemperatura(self, valorTemp):
        self.listaTemperaturas.append(valorTemp)
        print("Temperatura adicionada com sucesso!")

    def CalcularEstatisticaTemp(self):
        dados = np.array(self.listaTemperaturas)

        analise_estatistica = {
            "maxima": dados.max(),
            "minima": dados.min(),
            "media": dados.mean(),
        }

        return analise_estatistica

    def VerificarTemp(self):
        # condicional que verifica as temperatures entre 20 e 80 graus
        foraLimite = False
        for temp in self.listaTemperaturas:
            if temp < 20 or temp > 80:
                print(f"Alerta de temperatura {temp}°C")
                foraLimite = True
                break
        if not foraLimite:
            print("Todas as temperaturas estão dentro do necessário.")


monitor = MonitorDeTemperatura()
print("Bem-vindo ao sistema de monitoramento de temperaturas!")

valoresTemp = int(input("Quantas temperaturas você deseja adicionar? "))

for temp in range(valoresTemp):
    valorTempInformado = float(input(f"Digite o valor da {temp + 1}° temperatura : "))
    monitor.AdicionarTemperatura(valorTempInformado)

estatistica = monitor.CalcularEstatisticaTemp()

print("\nAnálise Estatística das Temperaturas:")
print(f"Temperatura Máxima: {estatistica['maxima']}°C")
print(f"Temperatura Mínima: {estatistica['minima']}°C")
print(f"Temperatura Média: {estatistica['media']:.2f}°C")
