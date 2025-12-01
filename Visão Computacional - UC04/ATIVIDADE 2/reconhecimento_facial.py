"""
Script complementar para reconhecimento facial
Atividade 2 - Visão Computacional UC04

Este script pode ser executado independentemente do notebook
para realizar análise facial com DeepFace
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class AnalisadorFacial:
    """
    Classe para análise facial usando DeepFace
    """
    
    def __init__(self):
        """
        Inicializa o analisador facial
        """
        self.traducao_emocoes = {
            'angry': 'Raiva',
            'disgust': 'Nojo',
            'fear': 'Medo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Surpresa',
            'neutral': 'Neutro'
        }
        
        self.traducao_generos = {
            'Man': 'Homem',
            'Woman': 'Mulher'
        }
    
    def carregar_imagem(self, caminho):
        """
        Carrega imagem e converte BGR para RGB
        """
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
        
        img_bgr = cv2.imread(caminho)
        if img_bgr is None:
            raise Exception(f"Erro ao carregar imagem: {caminho}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, img_bgr
    
    def redimensionar_imagem(self, imagem, largura_max=800):
        """
        Redimensiona imagem mantendo proporção
        """
        altura, largura = imagem.shape[:2]
        if largura > largura_max:
            proporcao = largura_max / largura
            nova_largura = largura_max
            nova_altura = int(altura * proporcao)
            return cv2.resize(imagem, (nova_largura, nova_altura))
        return imagem
    
    def criar_imagem_exemplo(self):
        """
        Cria uma imagem de exemplo com rosto simples
        """
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        
        # Desenha rosto básico
        cv2.circle(img, (200, 200), 80, (220, 180, 140), -1)
        cv2.circle(img, (180, 180), 8, (0, 0, 0), -1)
        cv2.circle(img, (220, 180), 8, (0, 0, 0), -1)
        cv2.ellipse(img, (200, 220), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        
        return img
    
    def analisar_face(self, caminho_imagem):
        """
        Realiza análise facial completa
        """
        try:
            print("Iniciando análise facial...")
            print("Primeira execução pode demorar (download de modelos)...")
            
            resultados = DeepFace.analyze(
                img_path=caminho_imagem,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            
            if not isinstance(resultados, list):
                resultados = [resultados]
            
            return resultados
            
        except Exception as e:
            print(f"Erro na análise: {str(e)}")
            try:
                # Tentativa alternativa
                resultados = DeepFace.analyze(
                    img_path=caminho_imagem,
                    actions=['emotion'],
                    enforce_detection=False
                )
                if not isinstance(resultados, list):
                    resultados = [resultados]
                return resultados
            except:
                return []
    
    def desenhar_deteccoes(self, img_rgb, resultados):
        """
        Desenha retângulos e informações nos rostos detectados
        """
        img_resultado = img_rgb.copy()
        
        for i, resultado in enumerate(resultados):
            try:
                regiao = resultado['region']
                x, y, w, h = regiao['x'], regiao['y'], regiao['w'], regiao['h']
                
                # Desenha retângulo
                cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 3)
                
                # Prepara informações
                idade = int(resultado.get('age', 0))
                genero = self.traducao_generos.get(resultado.get('dominant_gender', ''), 'N/A')
                emocao = self.traducao_emocoes.get(resultado.get('dominant_emotion', ''), 'N/A')
                raca = resultado.get('dominant_race', 'N/A')
                
                # Lista de textos
                textos = [
                    f"Idade: {idade}",
                    f"Genero: {genero}",
                    f"Emocao: {emocao}",
                    f"Raca: {raca}"
                ]
                
                # Desenha textos com fundo
                for j, texto in enumerate(textos):
                    y_texto = y - 10 - (len(textos) - j - 1) * 25
                    if y_texto < 25:
                        y_texto = y + h + 25 + j * 25
                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Fundo branco
                    cv2.rectangle(img_resultado, (x, y_texto - text_height - 5),
                                (x + text_width + 10, y_texto + 5), (255, 255, 255), -1)
                    
                    # Texto preto
                    cv2.putText(img_resultado, texto, (x + 5, y_texto),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            except KeyError:
                # Se faltar alguma informação, desenha apenas rótulo básico
                cv2.putText(img_resultado, f"Rosto {i+1}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return img_resultado
    
    def exibir_resultados(self, img_original, img_resultado, resultados):
        """
        Exibe comparação entre imagem original e com análises
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(img_original)
        axes[0].set_title('Imagem Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_resultado)
        title = f'Análise Facial - {len(resultados)} rosto(s) detectado(s)'
        axes[1].set_title(title, fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def salvar_resultado(self, img_resultado, nome_arquivo='resultado_analise.jpg'):
        """
        Salva imagem resultado
        """
        img_bgr = cv2.cvtColor(img_resultado, cv2.COLOR_RGB2BGR)
        cv2.imwrite(nome_arquivo, img_bgr)
        print(f"Resultado salvo como '{nome_arquivo}'")
    
    def imprimir_detalhes(self, resultados):
        """
        Imprime detalhes da análise
        """
        print(f"\n=== RESULTADOS DA ANÁLISE ===")
        print(f"Número de rostos detectados: {len(resultados)}")
        
        for i, resultado in enumerate(resultados):
            print(f"\n--- Rosto {i+1} ---")
            
            if 'age' in resultado:
                print(f"Idade estimada: {int(resultado['age'])} anos")
            
            if 'dominant_gender' in resultado:
                genero = self.traducao_generos.get(resultado['dominant_gender'], resultado['dominant_gender'])
                confianca = resultado.get('gender', {}).get(resultado['dominant_gender'], 0)
                print(f"Gênero: {genero} (confiança: {confianca:.1f}%)")
            
            if 'dominant_emotion' in resultado:
                emocao = self.traducao_emocoes.get(resultado['dominant_emotion'], resultado['dominant_emotion'])
                confianca = resultado.get('emotion', {}).get(resultado['dominant_emotion'], 0)
                print(f"Emoção: {emocao} (confiança: {confianca:.1f}%)")
            
            if 'dominant_race' in resultado:
                raca = resultado['dominant_race']
                confianca = resultado.get('race', {}).get(raca, 0)
                print(f"Raça/Etnia: {raca} (confiança: {confianca:.1f}%)")
            
            if 'region' in resultado:
                regiao = resultado['region']
                print(f"Posição: x={regiao['x']}, y={regiao['y']}, tamanho={regiao['w']}x{regiao['h']}")

def processar_imagem_webcam():
    """
    Processa imagem da webcam em tempo real
    """
    analisador = AnalisadorFacial()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Webcam não disponível")
        return
    
    print("Pressione 'q' para sair, 's' para salvar análise")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Análise rápida (apenas emoção para performance)
            resultados = DeepFace.analyze(
                img_path=frame_rgb,
                actions=['emotion'],
                enforce_detection=False
            )
            
            if not isinstance(resultados, list):
                resultados = [resultados]
            
            # Desenha resultados
            frame_resultado = analisador.desenhar_deteccoes(frame_rgb, resultados)
            frame_resultado_bgr = cv2.cvtColor(frame_resultado, cv2.COLOR_RGB2BGR)
            
        except:
            frame_resultado_bgr = frame
        
        cv2.imshow('Análise Facial - Webcam', frame_resultado_bgr)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('captura_analise.jpg', frame_resultado_bgr)
            print("Captura salva!")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Função principal
    """
    print("=== Reconhecimento Facial com DeepFace ===\n")
    
    analisador = AnalisadorFacial()
    
    # Procura imagens na pasta
    pasta_imagens = "imagens"
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp']
    
    imagens_encontradas = []
    if os.path.exists(pasta_imagens):
        for ext in extensoes:
            imagens_encontradas.extend(Path(pasta_imagens).glob(f'*{ext}'))
            imagens_encontradas.extend(Path(pasta_imagens).glob(f'*{ext.upper()}'))
    
    if imagens_encontradas:
        caminho_imagem = str(imagens_encontradas[0])
        print(f"Processando: {caminho_imagem}")
        
        # Carrega imagem
        img_rgb, img_bgr = analisador.carregar_imagem(caminho_imagem)
        img_rgb = analisador.redimensionar_imagem(img_rgb)
        
    else:
        print("Nenhuma imagem encontrada. Criando exemplo...")
        
        # Cria imagem de exemplo
        img_rgb = analisador.criar_imagem_exemplo()
        
        # Salva exemplo
        os.makedirs(pasta_imagens, exist_ok=True)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        caminho_imagem = os.path.join(pasta_imagens, 'exemplo.jpg')
        cv2.imwrite(caminho_imagem, img_bgr)
    
    # Realiza análise
    resultados = analisador.analisar_face(caminho_imagem)
    
    if resultados:
        # Desenha detecções
        img_resultado = analisador.desenhar_deteccoes(img_rgb, resultados)
        
        # Exibe resultados
        analisador.exibir_resultados(img_rgb, img_resultado, resultados)
        
        # Salva resultado
        analisador.salvar_resultado(img_resultado)
        
        # Imprime detalhes
        analisador.imprimir_detalhes(resultados)
        
    else:
        print("Nenhum rosto detectado ou erro na análise.")
    
    # Opção webcam
    opcao = input("\nTestar com webcam? (s/n): ").lower().strip()
    if opcao == 's':
        processar_imagem_webcam()
    
    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()