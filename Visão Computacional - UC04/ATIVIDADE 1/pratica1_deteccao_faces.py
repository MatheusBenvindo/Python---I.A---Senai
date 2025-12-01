"""
Prática 1 - Detecção de Faces com OpenCV
Visão Computacional - UC04

Objetivos:
- Manipular imagens digitais como arrays
- Realizar operações básicas como conversão de cores
- Utilizar classificadores Haar Cascades para detectar faces
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class DetectorFaces:
    """
    Classe para detecção de faces usando Haar Cascades do OpenCV
    """
    
    def __init__(self):
        """
        Inicializa o detector com os classificadores Haar Cascade
        """
        # Carrega o classificador Haar Cascade para faces
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Carrega o classificador para olhos (opcional)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Verifica se os classificadores foram carregados corretamente
        if self.face_cascade.empty():
            raise Exception("Erro ao carregar o classificador de faces")
        if self.eye_cascade.empty():
            print("Aviso: Classificador de olhos não carregado")
    
    def carregar_imagem(self, caminho_imagem):
        """
        Carrega uma imagem do disco
        
        Args:
            caminho_imagem (str): Caminho para a imagem
            
        Returns:
            numpy.ndarray: Imagem carregada
        """
        if not os.path.exists(caminho_imagem):
            raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")
        
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            raise Exception(f"Erro ao carregar a imagem: {caminho_imagem}")
        
        return imagem
    
    def converter_para_cinza(self, imagem):
        """
        Converte imagem colorida para escala de cinza
        
        Args:
            imagem (numpy.ndarray): Imagem colorida (BGR)
            
        Returns:
            numpy.ndarray: Imagem em escala de cinza
        """
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    def detectar_faces(self, imagem, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detecta faces na imagem usando Haar Cascade
        
        Args:
            imagem (numpy.ndarray): Imagem de entrada
            scale_factor (float): Fator de escala para detecção multi-escala
            min_neighbors (int): Número mínimo de vizinhos para validar detecção
            min_size (tuple): Tamanho mínimo da face
            
        Returns:
            numpy.ndarray: Array com coordenadas das faces detectadas [(x, y, w, h), ...]
        """
        # Converte para escala de cinza se necessário
        if len(imagem.shape) == 3:
            imagem_cinza = self.converter_para_cinza(imagem)
        else:
            imagem_cinza = imagem
        
        # Detecta faces
        faces = self.face_cascade.detectMultiScale(
            imagem_cinza,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detectar_olhos(self, imagem_face):
        """
        Detecta olhos em uma região de face
        
        Args:
            imagem_face (numpy.ndarray): Região da face extraída
            
        Returns:
            numpy.ndarray: Array com coordenadas dos olhos detectados
        """
        if len(imagem_face.shape) == 3:
            face_cinza = self.converter_para_cinza(imagem_face)
        else:
            face_cinza = imagem_face
        
        olhos = self.eye_cascade.detectMultiScale(
            face_cinza,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        )
        
        return olhos
    
    def desenhar_deteccoes(self, imagem, faces, desenhar_olhos=True):
        """
        Desenha retângulos ao redor das faces e olhos detectados
        
        Args:
            imagem (numpy.ndarray): Imagem original
            faces (numpy.ndarray): Coordenadas das faces detectadas
            desenhar_olhos (bool): Se deve detectar e desenhar olhos
            
        Returns:
            numpy.ndarray: Imagem com detecções desenhadas
        """
        imagem_resultado = imagem.copy()
        
        for (x, y, w, h) in faces:
            # Desenha retângulo ao redor da face
            cv2.rectangle(imagem_resultado, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Adiciona texto
            cv2.putText(imagem_resultado, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Detecta olhos na região da face
            if desenhar_olhos and not self.eye_cascade.empty():
                roi_face = imagem[y:y + h, x:x + w]
                olhos = self.detectar_olhos(roi_face)
                
                for (ex, ey, ew, eh) in olhos:
                    # Ajusta coordenadas dos olhos para a imagem completa
                    cv2.rectangle(imagem_resultado, (x + ex, y + ey), 
                                (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        
        return imagem_resultado
    
    def mostrar_resultados(self, imagem_original, imagem_detectada, faces):
        """
        Mostra os resultados da detecção usando matplotlib
        
        Args:
            imagem_original (numpy.ndarray): Imagem original
            imagem_detectada (numpy.ndarray): Imagem com detecções
            faces (numpy.ndarray): Array com faces detectadas
        """
        # Converte BGR para RGB para matplotlib
        img_original_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
        img_detectada_rgb = cv2.cvtColor(imagem_detectada, cv2.COLOR_BGR2RGB)
        
        # Cria subplot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Imagem original
        axes[0].imshow(img_original_rgb)
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')
        
        # Imagem com detecções
        axes[1].imshow(img_detectada_rgb)
        axes[1].set_title(f'Faces Detectadas: {len(faces)}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Informações sobre as detecções
        print(f"\nResultados da Detecção:")
        print(f"Número de faces detectadas: {len(faces)}")
        
        for i, (x, y, w, h) in enumerate(faces):
            print(f"Face {i+1}: posição=({x}, {y}), tamanho=({w}x{h})")

def criar_imagem_exemplo():
    """
    Cria uma imagem de exemplo com formas geométricas para teste básico
    """
    # Cria uma imagem em branco
    imagem = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Adiciona fundo
    imagem[:] = (240, 240, 240)  # Cinza claro
    
    # Desenha algumas formas para simular uma face simples
    cv2.circle(imagem, (300, 200), 80, (200, 180, 160), -1)  # Face
    cv2.circle(imagem, (280, 180), 8, (0, 0, 0), -1)  # Olho esquerdo
    cv2.circle(imagem, (320, 180), 8, (0, 0, 0), -1)  # Olho direito
    cv2.ellipse(imagem, (300, 220), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Boca
    
    return imagem

def processar_imagem_webcam():
    """
    Captura e processa imagem da webcam em tempo real
    """
    detector = DetectorFaces()
    
    # Tenta abrir a webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam")
        return
    
    print("Pressione 'q' para sair, 's' para salvar uma captura")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detecta faces
        faces = detector.detectar_faces(frame)
        
        # Desenha detecções
        frame_com_deteccao = detector.desenhar_deteccoes(frame, faces)
        
        # Mostra o resultado
        cv2.imshow('Detecção de Faces - Webcam', frame_com_deteccao)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Salva a captura
            cv2.imwrite('captura_webcam.jpg', frame_com_deteccao)
            print("Captura salva como 'captura_webcam.jpg'")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Função principal - demonstra o uso do detector de faces
    """
    print("=== Prática 1: Detecção de Faces com OpenCV ===\n")
    
    # Cria instância do detector
    detector = DetectorFaces()
    
    # Verifica se existe alguma imagem na pasta imagens
    pasta_imagens = Path("imagens")
    
    if pasta_imagens.exists():
        imagens_encontradas = list(pasta_imagens.glob("*.jpg")) + list(pasta_imagens.glob("*.png"))
        
        if imagens_encontradas:
            # Processa a primeira imagem encontrada
            caminho_imagem = str(imagens_encontradas[0])
            print(f"Processando imagem: {caminho_imagem}")
            
            try:
                # Carrega imagem
                imagem = detector.carregar_imagem(caminho_imagem)
                
                # Detecta faces
                faces = detector.detectar_faces(imagem)
                
                # Desenha detecções
                imagem_com_deteccao = detector.desenhar_deteccoes(imagem, faces)
                
                # Mostra resultados
                detector.mostrar_resultados(imagem, imagem_com_deteccao, faces)
                
                # Salva resultado
                cv2.imwrite("resultado_deteccao.jpg", imagem_com_deteccao)
                print("Resultado salvo como 'resultado_deteccao.jpg'")
                
            except Exception as e:
                print(f"Erro ao processar imagem: {e}")
        else:
            print("Nenhuma imagem encontrada na pasta 'imagens'")
    
    # Se não há imagens, cria uma de exemplo
    if not pasta_imagens.exists() or not list(pasta_imagens.glob("*.jpg")):
        print("\nCriando imagem de exemplo...")
        
        # Cria e salva imagem de exemplo
        imagem_exemplo = criar_imagem_exemplo()
        os.makedirs("imagens", exist_ok=True)
        cv2.imwrite("imagens/exemplo.jpg", imagem_exemplo)
        
        # Processa a imagem de exemplo
        faces = detector.detectar_faces(imagem_exemplo)
        imagem_com_deteccao = detector.desenhar_deteccoes(imagem_exemplo, faces)
        
        # Mostra resultados
        detector.mostrar_resultados(imagem_exemplo, imagem_com_deteccao, faces)
    
    # Opção de usar webcam
    resposta = input("\nDeseja testar com a webcam? (s/n): ").lower().strip()
    if resposta == 's':
        processar_imagem_webcam()
    
    print("\n=== Prática concluída! ===")

if __name__ == "__main__":
    main()