"""
Script para criar imagens de exemplo para teste
"""

import cv2
import numpy as np
import os

def criar_imagem_rosto_exemplo():
    """
    Cria uma imagem com rosto mais realista para teste
    """
    # Cria imagem base
    img = np.ones((500, 500, 3), dtype=np.uint8) * 240
    
    # Face principal (oval)
    cv2.ellipse(img, (250, 250), (100, 130), 0, 0, 360, (220, 180, 140), -1)
    
    # Olhos
    cv2.circle(img, (220, 220), 12, (255, 255, 255), -1)  # Olho esquerdo - branco
    cv2.circle(img, (280, 220), 12, (255, 255, 255), -1)  # Olho direito - branco
    cv2.circle(img, (220, 220), 8, (0, 0, 0), -1)  # Pupila esquerda
    cv2.circle(img, (280, 220), 8, (0, 0, 0), -1)  # Pupila direita
    
    # Sobrancelhas
    cv2.ellipse(img, (220, 200), (15, 8), 0, 0, 180, (139, 69, 19), 3)
    cv2.ellipse(img, (280, 200), (15, 8), 0, 0, 180, (139, 69, 19), 3)
    
    # Nariz
    pts = np.array([[250, 240], [245, 270], [255, 270]], np.int32)
    cv2.fillPoly(img, [pts], (200, 160, 120))
    
    # Boca sorrindo
    cv2.ellipse(img, (250, 290), (25, 15), 0, 0, 180, (0, 0, 0), 3)
    
    # Cabelo
    cv2.ellipse(img, (250, 180), (110, 80), 0, 0, 180, (139, 69, 19), -1)
    
    return img

def criar_multiplas_faces():
    """
    Cria imagem com múltiplas faces para teste
    """
    img = np.ones((400, 600, 3), dtype=np.uint8) * 250
    
    # Face 1
    cv2.circle(img, (150, 200), 60, (220, 180, 140), -1)
    cv2.circle(img, (135, 185), 6, (0, 0, 0), -1)
    cv2.circle(img, (165, 185), 6, (0, 0, 0), -1)
    cv2.ellipse(img, (150, 210), (15, 8), 0, 0, 180, (0, 0, 0), 2)
    
    # Face 2
    cv2.circle(img, (450, 200), 60, (200, 160, 120), -1)
    cv2.circle(img, (435, 185), 6, (0, 0, 0), -1)
    cv2.circle(img, (465, 185), 6, (0, 0, 0), -1)
    cv2.ellipse(img, (450, 210), (15, 8), 0, 0, 180, (0, 0, 0), 2)
    
    return img

# Cria pasta de imagens se não existir
pasta_imagens = "imagens"
os.makedirs(pasta_imagens, exist_ok=True)

# Cria e salva imagens de exemplo
print("Criando imagens de exemplo...")

# Imagem 1: Rosto único mais detalhado
img1 = criar_imagem_rosto_exemplo()
cv2.imwrite(os.path.join(pasta_imagens, "rosto_exemplo.jpg"), img1)

# Imagem 2: Múltiplas faces
img2 = criar_multiplas_faces()
cv2.imwrite(os.path.join(pasta_imagens, "multiplas_faces.jpg"), img2)

print("Imagens criadas na pasta 'imagens/':")
print("- rosto_exemplo.jpg")
print("- multiplas_faces.jpg")