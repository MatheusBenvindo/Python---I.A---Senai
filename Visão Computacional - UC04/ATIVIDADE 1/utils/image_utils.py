"""
Utilitários para processamento de imagens
Visão Computacional - UC04
"""

import cv2
import numpy as np
import os
from pathlib import Path

def listar_imagens(pasta="imagens", extensoes=[".jpg", ".jpeg", ".png", ".bmp"]):
    """
    Lista todas as imagens em uma pasta
    
    Args:
        pasta (str): Caminho da pasta
        extensoes (list): Lista de extensões aceitas
        
    Returns:
        list: Lista de caminhos para as imagens
    """
    pasta_path = Path(pasta)
    if not pasta_path.exists():
        return []
    
    imagens = []
    for ext in extensoes:
        imagens.extend(pasta_path.glob(f"*{ext}"))
        imagens.extend(pasta_path.glob(f"*{ext.upper()}"))
    
    return [str(img) for img in imagens]

def redimensionar_imagem(imagem, largura_max=800, altura_max=600):
    """
    Redimensiona imagem mantendo proporção
    
    Args:
        imagem (numpy.ndarray): Imagem a ser redimensionada
        largura_max (int): Largura máxima
        altura_max (int): Altura máxima
        
    Returns:
        numpy.ndarray: Imagem redimensionada
    """
    altura, largura = imagem.shape[:2]
    
    # Calcula proporção
    proporcao = min(largura_max/largura, altura_max/altura)
    
    if proporcao < 1:
        nova_largura = int(largura * proporcao)
        nova_altura = int(altura * proporcao)
        return cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    
    return imagem

def melhorar_contraste(imagem, alpha=1.5, beta=0):
    """
    Melhora o contraste da imagem
    
    Args:
        imagem (numpy.ndarray): Imagem de entrada
        alpha (float): Controle de contraste (1.0-3.0)
        beta (int): Controle de brilho (0-100)
        
    Returns:
        numpy.ndarray: Imagem com contraste melhorado
    """
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

def aplicar_filtro_gaussiano(imagem, kernel_size=5):
    """
    Aplica filtro gaussiano para suavizar a imagem
    
    Args:
        imagem (numpy.ndarray): Imagem de entrada
        kernel_size (int): Tamanho do kernel (deve ser ímpar)
        
    Returns:
        numpy.ndarray: Imagem suavizada
    """
    return cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 0)

def equalizar_histograma(imagem):
    """
    Equaliza o histograma para melhorar o contraste
    
    Args:
        imagem (numpy.ndarray): Imagem em escala de cinza
        
    Returns:
        numpy.ndarray: Imagem com histograma equalizado
    """
    if len(imagem.shape) == 3:
        # Se for colorida, converte para escala de cinza
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    return cv2.equalizeHist(imagem)

def salvar_imagem_com_timestamp(imagem, prefixo="captura", pasta="capturas"):
    """
    Salva imagem com timestamp no nome
    
    Args:
        imagem (numpy.ndarray): Imagem a ser salva
        prefixo (str): Prefixo do nome do arquivo
        pasta (str): Pasta onde salvar
        
    Returns:
        str: Caminho do arquivo salvo
    """
    from datetime import datetime
    
    # Cria pasta se não existir
    os.makedirs(pasta, exist_ok=True)
    
    # Gera nome com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{prefixo}_{timestamp}.jpg"
    caminho_completo = os.path.join(pasta, nome_arquivo)
    
    # Salva imagem
    cv2.imwrite(caminho_completo, imagem)
    
    return caminho_completo

def criar_grade_imagens(imagens, titulos=None, cols=2):
    """
    Cria uma grade de imagens para comparação
    
    Args:
        imagens (list): Lista de imagens
        titulos (list): Lista de títulos (opcional)
        cols (int): Número de colunas
        
    Returns:
        numpy.ndarray: Imagem com grade
    """
    if not imagens:
        return None
    
    # Redimensiona todas as imagens para o mesmo tamanho
    altura_ref, largura_ref = imagens[0].shape[:2]
    imagens_redim = []
    
    for img in imagens:
        if img.shape[:2] != (altura_ref, largura_ref):
            img_redim = cv2.resize(img, (largura_ref, altura_ref))
        else:
            img_redim = img.copy()
        imagens_redim.append(img_redim)
    
    # Calcula número de linhas
    linhas = len(imagens_redim) // cols
    if len(imagens_redim) % cols > 0:
        linhas += 1
    
    # Cria grade
    grade = []
    for linha in range(linhas):
        linha_imgs = []
        for col in range(cols):
            idx = linha * cols + col
            if idx < len(imagens_redim):
                img = imagens_redim[idx]
                
                # Adiciona título se fornecido
                if titulos and idx < len(titulos):
                    img_com_titulo = img.copy()
                    cv2.putText(img_com_titulo, titulos[idx], (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    linha_imgs.append(img_com_titulo)
                else:
                    linha_imgs.append(img)
            else:
                # Preenche com imagem preta se necessário
                img_vazia = np.zeros_like(imagens_redim[0])
                linha_imgs.append(img_vazia)
        
        # Concatena horizontalmente
        linha_concatenada = np.hstack(linha_imgs)
        grade.append(linha_concatenada)
    
    # Concatena verticalmente
    resultado = np.vstack(grade)
    
    return resultado

def verificar_webcam(indice=0):
    """
    Verifica se a webcam está disponível
    
    Args:
        indice (int): Índice da webcam
        
    Returns:
        bool: True se webcam disponível
    """
    cap = cv2.VideoCapture(indice)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        return ret
    return False

def obter_informacoes_imagem(imagem):
    """
    Obtém informações detalhadas sobre uma imagem
    
    Args:
        imagem (numpy.ndarray): Imagem para análise
        
    Returns:
        dict: Dicionário com informações da imagem
    """
    if imagem is None:
        return None
    
    altura, largura = imagem.shape[:2]
    canais = imagem.shape[2] if len(imagem.shape) == 3 else 1
    
    info = {
        'dimensoes': (altura, largura),
        'canais': canais,
        'tipo': imagem.dtype,
        'tamanho_bytes': imagem.nbytes,
        'valor_min': imagem.min(),
        'valor_max': imagem.max(),
        'valor_medio': imagem.mean(),
        'formato': 'RGB' if canais == 3 else 'Escala de Cinza'
    }
    
    return info