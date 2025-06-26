import cv2
import numpy as np
import math

class VisionController:
    def __init__(self):
        self.camera_height = 200  # Altura da câmera em mm
        self.pixels_to_mm = 0.5   # Conversão pixel para mm
        
    def pixel_to_world(self, pixel_x, pixel_y, frame_width=640, frame_height=480):
        """
        Converte coordenadas de pixel para coordenadas do mundo real
        """
        # Centro da imagem como origem
        world_x = (pixel_x - frame_width/2) * self.pixels_to_mm
        world_y = (frame_height/2 - pixel_y) * self.pixels_to_mm
        world_z = 0  # Assume objetos na mesa
        
        return world_x, world_y, world_z

def detect_gripper(frame):
    """
    Detecta a garra vermelha e retorna sua posição e orientação
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Faixa de cor vermelha para a garra
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 100, 100) 
    upper_red2 = (180, 255, 255)
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:  # Filtro de área mínima
            # Calcula o retângulo rotacionado para obter orientação
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Centro da garra
            gx, gy = int(rect[0][0]), int(rect[0][1])
            
            # Ângulo da garra (orientação)
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle += 90
                
            return gx, gy, angle, box
    
    return None, None, None, None

def calculate_wrist_angle(target_x, target_y, gripper_x, gripper_y):
    """
    Calcula o ângulo do pulso baseado na orientação desejada para pegar o objeto
    """
    # Vetor da garra para o objeto
    dx = target_x - gripper_x
    dy = target_y - gripper_y
    
    # Ângulo desejado (garra apontando para o objeto)
    desired_angle = math.degrees(math.atan2(dy, dx))
    
    # Normaliza para 0-180 graus
    wrist_angle = (desired_angle + 90) % 180
    
    return wrist_angle
