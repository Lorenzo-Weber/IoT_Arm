import cv2
import numpy as np
import math

class VisionController:
    def __init__(self):
        self.camera_height = 200  # Altura da câmera em mm
        self.pixels_to_mm = 0.5   # Conversão pixel para mm
        
    def pixel_to_world(self, pixel_x, pixel_y, frame_width=640, frame_height=480):
        world_x = (pixel_x - frame_width/2) * self.pixels_to_mm
        world_y = (frame_height/2 - pixel_y) * self.pixels_to_mm
        world_z = 0  # Assume objetos na mesa
        
        return world_x, world_y, world_z

def detect_gripper(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = (100, 100, 100)  
    upper_blue = (130, 255, 255)
    
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:  
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            gx, gy = int(rect[0][0]), int(rect[0][1])
            
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle += 90
                
            return gx, gy, angle, box
    
    return None, None, None, None

def calculate_wrist_angle(target_x, target_y, gripper_x, gripper_y):
    dx = target_x - gripper_x
    dy = target_y - gripper_y
    
    desired_angle = math.degrees(math.atan2(dy, dx))
    
    wrist_angle = (desired_angle + 90) % 180
    
    return wrist_angle
