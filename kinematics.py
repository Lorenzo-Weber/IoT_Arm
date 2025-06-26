import math

def inverse_kinematics_5dof(x, y, z, l1, l2, l3, wrist_angle, gripper_angle):
    """
    Cinemática inversa para braço de 5 DOF
    x, y, z: posição desejada do efetuador final
    l1, l2, l3: comprimentos dos elos
    wrist_angle: ângulo desejado do pulso
    gripper_angle: ângulo da garra
    """
    # Ângulo da base
    base_angle = math.degrees(math.atan2(y, x))
    
    # Distância no plano XY
    r = math.sqrt(x**2 + y**2)
    
    # Ajusta Z para compensar offset do pulso
    z_adjusted = z - l3 * math.sin(math.radians(wrist_angle))
    r_adjusted = r - l3 * math.cos(math.radians(wrist_angle))
    
    # Distância total até o pulso
    d = math.sqrt(r_adjusted**2 + z_adjusted**2)
    
    if d > (l1 + l2) or d < abs(l1 - l2):
        raise ValueError("Posição fora do alcance")
    
    # Ângulos usando lei dos cossenos
    alpha = math.atan2(z_adjusted, r_adjusted)
    beta = math.acos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    gamma = math.acos((l1**2 + l2**2 - d**2) / (2 * l1 * l2))
    
    shoulder_angle = math.degrees(alpha + beta)
    elbow_angle = 180 - math.degrees(gamma)
    
    return base_angle, shoulder_angle, elbow_angle, wrist_angle, gripper_angle
