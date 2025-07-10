import time
import cv2
import numpy as np
import math
import paho.mqtt.client as mqtt
import json
import threading

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp  # Ganho proporcional
        self.ki = ki  # Ganho integral
        self.kd = kd  # Ganho derivativo
        self.setpoint = setpoint  # Valor desejado
        
        self.previous_error = 0
        self.integral = 0
        self.previous_time = time.time()
        
    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt <= 0:
            dt = 1e-6  # Evita divisão por zero
            
        error = self.setpoint - current_value
        
        proportional = self.kp * error
        
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        derivative = (error - self.previous_error) / dt
        derivative_term = self.kd * derivative
        
        output = proportional + integral_term + derivative_term
        
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
        
    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()

class GripperController:
    def __init__(self):
        self.pid_x = PIDController(kp=1.5, ki=0.1, kd=0.3)
        self.pid_y = PIDController(kp=1.5, ki=0.1, kd=0.3)
        self.pid_angle = PIDController(kp=2.0, ki=0.05, kd=0.4)
        
        self.max_velocity = 50  # mm/s
        self.max_angular_velocity = 30  # graus/s
        
    def move_to_target(self, current_pos, target_pos):
        cx, cy, current_angle = current_pos
        tx, ty, target_angle = target_pos
        
        self.pid_x.set_setpoint(tx)
        self.pid_y.set_setpoint(ty)
        self.pid_angle.set_setpoint(target_angle)
        
        correction_x = self.pid_x.update(cx)
        correction_y = self.pid_y.update(cy)
        correction_angle = self.pid_angle.update(current_angle)
        
        correction_x = np.clip(correction_x, -self.max_velocity, self.max_velocity)
        correction_y = np.clip(correction_y, -self.max_velocity, self.max_velocity)
        correction_angle = np.clip(correction_angle, -self.max_angular_velocity, self.max_angular_velocity)
        
        return correction_x, correction_y, correction_angle
    
    def is_at_target(self, current_pos, target_pos, tolerance=5):
        cx, cy, current_angle = current_pos
        tx, ty, target_angle = target_pos
        
        distance_error = np.sqrt((tx - cx)**2 + (ty - cy)**2)
        angle_error = abs(target_angle - current_angle)
        
        return distance_error < tolerance and angle_error < 5
    
def detect_gripper(frame):
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

class MQTTRoboticArmController:
    def __init__(self, mqtt_broker='localhost', mqtt_port=1883, client_id='robot_controller'):
        self.pid_base = PIDController(kp=2.0, ki=0.1, kd=0.4)
        self.pid_shoulder = PIDController(kp=2.2, ki=0.12, kd=0.5)
        self.pid_elbow = PIDController(kp=1.8, ki=0.08, kd=0.3)
        self.pid_wrist = PIDController(kp=1.5, ki=0.05, kd=0.25)
        self.pid_gripper = PIDController(kp=1.2, ki=0.03, kd=0.15)
        
        self.current_angles = [90, 90, 90, 90, 0]  # Posições atuais estimadas
        self.target_angles = [90, 90, 90, 90, 0]   # Posições desejadas
        self.esp32_angles = [90, 90, 90, 90, 0]    # Posições reais do ESP32
        self.command_angles = [90, 90, 90, 90, 0]  # Últimos ângulos enviados
        
        self.max_angle_step = [3, 2.5, 4, 5, 6]  # Máximo incremento por comando
        
        # Parâmetros do braço (mm)
        self.l1 = 100
        self.l2 = 100  
        self.l3 = 50
        
        # Estado de movimento
        self.position_updated = False
        self.last_feedback_time = 0
        
        # Configuração MQTT (compatível com paho-mqtt v2.0)
        self.mqtt_client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # Tópicos MQTT
        self.topics = {
            'command': 'robot/servo/absolute',     # Comando de ângulos absolutos
            'feedback': 'robot/servo/feedback',    # Feedback de posições
            'status': 'robot/status'               # Status geral
        }
        
        # Conecta ao broker MQTT
        self.connect_mqtt(mqtt_broker, mqtt_port)
    
    def connect_mqtt(self, broker, port):
        """Conecta ao broker MQTT com retry"""
        try:
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            print(f"Conectando ao MQTT: {broker}:{port}")
        except Exception as e:
            print(f"Erro na conexao MQTT: {e}")
            self.mqtt_client = None
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback de conexão MQTT"""
        if rc == 0:
            print("MQTT conectado com sucesso")
            client.subscribe(self.topics['feedback'])
            client.subscribe(self.topics['status'])
            
            # Solicita posição atual do ESP32
            self.request_current_position()
        else:
            print(f"Falha na conexao MQTT: codigo {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Processa mensagens MQTT"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            if topic == self.topics['feedback']:
                # Atualiza posições reais do ESP32
                self.esp32_angles = payload['angles']
                self.position_updated = True
                self.last_feedback_time = time.time()
                        
            elif topic == self.topics['status']:
                pass  # Remove lógica de movimento
                    
        except Exception as e:
            print(f"Erro ao processar MQTT: {e}")
    
    def request_current_position(self):
        """Solicita posição atual do ESP32"""
        if self.mqtt_client:
            request = {'action': 'get_position', 'timestamp': time.time()}
            self.mqtt_client.publish('robot/request', json.dumps(request))
    
    def move_to_position(self, target_x, target_y, target_z, wrist_angle, gripper_angle):
        """
        Move braço para posição 3D usando ângulos absolutos
        """
        try:
            # Calcula ângulos alvo da cinemática inversa
            target_angles = list(inverse_kinematics_5dof(
                target_x, target_y, target_z, 
                self.l1, self.l2, self.l3, 
                wrist_angle, gripper_angle
            ))
            
            # Usa feedback real do ESP32 quando disponível
            current_positions = self.esp32_angles if self.position_updated else self.current_angles
            
            # Calcula correções PID
            pids = [self.pid_base, self.pid_shoulder, self.pid_elbow, self.pid_wrist, self.pid_gripper]
            new_angles = []
            corrections = []
            
            for i, (pid, current, target) in enumerate(zip(pids, current_positions, target_angles)):
                pid.set_setpoint(target)
                correction = pid.update(current)
                corrections.append(correction)
                
                # Calcula novo ângulo absoluto com limitação de velocidade
                desired_angle = current + correction * 0.1  # Suavização
                
                # Limita velocidade de mudança
                max_change = self.max_angle_step[i]
                if desired_angle > current + max_change:
                    desired_angle = current + max_change
                elif desired_angle < current - max_change:
                    desired_angle = current - max_change
                
                # Limita aos ranges dos servos
                final_angle = np.clip(desired_angle, 0, 180)
                new_angles.append(final_angle)
            
            # Envia sempre os novos ângulos
            self.current_angles = new_angles
            self.send_absolute_angles(new_angles)
            
            return corrections
            
        except ValueError as e:
            print(f"Cinematica: {e}")
            return [0, 0, 0, 0, 0]
    
    def send_absolute_angles(self, angles):
        """
        Envia ângulos absolutos para ESP32 via MQTT
        """
        if self.mqtt_client:
            # Arredonda para economizar precisão desnecessária
            rounded_angles = [round(angle, 1) for angle in angles]
            
            command = {
                'angles': rounded_angles,
                'mode': 'absolute',  # Confirma que são ângulos absolutos
                'timestamp': time.time(),
                'sequence': int(time.time() * 1000) % 100000
            }
            
            self.command_angles = rounded_angles  # Salva para comparação
            message = json.dumps(command)
            
            result = self.mqtt_client.publish(self.topics['command'], message)
            if result.rc == 0:
                print(f"ESP32: {[f'{a:.1f}°' for a in rounded_angles]}")
            else:
                print(f"Falha no envio MQTT: {result.rc}")
        else:
            print(f"Simulação: {[f'{a:.1f}°' for a in angles]}")
    
    def is_at_target(self, target_x, target_y, target_z, wrist_angle, gripper_angle, tolerance=5):
        """Verifica se está na posição alvo"""
        try:
            target_angles = inverse_kinematics_5dof(
                target_x, target_y, target_z, 
                self.l1, self.l2, self.l3, 
                wrist_angle, gripper_angle
            )
            
            current_positions = self.esp32_angles if self.position_updated else self.current_angles
            
            for current, target in zip(current_positions, target_angles):
                if abs(current - target) > tolerance:
                    return False
            return True
            
        except ValueError:
            return False
    
    def move_servo_absolute(self, servo_index, target_angle):
        """
        Move um servo específico para ângulo absoluto
        """
        if 0 <= servo_index < 5:
            target_angle = np.clip(target_angle, 0, 180)
            new_angles = self.current_angles.copy()
            new_angles[servo_index] = target_angle
            self.send_absolute_angles(new_angles)
            print(f"Servo {servo_index} → {target_angle:.1f}°")
    
    def close_gripper(self):
        """Fecha garra (ângulo absoluto)"""
        print("Fechando garra...")
        self.move_servo_absolute(4, 90)  # Servo 4 para 90°

    def open_gripper(self):
        """Abre garra (ângulo absoluto)"""
        print("Abrindo garra...")
        self.move_servo_absolute(4, 0)   # Servo 4 para 0°

    def get_health_status(self):
        """Retorna status de saúde da conexão"""
        current_time = time.time()
        mqtt_ok = self.mqtt_client and self.mqtt_client.is_connected()
        feedback_ok = (current_time - self.last_feedback_time) < 0.1 if self.position_updated else False
        
        return {
            'mqtt_connected': mqtt_ok,
            'feedback_recent': feedback_ok,
            'position_updated': self.position_updated
        }
    
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

# Inicialização com IP do seu broker MQTT
arm_controller = MQTTRoboticArmController(
    mqtt_broker='localhost',  # Mude para IP do seu broker
    mqtt_port=1883,
    client_id='robot_arm_vision_v2'
)
vision_controller = VisionController()
target_reached = False
gripper_closed = False

cap = cv2.VideoCapture(0)

# Estados da máquina de estados
STATE_SEARCHING = 0
STATE_APPROACHING = 1
STATE_GRASPING = 2
STATE_LIFTING = 3
current_state = STATE_SEARCHING

# Controle de timing para estados
state_change_time = time.time()
state_timeout = 10.0  # Timeout para cada estado

print("Sistema iniciado - Aguardando conexão com ESP32...")
time.sleep(3)  # Aguarda conexão MQTT

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detecta objeto verde (alvo)
        lower_green = (50, 100, 100)
        upper_green = (70, 255, 255)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gx, gy, gripper_orientation, gripper_box = detect_gripper(frame)
        
        # Desenha detecções
        if gripper_box is not None:
            cv2.drawContours(frame, [gripper_box], 0, (0, 0, 255), 2)
            cv2.circle(frame, (gx, gy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Garra: ({gx}, {gy})", (gx+10, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Timeout de estado para evitar travamento
        if current_time - state_change_time > state_timeout:
            print("⏰ Timeout de estado - Reiniciando...")
            current_state = STATE_SEARCHING
            state_change_time = current_time
            arm_controller.is_moving = False

        # Processa objeto alvo
        if green_contours and gx is not None:
            c = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"Alvo: ({cx}, {cy})", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # Converte para coordenadas do mundo
                    target_x, target_y, target_z = vision_controller.pixel_to_world(cx, cy)
                    
                    # Máquina de estados com controle de timing
                    if current_state == STATE_SEARCHING:
                        print("Objeto detectado - Aproximando...")
                        current_state = STATE_APPROACHING
                        state_change_time = current_time
                        arm_controller.open_gripper()
                    
                    elif current_state == STATE_APPROACHING:
                        wrist_angle = calculate_wrist_angle(cx, cy, gx, gy)
                        approach_z = target_z + 30
                        
                        corrections = arm_controller.move_to_position(
                            target_x, target_y, approach_z, wrist_angle, 0
                        )
                        
                        if arm_controller.is_at_target(target_x, target_y, approach_z, wrist_angle, 0):
                            print("Posicao de aproximacao alcancada - Descendo...")
                            current_state = STATE_GRASPING
                            state_change_time = current_time
                    
                    elif current_state == STATE_GRASPING:
                        wrist_angle = calculate_wrist_angle(cx, cy, gx, gy)
                        
                        corrections = arm_controller.move_to_position(
                            target_x, target_y, target_z, wrist_angle, 0
                        )
                        
                        if arm_controller.is_at_target(target_x, target_y, target_z, wrist_angle, 0):
                            print("Fechando garra...")
                            arm_controller.close_gripper()
                            current_state = STATE_LIFTING
                            state_change_time = current_time
                            gripper_closed = True
                    
                    elif current_state == STATE_LIFTING:
                        if gripper_closed:
                            lift_z = target_z + 50
                            wrist_angle = calculate_wrist_angle(cx, cy, gx, gy)
                            
                            arm_controller.move_to_position(
                                target_x, target_y, lift_z, wrist_angle, 90
                            )
                            
                            if arm_controller.is_at_target(target_x, target_y, lift_z, wrist_angle, 90):
                                print("Objeto capturado com sucesso!")
                                time.sleep(2)
                                
                                # Reinicia ciclo
                                arm_controller.open_gripper()
                                current_state = STATE_SEARCHING
                                state_change_time = current_time
                                gripper_closed = False

        # Timeout de estado para evitar travamento
        if current_time - state_change_time > state_timeout:
            print("Timeout de estado - Reiniciando...")
            current_state = STATE_SEARCHING
            state_change_time = current_time
            arm_controller.is_moving = False

        # Interface visual
        states_text = ['Procurando', 'Aproximando', 'Pegando', 'Levantando']
        cv2.putText(frame, f"Estado: {states_text[current_state]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status detalhado do sistema
        health = arm_controller.get_health_status()
        status_color = (0, 255, 0) if health['mqtt_connected'] else (0, 0, 255)
        status_text = "MQTT OK" if health['mqtt_connected'] else "MQTT Falhou"
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Feedback do ESP32
        if health['feedback_recent']:
            angles_text = f"ESP32: {[f'{a:.0f}°' for a in arm_controller.esp32_angles]}"
            cv2.putText(frame, angles_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Indicador de movimento
        if health['is_moving']:
            cv2.putText(frame, "Movendo...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Controle Robotico MQTT + PID", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Parada de emergência
            arm_controller.emergency_stop()

except KeyboardInterrupt:
    print("\nInterrompido pelo usuário")
    
finally:
    cap.release()
    cv2.destroyAllWindows()
    arm_controller.disconnect()
    print("Sistema finalizado")