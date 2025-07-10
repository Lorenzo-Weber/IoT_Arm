import time
import numpy as np
import paho.mqtt.client as mqtt
import json
from pid_controller import PIDController
from kinematics import inverse_kinematics_5dof

class MQTTRoboticArmController:
    def __init__(self, mqtt_broker='localhost', mqtt_port=1883, client_id='robot_controller'):
        self.pid_base = PIDController(kp=2.0, ki=0.1, kd=0.4)
        self.pid_shoulder = PIDController(kp=2.2, ki=0.12, kd=0.5)
        self.pid_elbow = PIDController(kp=1.8, ki=0.08, kd=0.3)
        self.pid_wrist = PIDController(kp=1.5, ki=0.05, kd=0.25)
        self.pid_gripper = PIDController(kp=1.2, ki=0.03, kd=0.15)

        self.ids = ["base", "ombro", "cotovelo", "pulso", "garra"]
        
        self.current_angles = [90, 90, 90, 90, 0]  # Posições atuais estimadas
        self.target_angles = [90, 90, 90, 90, 0]   # Posições desejadas
        self.esp32_angles = [90, 90, 90, 90, 0]    # Posições reais do ESP32
        self.command_angles = [90, 90, 90, 90, 0]  # Últimos ângulos enviados
        
        self.max_angle_step = [180, 180, 180, 180, 180]  # Máximo incremento por comando
        
        # Parâmetros do braço (mm)
        self.l1 = 50
        self.l2 = 50  
        self.l3 = 25
        
        self.movement_delay = 0.10  # Delay entre comandos MQTT
        self.last_command_time = 0
        self.position_tolerance = 2  # Tolerância
        self.movement_timeout = 8.0
        
        # Estado de movimento
        self.is_moving = False
        self.movement_start_time = 0
        self.position_updated = False
        self.last_feedback_time = 0
        
        self.mqtt_client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        self.connect_mqtt(mqtt_broker, mqtt_port)
    
    def connect_mqtt(self, broker, port):
        try:
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            print(f"Conectando ao MQTT: {broker}:{port}")
        except Exception as e:
            print(f"Erro na conexao MQTT: {e}")
            self.mqtt_client = None
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("MQTT conectado com sucesso")
            for servo_id in self.ids:
                feedback_topic = f'braco/servo/{servo_id}/pos'
                command_topic = f'controle/abs/{servo_id}'
                relative_topic = f'controle/rel/{servo_id}'
                client.subscribe(feedback_topic)
                # client.subscribe(command_topic)
                # client.subscribe(relative_topic)
                print(f"Subscrito ao tópico: {feedback_topic}")
            
            print("Aguardando feedback dos servos...")
        else:
            print(f"Falha na conexao MQTT: codigo {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        try:
            topic = msg.topic

            angle_str = str(msg.payload[0]) if len(msg.payload) > 0 else "0"

            print(f"Mensagem recebida no tópico {topic}: {angle_str}")

            angle_value = int(angle_str) if angle_str.isdigit() else 0
            
            for i, servo_id in enumerate(self.ids):
                feedback_topic = f'braco/servo/{servo_id}/pos'
                if topic == feedback_topic:
                    self.esp32_angles[i] = angle_value
                    self.position_updated = True
                    self.last_feedback_time = time.time()
                    
                    if self.is_moving and self.check_movement_complete():
                        self.is_moving = False
                        print("Movimento concluído")
                    break
                    
        except Exception as e:
            print(f"Erro ao processar MQTT: {e}")
    
    def check_movement_complete(self):
        for esp32_angle, target_angle in zip(self.esp32_angles, self.command_angles):
            if abs(esp32_angle - target_angle) > self.position_tolerance:
                return False
        return True
    
    def move_to_position(self, target_x, target_y, target_z, wrist_angle, gripper_angle):
        current_time = time.time()
        if current_time - self.last_command_time < self.movement_delay:
            return [0, 0, 0, 0, 0]
        
        try:
            target_angles = list(inverse_kinematics_5dof(
                target_x, target_y, target_z, 
                self.l1, self.l2, self.l3, 
                wrist_angle, gripper_angle
            ))
            
            current_positions = self.esp32_angles if self.position_updated else self.current_angles
            
            pids = [self.pid_base, self.pid_shoulder, self.pid_elbow, self.pid_wrist, self.pid_gripper]
            relative_angles = []
            corrections = []
            
            for i, (pid, current, target) in enumerate(zip(pids, current_positions, target_angles)):
                pid.set_setpoint(target)
                correction = pid.update(current)
                corrections.append(correction)
                
                # Calcula movimento relativo direto baseado no erro
                error = target - current
                
                # Limita o movimento relativo máximo por iteração
                max_change = self.max_angle_step[i]
                if error > max_change:
                    relative_move = max_change
                elif error < -max_change:
                    relative_move = -max_change
                else:
                    relative_move = error
                
                # Aplica fator de suavização
                relative_move = relative_move * 0.3
                
                relative_angles.append(relative_move)
            
            # Verifica se há movimento significativo para enviar
            significant_change = any(abs(rel) > 0.5 for rel in relative_angles)
            
            if significant_change:
                self.send_relative_angles(relative_angles)
                self.last_command_time = current_time
                
                # Atualiza estimativa de posição atual
                for i, rel in enumerate(relative_angles):
                    self.current_angles[i] = np.clip(self.current_angles[i] + rel, 0, 180)
                
                if any(abs(rel) > 2 for rel in relative_angles):
                    self.is_moving = False
                    self.movement_start_time = current_time
            
            return corrections
            
        except ValueError as e:
            print(f"Cinematica: {e}")
            return [0, 0, 0, 0, 0]
    
    def send_relative_angles(self, relative_angles):
        if self.mqtt_client:
            for i, (servo_id, rel_angle) in enumerate(zip(self.ids, relative_angles)):
                if abs(rel_angle) > 0.5:  # Só envia se há movimento significativo
                    command_topic = f'controle/rel/{servo_id}'
                    
                    if rel_angle >= 0:
                        uint8_value = min(127, int(round(abs(rel_angle))))
                    else:
                        uint8_value = max(128, 255 - int(round(abs(rel_angle))) + 1)
                    
                    message = bytes([uint8_value])
                    result = self.mqtt_client.publish(command_topic, message)
                    
                    if result.rc != 0:
                        print(f"Falha no envio MQTT para {servo_id}: {result.rc}")
                    else:
                        sign = "+" if rel_angle >= 0 else "-"
                        print(f"ESP32 {servo_id}: {sign}{abs(rel_angle):.0f}° (uint8: {uint8_value})")
        else:
            moving_servos = [f'{self.ids[i]}={int(round(a)):+d}°' for i, a in enumerate(relative_angles) if abs(a) > 0.5]
            if moving_servos:
                print(f"Simulação rel: {moving_servos}")

    def send_absolute_angles(self, angles):
        if self.mqtt_client:
            current_positions = self.esp32_angles if self.position_updated else self.current_angles
            relative_moves = []
            
            for current, target in zip(current_positions, angles):
                relative_moves.append(target - current)
            
            self.send_relative_angles(relative_moves)
            self.command_angles = [int(round(angle)) for angle in angles]
        else:
            print(f"Simulação abs: {[f'{self.ids[i]}={int(round(a))}°' for i, a in enumerate(angles)]}")

    def is_at_target(self, target_x, target_y, target_z, wrist_angle, gripper_angle, tolerance=3):
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
        if 0 <= servo_index < 5:
            target_angle = np.clip(target_angle, 0, 180)
            servo_id = self.ids[servo_index]
            
            current_angle = self.esp32_angles[servo_index] if self.position_updated else self.current_angles[servo_index]
            relative_move = target_angle - current_angle
            
            if abs(relative_move) > 0.5:  # Só move se diferença significativa
                command_topic = f'controle/rel/{servo_id}'
                
                if self.mqtt_client:
                    # Converte para uint8
                    if relative_move >= 0:
                        uint8_value = min(127, int(round(abs(relative_move))))
                    else:
                        uint8_value = max(128, 255 - int(round(abs(relative_move))) + 1)
                    
                    message = bytes([uint8_value])
                    result = self.mqtt_client.publish(command_topic, message)
                    if result.rc == 0:
                        sign = "+" if relative_move >= 0 else "-"
                        print(f"Servo {servo_id} → {sign}{abs(relative_move):.0f}° (uint8: {uint8_value})")
                        self.current_angles[servo_index] = target_angle
                        self.command_angles[servo_index] = int(round(target_angle))
                    else:
                        print(f"Falha no envio MQTT para {servo_id}: {result.rc}")
                else:
                    print(f"Simulação - Servo {servo_id} → {relative_move:+.0f}°")

    def close_gripper(self):
        print("Fechando garra...")
        if self.mqtt_client:
            command_topic = f'controle/rel/{self.ids[4]}'
            uint8_value = 90 
            message = bytes([uint8_value])
            self.mqtt_client.publish(command_topic, message)
            print(f"Garra fechando: +90° (uint8: {uint8_value})")

    def open_gripper(self):
        print("Abrindo garra...")
        if self.mqtt_client:
            command_topic = f'controle/rel/{self.ids[4]}'
            uint8_value = 218   
            message = bytes([uint8_value])
            self.mqtt_client.publish(command_topic, message)
            print(f"Garra abrindo: -90° (uint8: {uint8_value})")

    def wait_for_movement(self, timeout=None):
        if timeout is None:
            timeout = self.movement_timeout
            
        start_time = time.time()
        while self.is_moving and (time.time() - start_time) < timeout:
            time.sleep(0.05)
            
        if self.is_moving:
            print("Timeout - continuando...")
            self.is_moving = False
            return False
        return True
    
    def get_health_status(self):
        current_time = time.time()
        mqtt_ok = self.mqtt_client and self.mqtt_client.is_connected()
        feedback_ok = (current_time - self.last_feedback_time) < 0 if self.position_updated else False
        
        return {
            'mqtt_connected': mqtt_ok,
            'feedback_recent': feedback_ok,
            'position_updated': self.position_updated,
            'is_moving': self.is_moving
        }
    
    def emergency_stop(self):
        if self.mqtt_client:
            for servo_id in self.ids:
                stop_topic = f'controle/abs/{servo_id}'
                self.mqtt_client.publish(stop_topic, "90")
        self.is_moving = False
        print("PARADA DE EMERGENCIA")
    
    def disconnect(self):
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
