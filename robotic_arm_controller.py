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
        
        self.max_angle_step = [3, 2.5, 4, 5, 6]  # Máximo incremento por comando
        
        # Parâmetros do braço (mm)
        self.l1 = 100
        self.l2 = 100  
        self.l3 = 50
        
        # Controle de timing otimizado
        self.movement_delay = 0.15  # Delay entre comandos MQTT
        self.last_command_time = 0
        self.position_tolerance = 2  # Tolerância mais precisa
        self.movement_timeout = 8.0
        
        # Estado de movimento
        self.is_moving = False
        self.movement_start_time = 0
        self.position_updated = False
        self.last_feedback_time = 0
        
        # Configuração MQTT
        self.mqtt_client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
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
            # Subscreve aos tópicos de feedback de cada servo
            for servo_id in self.ids:
                feedback_topic = f'braco/servo/{servo_id}/pos'
                client.subscribe(feedback_topic)
                print(f"Subscrito ao tópico: {feedback_topic}")
            
            # Remove a solicitação automática de posição para evitar envio de dados desnecessários
            print("Aguardando feedback dos servos...")
        else:
            print(f"Falha na conexao MQTT: codigo {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Processa mensagens MQTT"""
        try:
            topic = msg.topic
            # Recebe apenas o ângulo como string, não JSON
            angle_str = msg.payload.decode()
            angle_value = float(angle_str)
            
            # Verifica se é feedback de posição de algum servo
            for i, servo_id in enumerate(self.ids):
                feedback_topic = f'braco/servo/{servo_id}/pos'
                if topic == feedback_topic:
                    # Atualiza posição específica do servo
                    self.esp32_angles[i] = angle_value
                    self.position_updated = True
                    self.last_feedback_time = time.time()
                    
                    # Verifica conclusão do movimento
                    if self.is_moving and self.check_movement_complete():
                        self.is_moving = False
                        print("Movimento concluído")
                    break
                    
        except Exception as e:
            print(f"Erro ao processar MQTT: {e}")
    
    def request_current_position(self):
        """Remove função que enviava JSON desnecessário"""
        pass
    
    def check_movement_complete(self):
        """Verifica se movimento foi concluído com base no feedback"""
        for esp32_angle, target_angle in zip(self.esp32_angles, self.command_angles):
            if abs(esp32_angle - target_angle) > self.position_tolerance:
                return False
        return True
    
    def move_to_position(self, target_x, target_y, target_z, wrist_angle, gripper_angle):
        """Move braço para posição 3D usando ângulos absolutos"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_command_time < self.movement_delay:
            return [0, 0, 0, 0, 0]
        
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
            
            # Verifica se mudança é significativa
            significant_change = any(
                abs(new - current) > 0.5 
                for new, current in zip(new_angles, current_positions)
            )
            
            if significant_change:
                self.current_angles = new_angles
                self.send_absolute_angles(new_angles)
                self.last_command_time = current_time
                
                # Só marca como "movendo" se mudança for grande
                if any(abs(new - current) > 2 for new, current in zip(new_angles, current_positions)):
                    self.is_moving = True
                    self.movement_start_time = current_time
            
            return corrections
            
        except ValueError as e:
            print(f"Cinematica: {e}")
            return [0, 0, 0, 0, 0]
    
    def send_absolute_angles(self, angles):
        """Envia ângulos absolutos para ESP32 via MQTT"""
        if self.mqtt_client:
            # Arredonda para valores inteiros
            rounded_angles = [int(round(angle)) for angle in angles]
            
            # Envia comando para cada servo individualmente
            for i, (servo_id, angle) in enumerate(zip(self.ids, rounded_angles)):
                command_topic = f'controle/abs/{servo_id}'
                
                # Envia apenas o ângulo como string inteira
                message = str(angle)
                result = self.mqtt_client.publish(command_topic, message)
                
                if result.rc != 0:
                    print(f"Falha no envio MQTT para {servo_id}: {result.rc}")
            
            self.command_angles = rounded_angles
            print(f"ESP32: {[f'{self.ids[i]}={a}°' for i, a in enumerate(rounded_angles)]}")
        else:
            print(f"Simulação: {[f'{self.ids[i]}={int(round(a))}°' for i, a in enumerate(angles)]}")
    
    def is_at_target(self, target_x, target_y, target_z, wrist_angle, gripper_angle, tolerance=5):
        """Verifica se está na posição alvo"""
        if self.is_moving:
            return False
            
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
        """Move um servo específico para ângulo absoluto"""
        if 0 <= servo_index < 5:
            target_angle = np.clip(target_angle, 0, 180)
            servo_id = self.ids[servo_index]
            command_topic = f'controle/abs/{servo_id}'
            
            if self.mqtt_client:
                # Envia apenas o ângulo como string inteira
                message = str(int(round(target_angle)))
                result = self.mqtt_client.publish(command_topic, message)
                if result.rc == 0:
                    print(f"Servo {servo_id} → {int(round(target_angle))}°")
                    self.current_angles[servo_index] = int(round(target_angle))
                    self.command_angles[servo_index] = int(round(target_angle))
                else:
                    print(f"Falha no envio MQTT para {servo_id}: {result.rc}")
            else:
                print(f"Simulação - Servo {servo_id} → {int(round(target_angle))}°")
    
    def close_gripper(self):
        """Fecha garra"""
        print("Fechando garra...")
        self.move_servo_absolute(4, 90)
        self.wait_for_movement(timeout=2.0)
    
    def open_gripper(self):
        """Abre garra"""
        print("Abrindo garra...")
        self.move_servo_absolute(4, 0)
        self.wait_for_movement(timeout=2.0)
    
    def wait_for_movement(self, timeout=None):
        """Aguarda movimento com timeout"""
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
        """Retorna status de saúde da conexão"""
        current_time = time.time()
        mqtt_ok = self.mqtt_client and self.mqtt_client.is_connected()
        feedback_ok = (current_time - self.last_feedback_time) < 5.0 if self.position_updated else False
        
        return {
            'mqtt_connected': mqtt_ok,
            'feedback_recent': feedback_ok,
            'position_updated': self.position_updated,
            'is_moving': self.is_moving
        }
    
    def emergency_stop(self):
        """Para todos os movimentos"""
        if self.mqtt_client:
            # Envia parada de emergência para todos os servos (volta para posição neutra)
            for servo_id in self.ids:
                stop_topic = f'controle/abs/{servo_id}'
                self.mqtt_client.publish(stop_topic, "90")
        self.is_moving = False
        print("PARADA DE EMERGENCIA")
    
    def disconnect(self):
        """Desconecta do MQTT"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
