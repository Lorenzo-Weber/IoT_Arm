import time
import cv2
from robotic_arm_controller import MQTTRoboticArmController
from vision_controller import VisionController, detect_gripper, calculate_wrist_angle

def main():
    # Configurações
    MQTT_BROKER = '192.168.2.145'
    MQTT_PORT = 1883
    
    # Estados da máquina de estados
    STATE_SEARCHING = 0
    STATE_APPROACHING = 1
    STATE_GRASPING = 2
    STATE_LIFTING = 3
    
    # Inicialização dos controladores
    print("Inicializando sistema...")
    arm_controller = MQTTRoboticArmController(
        mqtt_broker=MQTT_BROKER,
        mqtt_port=MQTT_PORT,
        client_id='CAMERACONTROLLER'
    )
    vision_controller = VisionController()
    
    # Variáveis de estado
    current_state = STATE_SEARCHING
    target_reached = False
    gripper_closed = False
    state_change_time = time.time()
    state_timeout = 10.0
    
    # Inicializa câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return
    
    print("Sistema iniciado - Aguardando conexão com ESP32...")
    time.sleep(3)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame da câmera")
                break
                
            current_time = time.time()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detecta objeto verde (alvo)
            lower_green = (50, 100, 100)
            upper_green = (70, 255, 255)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detecta garra vermelha
            gx, gy, gripper_orientation, gripper_box = detect_gripper(frame)
            
            # Desenha detecções
            if gripper_box is not None:
                cv2.drawContours(frame, [gripper_box], 0, (0, 0, 255), 2)
                cv2.circle(frame, (gx, gy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Garra: ({gx}, {gy})", (gx+10, gy-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Timeout de estado para evitar travamento
            if current_time - state_change_time > state_timeout:
                print("Timeout de estado - Reiniciando...")
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
                        cv2.putText(frame, f"Alvo: ({cx}, {cy})", (cx+10, cy-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        # Converte para coordenadas do mundo
                        target_x, target_y, target_z = vision_controller.pixel_to_world(cx, cy)
                        
                        # Máquina de estados
                        if current_state == STATE_SEARCHING:
                            print("Objeto detectado - Aproximando...")
                            current_state = STATE_APPROACHING
                            state_change_time = current_time
                            arm_controller.open_gripper()
                        
                        elif current_state == STATE_APPROACHING:
                            if not arm_controller.is_moving:
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
                            if not arm_controller.is_moving:
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
                            if gripper_closed and not arm_controller.is_moving:
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

            # Interface visual
            states_text = ['Procurando', 'Aproximando', 'Pegando', 'Levantando']
            cv2.putText(frame, f"Estado: {states_text[current_state]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status do sistema
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

if __name__ == "__main__":
    main()
