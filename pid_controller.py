import time

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
            
        # Calcula o erro
        error = self.setpoint - current_value
        
        # Termo Proporcional
        proportional = self.kp * error
        
        # Termo Integral
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # Termo Derivativo
        derivative = (error - self.previous_error) / dt
        derivative_term = self.kd * derivative
        
        # Saída PID
        output = proportional + integral_term + derivative_term
        
        # Atualiza valores para próxima iteração
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
        
    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()
