# math_core.py
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class LKF:
    def __init__(self, dt, process_noise, meas_noise):
        self.dt = dt
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1)) 
        self.x[2] = 5.0 # Стартуем сразу на высоте 5м, чтобы не скрести дно
        
        # Physics Model
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement Model
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        self.P = np.eye(6) * 100.0
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(3) * meas_noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Ограничение фильтра (чтобы линия оценки тоже не уходила под пол)
        if self.x[2] < 1.0: self.x[2] = 1.0
            
        return self.x.flatten().tolist()

    def update(self, z_meas):
        z = np.array(z_meas).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x.flatten().tolist()

class AUVSimulator:
    def __init__(self, dt):
        self.dt = dt
        # Start positions
        self.true_pos = np.array([0.0, 0.0, 5.0]) 
        self.velocity_cmd = np.array([0.0, 0.0, 0.0]) 
        self.current_drift = np.array([0.0, 0.0, 0.0])
        self.noise_std = 2.0
        self.radius = 1.0 # Радиус корпуса робота (чтобы не центром лежал на песке)

    def step(self):
        # 1. Расчет новой позиции
        total_vel = self.velocity_cmd + self.current_drift
        new_pos = self.true_pos + total_vel * self.dt
        
        # 2. ФИЗИКА ДНА (COLLISION DETECTION)
        # Если Z < радиуса, значит мы ударились о дно (z=0)
        if new_pos[2] < self.radius:
            new_pos[2] = self.radius
            # Если скорость направлена вниз, обнуляем её (удар гасит импульс)
            if total_vel[2] < 0:
                total_vel[2] = 0
        
        self.true_pos = new_pos

        # 3. Генерация шума
        noise = np.random.normal(0, self.noise_std, 3)
        measured_pos = self.true_pos + noise
        
        return {
            "true_x": float(self.true_pos[0]),
            "true_y": float(self.true_pos[1]),
            "true_z": float(self.true_pos[2]),
            "meas_x": float(measured_pos[0]),
            "meas_y": float(measured_pos[1]),
            "meas_z": float(measured_pos[2]),
            "total_vx": float(total_vel[0]),
            "total_vy": float(total_vel[1]),
            "total_vz": float(total_vel[2])
        }

    def update_params(self, vx, vy, vz, dx, dy, dz, noise):
        self.velocity_cmd = np.array([vx, vy, vz])
        self.current_drift = np.array([dx, dy, dz])
        self.noise_std = noise

dt = 0.1
sim = AUVSimulator(dt)
kf = LKF(dt, process_noise=0.1, meas_noise=2.0)

class ControlParams(BaseModel):
    cmd_vx: float
    cmd_vy: float
    cmd_vz: float
    drift_x: float
    drift_y: float
    drift_z: float
    noise: float

@app.post("/step")
def step_simulation(params: ControlParams):
    sim.update_params(params.cmd_vx, params.cmd_vy, params.cmd_vz, 
                      params.drift_x, params.drift_y, params.drift_z, params.noise)
    
    sim_data = sim.step()
    kf.predict()
    est = kf.update([sim_data["meas_x"], sim_data["meas_y"], sim_data["meas_z"]])
    
    return {
        "simulation": sim_data,
        "estimation": {
            "est_x": est[0], "est_y": est[1], "est_z": est[2],
            "est_vx": est[3], "est_vy": est[4], "est_vz": est[5]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)