from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random
import math

app = FastAPI()

# нулевая матрица r x c 
def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


# единичная матрица n x n
def eye(n):
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0

    return I


def mat_copy(A):
    return [row[:] for row in A]


def mat_add(A, B):
    r = len(A)
    c = len(A[0])

    C = zeros(r, c)

    for i in range(r):
        for j in range(c):
            C[i][j] = A[i][j] + B[i][j]

    return C


def mat_sub(A, B):
    r = len(A)
    c = len(A[0])

    C = zeros(r, c)

    for i in range(r):
        for j in range(c):
            C[i][j] = A[i][j] - B[i][j]

    return C


def mat_mul(A, B):
    # A: m x n
    # B: n x p
    # C: m x p
    m = len(A)
    n = len(A[0])

    n2 = len(B)
    p = len(B[0])

    if n != n2: 
        return zeros(m, p) 
    
    C = zeros(m, p)
    for i in range(m):
        ai = A[i]
        for k in range(n):
            aik = ai[k]
            bk = B[k]
            for j in range(p):
                C[i][j] += aik * bk[j]
    
    return C


def transpose(A):
    r = len(A)
    c = len(A[0])
    T = zeros(c, r)

    for i in range(r):
        for j in range(c):
            T[j][i] = A[i][j]

    return T

# умножение на скаляр
def scalar_mult(A, s):
    r = len(A); c = len(A[0])
    B = zeros(r, c)
    for i in range(r):
        for j in range(c):
            B[i][j] = A[i][j] * s
    return B

# разбиваем матрицу на L, U и P
def lup_decompose(A):
    n = len(A)
    LU = mat_copy(A)
    perm = list(range(n))

    for k in range(n):
        pivot = k
        maxv = abs(LU[k][k])
        for i in range(k+1, n):
            if abs(LU[i][k]) > maxv:
                maxv = abs(LU[i][k])
                pivot = i

        if maxv < 1e-12:
            # для обработки выхода за пределы диапазона в питоне
            pass 

        if pivot != k:
            LU[k], LU[pivot] = LU[pivot], LU[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]

        for i in range(k+1, n):
            if abs(LU[k][k]) > 1e-12:
                LU[i][k] = LU[i][k] / LU[k][k]
                for j in range(k+1, n):
                    LU[i][j] -= LU[i][k] * LU[k][j]

    return LU, perm

# решаем систему LUx = Pb
def lup_solve(LU, perm, b):
    n = len(LU)
    pb = [b[perm[i]] for i in range(n)]

    y = [0.0]*n
    for i in range(n):
        s = pb[i]
        for k in range(i):
            s -= LU[i][k] * y[k]
        y[i] = s

    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = y[i]
        for k in range(i+1, n):
            s -= LU[i][k] * x[k]
        if abs(LU[i][i]) > 1e-12:
            x[i] = s / LU[i][i]
        else:
            x[i] = 0.0

    return x

def invert_matrix(A):
    n = len(A)
    LU, perm = lup_decompose(A)
    invA = zeros(n, n)
    I = eye(n)

    for col in range(n):
        b = [I[i][col] for i in range(n)]
        x = lup_solve(LU, perm, b)
        for i in range(n):
            invA[i][col] = x[i]

    return invA

def vec_to_col(v):
    return [[x] for x in v]

def col_to_vec(col):
    return [row[0] for row in col]


class LKF:
    def __init__(self, dt, process_noise, meas_noise):
        self.dt = dt
        # Состояние: [x, y, z, vx, vy, vz]
        self.x = vec_to_col([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
        
        # Матрица перехода
        # x_new = x + v_x*dt
        self.F = eye(6)
        self.F[0][3] = dt; self.F[1][4] = dt; self.F[2][5] = dt
        
        # Матрица наблюдений (мы измеряем только координаты x,y,z)
        self.H = zeros(3, 6)
        self.H[0][0] = 1.0; self.H[1][1] = 1.0; self.H[2][2] = 1.0
        
        # Ковариация ошибки P (начальная неуверенность)
        self.P = scalar_mult(eye(6), 50.0)
        
        # Шум процесса Q (насколько мы не доверяем нашей модели "постоянной скорости")
        self.Q = scalar_mult(eye(6), process_noise)
        
        # Шум измерений R
        self.R = scalar_mult(eye(3), meas_noise)

    def predict(self):
        # 1. x = F * x
        self.x = mat_mul(self.F, self.x)
        
        # 2. P = F * P * F^T + Q
        FP = mat_mul(self.F, self.P)
        FPFt = mat_mul(FP, transpose(self.F))
        self.P = mat_add(FPFt, self.Q)
        
        # чтобы фильтр не улетал под песочек
        if self.x[2][0] < 0.0: self.x[2][0] = 0.0
            
        return col_to_vec(self.x)

    def update(self, z_meas):
        z = vec_to_col(z_meas) 
        
        # y = z - H * x (Инновация)
        # Разница между датчиком и ожиданием модели
        Hx = mat_mul(self.H, self.x)
        y = mat_sub(z, Hx)
        
        # S = H * P * H^T + R (Ковариация инновации)
        # S показывает, насколько мы уверены в инновации
        HPHt = mat_mul(mat_mul(self.H, self.P), transpose(self.H))
        S = mat_add(HPHt, self.R)
        
        # K = P * H^T * inv(S) (кожефициент Калмана)
        # K показывает, насколько мы доверяем измерению по сравнению с моделью
        invS = invert_matrix(S)
        K = mat_mul(mat_mul(self.P, transpose(self.H)), invS)
        
        # x = x + K * y
        Ky = mat_mul(K, y)
        self.x = mat_add(self.x, Ky)
        
        # P = (I - K * H) * P
        I = eye(6)
        KH = mat_mul(K, self.H)
        ImKH = mat_sub(I, KH)
        self.P = mat_mul(ImKH, self.P)
        
        return col_to_vec(self.x)


class AUVSimulator:
    def __init__(self, dt):
        self.dt = dt
        self.pos = [0.0, 0.0, 5.0]   # Позиция 
        self.vel = [0.0, 0.0, 0.0]   # Скорость 
        
        # двигатель
        self.force_cmd = [0.0, 0.0, 0.0] 
        # вектор течения
        self.drift_vel = [0.0, 0.0, 0.0]
        
        self.noise_std = 2.0
        self.radius = 1.0

        self.mass = 15.0        # масса громозеки. Создает инерцию.
        self.drag_coeff = 3.0   # Коэф. вязкого трения (кг/с). Чем выше, тем труднее плыть.
        self.thrust_factor = 8.0 # Усиление джойстика (чтобы были силы побольше)

    def update_params(self, vx, vy, vz, dx, dy, dz, noise):
        # Интерпретируем вход джойстика как КОМАНДУ ТЯГИ, а не скорости
        self.force_cmd = [
            vx * self.thrust_factor, 
            vy * self.thrust_factor, 
            vz * self.thrust_factor
        ]

        # Течение остается скоростью
        self.drift_vel = [dx, dy, dz]
        self.noise_std = noise

    def step(self):
        # Симуляция пошаговой физики
        
        # Истинная скорость относительно земли (для возврата)
        true_vel_vector = [0.0]*3 

        for i in range(3):
            # 1. Сила сопротивления (направлена против скорости)
            # F_drag = - k * v
            f_drag = -self.drag_coeff * self.vel[i]
            
            # 2. Результирующая сила
            # F_net = F_thrust + F_drag
            f_net = self.force_cmd[i] + f_drag
            
            # 3. Второй закон Ньютона: a = F / m
            acc = f_net / self.mass
            
            # 4. Интегрируем скорость: v = v0 + a * dt
            self.vel[i] += acc * self.dt
            
            # 5. Итоговая скорость = Скорость аппарата + Скорость течения
            v_total = self.vel[i] + self.drift_vel[i]
            true_vel_vector[i] = v_total
            
            # 6. Интегрируем позицию: x = x0 + v_total * dt
            self.pos[i] += v_total * self.dt

        # коллизия
        if self.pos[2] < self.radius:
            self.pos[2] = self.radius
            # Неупругое соударение - гасим вертикальную скорость и силу
            if self.vel[2] < 0: self.vel[2] = 0.0
            
        # Генерация синтетических наблюдений (Истина + Гауссов шум)
        def gauss(mu, sigma):
            u1 = random.random(); u2 = random.random()
            if u1 < 1e-12: u1 = 1e-12
            z = math.sqrt(-2.0*math.log(u1)) * math.cos(2*math.pi*u2)
            return mu + z * sigma

        meas_noise = [gauss(0.0, self.noise_std) for _ in range(3)]
        meas_pos = [self.pos[i] + meas_noise[i] for i in range(3)]

        return {
            "true_x": self.pos[0],
            "true_y": self.pos[1],
            "true_z": self.pos[2],
            "meas_x": meas_pos[0],
            "meas_y": meas_pos[1],
            "meas_z": meas_pos[2],
            "total_vx": true_vel_vector[0],
            "total_vy": true_vel_vector[1],
            "total_vz": true_vel_vector[2]
        }


dt = 0.1
sim = AUVSimulator(dt)
kf = LKF(dt, process_noise=0.5, meas_noise=2.0)

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
    # 1. Обновляем параметры симулятора (тяга, течение, уровень шума)
    sim.update_params(params.cmd_vx, params.cmd_vy, params.cmd_vz,
                      params.drift_x, params.drift_y, params.drift_z, params.noise)
    
    # 2. Шаг "реальной" физики
    sim_data = sim.step()
    
    # 3. Шаг Фильтра Калмана
    # a) Predict: Фильтр думает, куда объект переместится (исходя из пред. скорости)
    kf.predict()
    
    # b) Update: Фильтр корректирует предсказание на основе "шумного" измерения
    est = kf.update([sim_data["meas_x"], sim_data["meas_y"], sim_data["meas_z"]])
    
    return {
        "simulation": sim_data,
        "estimation": {
            "est_x": est[0], "est_y": est[1], "est_z": est[2],
            "est_vx": est[3], "est_vy": est[4], "est_vz": est[5]
        }
    }

if __name__ == "__main__":
    print("Starting Physics Engine & Kalman Filter...")
    uvicorn.run(app, host="127.0.0.1", port=8000)