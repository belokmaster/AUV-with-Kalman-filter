import uvicorn
import random
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def eye(n):
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def mat_copy(A):
    return [row[:] for row in A]


def mat_add(A, B):
    r, c = len(A), len(A[0])
    C = zeros(r, c)
    for i in range(r):
        for j in range(c):
            C[i][j] = A[i][j] + B[i][j]

    return C


def mat_sub(A, B):
    r, c = len(A), len(A[0])
    C = zeros(r, c)
    for i in range(r):
        for j in range(c):
            C[i][j] = A[i][j] - B[i][j]

    return C


def mat_mul(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])

    if n != n2:
        raise ValueError(f"Dimension mismatch: cols_A={n} != rows_B={n2}")

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
    r, c = len(A), len(A[0])

    T = zeros(c, r)
    for i in range(r):
        for j in range(c):
            T[j][i] = A[i][j]

    return T


def scalar_mult(A, s):
    r, c = len(A), len(A[0])

    B = zeros(r, c)
    for i in range(r):
        for j in range(c):
            B[i][j] = A[i][j] * s

    return B


def lup_decompose(A, epsilon=1e-12):
    n = len(A)

    if len(A[0]) != n:
        raise ValueError("Matrix must be square")

    LU = mat_copy(A)
    perm = list(range(n))

    for k in range(n):
        pivot, maxv = k, abs(LU[k][k])

        for i in range(k + 1, n):
            if abs(LU[i][k]) > maxv:
                maxv, pivot = abs(LU[i][k]), i

        if pivot != k:
            LU[k], LU[pivot] = LU[pivot], LU[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]

        if abs(LU[k][k]) <= epsilon:
            raise ValueError("Matrix is singular (close to zero pivot)")

        for i in range(k + 1, n):
            LU[i][k] /= LU[k][k]
            for j in range(k + 1, n):
                LU[i][j] -= LU[i][k] * LU[k][j]

    return LU, perm


def lup_solve(LU, perm, b):
    n = len(LU)
    pb = [b[perm[i]] for i in range(n)]
    y = [0.0] * n

    for i in range(n):
        s = pb[i]
        for k in range(i):
            s -= LU[i][k] * y[k]
        y[i] = s

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= LU[i][k] * x[k]
        x[i] = s / LU[i][i]

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


def symmetrize(M):
    Mt = transpose(M)
    return scalar_mult(mat_add(M, Mt), 0.5)


def vec_to_col(v):
    return [[x] for x in v]


def col_to_vec(col):
    return [row[0] for row in col]


class LKF:
    """
    Линейный KF с augmented state:
      x = [p_x,p_y,p_z,  v_x,v_y,v_z,  b_x,b_y,b_z]^T

    Модель динамики (дискретная, с линейным drag):
      a = (F - drag * v) / mass
      v_{k+1} = v_k + a * dt
      p_{k+1} = p_k + v_k*dt + 0.5 * a * dt^2
      b_{k+1} = b_k  (random walk)
    Управление: F (force vector, Н)
    IMU измеряет: z_imu = a + b + noise = - (drag/m) v + (1/m) F + b + noise
    Эхолокатор: z_pos = p + noise_pos
    """

    def __init__(self, dt, process_noise, meas_noise,
                 imu_noise=0.1, bias_walk=0.01, mass=15.0, drag_coeff=3.0):
        """
        dt: шаг
        process_noise: sigma_a (std шума ускорения) — для Q p/v-блока
        meas_noise: sigma_pos (std эхолокатора, позиция)
        imu_noise: sigma_imu (std акселерометра)
        bias_walk: std блуждания bias (в единицах акселератора) (м/с^2 / sqrt(s))
        mass, drag_coeff: модельные параметры (должны соответствовать симулятору)
        """
        self.dt = dt

        self.mass = mass
        self.drag = drag_coeff

        self.x = vec_to_col([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.F = zeros(9, 9)
        self.B = zeros(9, 3)
        self._build_FB(dt, mass, drag_coeff)

        # H_pos (3x9) для эхолокатора (позиция)
        self.H_pos = zeros(3, 9)
        for i in range(3):
            self.H_pos[i][i] = 1.0

        # H_imu (3x9) для IMU: будет = [0, -drag/m * I3, I3]
        self.H_imu = zeros(3, 9)
        coef = - (drag_coeff / mass)
        for i in range(3):
            self.H_imu[i][3 + i] = coef  # maps v -> -drag/m * v
            self.H_imu[i][6 + i] = 1.0  # maps bias -> b

        # начальная ковариация
        self.P = scalar_mult(eye(9), 1e2)  # более крупная неопределённость

        # Q: процессный шум
        # верхний-left 6x6 как ранее (p,v) из sigma_a
        # bias block (3x3) = (bias_walk^2 * dt) * I3
        self.sigma_a = process_noise
        self.bias_walk = bias_walk
        self.Q = self._build_Q(process_noise, bias_walk, dt)

        # измерительные ковариации
        self.R_pos = scalar_mult(eye(3), meas_noise * meas_noise)  # эхолокатор (побольше)
        self.R_imu = scalar_mult(eye(3), imu_noise * imu_noise)  # IMU шума (меньше)

    def _build_FB(self, dt, mass, drag):
        """
        Строим F и B по выводам:
        p_{k+1} = p_k + alpha_p * v_k + beta_p * F
        v_{k+1} = alpha_v * v_k + beta_v * F
        b_{k+1} = b_k
        где
          alpha_v = 1 - (drag/mass)*dt
          beta_v  = dt / mass
          alpha_p = dt - 0.5*(drag/mass)*dt^2
          beta_p  = 0.5 * dt^2 / mass
        """
        # нули
        self.F = zeros(9, 9)
        self.B = zeros(9, 3)
        alpha_v = 1.0 - (drag / mass) * dt
        beta_v = dt / mass
        alpha_p = dt - 0.5 * (drag / mass) * (dt * dt)
        beta_p = 0.5 * (dt * dt) / mass

        # p-rows
        for i in range(3):
            self.F[i][i] = 1.0
            self.F[i][3 + i] = alpha_p  # coefficient on v_k
            # B for p
            self.B[i][i] = beta_p

        # v-rows
        for i in range(3):
            self.F[3 + i][3 + i] = alpha_v
            self.B[3 + i][i] = beta_v

        # bias-rows (random walk)
        for i in range(3):
            self.F[6 + i][6 + i] = 1.0
            # no B for bias

    def _build_Q(self, sigma_a, bias_walk, dt):
        """
        Построение Q (9x9):
         - для (p,v) используем структуру, зависящую от sigma_a (как ранее)
         - для bias: variance = bias_walk^2 * dt (random walk дискретизация)
         - cross-terms 0
        """
        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        dt4 = dt2 * dt2
        a = dt4 / 4.0
        b = dt3 / 2.0
        c = dt2
        s2 = sigma_a * sigma_a
        Q = zeros(9, 9)
        for axis in range(3):
            pi = axis
            vi = 3 + axis
            Q[pi][pi] = a * s2
            Q[pi][vi] = b * s2
            Q[vi][pi] = b * s2
            Q[vi][vi] = c * s2
        # bias block
        bias_var = (bias_walk * bias_walk) * self.dt
        for i in range(3):
            Q[6 + i][6 + i] = bias_var
        return Q

    def predict(self, force_cmd=None):
        """
        Прогноз состояния: x = F x + B force_cmd
        force_cmd: [Fx,Fy,Fz] в Н (если None — считаем нулевым)
        """
        if force_cmd is None:
            force_cmd = [0.0, 0.0, 0.0]
        xu = mat_mul(self.F, self.x)  # 9x1
        bu = mat_mul(self.B, vec_to_col(force_cmd))  # 9x1
        self.x = mat_add(xu, bu)
        # P = F P F^T + Q
        self.P = mat_add(mat_mul(mat_mul(self.F, self.P), transpose(self.F)), self.Q)
        self.P = symmetrize(self.P)
        return col_to_vec(self.x)

    def update_imu(self, imu_meas, force_cmd):
        """
        Обновление по IMU (высокая частота).
        imu_meas: [ax_meas, ay_meas, az_meas] (в м/с^2)
        force_cmd: [Fx,Fy,Fz] (в Н) — нужен, тк IMU измерение содержит + (1/m) * F
        Модель измерения:
          imu_meas - (1/m) * F = H_imu * x + noise
        где H_imu = [0, -drag/m I3, I3]
        """
        # корректируем измерение вычитая известную часть (1/m)*F
        F_term = [(1.0 / self.mass) * f for f in force_cmd]
        z_tilde = [imu_meas[i] - F_term[i] for i in range(3)]

        z = vec_to_col(z_tilde)  # 3x1
        y = mat_sub(z, mat_mul(self.H_imu, self.x))  # 3x1

        S = mat_add(mat_mul(mat_mul(self.H_imu, self.P), transpose(self.H_imu)), self.R_imu)
        PHt = mat_mul(self.P, transpose(self.H_imu))  # 9x3

        S_inv = invert_matrix(S)
        K = mat_mul(PHt, S_inv)  # 9x3

        self.x = mat_add(self.x, mat_mul(K, y))

        # Joseph form
        I9 = eye(9)
        KH = mat_mul(K, self.H_imu)
        temp = mat_sub(I9, KH)
        term1 = mat_mul(mat_mul(temp, self.P), transpose(temp))
        KRKt = mat_mul(mat_mul(K, self.R_imu), transpose(K))
        self.P = mat_add(term1, KRKt)
        self.P = symmetrize(self.P)
        return col_to_vec(self.x)

    def update_pos(self, z_meas):
        """
        Обновление по позиции (эхо/сонар) — редкое измерение.
        z_meas: [px,py,pz]
        """
        z = vec_to_col(z_meas)
        y = mat_sub(z, mat_mul(self.H_pos, self.x))  # 3x1

        S = mat_add(mat_mul(mat_mul(self.H_pos, self.P), transpose(self.H_pos)), self.R_pos)
        PHt = mat_mul(self.P, transpose(self.H_pos))  # 9x3

        S_inv = invert_matrix(S)
        K = mat_mul(PHt, S_inv)  # 9x3

        self.x = mat_add(self.x, mat_mul(K, y))

        # Joseph form
        I9 = eye(9)
        KH = mat_mul(K, self.H_pos)
        temp = mat_sub(I9, KH)
        term1 = mat_mul(mat_mul(temp, self.P), transpose(temp))
        KRKt = mat_mul(mat_mul(K, self.R_pos), transpose(K))
        self.P = mat_add(term1, KRKt)
        self.P = symmetrize(self.P)
        return col_to_vec(self.x)


class AUVSimulator:
    def __init__(self, dt):
        self.dt = dt
        self.pos = [0.0, 0.0, 5.0]
        self.vel = [0.0, 0.0, 0.0]
        self.force_cmd = [0.0, 0.0, 0.0]
        self.drift_vel = [0.0, 0.0, 0.0]
        self.noise_std = 2.0            # sigma для позиционного эхолокатора
        self.radius = 1.0
        self.mass = 15.0
        self.drag_coeff = 3.0
        self.thrust_factor = 8.0

        # IMU параметры (реалистичный акселерометр)
        self.imu_noise_std = 0.1         # std шума акселерометра (м/с^2)
        self.imu_bias = [0.0, 0.0, 0.0]  # начальный bias (м/с^2)
        self.imu_bias_walk_std = 0.01    # std random-walk bias (м/с^2 / sqrt(s))

        # счётчик шагов (можно использовать для редких обновлений)
        self.step_count = 0

    def update_params(self, vx, vy, vz, dx, dy, dz, noise,
                      imu_noise_std=None, imu_bias_walk_std=None):
        # force_cmd в Ньютонах
        self.force_cmd = [vx * self.thrust_factor, vy * self.thrust_factor, vz * self.thrust_factor]
        self.drift_vel = [dx, dy, dz]
        self.noise_std = noise
        if imu_noise_std is not None:
            self.imu_noise_std = imu_noise_std
        if imu_bias_walk_std is not None:
            self.imu_bias_walk_std = imu_bias_walk_std

    def step(self):
        self.step_count += 1
        true_vel_vector = [0.0] * 3
        acc_vector = [0.0] * 3
        for i in range(3):
            f_drag = -self.drag_coeff * self.vel[i]
            f_net = self.force_cmd[i] + f_drag
            acc = f_net / self.mass
            acc_vector[i] = acc
            self.vel[i] += acc * self.dt
            v_total = self.vel[i] + self.drift_vel[i]
            true_vel_vector[i] = v_total
            self.pos[i] += v_total * self.dt

        if self.pos[2] < self.radius:
            self.pos[2] = self.radius
            if self.vel[2] < 0:
                self.vel[2] = 0.0

        # --- bias random-walk (масштабируем через sqrt(dt)) ---
        for i in range(3):
            u1 = random.random()
            u2 = random.random()
            if u1 < 1e-12: u1 = 1e-12
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            self.imu_bias[i] += z * self.imu_bias_walk_std * math.sqrt(self.dt)

        # Функция для генерации гаусс. шума
        def gauss(mu, sigma):
            u1 = random.random()
            u2 = random.random()
            if u1 < 1e-12: u1 = 1e-12
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return mu + z * sigma

        # --- IMU измерение: true acc + bias + noise ---
        imu_acc = [0.0] * 3
        for i in range(3):
            imu_acc[i] = acc_vector[i] + self.imu_bias[i] + gauss(0.0, self.imu_noise_std)

        # --- измерение позиции (эхо) — шумное ---
        meas_noise = [gauss(0.0, self.noise_std) for _ in range(3)]
        meas_pos = [self.pos[i] + meas_noise[i] for i in range(3)]

        return {
            "true_x": self.pos[0], "true_y": self.pos[1], "true_z": self.pos[2],
            "meas_x": meas_pos[0], "meas_y": meas_pos[1], "meas_z": meas_pos[2],
            "total_vx": true_vel_vector[0], "total_vy": true_vel_vector[1], "total_vz": true_vel_vector[2],
            "acc_x": acc_vector[0], "acc_y": acc_vector[1], "acc_z": acc_vector[2],
            # возвращаем IMU как список (ключ "imu_acc")
            "imu_acc": imu_acc,
            # для отладки можно вернуть bias отдельно
            "imu_bias": list(self.imu_bias)
        }


app = FastAPI()

DT = 0.1
sim = AUVSimulator(DT)

# фильтр инициализируется текущим уровнем шума эхолокатора
kf = LKF(DT, process_noise=0.5, meas_noise=sim.noise_std)


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
    sim.update_params(
        params.cmd_vx, params.cmd_vy, params.cmd_vz,
        params.drift_x, params.drift_y, params.drift_z,
        params.noise
    )

    sim_data = sim.step()

    # 1) predict: используем текущую силу (force_cmd в Н)
    kf.predict(force_cmd=sim.force_cmd)

    # 2) update по IMU (высокая частота) — IMU noisy измерение
    # передаём сам imu_meas и текущую силу (KF вычитает 1/m * F внутри)
    kf.update_imu(sim_data["imu_acc"], sim.force_cmd)

    # 3) update по эхолокатору (позиция) — редкое, но в этом endpoint делаем сразу
    est = kf.update_pos([
        sim_data["meas_x"],
        sim_data["meas_y"],
        sim_data["meas_z"]
    ])

    return {
        "simulation": sim_data,
        "estimation": {
            "est_x": est[0], "est_y": est[1], "est_z": est[2],
            "est_vx": est[3], "est_vy": est[4], "est_vz": est[5]
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            sim.update_params(
                vx=float(data.get("cmd_vx", 0.0)),
                vy=float(data.get("cmd_vy", 0.0)),
                vz=float(data.get("cmd_vz", 0.0)),
                dx=float(data.get("drift_x", 0.0)),
                dy=float(data.get("drift_y", 0.0)),
                dz=float(data.get("drift_z", 0.0)),
                noise=float(data.get("noise", sim.noise_std))
            )

            sim_data = sim.step()

            # 1) predict по текущей силе
            kf.predict(force_cmd=sim.force_cmd)

            # 2) update IMU (высокая частота)
            kf.update_imu(sim_data["imu_acc"], sim.force_cmd)

            # 3) update position (можно делать реже — здесь делаем каждый шаг)
            est = kf.update_pos([
                sim_data["meas_x"],
                sim_data["meas_y"],
                sim_data["meas_z"]
            ])

            await websocket.send_json({
                "simulation": sim_data,
                "estimation": {
                    "est_x": est[0], "est_y": est[1], "est_z": est[2],
                    "est_vx": est[3], "est_vy": est[4], "est_vz": est[5]
                }
            })

    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        print(f"Ошибка в WebSocket: {e}")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Сервер запущен: http://127.0.0.1:8000")

    uvicorn.run(app, host="127.0.0.1", port=8000)


