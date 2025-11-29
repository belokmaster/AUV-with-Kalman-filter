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


def vec_to_col(v):
    return [[x] for x in v]


def col_to_vec(col):
    return [row[0] for row in col]


class LKF:
    def __init__(self, dt, process_noise, meas_noise):
        self.dt = dt
        self.x = vec_to_col([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
        self.F = eye(6)
        self.F[0][3] = dt
        self.F[1][4] = dt
        self.F[2][5] = dt
        self.H = zeros(3, 6)
        self.H[0][0], self.H[1][1], self.H[2][2] = 1.0, 1.0, 1.0
        self.P = scalar_mult(eye(6), 50.0)
        self.Q = scalar_mult(eye(6), process_noise)
        self.R = scalar_mult(eye(3), meas_noise)


    def predict(self):
        self.x = mat_mul(self.F, self.x)
        self.P = mat_add(mat_mul(mat_mul(self.F, self.P), transpose(self.F)), self.Q)
        if self.x[2][0] < 0.0:
            self.x[2][0] = 0.0
        return col_to_vec(self.x)


    def update(self, z_meas):
        z = vec_to_col(z_meas)
        y = mat_sub(z, mat_mul(self.H, self.x))
        S = mat_add(mat_mul(mat_mul(self.H, self.P), transpose(self.H)), self.R)
        K = mat_mul(mat_mul(self.P, transpose(self.H)), invert_matrix(S))
        self.x = mat_add(self.x, mat_mul(K, y))
        self.P = mat_mul(mat_sub(eye(6), mat_mul(K, self.H)), self.P)
        return col_to_vec(self.x)


class AUVSimulator:
    def __init__(self, dt):
        self.dt = dt
        self.pos = [0.0, 0.0, 5.0]
        self.vel = [0.0, 0.0, 0.0]
        self.force_cmd = [0.0, 0.0, 0.0]
        self.drift_vel = [0.0, 0.0, 0.0]
        self.noise_std = 2.0
        self.radius = 1.0
        self.mass = 15.0
        self.drag_coeff = 3.0
        self.thrust_factor = 8.0


    def update_params(self, vx, vy, vz, dx, dy, dz, noise):
        self.force_cmd = [vx * self.thrust_factor, vy * self.thrust_factor, vz * self.thrust_factor]
        self.drift_vel = [dx, dy, dz]
        self.noise_std = noise


    def step(self):
        true_vel_vector = [0.0] * 3
        for i in range(3):
            f_drag = -self.drag_coeff * self.vel[i]
            f_net = self.force_cmd[i] + f_drag
            acc = f_net / self.mass
            self.vel[i] += acc * self.dt
            v_total = self.vel[i] + self.drift_vel[i]
            true_vel_vector[i] = v_total
            self.pos[i] += v_total * self.dt
        if self.pos[2] < self.radius:
            self.pos[2] = self.radius
            if self.vel[2] < 0:
                self.vel[2] = 0.0


        def gauss(mu, sigma):
            u1 = random.random()
            u2 = random.random()
            if u1 < 1e-12: u1 = 1e-12
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return mu + z * sigma


        meas_noise = [gauss(0.0, self.noise_std) for _ in range(3)]
        meas_pos = [self.pos[i] + meas_noise[i] for i in range(3)]
        return {
            "true_x": self.pos[0], "true_y": self.pos[1], "true_z": self.pos[2],
            "meas_x": meas_pos[0], "meas_y": meas_pos[1], "meas_z": meas_pos[2],
            "total_vx": true_vel_vector[0], "total_vy": true_vel_vector[1], "total_vz": true_vel_vector[2]
        }


app = FastAPI()

DT = 0.1
sim = AUVSimulator(DT)
kf = LKF(DT, process_noise=0.5, meas_noise=2.0)


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
                      params.drift_x, params.drift_y, params.drift_z,
                      params.noise)
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
                noise=float(data.get("noise", 2.0))
            )
            sim_data = sim.step()
            kf.predict()
            est = kf.update([sim_data["meas_x"], sim_data["meas_y"], sim_data["meas_z"]])
            response = {
                "simulation": sim_data,
                "estimation": {
                    "est_x": est[0], "est_y": est[1], "est_z": est[2],
                    "est_vx": est[3], "est_vy": est[4], "est_vz": est[5]
                }
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        print(f"Ошибка в WebSocket: {e}")


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    print("Сервер запущен: http://127.0.0.1:8000")

    uvicorn.run(app, host="127.0.0.1", port=8000)


