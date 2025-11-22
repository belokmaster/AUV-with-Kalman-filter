from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random
import math

app = FastAPI()

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
    r = len(A); c = len(A[0])
    C = zeros(r, c)
    for i in range(r):
        for j in range(c):
            C[i][j] = A[i][j] + B[i][j]
    return C

def mat_sub(A, B):
    r = len(A); c = len(A[0])
    C = zeros(r, c)
    for i in range(r):
        for j in range(c):
            C[i][j] = A[i][j] - B[i][j]
    return C

def mat_mul(A, B):
    # A: m x n, B: n x p -> C: m x p
    m = len(A); n = len(A[0]); n2 = len(B); p = len(B[0])
    assert n == n2, "mat_mul dims"
    C = zeros(m, p)
    for i in range(m):
        ai = A[i]
        for k in range(n):
            aik = ai[k]
            bk = B[k]
            for j in range(p):
                C[i][j] += aik * bk[j]
    return C

def mat_vec_mul(A, v):
    # A: m x n, v: n x 1 (list)
    m = len(A); n = len(A[0])
    assert n == len(v)
    res = [0.0]*m
    for i in range(m):
        s = 0.0
        ai = A[i]
        for j in range(n):
            s += ai[j] * v[j]
        res[i] = s
    return res

def transpose(A):
    r = len(A); c = len(A[0])
    T = zeros(c, r)
    for i in range(r):
        for j in range(c):
            T[j][i] = A[i][j]
    return T

def scalar_mult(A, s):
    r = len(A); c = len(A[0])
    B = zeros(r, c)
    for i in range(r):
        for j in range(c):
            B[i][j] = A[i][j] * s
    return B

#  LUP-разложение 
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
            raise ValueError("Matrix is singular to working precision")
        if pivot != k:
            LU[k], LU[pivot] = LU[pivot], LU[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
        # Eliminate
        for i in range(k+1, n):
            LU[i][k] = LU[i][k] / LU[k][k]
            for j in range(k+1, n):
                LU[i][j] -= LU[i][k] * LU[k][j]
    return LU, perm

def lup_solve(LU, perm, b):
    # Solve LU x = Pb. b can be a vector (len n) or matrix n x m (list of rows)
    n = len(LU)
    # apply permutation to b
    if isinstance(b[0], list):
        # matrix
        m = len(b[0])
        pb = [b[perm[i]][:] for i in range(n)]
    else:
        # vector
        pb = [b[perm[i]] for i in range(n)]

    # forward solve Ly = pb
    if isinstance(pb[0], list):
        m = len(pb[0])
        y = zeros(n, m)
        for i in range(n):
            for j in range(m):
                s = pb[i][j]
                for k in range(i):
                    s -= LU[i][k] * y[k][j]
                y[i][j] = s
    else:
        y = [0.0]*n
        for i in range(n):
            s = pb[i]
            for k in range(i):
                s -= LU[i][k] * y[k]
            y[i] = s

    # backward solve Ux = y
    if isinstance(y[0], list):
        m = len(y[0])
        x = zeros(n, m)
        for i in range(n-1, -1, -1):
            for j in range(m):
                s = y[i][j]
                for k in range(i+1, n):
                    s -= LU[i][k] * x[k][j]
                x[i][j] = s / LU[i][i]
    else:
        x = [0.0]*n
        for i in range(n-1, -1, -1):
            s = y[i]
            for k in range(i+1, n):
                s -= LU[i][k] * x[k]
            x[i] = s / LU[i][i]
    return x

def invert_matrix(A):
    # A: n x n, return inverse n x n using LUP
    n = len(A)
    LU, perm = lup_decompose(A)
    # build identity as matrix of columns for RHS
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    # lup_solve expects b as rows; we need to pass columns transposed; easier: solve for each column of identity
    invA = zeros(n, n)
    for col in range(n):
        # build b vector as column of I
        b = [I[i][col] for i in range(n)]
        x = lup_solve(LU, perm, b)  # vector
        for i in range(n):
            invA[i][col] = x[i]
    return invA

# -------------------------
#  Вспомогательные преобразования (векторы <-> матрицы)
# -------------------------
def vec_to_col(v):
    return [[x] for x in v]

def col_to_vec(col):
    return [row[0] for row in col]

# -------------------------
#  LKF (ручной)
# -------------------------
class LKF:
    def __init__(self, dt, process_noise, meas_noise):
        self.dt = dt
        # state vector as column (6x1)
        self.x = vec_to_col([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])  # start z=5
        # F 6x6
        self.F = eye(6)
        self.F[0][3] = dt
        self.F[1][4] = dt
        self.F[2][5] = dt
        # H 3x6
        self.H = zeros(3, 6)
        self.H[0][0] = 1.0
        self.H[1][1] = 1.0
        self.H[2][2] = 1.0
        # Covariances
        self.P = scalar_mult(eye(6), 100.0)
        self.Q = scalar_mult(eye(6), process_noise)
        self.R = scalar_mult(eye(3), meas_noise)

    def predict(self):
        # x = F x
        self.x = mat_mul(self.F, self.x)  # 6x6 * 6x1 -> 6x1
        # P = F P F^T + Q
        self.P = mat_add(mat_mul(mat_mul(self.F, self.P), transpose(self.F)), self.Q)
        # floor on z
        if self.x[2][0] < 1.0:
            self.x[2][0] = 1.0
        return col_to_vec(self.x)

    def update(self, z_meas):
        # z_meas: list len 3
        z = vec_to_col(z_meas)  # 3x1
        # y = z - H x
        Hx = mat_mul(self.H, self.x)  # 3x1
        y = mat_sub(z, Hx)  # 3x1
        # S = H P H^T + R  (3x3)
        S = mat_add(mat_mul(mat_mul(self.H, self.P), transpose(self.H)), self.R)
        # PHt = P H^T  (6x3)
        PHt = mat_mul(self.P, transpose(self.H))
        # invS via LUP
        invS = invert_matrix(S)  # 3x3
        # K = PHt * invS  (6x3)
        K = mat_mul(PHt, invS)
        # x = x + K y
        Ky = mat_mul(K, y)  # 6x1
        self.x = mat_add(self.x, Ky)
        # P = (I - K H) P
        I6 = eye(6)
        KH = mat_mul(K, self.H)  # 6x6
        IminusKH = mat_sub(I6, KH)
        self.P = mat_mul(IminusKH, self.P)
        return col_to_vec(self.x)

# -------------------------
#  AUV Simulator (без numpy)
# -------------------------
class AUVSimulator:
    def __init__(self, dt):
        self.dt = dt
        self.true_pos = [0.0, 0.0, 5.0]  # x,y,z
        self.velocity_cmd = [0.0, 0.0, 0.0]
        self.current_drift = [0.0, 0.0, 0.0]
        self.noise_std = 2.0
        self.radius = 1.0

    def step(self):
        total_vel = [self.velocity_cmd[i] + self.current_drift[i] for i in range(3)]
        new_pos = [self.true_pos[i] + total_vel[i]*self.dt for i in range(3)]
        # collision with bottom
        if new_pos[2] < self.radius:
            new_pos[2] = self.radius
            if total_vel[2] < 0:
                total_vel[2] = 0.0
        self.true_pos = new_pos
        # gaussian noise (Box-Muller)
        def gauss(mu, sigma):
            # simple Box-Muller
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2.0*math.log(max(u1, 1e-12))) * math.cos(2*math.pi*u2)
            return mu + z0 * sigma
        noise = [gauss(0.0, self.noise_std) for _ in range(3)]
        measured_pos = [self.true_pos[i] + noise[i] for i in range(3)]
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
        self.velocity_cmd = [vx, vy, vz]
        self.current_drift = [dx, dy, dz]
        self.noise_std = noise

# -------------------------
#  Инстансы и FastAPI endpoint (API тот же)
# -------------------------
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
