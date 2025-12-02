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

# PA = LU
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

# решение Ax = b через LUP разложение
# PA = LU
# Сначала решается Ly=Pb, после Ux=y
def lup_solve(LU, perm, b):
    n = len(LU)
    pb = [b[perm[i]] for i in range(n)]

    # Ly = Pb
    y = [0.0] * n
    for i in range(n):
        s = pb[i]
        for k in range(i):
            s -= LU[i][k] * y[k]
        y[i] = s

    # решаем Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= LU[i][k] * x[k]
        x[i] = s / LU[i][i]

    return x

# обратная матрица через LUP разложение
# AX=I
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

# симметризирует матрицу для избежения ошибок
def symmetrize(M):
    Mt = transpose(M)
    return scalar_mult(mat_add(M, Mt), 0.5)


def vec_to_col(v):
    return [[x] for x in v]


def col_to_vec(col):
    return [row[0] for row in col]


class LKF:
    """
    Линейный Калмановский фильтр для состояния:
        x = [p_x,p_y,p_z,  v_x,v_y,v_z,  b_x,b_y,b_z]^T
        p - позиция, v - скорость, b - смещение акселерометра (bias)

    Дискретная модель движения с линейным сопротивлением (drag):
        a = (F - drag * v) / mass

        v_{k+1} = v_k + a * dt
        p_{k+1} = p_k + v_k*dt + 0.5 * a * dt^2
        b_{k+1} = b_k  (random walk)

    Два типа датчиков:
        1. IMU (Inertial Measurement Unit) - акселерометр. Он измеряет ускорение.
        - Модель измерения IMU: z_imu = a_истинное + b + шум_imu
        - Подставляя `a`, получаем: z_imu = (F/m - (drag/m)*v) + b + шум_imu
        2. Эхолокатор (Sonar/Echosounder) - измеряет позицию.
        - Модель измерения: z_pos = p_истинное + шум_pos
    
    Фильтр Калмана использует эти модели для получения оптимальной оценки состояния `x`
    """

    def __init__(self, dt, process_noise, meas_noise,
                 imu_noise=0.1, bias_walk=0.01, mass=15.0, drag_coeff=3.0):
        """
        Параметры:
            - dt: шаг
            - process_noise (sigma_a): стандартное отклонение шума процесса (немоделируемые ускорения).
            - meas_noise (sigma_pos): стандартное отклонение шума измерения эхолокатора.
            - imu_noise (sigma_imu): стандартное отклонение шума IMU.
            - bias_walk: стандартное отклонение случайного блуждания смещения (bias).
            - mass, drag_coeff: физические параметры модели.
        """

        self.dt = dt

        self.mass = mass # масса громозяки
        self.drag = drag_coeff # коэффицент сопротивления воды

        self.x = vec_to_col([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.F = zeros(9, 9) # F: [9x9] - Матрица перехода состояния.
        self.B = zeros(9, 3) # B: [9x3] - Матрица управления.
        self._build_FB(dt, mass, drag_coeff)

        # H_pos: [3x9] - Матрица для измерения позиции. Связывает состояние `x` с измерением `z_pos`.
        # z_pos = H_pos * x. Так как измеряем только позицию, H_pos = [I_3x3 | 0_3x3 | 0_3x3]
        self.H_pos = zeros(3, 9)
        for i in range(3):
            self.H_pos[i][i] = 1.0

        # IMU это акселерометр
        # H_imu: [3x9] - Матрица для измерения IMU.
        self.H_imu = zeros(3, 9)
        coef = - (drag_coeff / mass)
        for i in range(3):
            self.H_imu[i][3 + i] = coef  # maps v -> -drag/m * v
            self.H_imu[i][6 + i] = 1.0  # maps bias -> b

        # P: [9x9] - Ковариационная матрица ошибки оценки.
        # Показывает нашу "уверенность" в оценке `x`. Большие значения на диагонали
        # означают высокую неопределенность.
        self.P = scalar_mult(eye(9), 1e2)  

        # Q: [9x9] - Ковариационная матрица шума процесса.
        # верхний-left 6x6 как ранее (p,v) из sigma_a
        # bias block (3x3) = (bias_walk^2 * dt) * I3
        self.sigma_a = process_noise
        self.bias_walk = bias_walk
        self.Q = self._build_Q(process_noise, bias_walk, dt)

        # R: [3x3] - Ковариационные матрицы шума измерений.
        # R_pos: [3x3] - для эхолокатора.
        self.R_pos = scalar_mult(eye(3), meas_noise * meas_noise)  # эхолокатор (побольше)
        # R_imu: [3x3] - для IMU. акселерометр
        self.R_imu = scalar_mult(eye(3), imu_noise * imu_noise)  # IMU шума (меньше)

    def _build_FB(self, dt, mass, drag):
        """
        Построение матриц F и B, которые описывают детерминированную часть модели движения.
            x_k = F * x_{k-1} + B * u_{k-1}

        Вывод формул из непрерывной модели:
            a = (F_u - drag * v) / mass
            p_dot = v
            v_dot = a

        Дискретизация дает:
          v_k = v_{k-1} + ( (F_u / m) - (drag/m)*v_{k-1} )*dt = (1 - (drag/m)*dt)*v_{k-1} + (dt/m)*F_u
          p_k = p_{k-1} + v_{k-1}*dt + 0.5*a*dt^2
              = p_{k-1} + v_{k-1}*dt + 0.5*((F_u/m) - (drag/m)*v_{k-1})*dt^2
              = p_{k-1} + (dt - 0.5*(drag/m)*dt^2)*v_{k-1} + (0.5*dt^2/m)*F_u
          
          b_k = b_{k-1} (смешение меняется только за счет шума, не управления)
        
        Отсюда получаем коэффициенты для матриц F и B.
        """
        # нули
        self.F = zeros(9, 9)
        self.B = zeros(9, 3)
        
        # Коэффициенты, выведенные выше
        alpha_v = 1.0 - (drag / mass) * dt
        beta_v = dt / mass
        alpha_p = dt - 0.5 * (drag / mass) * (dt * dt)
        beta_p = 0.5 * (dt * dt) / mass

        # Заполняем F (матрица перехода) и B (матрица управления)
        # Блок для позиции (p_k = 1*p_{k-1} + alpha_p*v_{k-1} + beta_p*F_u)
        for i in range(3):
            self.F[i][i] = 1.0
            self.F[i][3 + i] = alpha_p  # p зависит от v
            self.B[i][i] = beta_p # p зависит от F_u

        # Блок для скорости (v_k = alpha_v*v_{k-1} + beta_v*F_u)
        for i in range(3):
            self.F[3 + i][3 + i] = alpha_v # v зависит от v
            self.B[3 + i][i] = beta_v # v зависит от F_u

        # Блок для смещения (b_k = 1*b_{k-1})
        for i in range(3):
            self.F[6 + i][6 + i] = 1.0


    def _build_Q(self, sigma_a, bias_walk, dt):
        """
        Построение матрицы ковариации шума процесса Q.
        Эта матрица моделирует неопределенность, которая добавляется на каждом шаге
        из-за неточностей модели.

        - Блок для (p, v) выводится из предположения о постоянном случайном ускорении
          с дисперсией sigma_a^2 в течение интервала dt.
        - Блок для смещения (bias) моделируется как случайное блуждание (random walk),
          его дисперсия растет как (bias_walk^2 * dt).
        """

        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        dt4 = dt2 * dt2

        # Коэффициенты для p,v блока
        a = dt4 / 4.0
        b = dt3 / 2.0
        c = dt2
        s2 = sigma_a * sigma_a

        Q = zeros(9, 9)
        for axis in range(3):
            pi = axis
            vi = 3 + axis

            # Cov(p, p) = (dt^4/4) * sigma_a^2
            Q[pi][pi] = a * s2
            # Cov(p, v) = (dt^3/2) * sigma_a^2
            Q[pi][vi] = b * s2
            # Cov(v, p) = Cov(p, v)
            Q[vi][pi] = b * s2
            # Cov(v, v) = (dt^2) * sigma_a^2
            Q[vi][vi] = c * s2

        # Заполняем диагональный блок для смещения (bias)
        bias_var = (bias_walk * bias_walk) * self.dt
        for i in range(3):
            Q[6 + i][6 + i] = bias_var

        return Q

    def predict(self, force_cmd=None):
        """
        На этом шаге мы предсказываем состояние и его неопределенность на следующий момент времени,
        используя только нашу модель движения (без новых измерений).
        
        1. Прогноз состояния (State Prediction):
           x_k|k-1 = F * x_{k-1|k-1} + B * u_{k-1}
           - x_k|k-1: прогноз состояния на шаг k, основанный на данных до шага k-1.
           - x_{k-1|k-1}: оценка состояния на предыдущем шаге.
           - u: вектор управления (сила).

        2. Прогноз ковариации ошибки (Error Covariance Prediction):
           P_k|k-1 = F * P_{k-1|k-1} * F^T + Q
           - P_k|k-1: прогнозируемая ковариация.
           - P_{k-1|k-1}: ковариация на предыдущем шаге.
           - F * P * F^T: как неопределенность распространяется через нашу модель.
           - Q: добавляем неопределенность из-за неточности модели.
        """

        if force_cmd is None:
            force_cmd = [0.0, 0.0, 0.0]
        
        # 1. Прогноз состояния: x_k|k-1 = F * x + B * u
        # self.x: [9x1], F:[9x9], B:[9x3], u:[3x1]
        xu = mat_mul(self.F, self.x)  # 9x1
        bu = mat_mul(self.B, vec_to_col(force_cmd))  # 9x1
        self.x = mat_add(xu, bu)

        # 2. Прогноз ковариации: P_k|k-1 = F * P * F^T + Q
        # P: [9x9], F:[9x9], F^T:[9x9], Q:[9x9]
        self.P = mat_add(mat_mul(mat_mul(self.F, self.P), transpose(self.F)), self.Q)
        
        # Симметризуем P для численной стабильности
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

        # Модель измерения IMU: z_imu = (-drag/m * v + b) + (1/m) * F + noise
        # Нам нужно "очистить" измерение от известного управляющего воздействия.
        # z_tilde = z_imu - (1/m)*F = H_imu * x_истинное + noise
        F_term = [(1.0 / self.mass) * f for f in force_cmd]
        z_tilde = [imu_meas[i] - F_term[i] for i in range(3)]

        z = vec_to_col(z_tilde)  # z: [3x1] - очищенное измерение

        # 1. Вычисление невязки (Innovation):
        #    y = z - H * x_k|k-1
        #    y: [3x1] - разница между реальным измерением и предсказанным.
        #    H_imu: [3x9], x: [9x1]
        y = mat_sub(z, mat_mul(self.H_imu, self.x))  # 3x1

        # 2. Ковариация невязки (Innovation Covariance):
        #    S = H * P_k|k-1 * H^T + R
        #    S: [3x3] - неопределенность нашего предсказанного измерения.
        #    P: [9x9], H_imu^T: [9x3], R_imu: [3x3]
        S = mat_add(mat_mul(mat_mul(self.H_imu, self.P), transpose(self.H_imu)), self.R_imu)

        # 3. Вычисление коэффициента Калмана (Kalman Gain):
        #    K = P_k|k-1 * H^T * S^{-1}
        #    K: [9x3] - "оптимальный вес", который определяет, насколько сильно мы доверяем
        #    новому измерению по сравнению с нашим прогнозом.
        PHt = mat_mul(self.P, transpose(self.H_imu)) 
        S_inv = invert_matrix(S)
        K = mat_mul(PHt, S_inv)  # 9x3

        # 4. Обновление оценки состояния (State Update):
        #    x_k|k = x_k|k-1 + K * y
        #    Корректируем наш прогноз на основе невязки, взвешенной с помощью K.
        self.x = mat_add(self.x, mat_mul(K, y))

        # 5. Обновление ковариации ошибки (Covariance Update) - ФОРМА ЙОЗЕФА:
        #    P_k|k = (I - K*H) * P_k|k-1 * (I - K*H)^T + K * R * K^T
        #    Эта форма более численно устойчива, чем стандартная P = (I - K*H)*P,
        #    так как она гарантирует, что матрица P останется симметричной и
        #    положительно определенной, предотвращая расходимость фильтра из-за ошибок округления.
        I9 = eye(9)
        KH = mat_mul(K, self.H_imu)
        temp = mat_sub(I9, KH)
        term1 = mat_mul(mat_mul(temp, self.P), transpose(temp))
        KRKt = mat_mul(mat_mul(K, self.R_imu), transpose(K))
        self.P = mat_add(term1, KRKt)

        # Снова симметризуем для надежности
        self.P = symmetrize(self.P)

        return col_to_vec(self.x)

    def update_pos(self, z_meas):
        """
        Этот метод полностью аналогичен `update_imu`, но использует другие матрицы:
        - Матрицу измерения H_pos вместо H_imu.
        - Ковариацию шума измерения R_pos вместо R_imu.
        Все шаги (вычисление невязки, S, K, обновление x и P) идентичны по своей логике.
        """

        z = vec_to_col(z_meas) # z: [3x1] - измерение позиции

        # 1. Невязка: y = z - H_pos * x
        y = mat_sub(z, mat_mul(self.H_pos, self.x))  # 3x1

        # 2. Ковариация невязки: S = H_pos * P * H_pos^T + R_pos
        S = mat_add(mat_mul(mat_mul(self.H_pos, self.P), transpose(self.H_pos)), self.R_pos)

        # 3. Коэффициент Калмана: K = P * H_pos^T * S^{-1}
        PHt = mat_mul(self.P, transpose(self.H_pos))  # 9x3
        S_inv = invert_matrix(S)
        K = mat_mul(PHt, S_inv)  # 9x3

        # 4. Обновление состояния: x = x + K * y
        self.x = mat_add(self.x, mat_mul(K, y))

        # 5. Обновление ковариации (Форма Йозефа)
        #    P_k|k = (I - K*H) * P_k|k-1 * (I - K*H)^T + K * R * K^T
        #    Эта форма более численно устойчива, чем стандартная P = (I - K*H)*P
        I9 = eye(9)
        KH = mat_mul(K, self.H_pos)
        temp = mat_sub(I9, KH)
        term1 = mat_mul(mat_mul(temp, self.P), transpose(temp))
        KRKt = mat_mul(mat_mul(K, self.R_pos), transpose(K))
        self.P = mat_add(term1, KRKt)
        self.P = symmetrize(self.P)

        return col_to_vec(self.x)


class AUVSimulator:
    """
    Симулятор подводного аппарата для генерации "реальных" и "измеренных" данных.
    Он моделирует физику движения и работу датчиков с шумами.
    """

    def __init__(self, dt):
        self.dt = dt
        # Истинное состояние аппарата
        self.pos = [0.0, 0.0, 5.0] # [p_x, p_y, p_z]
        self.vel = [0.0, 0.0, 0.0] # [v_x, v_y, v_z]

        # Параметры управления и среды
        self.force_cmd = [0.0, 0.0, 0.0]
        self.drift_vel = [0.0, 0.0, 0.0]
        self.noise_std = 2.0            # sigma для позиционного эхолокатора

        self.radius = 1.0
        self.mass = 15.0
        self.drag_coeff = 3.0
        self.thrust_factor = 8.0 # Коэффициент для преобразования команды в силу

        # IMU параметры (акселерометр)
        self.imu_noise_std = 0.1         # Шум измерения IMU
        self.imu_bias = [0.0, 0.0, 0.0]  # Истинное смещение (bias)
        self.imu_bias_walk_std = 0.01    # "Скорость" изменения смещения

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
            # Полная скорость = скорость аппарата + скорость течения
            v_total = self.vel[i] + self.drift_vel[i]
            true_vel_vector[i] = v_total
            self.pos[i] += v_total * self.dt

        # Ограничение по глубине (не может выплыть выше поверхности)
        if self.pos[2] < self.radius:
            self.pos[2] = self.radius
            if self.vel[2] < 0:
                self.vel[2] = 0.0

        # Имитация "дрейфа" смещения (bias) акселерометра
        for i in range(3):
            # Генерация нормального шума через метод Бокса-Мюллера. устойчивое
            u1 = random.random()
            u2 = random.random()
            if u1 < 1e-12: 
                u1 = 1e-12
            
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            self.imu_bias[i] += z * self.imu_bias_walk_std * math.sqrt(self.dt)

        # Функция для генерации гаусс. шума
        def gauss(mu, sigma):
            u1 = random.random()
            u2 = random.random()
            if u1 < 1e-12: 
                u1 = 1e-12

            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return mu + z * sigma

        # Измерение IMU = истинное ускорение + смещение + шум
        imu_acc = [0.0] * 3
        for i in range(3):
            imu_acc[i] = acc_vector[i] + self.imu_bias[i] + gauss(0.0, self.imu_noise_std)

        # Измерение позиции = истинная позиция + шум
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

    kf.predict(force_cmd=sim.force_cmd)

    kf.update_imu(sim_data["imu_acc"], sim.force_cmd)

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


