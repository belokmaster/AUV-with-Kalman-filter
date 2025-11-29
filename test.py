import pytest


def matrix_print(A):
    for i in range(len(A)):
        el = []
        for j in range(len(A[0])):
            el.append(A[i][j])
        print(el)


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


def lup_decompose_trace(A, epsilon=1e-12):
    A = mat_copy(A)
    n = len(A)
    LU = mat_copy(A)
    perm = list(range(n))
    print("Initial A:")
    matrix_print(A)
    for k in range(n):
        pivot = k
        maxv = abs(LU[k][k])
        for i in range(k + 1, n):
            if abs(LU[i][k]) > maxv:
                maxv, pivot = abs(LU[i][k]), i
        print(f"\nStep k={k+1}: pivot row = {pivot+1}, pivot value = {LU[pivot][k]}")
        if pivot != k:
            print(f" swap rows {k+1} and {pivot+1}")
            LU[k], LU[pivot] = LU[pivot], LU[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
            print(" LU after swap:")
            matrix_print(LU)
        if abs(LU[k][k]) <= epsilon:
            raise ValueError("singular")
        for i in range(k + 1, n):
            LU[i][k] /= LU[k][k]
            factor = LU[i][k]
            print(f"  multiplier l[{i+1},{k+1}] = {factor}")
            for j in range(k + 1, n):
                LU[i][j] -= factor * LU[k][j]
            print(f"  LU after eliminating row {i+1}:")
            matrix_print(LU)
    print("\nFinal LU:")
    matrix_print(LU)
    print("perm:", [x + 1 for x in perm])
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


def invert_matrix_with_trace(A):
    n = len(A)
    LU, perm = lup_decompose_trace(A)
    invA = zeros(n, n)
    I = eye(n)
    for col in range(n):
        b = [I[i][col] for i in range(n)]
        x = lup_solve(LU, perm, b)
        for i in range(n):
            invA[i][col] = x[i]
    return invA, LU, perm


def approx_eq(A, B, tol=1e-9):
    r = len(A)
    c = len(A[0])
    for i in range(r):
        for j in range(c):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


A3 = [
    [0.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]

A4 = [
    [0.0, 2.0, 1.0, 3.0],
    [4.0, 5.0, 6.0, 7.0],
    [1.0, 0.0, 8.0, 9.0],
    [2.0, 3.0, 4.0, 5.0]
]

singularA4 = [
    [0.0, 2.0, 1.0, 3.0],
    [0.0, 4.0, 2.0, 6.0],
    [1.0, 0.0, 8.0, 9.0],
    [2.0, 3.0, 4.0, 5.0]
]

print("\n======================")
print("========= A3 =========")
print("======================")
invA3, LU3, perm3 = invert_matrix_with_trace(A3)
print("inv(A3):")
matrix_print(invA3)
print("Check A3 * invA3 = I ?")
matrix_print(mat_mul(A3, invA3))
assert approx_eq(mat_mul(A3, invA3), eye(3)), "A3 inverse failed"

print("\n======================")
print("========= A4 =========")
print("======================")
invA4, LU4, perm4 = invert_matrix_with_trace(A4)
print("inv(A4):")
matrix_print(invA4)
print("Check A4 * invA4 = I ?")
matrix_print(mat_mul(A4, invA4))
assert approx_eq(mat_mul(A4, invA4), eye(4)), "A4 inverse failed"

print("\n======================")
print("======== sing ========")
print("======================")
with pytest.raises(ValueError):
    lup_decompose(singularA4)
print("Correctly caught ValueError for singular matrix")

print("\nAll checks passed.")
