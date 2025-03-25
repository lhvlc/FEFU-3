import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

phi1 = lambda x, t: 2 * t ** 2 * (x ** 2 - x)
psi1 = lambda x: x ** 2 * (x - 1)
gamma_0_1 = lambda t: 0
gamma_1_1 = lambda t: 0

phi2 = lambda x, t: 0
psi2 = lambda x: 0
gamma_0_2 = lambda t: -2 * t
gamma_1_2 = lambda t: t ** 2

def solve_explicit(M, N, l, T, a, phi, psi, gamma_0, gamma_1):
    h = l / M
    tau = T / N
    r = (a ** 2 * tau) / (h ** 2)
    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)

    u = np.zeros((N + 1, M + 1))
    u[0, :] = psi(x)

    for n in range(0, min(N, 6)):
        u[n + 1, 0] = gamma_0(t[n + 1])
        u[n + 1, M] = gamma_1(t[n + 1])
        for m in range(1, M):
            u[n + 1, m] = u[n, m] + r * (u[n, m + 1] - 2 * u[n, m] + u[n, m - 1]) + tau * phi(x[m], t[n])
    return x, t, u

def solve_implicit(M, N, l, T, a, phi, psi, gamma_0, gamma_1):
    h = l / M
    tau = T / N
    r = (a ** 2 * tau) / (h ** 2)
    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)

    A = np.zeros((3, M - 1))
    A[0, 1:] = -r
    A[1, :] = 1 + 2 * r
    A[2, :-1] = -r

    u = np.zeros((N + 1, M + 1))
    u[0, :] = psi(x)

    for n in range(0, min(N, 6)):
        b = u[n, 1:M] + tau * phi(x[1:M], t[n + 1])
        b[0] += r * gamma_0(t[n + 1])
        b[-1] += r * gamma_1(t[n + 1])

        u[n + 1, 1:M] = solve_banded((1, 1), A, b)
        u[n + 1, 0] = gamma_0(t[n + 1])
        u[n + 1, M] = gamma_1(t[n + 1])
    return x, t, u

def solve_crank_nicolson(M, N, l, T, a, phi, psi, gamma_0, gamma_1):
    h = l / M
    tau = T / N
    r = (a ** 2 * tau) / (2 * h ** 2)
    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)

    A = np.zeros((3, M - 1))
    A[0, 1:] = -r
    A[1, :] = 1 + 2 * r
    A[2, :-1] = -r

    u = np.zeros((N + 1, M + 1))
    u[0, :] = psi(x)

    for n in range(0, min(N, 6)):
        b = (1 - 2 * r) * u[n, 1:M] + r * (u[n, 2:M + 1] + u[n, 0:M - 1]) + tau * phi(x[1:M], t[n + 1])
        b[0] += r * gamma_0(t[n + 1])
        b[-1] += r * gamma_1(t[n + 1])

        u[n + 1, 1:M] = solve_banded((1, 1), A, b)
        u[n + 1, 0] = gamma_0(t[n + 1])
        u[n + 1, M] = gamma_1(t[n + 1])
    return x, t, u

M = 25 #число узлов по x
N = 50 #число узлов по t
l = 1
T = 0.1
a = 1

#решение для первого набора условий
x_exp1, t_exp1, u_exp1 = solve_explicit(M, N, l, T, a, phi1, psi1, gamma_0_1, gamma_1_1)
x_imp1, t_imp1, u_imp1 = solve_implicit(M, N, l, T, a, phi1, psi1, gamma_0_1, gamma_1_1)
x_cn1, t_cn1, u_cn1 = solve_crank_nicolson(M, N, l, T, a, phi1, psi1, gamma_0_1, gamma_1_1)

#решение для второго набора условий
x_exp2, t_exp2, u_exp2 = solve_explicit(M, N, l, T, a, phi2, psi2, gamma_0_2, gamma_1_2)
x_imp2, t_imp2, u_imp2 = solve_implicit(M, N, l, T, a, phi2, psi2, gamma_0_2, gamma_1_2)
x_cn2, t_cn2, u_cn2 = solve_crank_nicolson(M, N, l, T, a, phi2, psi2, gamma_0_2, gamma_1_2)

plt.figure(figsize=(12, 10))
methods = [(u_exp1, "Явная схема (условие 1)"), (u_imp1, "Неявная схема (условие 1)"),
           (u_cn1, "Кранк-Николсон (условие 1)"),
           (u_exp2, "Явная схема (условие 2)"), (u_imp2, "Неявная схема (условие 2)"),
           (u_cn2, "Кранк-Николсон (условие 2)")]
for i, (u, label) in enumerate(methods):
    plt.subplot(2, 3, i + 1)
    plt.imshow(u[:6, :], aspect='auto', cmap='hot', origin='lower', extent=[0, l, 0, 6 * T / N])
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(label)
plt.tight_layout()
plt.show()
