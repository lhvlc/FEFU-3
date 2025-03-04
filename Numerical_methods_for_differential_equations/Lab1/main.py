import numpy as np
import matplotlib.pyplot as plt
#Вариант 9
def f(x, y):
    return 2 * y - 2

def exact_solution(x):
    return 1 - np.exp(2 * x)

def runge_kutta_3(f, x0, y0, h, x_end):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros_like(x_values)
    y_values[0] = y0

    for i in range(len(x_values) - 1):
        x_n = x_values[i]
        y_n = y_values[i]

        k1 = f(x_n, y_n)
        k2 = f(x_n + h / 2, y_n + h / 2 * k1)
        k3 = f(x_n + h, y_n - h * k1 + 2 * h * k2)

        #обновление значения
        y_values[i + 1] = y_n + h / 6 * (k1 + 4 * k2 + k3)

    return x_values, y_values

#параметры задачи
x0 = 0
y0 = 0
x_end = 1
h = 0.1

#с шагом h
x_h, y_h = runge_kutta_3(f, x0, y0, h, x_end)

#с шагом h/2
h_half = h / 2
x_h2, y_h2 = runge_kutta_3(f, x0, y0, h_half, x_end)

#точное решение
x_exact = np.linspace(x0, x_end, 1000)
y_exact = exact_solution(x_exact)

#графики
plt.plot(x_h, y_h, 'o-', label="Рунге-Кутта 3 (h)")
plt.plot(x_h2, y_h2, 's-', label="Рунге-Кутта 3 (h/2)")
plt.plot(x_exact, y_exact, '-', label="Точное решение")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Сравнение численного и точного решения")
plt.grid()
plt.show()

#вывод разницы между решениями с h и h/2
for i in range(len(x_h)):
    print(f"x = {x_h[i]:.2f}, y_h = {y_h[i]:.6f}, y_h2 = {y_h2[2 * i]:.6f}, Разница = {abs(y_h[i] - y_h2[2 * i]):.6f}")
