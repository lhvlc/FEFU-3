import numpy as np
import matplotlib.pyplot as plt

def compute_solution(A, b, x0, r):
    #находим аналитическое решение
    x_star = -np.linalg.inv(A) @ b
    distance = np.linalg.norm(x_star - x0)

    #проверяем, находится ли решение в пределах ограничения
    if distance > r:
        print(f"Решение {x_star} выходит за пределы ограничения.")
    else:
        print(f"Решение: {x_star}")
        print(f"Расстояние до центра: {distance}")
        print(f"Радиус ограничения: {r}")

    return x_star, distance

def plot_function_and_solution(A, b, x0, r, x_star):
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    #вычисляем значения функции f
    F = 0.5 * (A[0, 0] * X1 ** 2 + 2 * A[0, 1] * X1 * X2 + A[1, 1] * X2 ** 2) + b[0] * X1 + b[1] * X2

    #настраиваем график
    fig, ax = plt.subplots()
    contour = ax.contour(X1, X2, F, levels=20, cmap='viridis')
    ax.plot(*x_star, 'ro', label='Оптимальное решение')

    #добавляем круг ограничения
    circle = plt.Circle(x0, r, color='r', fill=False, label='Ограничение')
    ax.add_artist(circle)

    #настройка графика
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
    ax.axis('equal')
    plt.title('Контур функции и ограничение')
    plt.show()

#параметры для 2x2 матрицы
A_2x2 = np.array([[2, 1], [1, 3]])
b_2x2 = np.array([-1, 2])
x0_2x2 = np.array([0, 0])
r_2x2 = 1

#параметры для 3x3 матрицы
A_3x3 = np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]])
b_3x3 = np.array([-1, 2, 0])
x0_3x3 = np.array([0, 0, 0])
r_3x3 = 1

#вычисление и визуализация для 2x2
print("Для матрицы 2x2:")
x_star_2x2, distance_2x2 = compute_solution(A_2x2, b_2x2, x0_2x2, r_2x2)
plot_function_and_solution(A_2x2, b_2x2, x0_2x2, r_2x2, x_star_2x2)

#вычисление и визуализация для 3x3
print("\nДля матрицы 3x3:")
x_star_3x3, distance_3x3 = compute_solution(A_3x3, b_3x3, x0_3x3, r_3x3)
