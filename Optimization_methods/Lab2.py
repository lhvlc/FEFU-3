import numpy as np

A = np.array([[5, 4, 2],
              [4, 6, 3],
              [2, 3, 4]])

b = np.array([1, 1, 1])

#функция для вычисления градиента
def gradient(x):
    return A @ x + b

#функция для вычисления значения функции f_0
def objective_function(x):
    return 0.5 * x.T @ A @ x + b.T @ x

#параметры градиентного спуска
alpha = 0.01 #скорость обучения
epsilon = 1e-6 #критерий остановки
max_iterations = 1000 #максимальное количество итераций

#инициализация
x = np.zeros(3) #начальная точка, можно выбрать другую
iterations = 0

while iterations < max_iterations:
    grad = gradient(x)
    x_new = x - alpha * grad

    #проверяем условие остановки
    if np.linalg.norm(x_new - x) < epsilon:
        break

    x = x_new
    iterations += 1

#результат
min_value = objective_function(x)
print("Минимум найден в точке:", x)
print("Значение функции в этой точке:", min_value)