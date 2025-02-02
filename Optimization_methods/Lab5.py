import numpy as np
from scipy.optimize import linprog

#задаем параметры задачи
m, n = 8, 6

#генерация случайных неотрицательных коэффициентов для A, b, c
np.random.seed(0)
A = np.random.rand(m, n)
b = np.random.rand(m)
c = np.random.rand(n)

#решение прямой задачи с использованием симплекс-метода
res_primal = linprog(-c, A_ub=A, b_ub=b, method='highs')

if res_primal.success:
    print("Прямое решение:")
    print("Значения переменных x:", res_primal.x)
    print("Максимальная целевая функция:", -res_primal.fun)
else:
    print("Не удалось найти оптимальное решение для прямой задачи.")

#двойственная задача: min b^T y при A^T y >= c, y >= 0
res_dual = linprog(b, A_ub=-A.T, b_ub=-c, method='highs')

if res_dual.success:
    print("\nДвойственное решение:")
    print("Значения переменных y:", res_dual.x)
    print("Минимальная целевая функция:", res_dual.fun)
else:
    print("Не удалось найти оптимальное решение для двойственной задачи.")
