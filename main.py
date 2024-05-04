# библиотека для удобной работы с матрицами
import numpy as np
# библиотека для удобной работы с матрицами
import matplotlib.pyplot as plt


def f(a, b1, x):
    return (1 / 2 * x.transpose() @ a @ x + b1.transpose() @ x)[0][0]
    # символ "@" - матричное умножение
    # функция transpose() транспонирует оси массива.


# инициализация:
A = np.array([[7, 1, 1, 1, 1, 1],
[1, 8, 1, 1, 1, 1],
[1, 1, 9, 1, 1, 1],
[1, 1, 1, 10, 1, 1],
[1, 1, 1, 1, 11, 1],
[1, 1, 1, 1, 1, 12]]
            #[[0, 2, 0, 1, 5, 7],
              #[1, 4, 5, 1, 3, 9],
              #[1, 4, 0, 1, 5, 2],
              #[4, 4, 5, 1, 2, 9],
              #[1, 4, 3, 4, 5, 9],
              #[8, 1, 5, 1, 8, 2]]
    )  # произвольная положительно определенная матрица

B = np.array([[0, 2, 0, 1, 5, 7],
              [1, 4, 5, 1, 3, 9],
              [1, 4, 0, 1, 5, 2],
              [4, 4, 5, 1, 2, 9],
              [1, 4, 3, 4, 5, 9],
              [8, 1, 5, 1, 8, 2]])
print("проверка положительной определенности B^T*B")
print(B.transpose() @ B)
# матрица для проверки положительной определенности матрицы А

b = np.array([[0, 6, 7, 1, 5, 4]]).transpose()  # произвольный ненулевой вектор
x_0 = np.array([[0, 0, 0, 0, 0, 0.1]]).transpose()  # произвольный начальный ненулевой вектор

x_star = -np.linalg.pinv(
    A) @ b  # linalg.pinv(A) - вычисляет матрицу обратную матрице А. Вычисление происходит на основе сингулярного (SVD) разложения матрицы.

l = 10 ** -4  # λ

accuracy = 10 ** -5  # ξ - точность условия выхода из цикла

x_exact = np.linalg.solve(1 / 2 * (A.transpose() + A), -b)  # вычисление точного значения x через первую производную
# функция linalg.solve() решает линейное матричное уравнение
# Решения вычисляются с использованием подпрограммы LAPACK _gesv.
# GESV вычисляет решение реальной системы линейных уравнений
#     А * Х = В,
#  Разложение LU с частичным поворотом и обменом строками
#  используется для факторизации A как
#     A = P * L * U,
#  где P — матрица перестановок, L — единичный нижний треугольник, а U — верхний треугольник.
#  Затем факторизованная форма A используется для решения система уравнений А*Х=В.
x_previous = x_0.copy()
x_current = x_previous - l * (1 / 2 * (A.transpose() + A) @ x_previous + b)
step = 1
f_per_step = np.array([f(A, b, x_current)])  # массив для построения графика
x_4 = 0
x_2 = 0
x_3_4 = 0

# реализация метода градиента:
while np.sqrt(np.sum((x_previous - x_current) ** 2)) > accuracy:
    step += 1
    x_previous = x_current.copy()
    x_current = x_previous - l * (1 / 2 * (A.transpose() + A) @ x_previous + b)
    f_per_step = np.append(f_per_step, f(A, b, x_current))  # сохранение результатов каждого шага для построения графика
    if step == 52380 // 4:
        x_4 = x_current
    if step == 52380 // 2:
        x_2 = x_current
    if step == 52380 * 3 // 4:
        x_3_4 = x_current
# вывод:
print('произвольная положительно определенная матрица')
print(A)
print("проверка положительной определенности B^T*B")
print(B.transpose() @ B)
print("произвольный ненулевой вектор размерности 6")
print(b)
print("произвольный начальный ненулевой вектор размера 6,отдаленный от точного решения,x")
print(x_0)
print('x_*=-A^(-1)*b')
print(x_star)
print("Приравнивая производную к нулю, получаем вектор x_точ:")
print(x_exact)
print('Количество шагов:')
print(step)
print('Промежуточные результаты:')
print('step/4:')
print(x_4)
print('step/2:')
print(x_2)
print('3step/4:')
print(x_3_4)
print('step:')
print(x_current)
print('Промежуточные значения функционала:')
print('f(x_(step/4) ):')
print(f(A, b, x_4))
print('f(x_(step/2) ):')
print(f(A, b, x_2))
print('f(x_(3step/4) ):')
print(f(A, b, x_3_4))
print('f(x_(step) ):')
print(f(A, b, x_current))
print('Значение функционала в точке x_*:')
print(f(A, b, x_star))
print("Погрешности метода градиента:")
print('|x_step - x_точ|= ')
print(abs(x_current - x_exact))
print('|f(x_step )-f(x_* )|=')
print(abs(f(A, b, x_current) - f(A, b, x_star)))
# построение графика:
plt.plot(range(0, step), f_per_step)
plt.xlabel('номер шага')
plt.ylabel('значение функции')
plt.show()
