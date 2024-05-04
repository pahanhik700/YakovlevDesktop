import numpy as np

# Создаем произвольные матрицы A, B и C
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [0, 1]])
C = np.array([[1, 0], [0, 3]])

# Вычисляем спектральные нормы
spectral_norm_A = np.linalg.norm(A, 2)
spectral_norm_B = np.linalg.norm(B, 2)
spectral_norm_C = np.linalg.norm(C, 2)

# Проверяем, что спектральная норма матрицы A меньше чем у матриц B и C
print(spectral_norm_A < spectral_norm_B and spectral_norm_A < spectral_norm_C)