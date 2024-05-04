import numpy as np


class SecondLab:

    A = 0
    b = 0
    r = 0
    x_0 = 0
    a = 0
    y = 0
    sign = 0

    def __init__(self, r, a, y, sign):
        np.random.seed(1)
        self.A = np.loadtxt("a.txt", usecols=(range(4)))
        self.b = np.loadtxt("b.txt", usecols=(range(1)), ndmin=2)
        self.x_0 = np.loadtxt("x_0.txt", usecols=(range(1)), ndmin=2)
        self.r = r
        self.a = a
        self.y = y
        self.sign = sign
        self.generate_matrix()
        self.generate_vector("b")
        self.generate_vector("x_0")

    def f(self, x: np.ndarray) -> float:
        res = .5 * x.transpose() @ self.A @ x + self.b.transpose() @ x
        return res[0][0]

    def generate_matrix(self):
        matrix = np.random.uniform(0.4, 0.7, (4, 4))
        np.savetxt("a.txt", matrix @ matrix, fmt='%.7f')

    def generate_vector(self, name: str):
        matrix = np.random.uniform(1, 2, (4, 1))
        np.savetxt(f"{name}.txt", matrix, fmt='%.7f')

    def lagrange_slae(self, x: np.ndarray) -> np.ndarray:
        return np.append((self.A + 2 * np.eye(4) * self.y) @ x + (self.b + 2 * self.y * self.x_0),
                         [[np.linalg.norm(x - self.x_0)**2 - self.r**2]], axis=0)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        J_1_1 = self.A + 2 * np.eye(4) * self.y
        J_1_2 = 2 * (x - self.x_0)
        J_2_1 = J_1_2.transpose()
        J_2_2 = [[0]]
        J_1 = np.append(J_1_1, J_1_2, axis=1)
        J_2 = np.append(J_2_1, J_2_2, axis=1)
        return np.append(J_1, J_2, axis=0)

    def newton(self, x_k: np.ndarray, epsilon=1e-6, max_iter=30, ):
        x_prev = x_k
        x_cur = x_prev - np.linalg.inv(self.jacobian(x_prev[0:-1])) @ self.lagrange_slae(x_prev[0:-1])
        it = 0
        while np.linalg.norm(x_cur[0:-1] - x_prev[0:-1]) > epsilon and it < max_iter:
            it += 1
            x_prev = x_cur
            x_cur = x_prev - np.linalg.inv(self.jacobian(x_prev[0:-1])) @ self.lagrange_slae(x_prev[0:-1])
        return x_cur

    def start_lab(self):
        x_ = np.append(self.x_0, [[self.y]], axis=0)

        print('')
        x_star = -np.linalg.inv(self.A) @ self.b
        f_in_x_star = self.f(x_star)
        print(f"x*:\n{x_star}")
        print(f"\nФункция в точке x* = {f_in_x_star}")
        print(f"\nx*-x_0:\n{x_star - self.x_0}")
        print(f"\n||x*-x_0|| = {np.linalg.norm(x_star - self.x_0)}\n")

        for i in range(8):
            self.sign = -self.sign
            x_k = x_.copy()
            x_k[i // 2][0] += self.sign * self.a
            print(f"\nНачальное приближение {i + 1}:\n{x_k[0:-1]}")
            res = self.newton(x_k)
            print("Значение x:")
            print(res[0:-1])
            print(f"Значение y = {res[4][0]}")
            print(f"Значение функции = {self.f(res[0:-1])}\n")


def execute_second_lab():
    s_l = SecondLab(5, 4, 3, 1)
    s_l.start_lab()


if __name__ == "__main__":
    execute_second_lab()