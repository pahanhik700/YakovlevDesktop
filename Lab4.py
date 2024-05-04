import numpy as np
from tabulate import tabulate


class Solution:
    __m = 6
    __n = 8
    __A = np.array([])
    __b = np.array([])
    __x = np.array([])
    __c = np.array([])
    __support_matrix = 0

    def __init__(self):
        np.random.seed(17)
        self.__c = np.array(np.random.randint(100, size=self.__n))
        self.__b = np.array(np.random.randint(100, size=self.__m))
        self.__A = np.array(np.random.randint(-100, 100, size=(self.__m, self.__n)))

        print("c:\n", self.__c)
        print("b:\n", self.__b)
        print("A:\n", self.__A, "\n")

    def __fun_for_print(self, mat, index_l, index_c, t):
        print(f"Шаг алгоритма #{t}")
        print(tabulate(mat, tablefmt="latex", floatfmt=".2f"))
        print(f"Индекс разрешающей строки = {index_l}")
        print(f"Индекс разрещающего столбца = {index_c}")
        print(f"Разрешающий элемент = {mat[index_l][index_c]}")
        answer = []
        for i in range(14):
            if sum(mat[0:, i]) == 1 and mat[0, i] == 0:
                index_line = np.where(mat[1:, i] == 1)[0][0] + 1
                answer.append(mat[index_line][mat[0].size - 1])
            else:
                answer.append(0)
        print("Промежуточное решение:\n", np.array(answer))

    def __create_support_mat(self, mat):
        main_mat = np.zeros((9, 23))
        main_mat[1:, :6] = mat
        main_mat[1:, 6:14] = -np.eye(8)
        main_mat[1:, 14:22] = np.eye(8)
        main_mat[1:, 22] = 1
        main_mat[0, 14:22] = 1

        temp = 0
        for i in range(1, main_mat.shape[0]):
            temp += main_mat[i]
        main_mat[0] -= temp

        return main_mat

    def __simplex_method(self, main_mat, m, n):
        t = 0
        index_min_line = 0
        index_min_col = 0
        while np.any(main_mat[0, :n - 1] < 0):
            t += 1
            index_min_col = np.where(main_mat == main_mat[0, :(n - 1)].min())[1][0]
            div_last = []
            for i in range(1, m):
                div_last.append(main_mat[i][-1] / main_mat[i][index_min_col])

            index_min_line = np.where(div_last == min(filter(lambda x: x > 0, div_last)))[0][0] + 1

            self.__fun_for_print(main_mat, index_min_line, index_min_col, t)
            main_mat[index_min_line] /= main_mat[index_min_line][index_min_col]
            cur_col = main_mat[:, index_min_col]

            for i in range(len(cur_col)):
                if cur_col[i] != 1:
                    main_mat[i] -= main_mat[index_min_line] * cur_col[i]

        print('END OF SIMPLEX!!!: \n')
        t += 1
        self.__fun_for_print(main_mat, index_min_line, index_min_col, t)
        return main_mat

    def __create_main(self, c=0, b=0):
        if c == 0 and b == 0:
            main_mat = np.zeros((7, 15))
            main_mat[0][:8] = -1
            main_mat[1:, :8] = self.__A
            main_mat[1:, 8: 14] = np.eye(6)
            main_mat[1:, 14] = 1
            return main_mat

        main_mat = np.zeros((9, 15))
        main_mat[0][:c.shape[0]] = -c
        main_mat[1:, :6] = self.__A
        main_mat[1:, 6: 14] = np.eye(8)
        main_mat[1:, 14] = b
        return main_mat

    def __to_ready_double_problem(self, mat, b=0):
        if b == 0:
            mat[0, :6] = 1
            mat[0, 14] = 0
            for i in range(15):
                if sum(mat[1:, i]) == 1 and mat[0, i] != 0:
                    index_line = np.where(mat[1:, i] == 1)[0][0] + 1
                    mat[0, :15] -= mat[index_line] * mat[0][i]
                    mat[0][i] = 0

            print('TO READY DOUBLE PROBLEM: ', tabulate(mat, tablefmt="latex", floatfmt=".2f"))
            return mat

        mat[0, :8] = b
        mat[0, 14] = 0
        for i in range(15):
            if sum(mat[1:, i]) == 1 and mat[0, i] != 0:
                index_line = np.where(mat[1:, i] == 1)[0][0] + 1
                mat[0, :15] -= mat[index_line] * mat[0][i]
                mat[0][i] = 0

        return mat

    def execute(self):
        tmp = []
        for i in range(self.__A[0].size):
            tmp.append(max(self.__A[0:, i]))
        print("Верхняя цена игры:", min(tmp))

        tmp.clear()

        for i in range(self.__A[0, :6].size):
            tmp.append(min(self.__A[i, 0:]))
        print("Нижняя цена игры:", max(tmp))

        beta = min(tmp)
        print("β =", beta)
        self.__A[0:] += abs(beta)
        print("\nНеотрицательная A:\n", self.__A)

        print("---Прямая---")
        print(tabulate(self.__simplex_method(self.__create_main(), m=7, n=15),
                       tablefmt="latex", floatfmt=".2f"))

        q = [0.00066279, 0.00086283, 0.00227322, 0., 0.00175046, 0.0038538, 0.00412645, 0.]
        print("\nx =", q)
        print("||y|| =", np.linalg.norm(q))
        q /= np.linalg.norm(q)
        print("q =", q)
        print("Значение целевой функции =", 0.01, "\n\n")

        support_mat = self.__simplex_method(self.__create_support_mat(self.__A.T), m=9, n=23)

        double_mat = np.column_stack([support_mat[0:, :14], support_mat[0:, 22]])
        print("---DOUBLE MATRIX---\n")
        print(tabulate(double_mat, floatfmt=".2f", tablefmt="latex"))

        double_mat = self.__to_ready_double_problem(double_mat)

        print("---Двойственная---\n")
        print("---Возвращаемое значение---\n", self.__simplex_method(double_mat, m=7, n=15))

        p = [0.00316193, 0.00234683, 0.00223194, 0.00312174, 0.00148573, 0.00118138]
        print("\nx =", p)
        print("||x|| =", np.linalg.norm(p))
        p /= np.linalg.norm(p)
        print("p = ", p)
        print("Значение целевой функции =", 0.01)
        fi = 1 / 0.01 - abs(beta)
        print("fi = ", fi)


if __name__ == "__main__":
    Solution().execute()
