import numpy as np
from tabulate import tabulate


class Solution:
    __m = 8
    __n = 6
    __A = np.array([])
    __b = np.array([])
    __c = np.array([])
    __x = np.array([])
    __support_matrix = 0

    def __init__(self):
        np.random.seed(16)
        self.__c = np.array(np.random.randint(100, size=self.__n))
        self.__b = np.array(np.random.randint(100, size=self.__m))
        self.__A = np.array(np.random.randint(100, size=(self.__m, self.__n)))

        print("c:\n", self.__c)
        print("b:\n", self.__b)
        print("A:\n", self.__A, "\n")

        self.__support_matrix = self.__create_support_mat()

    def __fun_for_print(self, mat, index_l, index_c, t):
        print(f"Шаг алгоритма #{t}")
        print(tabulate(mat, tablefmt="latex", floatfmt=".2f"))
        print(f"Индекс разрешающей строки = {index_l}")
        print(f"Индекс разрещающего столбца = {index_c}")
        print(f"Разрешающий элемент = {mat[index_l][index_c]}")
        answer = []

        for i in range(mat[0].size - 1):
            if sum(mat[0:, i]) == 1 and mat[0, i] == 0:
                index_line = np.where(mat[1:, i] == 1)[0][0] + 1
                answer.append(mat[index_line][mat[0].size - 1])
            else:
                answer.append(0)
        print("Промежуточное решение:\n", np.array(answer))

    def __create_support_mat(self):
        main_mat = np.zeros((7, 21))
        main_mat[1:, :8] = self.__A.T
        main_mat[1:, 8:14] = -np.eye(6)
        main_mat[1:, 14:20] = np.eye(6)
        main_mat[1:, 20] = self.__c
        main_mat[0, 14:20] = 1

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
        t += 1
        print("END OF SIMPLEX!!!: \n")
        self.__fun_for_print(main_mat, index_min_line, index_min_col, t)

        return main_mat

    def __create_main(self, c, b):
        main_mat = np.zeros((9, 15))
        main_mat[0][:c.shape[0]] = -c
        main_mat[1:, :6] = self.__A
        main_mat[1:, 6: 14] = np.eye(8)
        main_mat[1:, 14] = b
        return main_mat

    def __to_ready_double_problem(self, mat, b):
        mat[0, :8] = b
        mat[0, 14] = 0
        for i in range(15):
            if sum(mat[1:, i]) == 1 and mat[0, i] != 0:
                index_line = np.where(mat[1:, i] == 1)[0][0] + 1
                mat[0, :15] -= mat[index_line] * mat[0][i]
                mat[0][i] = 0

        return mat

    def execute(self):
        print("---Прямая---")
        print(tabulate(self.__simplex_method(self.__create_main(self.__c, self.__b), m=9, n=15),
                       tablefmt="latex", floatfmt=".2f"))

        support_mat = self.__simplex_method(self.__create_support_mat(), m=7, n=21)
        print("Последний шаг подготовки к решению двойственной задачи:\n")
        print(tabulate(support_mat, tablefmt="latex", floatfmt=".2f"))

        double_mat = np.column_stack([support_mat[0:, :14], support_mat[0:, 20]])
        double_mat = self.__to_ready_double_problem(double_mat, self.__b)

        print("---DOUBLE MATRIX---\n")
        print(tabulate(double_mat, tablefmt="latex", floatfmt=".2f"))

        print("---Двойственная---\n")
        print(tabulate(self.__simplex_method(double_mat, m=7, n=15), tablefmt="latex", floatfmt=".2f"))


if __name__ == "__main__":
    Solution().execute()
