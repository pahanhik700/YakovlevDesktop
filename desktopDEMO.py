import sys
from typing import List
import sympy as sp
import re
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGroupBox, QPushButton, QCheckBox, QLabel, QHBoxLayout, \
    QSpinBox
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def read_matrices():
    matrices = {}
    answers = {}
    i = 1
    for line in file:
        line = line.strip()
        if line.startswith("Type"):
            current_type = line
            if current_type not in matrices:  # Проверяем, существует ли уже список для данного типа матриц
                matrices[current_type] = []
                answers[current_type] = []
            if current_type == 'Type1':
                matrix1 = make_matrix()
                matrices[current_type].append(matrix1)
                answers[current_type].append(round(sp.det(matrix1), 2))
                matrix2 = make_matrix()
                matrices[current_type].append(matrix2)
                answers[current_type].append(round(sp.det(matrix2), 2))
            elif current_type == 'Type2':
                equ = file.readline().strip()
                equ = re.sub(r'(\d+)([A-Z])', r'\1*\2', equ)
                matrices[current_type].append(equ)
                A = make_matrix()
                B = make_matrix()
                C = make_matrix()
                matrices[current_type].append(A)
                matrices[current_type].append(B)
                matrices[current_type].append(C)
                #print(i)
                #i = i + 1
                #print(f'Матрица A {A}')
                #print(f'Матрица B {B}')
                #print(f'Матрица C {C}')
                #print(f'Выражение:{equ}')
                result = string_to_matrix_expression(equ, A, B, C, C)
                #print(result)
                answers[current_type].append(result)
            elif current_type == 'Type3':
                equ = file.readline().strip()
                equ = re.sub(r'(\d+)([A-Z])', r'\1*\2', equ)
                matrices[current_type].append(equ)
                A = make_matrix()
                E = make_matrix()
                matrices[current_type].append(A)
                matrices[current_type].append(E)
                print(i)
                i = i + 1
                print(f'Матрица A {A}')
                print(f'Матрица E {E}')
                print(f'Выражение:{equ}')
                result = string_to_matrix_expression(equ, A, B, C, E)
                print(result)
                answers[current_type].append(result)
    return matrices, answers


def make_matrix():
    rank = file.readline().strip().split()
    if len(rank) > 1:
        rank = int(rank[1])
    else:
        rank = int(rank[0])
    matrix = []
    for i in range(rank):
        row = file.readline().strip().split()
        matrix.append([int(x) for x in row])
    return sp.Matrix(matrix)


def parse_equation(equation_string):
    # Используем регулярное выражение для разбиения строки на части
    parts = re.findall(r'[-+]?[0-9]*[AQ-Z]|[AQ-Z]', equation_string)
    return parts


# def calculate_expression(parts, matrix_A, matrix_B, matrix_C):
#    result = None
#    for part in parts:
#        sign = -1 if part.startswith('-') else 1  # Определяем знак операции
#        matrix_name = part.lstrip('-+0123456789')  # Извлекаем имя матрицы
#        coefficient = int(part.strip('-+ABCDE')) if part.strip('-+ABCDE') else 1  # Извлекаем коэффициент, если он есть
#        if matrix_name == 'A':
#            matrix = matrix_A
#        elif matrix_name == 'B':
#            matrix = matrix_B
#        elif matrix_name == 'C':
#            matrix = matrix_C
#        elif matrix_name == 'E':
#            matrix = matrix_C
#        matrix *= sign * coefficient
#        if result is None:
#            result = matrix
#        else:
#            result += matrix
#    return result
def string_to_matrix_expression(equation_string, a, b, c, e):
    # Замена '^2' на '**2'
    equation_string = equation_string.replace('^2', '**2')
    # Определение символов-матриц
    matrix_symbols = {'A': a, 'B': b, 'C': c, 'E': e}
    symbols = {sym: sp.Symbol(sym) for sym in matrix_symbols}
    # Преобразование строки в матричное выражение
    matrix_expr = sp.sympify(equation_string)
    # Подстановка значений матриц
    matrix_expr = matrix_expr.subs({symbols[sym]: matrix_symbols[sym] for sym in matrix_symbols})
    return matrix_expr


# Пример использования
file_name = "tasks.txt"  # Путь к файлу с матрицами
file = open(file_name, 'r')
matrices, answers = read_matrices()
file.close()
print(matrices)
print(f'Ответы {answers}')


class PDFGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Генератор PDF')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.task_types = [
            'Определители',
            'Найти матрицу',
            'Найти значение матричного многочлена',
            'Найти произведение',
            'Определить, имеет ли данная матрица обратную',
            'Найти ранг матриц',
            'Решить матричное уравнение',
            'Решить систему уравнений'
            # Добавьте здесь другие типы задач, если необходимо
        ]

        self.task_checkboxes = []

        self.num_variants_spinbox = QSpinBox()
        self.num_variants_spinbox.setMinimum(1)
        self.num_variants_spinbox.setMaximum(100)  # Максимальное количество вариантов
        self.num_variants_spinbox.setValue(1)

        self.init_ui()

    def init_ui(self):
        group_box = QGroupBox('Выберите типы заданий')
        vbox = QVBoxLayout()

        for task_type in self.task_types:
            checkbox = QCheckBox(task_type)
            self.task_checkboxes.append(checkbox)
            vbox.addWidget(checkbox)

        group_box.setLayout(vbox)

        generate_button = QPushButton('Создать PDF')
        generate_button.clicked.connect(self.generate_pdf)

        num_variants_layout = QHBoxLayout()
        num_variants_layout.addWidget(QLabel('Количество вариантов:'))
        num_variants_layout.addWidget(self.num_variants_spinbox)

        self.layout.addWidget(group_box)
        self.layout.addLayout(num_variants_layout)
        self.layout.addWidget(generate_button)

        self.setLayout(self.layout)

    def generate_pdf(self):
        selected_tasks = [checkbox.text() for checkbox in self.task_checkboxes if checkbox.isChecked()]

        if not selected_tasks:
            error_label = QLabel('Выберите хотя бы один тип задания!')
            self.layout.addWidget(error_label)
            return

        num_variants = self.num_variants_spinbox.value()

        filename = 'tasks.pdf'

        pdfmetrics.registerFont(TTFont('Times-Roman', 'times.ttf'))  # Загрузка шрифта Times-Roman

        c = canvas.Canvas(filename, pagesize=letter)
        page_width, page_height = letter

        for i in range(num_variants):
            self.draw_variant(c, selected_tasks, page_width, page_height)
            c.showPage()  # Завершаем текущую страницу

        c.save()

        success_label = QLabel(f'PDF успешно создан: {filename}')
        self.layout.addWidget(success_label)

    def draw_variant(self, c, selected_tasks, page_width, page_height):
        c.setFont("Times-Roman", 12)  # Установка шрифта для текста
        y = page_height - 50  # Начальная позиция текста на странице
        # Рисуем первую страницу варианта
        for task_type in selected_tasks:
            c.drawString(100, y, task_type)
            tasks = self.generate_tasks_for_type(task_type)  # Получаем задания для выбранного типа
            for task in tasks:
                y -= 20
                if y <= 50:
                    c.showPage()  # Создаем новую страницу
                    y = page_height - 50  # Возвращаем y вниз страницы
                    c.setFont("Times-Roman", 12)  # Установка шрифта для текста
                c.drawString(100, y, task)
            y -= 20
            if y <= 50:
                c.showPage()  # Создаем новую страницу
                y = page_height - 50  # Возвращаем y вниз страницы
                c.setFont("Times-Roman", 12)  # Установка шрифта для текста

        c.showPage()  # Завершаем текущую страницу

    def generate_tasks_for_type(self, task_type):
        # Это место, где вы должны генерировать реальные задания для каждого типа
        # В этом примере мы просто возвращаем фиктивные задания
        if task_type == 'Определители':
            return ['Task 1.1', 'Task 1.2', 'Task 1.3']
        elif task_type == 'Найти матрицу':
            return ['Task 2.1', 'Task 2.2', 'Task 2.3']
        elif task_type == 'Найти значение матричного многочлена':
            return ['Task 3.1', 'Task 3.2', 'Task 3.3']
        elif task_type == 'Найти произведение':
            return ['Task 3.1', 'Task 3.2', 'Task 3.3']
        elif task_type == 'Определить, имеет ли данная матрица обратную':
            return ['Task 3.1', 'Task 3.2', 'Task 3.3']
        elif task_type == 'Найти ранг матриц':
            return ['Task 3.1', 'Task 3.2', 'Task 3.3']
        elif task_type == 'Решить матричное уравнение':
            return ['Task 3.1', 'Task 3.2', 'Task 3.3']
        elif task_type == 'Решить систему уравнений':
            return ['Task 3.1', 'Task 3.2', 'Task 3.3']
        else:
            return []

# if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    window = PDFGeneratorApp()
#    window.show()
#    sys.exit(app.exec_())
