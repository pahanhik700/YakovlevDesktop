import re

# Ваше выражение
expression = '3*A+2*B-4*C'

# Замена всех вхождений паттерна коэффициент+переменная на коэффициент*переменная
expression = re.sub(r'(\d+)([A-Z])', r'\1*\2', expression)

print(expression)