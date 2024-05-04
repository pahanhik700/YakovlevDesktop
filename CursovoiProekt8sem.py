import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import mean_absolute_error

# Задаем размеры входа, скрытого слоя и выхода
input_size = 1
hidden_size = 15
output_size = 1
# Количество эпох обучения
epochs = 5000

# Задаем структуру
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# экземпляр модели
model = MLP(input_size, hidden_size, output_size)

# Оптимизатор (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Определение функции потерь
criterion = nn.MSELoss()

# Функция, которую мы хотим аппроксимировать
def f(x):
    return np.sin(2 * np.pi * x)

# Создание обучающих данных
num_points = 100
x_cord = np.linspace(0, 1, num_points)
y_cord = f(x_cord)
x_cord_tensor = torch.tensor(x_cord, dtype=torch.float32).unsqueeze(1)  # добавляем размерность для одной оси
y_cord_tensor = torch.tensor(y_cord, dtype=torch.float32).unsqueeze(1)
# Список для сохранения значений функции потерь
losses = []

# Обучение модели
for epoch in range(epochs):
    # Forward pass
    outputs = model(x_cord_tensor)
    loss = criterion(outputs, y_cord_tensor)

    # Backward pass и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Сохранение значения функции потерь
    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

x_new = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=torch.float32).unsqueeze(1)

# Предсказание выходных значений на новых данных
with torch.no_grad():
    predictions = model(x_new)

# Проверка на тестовых данных
print("Predictions:")
for i, prediction in enumerate(predictions):
    y_true = f(x_new[i].item())
    print(f"x={x_new[i].item():.2f}, y_pred={prediction.item():.4f}, y_true={f(x_new[i].item()):.4f}")


print(f"MAE={mean_absolute_error(x_new, predictions):.4f}")

# Построение графика функции потерь
plt.plot(losses)
plt.xlabel('Поколение')
plt.ylabel('Ошибка')
plt.title('Training Loss')
plt.show()