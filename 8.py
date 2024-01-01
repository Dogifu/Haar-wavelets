import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Определение функции chx


def chx(x):
    return np.where((0 <= x) & (x <= 1), 1, 0)


# Создание массива x значений от 0 до 2 с шагом 0.01
x_values = np.arange(0, 2, 0.01)

# Определение вейвлета Хаара
haar_wavelet = np.array([1, -1])

# Применение вейвлета к функции chx (прямое вейвлет-преобразование)
wavelet_result = convolve(chx(x_values), haar_wavelet, mode='same')

# Обратное вейвлет-преобразование (инверсия вейвлета и свертка)
inverse_wavelet_result = convolve(
    wavelet_result, np.flip(haar_wavelet), mode='same')

# Построение графика
plt.figure(figsize=(10, 5))

# Исходный вейвлет
plt.subplot(1, 2, 1)
plt.plot(haar_wavelet, marker='o', label='Исходный вейвлет')
plt.title('Исходный вейвлет Хаара')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()

# Восстановленный вейвлет
plt.subplot(1, 2, 2)
plt.plot(np.flip(haar_wavelet), marker='o', label='Восстановленный вейвлет')
plt.title('Восстановленный вейвлет Хаара')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()

plt.tight_layout()
plt.show()
