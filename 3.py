import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Определение функции chx


def chx(x):
    return 1 if 0 <= x <= 1 else 0


# Создание массива x значений от 0 до 2 с шагом 0.01
x_values = np.arange(0, 2, 0.01)

# Определение функции chx для массива x_values
chx_values = np.array([chx(x) for x in x_values])

# Определение вейвлета Хаара
haar_wavelet = np.array([1, -1])

# Применение линейной свертки для построения линейности вейвлета
linear_wavelet_result = convolve(chx_values, haar_wavelet, mode='same')

# Построение графика
plt.plot(x_values, chx_values, label='chx')
plt.plot(x_values, linear_wavelet_result, label='Линейность вейвлета')
plt.title('Линейность вейвлета Хаара для функции chx')
plt.xlabel('x')
plt.ylabel('Значения')
plt.legend()
plt.show()
