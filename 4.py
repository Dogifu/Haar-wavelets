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

# Определение значения сдвига
shift_amount = 0.5

# Создание сдвинутого вейвлета
shifted_wavelet = np.roll(haar_wavelet, int(shift_amount * len(haar_wavelet)))

# Применение сдвинутого вейвлета к функции chx
shifted_wavelet_result = convolve(chx_values, shifted_wavelet, mode='same')

# Построение графика
plt.plot(x_values, chx_values, label='chx')
plt.plot(x_values, shifted_wavelet_result,
         label=f'Сдвиг вейвлета (на {shift_amount} единиц)')
plt.title('Сдвиг вейвлета Хаара для функции chx')
plt.xlabel('x')
plt.ylabel('Значения')
plt.legend()
plt.show()
