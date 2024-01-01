import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker

# Определение функции chx


def chx(x):
    return 1 if 0 <= x <= 1 else 0


# Создание массива x значений от 0 до 2 с шагом 0.01
x_values = np.arange(0, 2, 0.01)

# Определение функции chx для массива x_values
chx_values = np.array([chx(x) for x in x_values])

# Определение ширины вейвлета Хаара (в данном случае, можно использовать значение 1)
widths = np.arange(1, 31)

# Применение непрерывного вейвлет-преобразования (CWT) к функции chx
cwt_result = cwt(chx_values, ricker, widths)

# Построение графика
plt.imshow(np.abs(cwt_result), extent=[
           0, 2, 1, 31], cmap='jet', aspect='auto', interpolation='bilinear')
plt.colorbar(label='Амплитуда')
plt.title('Непрерывное вейвлет-преобразование для функции chx')
plt.xlabel('x')
plt.ylabel('Ширина вейвлета')
plt.show()
