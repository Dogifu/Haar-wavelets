import numpy as np
import matplotlib.pyplot as plt


def chx(x):
    # Пример: прямоугольная функция с центром в x и шириной 1
    return 1 if 0 <= x <= 1 else 0


def haar_wavelet_chx(x):
    return chx(x) - chx(x - 1)


# Создаем массив x значений от 0 до 2 с шагом 0.01
x_values = np.arange(0, 2, 0.01)

# Применяем вейвлет Хаара к каждому элементу массива x
y_values = [haar_wavelet_chx(x) for x in x_values]

# Строим график
plt.plot(x_values, y_values, label='Хаар-вейвлет для chx')
plt.title('График вейвлета Хаара для функции chx')
plt.xlabel('x')
plt.ylabel('Хаар-вейвлет(chx)')
plt.legend()
plt.show()
