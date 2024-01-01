import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, cwt, ricker


def chx(x):
    return 0.5 * (np.exp(x) + np.exp(-x))


def plot_haar_wavelet():
    x_values = np.arange(0, 1, 0.01)
    y_values = [1 if 0 <= x < 0.5 else -1 for x in x_values]

    plt.plot(x_values, y_values, label='Хаар-вейвлет')
    plt.title('График вейвлета Хаара')
    plt.xlabel('x')
    plt.ylabel('Хаар-вейвлет(x)')
    plt.legend()
    plt.show()


def plot_continuous_wavelet_transform():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    widths = np.arange(1, 31)

    cwt_result = cwt(chx_values, ricker, widths)

    plt.imshow(np.abs(cwt_result), extent=[
               0, 2, 1, 31], cmap='jet', aspect='auto', interpolation='bilinear')
    plt.colorbar(label='Амплитуда')
    plt.title('Непрерывное вейвлет-преобразование для функции chx')
    plt.xlabel('x')
    plt.ylabel('Ширина вейвлета')
    plt.show()


def plot_wavelet_shift():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    haar_wavelet = np.array([1, -1])
    shift_amount = 0.5

    shifted_wavelet = np.roll(haar_wavelet, int(
        shift_amount * len(haar_wavelet)))

    shifted_wavelet_result = convolve(chx_values, shifted_wavelet, mode='same')

    plt.plot(x_values, chx_values, label='chx')
    plt.plot(x_values, shifted_wavelet_result,
             label=f'Сдвиг вейвлета (на {shift_amount} единиц)')
    plt.title('Сдвиг вейвлета Хаара для функции chx')
    plt.xlabel('x')
    plt.ylabel('Значения')
    plt.legend()
    plt.show()


def plot_wavelet_scaling():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    haar_wavelet = np.array([1, -1])
    scale_factor = 2

    scaled_wavelet = np.repeat(haar_wavelet, scale_factor)

    scaled_wavelet_result = convolve(chx_values, scaled_wavelet, mode='same')

    plt.plot(x_values, chx_values, label='chx')
    plt.plot(x_values, scaled_wavelet_result,
             label=f'Масштабирование вейвлета (в {scale_factor} раза)')
    plt.title('Масштабирование вейвлета Хаара для функции chx')
    plt.xlabel('x')
    plt.ylabel('Значения')
    plt.legend()
    plt.show()


def plot_inverse_wavelet():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    haar_wavelet = np.array([1, -1])

    wavelet_result = convolve(chx_values, haar_wavelet, mode='same')
    inverse_wavelet_result = convolve(
        wavelet_result, np.flip(haar_wavelet), mode='same')

    plt.plot(x_values, chx_values, label='chx')
    plt.plot(x_values, inverse_wavelet_result,
             label='Обратное вейвлет-преобразование')
    plt.title('Обратное вейвлет-преобразование Хаара для функции chx')
    plt.xlabel('x')
    plt.ylabel('Значения')
    plt.legend()
    plt.show()


def plot_wavelet_linearity():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    haar_wavelet = np.array([1, -1])
    linear_combination = 0.5 * convolve(chx_values, haar_wavelet, mode='same')

    plt.plot(x_values, chx_values, label='chx')
    plt.plot(x_values, linear_combination,
             label='Линейная комбинация вейвлета')
    plt.title('Линейность вейвлета Хаара для функции chx')
    plt.xlabel('x')
    plt.ylabel('Значения')
    plt.legend()
    plt.show()


def plot_wavelet_localization():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    haar_wavelet = np.array([1, -1])
    localization_result = 0.5 * convolve(chx_values, haar_wavelet, mode='same')

    plt.plot(x_values, chx_values, label='chx')
    plt.plot(x_values, localization_result,
             label='Масштабно-временная локализация вейвлета')
    plt.title('Масштабно-временная локализация вейвлета Хаара для функции chx')
    plt.xlabel('x')
    plt.ylabel('Значения')
    plt.legend()
    plt.show()


def plot_wavelet_comparison():
    x_values = np.arange(0, 2, 0.01)
    chx_values = chx(x_values)
    haar_wavelet = np.array([1, -1])

    # Исходный вейвлет
    plt.subplot(2, 2, 1)
    plt.plot(haar_wavelet, marker='o', label='Исходный вейвлет')
    plt.title('Исходный вейвлет Хаара')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.legend()

    # Восстановленный вейвлет
    plt.subplot(2, 2, 2)
    plt.plot(np.flip(haar_wavelet), marker='o',
             label='Восстановленный вейвлет')
    plt.title('Восстановленный вейвлет Хаара')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.legend()

    # Сравнение исходного и восстановленного
    plt.subplot(2, 2, 3)
    plt.plot(haar_wavelet, marker='o', label='Исходный вейвлет')
    plt.plot(np.flip(haar_wavelet), marker='o',
             label='Восстановленный вейвлет')
    plt.title('Сравнение исходного и восстановленного вейвлетов')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.legend()

    # Функция chx
    plt.subplot(2, 2, 4)
    plt.plot(x_values, chx_values, label='chx')
    plt.title('Функция chx')
    plt.xlabel('x')
    plt.ylabel('Значение')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    print("Выберите действие:")
    print("1. Построить график вейвлета Хаара")
    print("2. Построить непрерывное вейвлет-преобразование для функции chx")
    print("3. Построить график сдвига вейвлета Хаара для функции chx")
    print("4. Построить график масштабирования вейвлета Хаара для функции chx")
    print("5. Построить обратное вейвлет-преобразование Хаара для функции chx")
    print("6. Построить линейность вейвлета Хаара для функции chx")
    print("7. Построить масштабно-временную локализацию вейвлета Хаара для функции chx")
    print("8. Построить сравнение исходного вейвлета из п1) и восстановленного из п7)")

    choice = input("Введите номер действия (1-8): ")

    if choice == '1':
        plot_haar_wavelet()
    elif choice == '2':
        plot_continuous_wavelet_transform()
    elif choice == '3':
        plot_wavelet_shift()
    elif choice == '4':
        plot_wavelet_scaling()
    elif choice == '5':
        plot_inverse_wavelet()
    elif choice == '6':
        plot_wavelet_linearity()
    elif choice == '7':
        plot_wavelet_localization()
    elif choice == '8':
        plot_wavelet_comparison()
    else:
        print("Некорректный выбор. Пожалуйста, введите число от 1 до 8.")


if __name__ == "__main__":
    main()
