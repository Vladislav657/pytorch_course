from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Разомкнутая система
num_open = [23.8, 14, 0]
den_open = [0.3267, 1.4256, 2.07, 1]


# Автоматический расчет замкнутой системы
def get_closed_loop_system(num_open, den_open):
    """
    Вычисляет передаточную функцию замкнутой системы
    с единичной отрицательной обратной связью
    W_closed(s) = W_open(s) / (1 + W_open(s))
    """
    # Приводим к одинаковой длине, дополняя нулями
    max_len = max(len(num_open), len(den_open))

    num_padded = np.pad(num_open, (max_len - len(num_open), 0), 'constant')
    den_padded = np.pad(den_open, (max_len - len(den_open), 0), 'constant')

    # Вычисляем знаменатель замкнутой системы: den_open + num_open
    den_closed = den_padded + num_padded

    # Убираем ведущие нули
    den_closed = np.trim_zeros(den_closed, 'f')
    if len(den_closed) == 0:
        den_closed = [0]

    return list(num_open), list(den_closed)


# Получаем коэффициенты замкнутой системы
num_closed, den_closed = get_closed_loop_system(num_open, den_open)

# Создаем системы
system_open = signal.TransferFunction(num_open, den_open)
system_closed = signal.TransferFunction(num_closed, den_closed)

# Вывод информации
print("РАЗОМКНУТАЯ СИСТЕМА:")
print(f"Числитель: {num_open}")
print(f"Знаменатель: {den_open}")
# print(f"W(s) = {num_open[0]} / ({den_open[0]}s⁴ + {den_open[1]}s³ + {den_open[2]}s² + {den_open[3]}s + {den_open[4]})")

print("\nЗАМКНУТАЯ СИСТЕМА:")
print(f"Числитель: {num_closed}")
print(f"Знаменатель: {den_closed}")
# print(
#     f"W(s) = {num_closed[0]} / ({den_closed[0]}s⁴ + {den_closed[1]}s³ + {den_closed[2]}s² + {den_closed[3]}s + {den_closed[4]})")

# Частотный диапазон
w_range = np.logspace(-1, 3, 1000)

# Создаем фигуру с несколькими subplots
plt.figure(figsize=(15, 10))

# 1. Диаграмма Найквиста (разомкнутая система)
plt.subplot(2, 3, 1)
w, frq_open = signal.freqresp(system_open, w_range)
plt.plot(frq_open.real, frq_open.imag, 'b-', linewidth=2, label='Разомкнутая')
plt.plot(frq_open.real[0], frq_open.imag[0], 'bo', markersize=6)
plt.plot(frq_open.real[-1], frq_open.imag[-1], 'bs', markersize=6)
plt.plot(-1, 0, 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2, label='Точка (-1, j0)')
plt.title('Диаграмма Найквиста\n(разомкнутая система)')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 2. ЛАЧХ сравнение
plt.subplot(2, 3, 2)
w, mag_open, phase_open = signal.bode(system_open, w_range)
w, mag_closed, phase_closed = signal.bode(system_closed, w_range)
plt.semilogx(w, mag_open, 'b-', linewidth=2, label='Разомкнутая')
plt.semilogx(w, mag_closed, 'r-', linewidth=2, label='Замкнутая')
plt.title('ЛАЧХ сравнение')
plt.ylabel('Амплитуда [дБ]')
plt.legend()
plt.grid(True, which='both', alpha=0.3)

# 3. ЛФЧХ сравнение
plt.subplot(2, 3, 3)
plt.semilogx(w, phase_open, 'b-', linewidth=2, label='Разомкнутая')
plt.semilogx(w, phase_closed, 'r-', linewidth=2, label='Замкнутая')
plt.title('ЛФЧХ сравнение')
plt.ylabel('Фаза [градусы]')
plt.xlabel('Частота [рад/с]')
plt.legend()
plt.grid(True, which='both', alpha=0.3)

# 4. АЧХ сравнение (линейная амплитуда)
plt.subplot(2, 3, 4)
amplitude_open = 10 ** (mag_open / 20)
amplitude_closed = 10 ** (mag_closed / 20)
plt.semilogx(w, amplitude_open, 'b-', linewidth=2, label='Разомкнутая')
plt.semilogx(w, amplitude_closed, 'r-', linewidth=2, label='Замкнутая')
plt.title('АЧХ сравнение (линейная амплитуда)')
plt.ylabel('Амплитуда')
plt.xlabel('Частота [рад/с]')
plt.legend()
plt.grid(True, which='both', alpha=0.3)

# 5. Переходная характеристика
plt.subplot(2, 3, 5)
t_open, y_open = signal.step(system_open)
t_closed, y_closed = signal.step(system_closed)
plt.plot(t_open, y_open, 'b-', linewidth=2, label='Разомкнутая')
plt.plot(t_closed, y_closed, 'r-', linewidth=2, label='Замкнутая')
plt.title('Переходная характеристика')
plt.ylabel('Амплитуда')
plt.xlabel('Время [с]')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. АФЧХ замкнутой системы
plt.subplot(2, 3, 6)
w, frq_closed = signal.freqresp(system_closed, w_range)
plt.plot(frq_closed.real, frq_closed.imag, 'r-', linewidth=2, label='Замкнутая')
plt.plot(frq_closed.real[0], frq_closed.imag[0], 'ro', markersize=6)
plt.plot(frq_closed.real[-1], frq_closed.imag[-1], 'rs', markersize=6)
plt.title('АФЧХ замкнутой системы')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.show()

# Анализ устойчивости
poles_open = np.roots(den_open)
poles_closed = np.roots(den_closed)

print("\nАНАЛИЗ УСТОЙЧИВОСТИ:")
print("Полюсы разомкнутой системы:", [f"{p:.4f}" for p in poles_open])
print("Устойчивость разомкнутой:", "Устойчива" if all(np.real(poles_open) < 0) else "Неустойчива")

print("\nПолюсы замкнутой системы:", [f"{p:.4f}" for p in poles_closed])
print("Устойчивость замкнутой:", "Устойчива" if all(np.real(poles_closed) < 0) else "Неустойчива")

# Критерий Найквиста
min_distance = np.min(np.sqrt((frq_open.real + 1) ** 2 + frq_open.imag ** 2))
print(f"\nКритерий Найквиста:")
print(f"Минимальное расстояние до точки (-1, j0): {min_distance:.4f}")
if min_distance > 0:
    print("Точка (-1, j0) не охватывается - система устойчива в замкнутом состоянии")
else:
    print("Точка (-1, j0) охватывается - система неустойчива в замкнутом состоянии")