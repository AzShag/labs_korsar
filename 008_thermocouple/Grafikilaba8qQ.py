import numpy as np
import matplotlib.pyplot as plt
import os

# Настройка шрифтов и графиков
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['grid.alpha'] = 0.5

BASE_PATH = r'c:\vscode'
k_list = [11, 21, 31, 41]
colors = {11: 'blue', 21: 'green', 31: 'orange', 41: 'red'}

def read_korsar_dat(filename, k_nodes):
    """Чтение файла с защитой от переноса строк"""
    with open(filename, 'r') as f:
        f.readline() # Пропускаем заголовок
        content = f.read()
    values = np.array(content.split(), dtype=float)
    num_cols = 1 + k_nodes + 7 
    return values.reshape(-1, num_cols)

# Создаем фигуру с 4-мя графиками
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Анализ нестационарного прогрева термопары (КОРСАР)', fontsize=16, fontweight='bold')

for k in k_list:
    filename = os.path.join(BASE_PATH, f'resultK{k}.dat')
    if not os.path.exists(filename):
        continue

    data = read_korsar_dat(filename, k)
    
    # Извлекаем данные
    time = data[:, 0]
    T_array = data[:, 1:1+k] # Массив температур по узлам (в Кельвинах)
    q_flux = data[:, -2]
    q_pow = data[:, -1]
    
    # Перевод в Цельсии
    T_center_C = T_array[:, 0] - 273.15      # Температура на оси
    T_surface_C = T_array[:, -1] - 273.15    # Температура на поверхности

    # Ограничиваем время для наглядности (первые 30 секунд)
    mask = time <= 30.0
    t_plot = time[mask]

    # --- 1. График: Центр vs Поверхность (строим только для K=41) ---
    if k == 41:
        axs[0, 0].plot(t_plot, T_center_C[mask], 'b-', linewidth=2, label='Температура на оси (центр)')
        axs[0, 0].plot(t_plot, T_surface_C[mask], 'r--', linewidth=2, label='Температура на поверхности')
        axs[0, 0].axhline(y=200, color='black', linestyle=':', label='Температура среды (200 °C)')
        axs[0, 0].set_title('Прогрев центра и поверхности (K=41)', fontsize=12)
        axs[0, 0].set_ylabel('Температура, °C', fontsize=11)
        axs[0, 0].legend()
        axs[0, 0].grid(True)

    # --- 2. График: Сходимость температуры на оси ---
    axs[0, 1].plot(t_plot, T_center_C[mask], color=colors[k], linewidth=1.5, label=f'K={k}')
    axs[0, 1].set_title('Сходимость температуры на оси по сетке', fontsize=12)
    axs[0, 1].set_ylabel('Температура, °C', fontsize=11)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- 3. График: Тепловой поток (q_flux) ---
    axs[1, 0].plot(t_plot, q_flux[mask], color=colors[k], linewidth=1.5, label=f'K={k}')
    axs[1, 0].set_title('Плотность теплового потока на поверхности', fontsize=12)
    axs[1, 0].set_xlabel('Время, с', fontsize=11)
    axs[1, 0].set_ylabel('Тепловой поток, Вт/м²', fontsize=11)
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- 4. График: Полная мощность (q_pow) ---
    axs[1, 1].plot(t_plot, q_pow[mask], color=colors[k], linewidth=1.5, label=f'K={k}')
    axs[1, 1].set_title('Суммарная мощность, передаваемая через поверхность', fontsize=12)
    axs[1, 1].set_xlabel('Время, с', fontsize=11)
    axs[1, 1].set_ylabel('Мощность, Вт', fontsize=11)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

# Общие настройки для осей X (время)
for ax in axs.flat:
    ax.set_xlim(0, 20) # Ограничим 20 секундами для детального рассмотрения переходного процесса
    ax.set_xlabel('Время, с', fontsize=11)

plt.tight_layout()
plt.subplots_adjust(top=0.92) # Оставляем место для главного заголовка
plt.show()