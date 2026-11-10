import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from scipy.optimize import brentq
import os

# Настройка шрифтов для корректного отображения русских букв
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. КОНСТАНТЫ И ФУНКЦИИ АНАЛИТИКИ
# ============================================================
BASE_PATH = r'c:\vscode'
R = 0.0025               # Радиус, м
a = 7.692308e-6          # Температуропроводность, м2/с
T0 = 293.15              # Начальная температура, К
T_inf = 473.15           # Температура среды, К
Bi = 0.0833333           # Число Био
dT_max = T_inf - T0      # Полный перепад температур

def get_mu_n(bi, n_max=100):
    """Поиск корней уравнения mu*J1(mu) - Bi*J0(mu) = 0"""
    mu = np.zeros(n_max)
    for n in range(n_max):
        a_guess = 0.1 + n * np.pi
        b_guess = a_guess + np.pi
        mu[n] = brentq(lambda x: x * j1(x) - bi * j0(x), a_guess, b_guess)
    return mu

def theta_exact(eta, fo, bi):
    """Точное решение (ряд Пеховича). Работает и для скалярных, и для векторных fo."""
    mu = get_mu_n(bi, n_max=50)
    # Инициализируем нулем (скаляром). Numpy сам расширит размерность при сложении
    theta = 0.0 
    for n in range(len(mu)):
        mu_n = mu[n]
        An = (2 * bi) / ((mu_n**2 + bi**2) * j0(mu_n))
        term = An * j0(mu_n * eta) * np.exp(-mu_n**2 * fo)
        theta = theta + term
        if np.max(np.abs(term)) < 1e-7: 
            break
    return theta

def theta_approx(eta, fo, bi):
    """Приближенное решение для Bi <= 0.1 из доп. сведений Пеховича"""
    return 1.0 - j0(np.sqrt(bi) * eta) * np.exp(-2.0 * bi * fo)

def read_korsar_dat(filename, k_nodes):
    """Чтение файла КОРСАР с защитой от переноса длинных строк"""
    with open(filename, 'r') as f:
        f.readline() # Пропускаем заголовок
        content = f.read()
    values = np.array(content.split(), dtype=float)
    num_cols = 1 + k_nodes + 7 # TAU + T(1:K) + TW, Fo, Bi, theta_c, theta_s, q_flux, q_pow
    return values.reshape(-1, num_cols)

# ============================================================
# 2. ОБРАБОТКА ДАННЫХ
# ============================================================
k_list = [11, 21, 31, 41]
colors = ['blue', 'green', 'orange', 'red']
taus = []

# Словари для хранения данных K=41 (для графиков Пеховича)
data_k41 = None 
Fo_k41 = None
T_center_k41 = None
T_surface_k41 = None

print("--- АНАЛИЗ ДАННЫХ ---")
for k, color in zip(k_list, colors):
    filename = os.path.join(BASE_PATH, f'resultK{k}.dat')
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        continue

    data = read_korsar_dat(filename, k)
    time = data[:, 0]
    T_array = data[:, 1:1+k]       
    Fo_array = data[:, 2+k]        

    # А) Поиск постоянной времени
    T_target = T0 + 0.632 * dT_max
    T_center = T_array[:, 0]
    idx = np.where(T_center >= T_target)[0][0]
    tau_k = np.interp(T_target, T_center[idx-1:idx+1], time[idx-1:idx+1])
    taus.append(tau_k)
    
    # Сохраняем данные K=41 для детального анализа Пеховича
    if k == 41:
        data_k41 = data
        Fo_k41 = Fo_array
        T_center_k41 = T_center
        T_surface_k41 = T_array[:, -1]

    # Б) Выбор момента для профилей (t = 2.0 с)
    t_plot = 2.0 
    idx_t = np.argmin(np.abs(time - t_plot))
    t_actual = time[idx_t]
    fo_actual = Fo_array[idx_t]
    r_nodes = np.linspace(0, R, k)
    eta_nodes = r_nodes / R
    T_korsar_prof = T_array[idx_t, :]

    # В) Ошибки (для профиля)
    theta_anal_prof = theta_exact(eta_nodes, fo_actual, Bi)
    T_anal_prof = T_inf + theta_anal_prof * (T0 - T_inf)
    error_rel = (np.abs(T_korsar_prof - T_anal_prof) / dT_max) * 100

    print(f"K={k:2d}: tau = {tau_k:.3f} с | Макс. ошибка в t=2с: {np.max(error_rel):.3f} %")

    # --- ИНИЦИАЛИЗАЦИЯ ГРАФИКОВ ---
    if k == 11: # Создаем фигуры только при первой итерации
        fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(14, 5))
        fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 5))
        fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(14, 5))
        fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- ПОСТРОЕНИЕ ГРАФИКОВ ---
    
    # ГРАФИК 1: Кривые нагрева и Сходимость tau
    mask_heat = time <= 20.0
    ax1_1.plot(time[mask_heat], T_center[mask_heat], '-', color=color, label=f'K={k}')
    ax1_2.plot(k, tau_k, 'o', color=color)
    
    # ГРАФИК 2: Профили температур (Размерный и Безразмерный)
    ax2_1.plot(r_nodes * 1000, T_korsar_prof, '-', color=color, label=f'KORSAR K={k}')
    theta_korsar_prof = (T_korsar_prof - T_inf) / (T0 - T_inf)
    ax2_2.plot(eta_nodes, theta_korsar_prof, '-', color=color, label=f'KORSAR K={k}')
    
    if k == 21: # Аналитику рисуем один раз
        ax2_1.plot(r_nodes * 1000, T_anal_prof, 'k--', linewidth=2, label='Аналитика')
        ax2_2.plot(eta_nodes, theta_anal_prof, 'k--', linewidth=2, label='Аналитика')

    # ГРАФИК 3: Ошибки по радиусу и по времени
    ax3_1.plot(eta_nodes, error_rel, '-o', color=color, markersize=4, label=f'K={k}')
    
    # Ошибка на оси во времени (считаем только для K=41 для скорости)
    if k == 41: 
        # Передаем eta=0.0 (скаляр) и Fo_array (массив), получаем массив
        theta_c_exact_time = theta_exact(0.0, Fo_array, Bi) 
        T_c_exact_time = T_inf + theta_c_exact_time * (T0 - T_inf)
        error_center_time = (np.abs(T_center - T_c_exact_time) / dT_max) * 100
        mask_err = time <= 10.0
        ax3_2.plot(time[mask_err], error_center_time[mask_err], '-', color=color, label=f'K={k}')

# Добавляем линию сходимости на График 1 (справа)
ax1_2.plot(k_list, taus, 'k--', alpha=0.5, label='Тренд')
ax1_2.set_xticks(k_list)

# ============================================================
# 3. ГРАФИКИ ПЕХОВИЧА (Только для K=41)
# ============================================================
# Вычисляем точную и приближенную аналитику для всех Fo
# Передаем скаляры для eta и массив для Fo, чтобы получить массивы функций от времени
theta_c_exact = theta_exact(0.0, Fo_k41, Bi)
theta_s_exact = theta_exact(1.0, Fo_k41, Bi)
theta_c_approx = theta_approx(0.0, Fo_k41, Bi)
theta_s_approx = theta_approx(1.0, Fo_k41, Bi)

# Безразмерные для КОРСАР
theta_c_k41 = (T_center_k41 - T_inf) / (T0 - T_inf)
theta_s_k41 = (T_surface_k41 - T_inf) / (T0 - T_inf)

# График 4-1: Безразмерная температура на оси (Theta от Fo)
ax4_1.semilogx(Fo_k41, 1 - theta_c_k41, 'b-', linewidth=2, label='KORSAR')
ax4_1.semilogx(Fo_k41, 1 - theta_c_exact, 'k--', label='Точный ряд')
ax4_1.semilogx(Fo_k41, theta_c_approx, 'r:', label='Приближение Bi<=0.1')

# График 4-2: Размерная температура на поверхности (T от t)
ax4_2.plot(time, T_surface_k41, 'b-', linewidth=2, label='KORSAR (поверхность)')
ax4_2.plot(time, T_inf + theta_s_exact*(T0-T_inf), 'k--', label='Аналитика (поверхность)')
# Для сравнения добавим центр
ax4_2.plot(time, T_center_k41, 'g-', linewidth=1, label='KORSAR (центр)')


# ============================================================
# 4. ОФОРМЛЕНИЕ ВСЕХ ГРАФИКОВ
# ============================================================

# Окно 1: Нагрев и Сходимость
ax1_1.set_xlabel('Время, с'); ax1_1.set_ylabel('Температура на оси, К')
ax1_1.set_title('Кривые нагрева (сходимость по K)')
ax1_1.axhline(y=T0 + 0.632*dT_max, color='black', linestyle=':', label='63.2%')
ax1_1.grid(True); ax1_1.legend()

ax1_2.set_xlabel('Число узлов K'); ax1_2.set_ylabel('Постоянная времени tau, с')
ax1_2.set_title('Сходимость постоянной времени')
ax1_2.grid(True); ax1_2.legend()

# Окно 2: Профили
ax2_1.set_xlabel('Радиус r, мм'); ax2_1.set_ylabel('Температура T, К')
ax2_1.set_title(f'Размерные профили (t={t_actual:.1f} с)')
ax2_1.grid(True); ax2_1.legend()

ax2_2.set_xlabel(r'Безразмерная координата \(\eta = r/R\)'); ax2_2.set_ylabel(r'Безразмерная \(\theta\)')
ax2_2.set_title(f'Безразмерные профили (Fo={fo_actual:.3f})')
ax2_2.grid(True); ax2_2.legend()

# Окно 3: Ошибки
ax3_1.set_xlabel(r'Безразмерная координата \(\eta = r/R\)'); ax3_1.set_ylabel('Относительная ошибка, %')
ax3_1.set_title(f'Отклонение по радиусу (t={t_actual:.1f} с)')
ax3_1.grid(True); ax3_1.legend()

ax3_2.set_xlabel('Время, с'); ax3_2.set_ylabel('Относительная ошибка на оси, %')
ax3_2.set_title('Ошибка КОРСАР на оси во времени')
ax3_2.grid(True); ax3_2.legend()

# Окно 4: Графики Пеховича
ax4_1.set_xlabel('Число Фурье, Fo'); ax4_1.set_ylabel(r'$\Theta_{\eta=0}$')
ax4_1.set_title('Безразмерная T на оси (График 43-1)')
ax4_1.grid(True, which="both", ls="--"); ax4_1.legend()

ax4_2.set_xlabel('Время, с'); ax4_2.set_ylabel('Температура T, К')
ax4_2.set_title('Размерная T на поверхности и оси')
ax4_2.set_xlim(0, 30)
ax4_2.grid(True); ax4_2.legend()

plt.tight_layout()
print("\nВсе графики построены. Закройте окна для завершения.")
plt.show()