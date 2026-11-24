import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from scipy.optimize import brentq
import os

# --- НАСТРОЙКА ГРАФИКОВ ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm' # Включает поддержку красивых формул

# --- 1. КОНСТАНТЫ ---
BASE_PATH = r'c:\vscode'
R = 0.0025               
a = 7.692308e-6          
T0 = 293.15              
T_inf = 473.15           
Bi = 0.0833333           
dT_max = T_inf - T0      

# --- 2. ФУНКЦИИ АНАЛИТИКИ ---
def get_mu_n(bi, n_max=100):
    mu = np.zeros(n_max)
    for n in range(n_max):
        a_guess = 0.1 + n * np.pi
        b_guess = a_guess + np.pi
        mu[n] = brentq(lambda x: x * j1(x) - bi * j0(x), a_guess, b_guess)
    return mu

def theta_exact(eta, fo, bi):
    mu = get_mu_n(bi, n_max=50)
    theta = 0.0
    for n in range(len(mu)):
        mu_n = mu[n]
        An = (2 * bi) / ((mu_n**2 + bi**2) * j0(mu_n))
        term = An * j0(mu_n * eta) * np.exp(-mu_n**2 * fo)
        theta = theta + term
        if np.max(np.abs(term)) < 1e-7: 
            break
    return theta

def read_korsar_dat(filename, k_nodes):
    with open(filename, 'r') as f:
        f.readline() # Пропускаем заголовок
        content = f.read()
    values = np.array(content.split(), dtype=float)
    num_cols = 1 + k_nodes + 7 
    return values.reshape(-1, num_cols)

# --- 3. НАСТРОЙКА СТИЛЕЙ ---
k_list = [11, 21, 31, 41]
colors = {11: 'blue', 21: 'green', 31: 'orange', 41: 'red'}
markers = {11: 'o', 21: '^', 31: 's', 41: 'D'} # Круг, треугольник, квадрат, ромб

taus = []

# Создаем фигуры
fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(14, 5))
fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 5))
fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 4. ОБРАБОТКА ДАННЫХ ---
print("--- АНАЛИЗ СКАЧКА ---")
for k in k_list:
    filename = os.path.join(BASE_PATH, f'resultK{k}.dat')
    if not os.path.exists(filename):
        print(f"Внимание: Файл {filename} не найден!")
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
    print(f"K={k}: tau = {tau_k:.4f} с")

    # Б) Профиль при t=2.0 с
    idx_t = np.argmin(np.abs(time - 2.0))
    t_actual = time[idx_t]
    fo_actual = Fo_array[idx_t]
    r_nodes = np.linspace(0, R, k)
    eta_nodes = r_nodes / R
    T_korsar_prof = T_array[idx_t, :]

    theta_anal_prof = theta_exact(eta_nodes, fo_actual, Bi)
    T_anal_prof = T_inf + theta_anal_prof * (T0 - T_inf)
    
    # В) Ошибка по радиусу (в момент t=2.0)
    error_rel = (np.abs(T_korsar_prof - T_anal_prof) / dT_max) * 100

    # Г) Ошибка на оси во времени
    theta_c_exact_time = theta_exact(0.0, Fo_array, Bi) 
    T_c_exact_time = T_inf + theta_c_exact_time * (T0 - T_inf)
    error_center_time = (np.abs(T_center - T_c_exact_time) / dT_max) * 100

    # --- ПОСТРОЕНИЕ ---
    
    # 1. Кривые нагрева (линия с редкими мелкими маркерами)
    mask_heat = time <= 20.0
    ax1_1.plot(time[mask_heat], T_center[mask_heat], 
               color=colors[k], marker=markers[k], markevery=4000, 
               markersize=3, linestyle='-', linewidth=1, label=f'K={k}')
               
    # 2. Сходимость tau (только точки)
    ax1_2.plot(k, tau_k, marker=markers[k], color=colors[k], 
               markersize=6, linestyle='None')

    # 3. Профили (ТОЛЬКО ТОЧКИ, без линий)
    ax2_1.plot(r_nodes * 1000, T_korsar_prof, marker=markers[k], color=colors[k], 
               markersize=4, linestyle='None', label=f'KORSAR K={k}')
               
    theta_korsar_prof = (T_korsar_prof - T_inf) / (T0 - T_inf)
    ax2_2.plot(eta_nodes, theta_korsar_prof, marker=markers[k], color=colors[k], 
               markersize=4, linestyle='None', label=f'KORSAR K={k}')
    
    if k == 21:
        # Аналитику оставляем линией
        ax2_1.plot(r_nodes * 1000, T_anal_prof, 'k--', linewidth=2, label='Аналитика')
        ax2_2.plot(eta_nodes, theta_anal_prof, 'k--', linewidth=2, label='Аналитика')

    # 4. Ошибки по радиусу (ЗДЕСЬ ДОБАВЛЕНЫ ЛИНИИ + МАРКЕРЫ)
    ax3_1.plot(eta_nodes, error_rel, marker=markers[k], color=colors[k], 
               markersize=4, linestyle='-', linewidth=1, label=f'K={k}')
    
    # 5. Ошибки по времени (линии с маркерами)
    mask_err = time <= 10.0
    ax3_2.plot(time[mask_err], error_center_time[mask_err], 
               color=colors[k], marker=markers[k], markevery=4000, 
               markersize=3, linestyle='-', linewidth=1, label=f'K={k}')


# --- 5. ОФОРМЛЕНИЕ ВСЕХ ГРАФИКОВ ---

# Линия тренда для tau
ax1_2.plot(k_list, taus, 'k--', alpha=0.3)

# Окно 1: Нагрев и Сходимость
ax1_1.set_xlabel('Время, с'); ax1_1.set_ylabel('Температура на оси, К')
ax1_1.set_title('Кривые нагрева (сходимость по K)')
ax1_1.axhline(y=T0 + 0.632*dT_max, color='black', linestyle=':', label='63.2%')
ax1_1.grid(True); ax1_1.legend()

ax1_2.set_xlabel('Число узлов K'); ax1_2.set_ylabel('Постоянная времени tau, с')
ax1_2.set_title('Сходимость постоянной времени')
ax1_2.set_xticks(k_list)
ax1_2.grid(True)

# Окно 2: Профили (ИСПОЛЬЗОВАНЫ ЗНАКИ $ ДЛЯ ГРЕЧЕСКИХ БУКВ)
ax2_1.set_xlabel('Радиус r, мм'); ax2_1.set_ylabel('Температура T, К')
ax2_1.set_title(f'Размерные профили (t={t_actual:.1f} с)')
ax2_1.grid(True); ax2_1.legend()

ax2_2.set_xlabel(r'Безразмерная координата $\eta = r/R$') 
ax2_2.set_ylabel(r'Безразмерная $\Theta$')
ax2_2.set_title(f'Безразмерные профили (Fo={fo_actual:.3f})')
ax2_2.grid(True); ax2_2.legend()

# Окно 3: Ошибки
ax3_1.set_xlabel(r'Безразмерная координата $\eta = r/R$')
ax3_1.set_ylabel('Относительная ошибка, %')
ax3_1.set_title(f'Отклонение по радиусу (t={t_actual:.1f} с)')
ax3_1.grid(True); ax3_1.legend()

ax3_2.set_xlabel('Время, с')
ax3_2.set_ylabel('Относительная ошибка на оси, %')
ax3_2.set_title('Ошибка КОРСАР на оси во времени')
ax3_2.grid(True); ax3_2.legend()

plt.tight_layout()
print("\nВсе графики построены.")
plt.show()