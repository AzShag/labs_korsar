import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Настройка шрифтов
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['grid.alpha'] = 0.5 

# ============================================================
# 1. КОНСТАНТЫ И НАСТРОЙКИ
# ============================================================
BASE_PATH = r'c:\vscode'
A_in = 5.0               
t_mean = 473.15          
tau_theor = 5.077        

# Частоты
frequencies = [0.01, 0.02, 0.03, 0.05, 0.08]

f_valid, K_exp, Phi_exp = [], [], []
K_theor_arr, Phi_theor_arr = [], []

def sine_func(t, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * t + phase) + offset

print(f"{'Частота (Гц)':<12} | {'K(КОРСАР)':<10} | {'K(Теор)':<10} | {'Ош. K,%':<10} | {'Фаза(КОРСАР)':<12} | {'Фаза(Теор)':<12} | {'Ош. Ф,°':<10}")
print("-" * 95)

# ============================================================
# 2. ОБРАБОТКА ДАННЫХ
# ============================================================
for f in frequencies:
    filename = os.path.join(BASE_PATH, f'res_{f}.dat')
    if not os.path.exists(filename):
        continue

    data = np.loadtxt(filename, skiprows=1)
    time = data[:, 0]
    T_env = data[:, 1]
    T_center = data[:, 2]

    # Отбрасываем переходный процесс (берем вторую половину файла)
    t_cut = time[-1] * 0.5 
    mask = time > t_cut
    time_fit = time[mask]
    T_c_fit = T_center[mask]
    T_e_fit = T_env[mask]

    # --- Расчет АЧХ и ФЧХ через curve_fit ---
    try:
        popt_e, _ = curve_fit(sine_func, time_fit, T_e_fit, p0=[A_in, f, 0, t_mean])
        popt_c, _ = curve_fit(sine_func, time_fit, T_c_fit, p0=[(max(T_c_fit)-min(T_c_fit))/2, f, 0, t_mean])
        
        K = popt_c[0] / popt_e[0] # Отношение амплитуд
        phase_e = np.degrees(popt_e[2])
        phase_c = np.degrees(popt_c[2])
        
        # Разница фаз (запаздывание)
        phase_deg = (phase_c - phase_e) % 360
        if phase_deg > 180: phase_deg -= 360
            
    except RuntimeError:
        K = np.nan
        phase_deg = np.nan

    # --- Теоретические значения ---
    omega_tau = 2 * np.pi * f * tau_theor
    K_th = 1.0 / np.sqrt(1.0 + omega_tau**2)
    Phi_th = -np.degrees(np.arctan(omega_tau))
    
    f_valid.append(f)
    K_exp.append(K)
    Phi_exp.append(phase_deg)
    K_theor_arr.append(K_th)
    Phi_theor_arr.append(Phi_th)
    
    err_K = abs(K - K_th) / K_th * 100 
    err_Phi = abs(phase_deg - Phi_th)
    
    print(f"{f:<12.4f} | {K:<10.4f} | {K_th:<10.4f} | {err_K:<10.2f} | {phase_deg:<12.2f} | {Phi_th:<12.2f} | {err_Phi:<10.2f}")

# Преобразование в массивы
f_valid = np.array(f_valid)
K_exp = np.array(K_exp)
Phi_exp = np.array(Phi_exp)
K_theor_arr = np.array(K_theor_arr)
Phi_theor_arr = np.array(Phi_theor_arr)
err_K_arr = np.abs(K_exp - K_theor_arr) / K_theor_arr * 100
err_Phi_arr = np.abs(Phi_exp - Phi_theor_arr)

# ============================================================
# 3. ПОСТРОЕНИЕ КРАСИВЫХ ГРАФИКОВ
# ============================================================
f_plot = np.logspace(np.log10(min(f_valid)*0.8), np.log10(max(f_valid)*1.2), 100)
K_plot = 1.0 / np.sqrt(1.0 + (2 * np.pi * f_plot * tau_theor)**2)
Phi_plot = -np.degrees(np.arctan(2 * np.pi * f_plot * tau_theor))
f_c_theor = 1.0 / (2 * np.pi * tau_theor)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

# АЧХ
ax1.semilogx(f_plot, K_plot, 'k--', linewidth=2, label=f'Теория (tau={tau_theor:.2f} с)')
ax1.semilogx(f_valid, K_exp, 'bo-', linewidth=1.5, markersize=8, label='Эксперимент (КОРСАР)')
ax1.axhline(y=1/np.sqrt(2), color='r', linestyle=':', linewidth=1.5, label='Уровень -3 дБ (0.707)')
ax1.axvline(x=f_c_theor, color='r', linestyle=':', linewidth=1.5, label=f'f_c = {f_c_theor:.4f} Гц')

ax1.set_ylabel('Коэффициент передачи, K', fontsize=12)
ax1.set_title('Амплитудно-частотная характеристика (АЧХ)', fontsize=14)
ax1.set_ylim(0, 1.1) 
ax1.grid(True, which="both", ls="--")
ax1.legend(loc='lower left', fontsize=11)

# ФЧХ
ax2.semilogx(f_plot, Phi_plot, 'k--', linewidth=2, label='Теория')
ax2.semilogx(f_valid, Phi_exp, 'go-', linewidth=1.5, markersize=8, label='Эксперимент (КОРСАР)')

ax2.set_xlabel('Частота, Гц', fontsize=12)
ax2.set_ylabel('Сдвиг фазы, градусы', fontsize=12)
ax2.set_title('Фазо-частотная характеристика (ФЧХ)', fontsize=14)
ax2.set_ylim(-100, 5) 
ax2.grid(True, which="both", ls="--")
ax2.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.show()

# ============================================================
# 4. ГРАФИКИ ОТКЛОНЕНИЙ
# ============================================================
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax3.semilogx(f_valid, err_K_arr, 'rs-', linewidth=1.5, markersize=8)
ax3.set_ylabel('Относительная ошибка K, %', fontsize=12)
ax3.set_title('Отклонение КОРСАР от теории (АЧХ)', fontsize=14)
ax3.set_ylim(bottom=0)
ax3.grid(True, which="both", ls="--")

ax4.semilogx(f_valid, err_Phi_arr, 'ms-', linewidth=1.5, markersize=8)
ax4.set_xlabel('Частота, Гц', fontsize=12)
ax4.set_ylabel('Абсолютная ошибка фазы, град', fontsize=12)
ax4.set_title('Отклонение КОРСАР от теории (ФЧХ)', fontsize=14)
ax4.set_ylim(bottom=0)
ax4.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()