import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from scipy.optimize import brentq
import os

# --- НАСТРОЙКА ГРАФИКОВ ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm' 

BASE_PATH = r'c:\vscode'
R = 0.0025
T0 = 293.15
T_inf = 473.15

cases = [
    {'file': 'resultK21.dat',       'Bi': 0.0833333, 'alpha': 1000,  'color': 'blue',  'label': r'$Bi=0.083$'},
    {'file': 'resultK21_a5000.dat', 'Bi': 0.4166667, 'alpha': 5000,  'color': 'green', 'label': r'$Bi=0.416$'},
    {'file': 'resultK21_bi1.dat',   'Bi': 1.0000000, 'alpha': 12000, 'color': 'red',   'label': r'$Bi=1.000$'}
]

# --- ФУНКЦИИ АНАЛИТИКИ ---
def get_mu_n(bi, n_max=100):
    mu = np.zeros(n_max)
    for n in range(n_max):
        mu[n] = brentq(lambda x: x * j1(x) - bi * j0(x), 0.1 + n * np.pi, 0.1 + (n + 1) * np.pi)
    return mu

def theta_exact(eta, fo, bi):
    mu = get_mu_n(bi, n_max=50)
    theta = 0.0
    for n in range(len(mu)):
        mu_n = mu[n]
        An = (2 * bi) / ((mu_n**2 + bi**2) * j0(mu_n))
        term = An * j0(mu_n * eta) * np.exp(-mu_n**2 * fo)
        theta = theta + term
        if np.max(np.abs(term)) < 1e-7: break
    return theta

def Theta_approx(eta, fo, bi):
    return 1.0 - j0(np.sqrt(bi) * eta) * np.exp(-2.0 * bi * fo)

def read_dat(filename, k_nodes):
    filepath = os.path.join(BASE_PATH, filename)
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f:
        f.readline()
        content = f.read()
    return np.array(content.split(), dtype=float).reshape(-1, 1 + k_nodes + 7)

for case in cases:
    case['data'] = read_dat(case['file'], 21)
    if case['data'] is not None:
        d = case['data']
        case['time'] = d[:, 0]
        case['T_center'] = d[:, 1]
        case['T_surface'] = d[:, 22]
        case['Fo'] = d[:, 23]

r_nodes = np.linspace(0, R, 21)
eta_nodes = r_nodes / R
t_plot = 2.0 

# ============================================================
# ОКНО 1: ПРОФИЛИ ТЕМПЕРАТУР (ТОЛЬКО КОРСАР)
# ============================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle(f'Профили температур (t={t_plot} с) для разных Bi', fontsize=16, fontweight='bold')

for case in cases:
    if case['data'] is None: continue
    idx = np.argmin(np.abs(case['time'] - t_plot))
    T_prof = case['data'][idx, 1:22]
    
    # Безразмерные (только КОРСАР)
    Theta_prof = (T_prof - T_inf) / (T0 - T_inf)
    ax1.plot(eta_nodes, Theta_prof, marker='o', markersize=4, linestyle='-', color=case['color'], label=f'KORSAR {case["label"]}')
    
    # Размерные (только КОРСАР)
    ax2.plot(r_nodes * 1000, T_prof, marker='o', markersize=4, linestyle='-', color=case['color'], label=f'KORSAR {case["label"]}')

ax1.set_xlabel(r'Безразмерная координата $\eta = r/R$', fontsize=12)
ax1.set_ylabel(r'Безразмерная $\Theta$', fontsize=12)
ax1.set_title('Безразмерные профили (КОРСАР)', fontsize=14)
ax1.grid(True); ax1.legend(fontsize=9)

ax2.set_xlabel('Радиус r, мм', fontsize=12)
ax2.set_ylabel('Температура T, К', fontsize=12)
ax2.set_title('Размерные профили (КОРСАР)', fontsize=14)
ax2.grid(True); ax2.legend(fontsize=9)
plt.tight_layout()


# ============================================================
# ОКНО 2: ПРОГРЕВ ВО ВРЕМЕНИ
# ============================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Динамика прогрева термопары', fontsize=16, fontweight='bold')

for case in cases:
    if case['data'] is None: continue
    mask = case['time'] <= 10.0 
    ax3.plot(case['time'][mask], case['T_center'][mask], color=case['color'], linewidth=2, label=case['label'])
    ax4.plot(case['time'][mask], case['T_surface'][mask], color=case['color'], linewidth=2, label=case['label'])

ax3.set_xlabel('Время, с', fontsize=12)
ax3.set_ylabel('Температура на оси, К', fontsize=12)
ax3.set_title('Прогрев центра', fontsize=14)
ax3.grid(True); ax3.legend()

ax4.set_xlabel('Время, с', fontsize=12)
ax4.set_ylabel('Температура поверхности, К', fontsize=12)
ax4.set_title('Прогрев наружной поверхности', fontsize=14)
ax4.grid(True); ax4.legend()
plt.tight_layout()


# ============================================================
# ОКНО 3: ГРАФИКИ ПЕХОВИЧА
# ============================================================
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Анализ сходимости с теорией', fontsize=16, fontweight='bold')

for case in cases:
    if case['data'] is None: continue
    Fo = case['Fo']
    Bi_val = case['Bi']
    
    # --- ЛЕВЫЙ ГРАФИК: График 43-1 ---
    theta_c_korsar = (case['T_center'] - T_inf) / (T0 - T_inf)
    ax5.semilogx(Fo, 1 - theta_c_korsar, color=case['color'], linewidth=2, label=f'KORSAR {case["label"]}')
    theta_c_exact = theta_exact(0.0, Fo, Bi_val)
    ax5.semilogx(Fo, 1 - theta_c_exact, color=case['color'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    if Bi_val < 0.1:
        Theta_c_approx = Theta_approx(0.0, Fo, Bi_val)
        ax5.semilogx(Fo, Theta_c_approx, color=case['color'], linestyle=':', linewidth=2, label=r'Приближение $Bi \leq 0.1$')

    # --- ПРАВЫЙ ГРАФИК: Размерная Т на поверхности и оси (во времени) ---
    mask = case['time'] <= 30.0
    t_arr = case['time'][mask]
    
    # KORSAR (сплошные линии: толстая - поверхность, тонкая - центр)
    ax6.plot(t_arr, case['T_surface'][mask], linestyle='-', linewidth=2, color=case['color'], label=f'KORSAR (поверх.) {case["label"]}')
    ax6.plot(t_arr, case['T_center'][mask], linestyle='-', linewidth=1, color=case['color'], alpha=0.6, label=f'KORSAR (центр) {case["label"]}')
    
    # Аналитика (треугольники: вверх - поверхность, вниз - центр)
    Fo_arr = case['Fo'][mask]
    T_surf_anal = T_inf + theta_exact(1.0, Fo_arr, Bi_val) * (T0 - T_inf)
    T_cent_anal = T_inf + theta_exact(0.0, Fo_arr, Bi_val) * (T0 - T_inf)
    
    # markevery=150 прореживает треугольники, чтобы они не слипались
    ax6.plot(t_arr, T_surf_anal, marker='^', markersize=6, linestyle='None', markevery=150, color=case['color'], label=f'Аналитика (поверх.) {case["label"]}')
    ax6.plot(t_arr, T_cent_anal, marker='v', markersize=6, linestyle='None', markevery=150, color=case['color'], label=f'Аналитика (центр) {case["label"]}')

ax5.set_xlabel('Число Фурье, Fo', fontsize=12)
ax5.set_ylabel(r'$\Theta_{\eta=0}$', fontsize=12)
ax5.set_title('Безразмерная Т на оси (График 43-1)', fontsize=14)
ax5.grid(True, which="both", ls="--"); ax5.legend(fontsize=9)

ax6.set_xlabel('Время, с', fontsize=12)
ax6.set_ylabel('Температура T, К', fontsize=12)
ax6.set_title('Размерная Т на поверхности и оси', fontsize=14)
ax6.grid(True); ax6.legend(fontsize=8)

plt.tight_layout()
plt.show()