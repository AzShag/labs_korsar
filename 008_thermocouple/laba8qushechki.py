import numpy as np
import os

BASE_PATH = r'c:\vscode'
k_list = [11, 21, 31, 41]
t_target = 2.0  # Момент времени для сравнения (t = 2.0 с)

# ИСПОЛЬЗУЕМ r"" ДЛЯ СТРОК С ФОРМУЛАМИ
print(r"| Число узлов \(K\) | Время \(t\), с | \(q\) (КОРСАР), Вт/м² | \(Q\) (КОРСАР), Вт |")
print("|---|---|---|---|")

for k in k_list:
    filename = os.path.join(BASE_PATH, f'resultK{k}.dat')
    if not os.path.exists(filename):
        print(f"| {k} | — | *файл не найден* | *файл не найден* |")
        continue

    # Читаем файл (пропускаем заголовок)
    with open(filename, 'r') as f:
        f.readline()
        content = f.read()
    
    values = np.array(content.split(), dtype=float)
    # Колонки: TAU + T(1:K) + TW + Fo + Bi + theta_c + theta_s + q_flux + q_pow
    num_cols = 1 + k + 7 
    data = values.reshape(-1, num_cols)
    
    time = data[:, 0]
    q_flux = data[:, -2]  # Предпоследний столбец
    q_pow  = data[:, -1]  # Последний столбец
    
    # Ищем ближайший момент времени к t_target
    idx = np.argmin(np.abs(time - t_target))
    t_actual = time[idx]
    q_val = q_flux[idx]
    Q_val = q_pow[idx]
    
    # Выводим строку таблицы Markdown
    print(f"| {k} | {t_actual:.2f} | {q_val:.2f} | {Q_val:.4f} |")

print("\n--- Для стационарного режима (конец расчета) ---")
# ИСПОЛЬЗУЕМ r"" ДЛЯ СТРОКИ С ФОРМУЛАМИ (двойной слеш здесь не нужен, так как строка сырая)
print(r"| Число узлов \(K\) | Время \(t\), с | \(q_{\text{ст}}\) (КОРСАР), Вт/м² | \(Q_{\text{ст}}\) (КОРСАР), Вт |")
print("|---|---|---|---|")

for k in k_list:
    filename = os.path.join(BASE_PATH, f'resultK{k}.dat')
    if not os.path.exists(filename): continue
    with open(filename, 'r') as f:
        f.readline()
        content = f.read()
    values = np.array(content.split(), dtype=float)
    num_cols = 1 + k + 7 
    data = values.reshape(-1, num_cols)
    time = data[:, 0]
    q_flux = data[:, -2]
    q_pow  = data[:, -1]
    print(f"| {k} | {time[-1]:.2f} | {q_flux[-1]:.2e} | {q_pow[-1]:.2e} |")