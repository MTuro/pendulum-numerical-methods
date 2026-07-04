import numpy as np
from pendulum import (f_nonlinear, f_linear, integrate_fixed, integrate_adaptive,
                      theta_analytic, period_analytic, detect_period,
                      time_for_n_periods, G, L)

theta0 = np.radians(5.0)  # angulo pequeno (deveria bater com a aproximacao linear)
y0 = [theta0, 0.0]

T_lin = period_analytic()
print(f"periodo linear teorico: {T_lin:.6f} s")

t_sim = time_for_n_periods(theta0, n_periods=10)
ts, ys = integrate_fixed(f_nonlinear, y0, 0.0, t_sim, h=0.001)
T_num, crossings = detect_period(ts, ys, n_periods=10)
print(f"periodo numerico (nao-linear, theta0=5deg, h=0.001): {T_num:.6f} s")
print(f"Diferenca: {abs(T_num - T_lin):.2e} s")
print(f"Numero de cruzamentos detectados: {len(crossings)}")

# teste passo adaptativo
ts_a, ys_a, hs_a, n_acc, n_rej = integrate_adaptive(f_nonlinear, y0, 0.0, t_sim, eps=1e-5)
T_num_a, _ = detect_period(ts_a, ys_a, n_periods=10)
print(f"\npasso adaptativo: {n_acc} passos aceitos, {n_rej} rejeitados")
print(f"periodo numerico (adaptativo): {T_num_a:.6f} s")
print(f"h medio: {np.mean(hs_a):.6f}, h min: {np.min(hs_a):.6f}, h max: {np.max(hs_a):.6f}")