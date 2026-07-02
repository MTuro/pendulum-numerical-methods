import numpy as np
import time

# variaveis fisicas
G = 9.81      # aceleracao da gravidade (m/s^2)
L = 1.0       # comprimento da haste (m)


def f_nonlinear(t, y, g=G, l=L):
    theta, omega = y
    return np.array([omega, -(g / l) * np.sin(theta)])


def f_linear(t, y, g=G, l=L):
    theta, omega = y
    return np.array([omega, -(g / l) * theta])


# passo unico de RK4
def rk4_step(f, t, y, h, **kwargs):
    k1 = f(t, y, **kwargs)
    k2 = f(t + h / 2.0, y + h / 2.0 * k1, **kwargs)
    k3 = f(t + h / 2.0, y + h / 2.0 * k2, **kwargs)
    k4 = f(t + h, y + h * k3, **kwargs)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)



# integracao com passo constante
def integrate_fixed(f, y0, t0, t_end, h, **kwargs):
    n_steps = int(np.ceil((t_end - t0) / h))
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, 2))
    ts[0] = t0
    ys[0] = y0

    t = t0
    y = np.array(y0, dtype=float)
    for i in range(1, n_steps + 1):
        step = min(h, t_end - t)
        y = rk4_step(f, t, y, step, **kwargs)
        t += step
        ts[i] = t
        ys[i] = y
    return ts, ys

def integrate_adaptive(f, y0, t0, t_end, eps=1e-5, h_init=0.01, h_min=1e-8, h_max=0.5, safety=0.9, **kwargs):
    ts = [t0]
    ys = [np.array(y0, dtype=float)]
    hs_accepted = []

    t = t0
    y = np.array(y0, dtype=float)
    h = h_init
    n_rejected = 0
    n_accepted = 0

    while t < t_end - 1e-14:
        if t + h > t_end:
            h = t_end - t

        while True:
            #h
            y_big = rk4_step(f, t, y, h, **kwargs)
            # h/2
            y_half = rk4_step(f, t, y, h / 2.0, **kwargs)
            y_half2 = rk4_step(f, t + h / 2.0, y_half, h / 2.0, **kwargs)

            # estimativa de erro local
            err_vec = (y_half2 - y_big) / 15.0
            err = np.linalg.norm(err_vec, ord=np.inf)

            if err <= eps or h <= h_min:
                # aceita o passo
                y_accept = y_half2 + err_vec  # extrapolacao de richardson
                t += h
                y = y_accept
                ts.append(t)
                ys.append(y)
                hs_accepted.append(h)
                n_accepted += 1

                # fator de crescimento do passo
                if err == 0.0:
                    factor = 4.0
                else:
                    factor = safety * (eps / err) ** 0.2
                factor = min(max(factor, 0.2), 4.0)
                h = min(h * factor, h_max)
                break
            else:
                # rejeita
                n_rejected += 1
                factor = safety * (eps / err) ** 0.2
                factor = min(max(factor, 0.1), 0.9)
                h = max(h * factor, h_min)

    return (np.array(ts), np.array(ys), np.array(hs_accepted),
            n_accepted, n_rejected)


# solucao analitica
def theta_analytic(t, theta0, g=G, l=L):
    omega_n = np.sqrt(g / l)
    return theta0 * np.cos(omega_n * t)


def period_analytic(g=G, l=L):
    return 2.0 * np.pi * np.sqrt(l / g)


def detect_period(ts, ys, n_periods=10):
    omega = ys[:, 1]
    crossing_times = []

    for i in range(1, len(ts)):
        v1, v2 = omega[i - 1], omega[i]
        if v1 == 0.0:
            crossing_times.append(ts[i - 1])
            continue
        if v1 * v2 <= 0.0:
            t1, t2 = ts[i - 1], ts[i]
            #interpolacao
            t_cross = t1 + abs(v1) / (abs(v1) + abs(v2)) * (t2 - t1)
            crossing_times.append(t_cross)

    if len(crossing_times) < 2:
        return None, crossing_times

    n_crossings_needed = 2 * n_periods
    n_available = len(crossing_times) - 1 

    if n_available >= n_crossings_needed:
        # tempo de n_periods periodos
        t_start = crossing_times[0]
        t_end = crossing_times[n_crossings_needed]
        T = (t_end - t_start) / n_periods
    else:
        # usa periodos completos disponiveis
        n_full = n_available // 2
        if n_full < 1:
            # so temos meio periodo 
            T = 2.0 * (crossing_times[1] - crossing_times[0])
        else:
            t_start = crossing_times[0]
            t_end = crossing_times[2 * n_full]
            T = (t_end - t_start) / n_full

    return T, crossing_times

def time_for_n_periods(theta0, n_periods=10, g=G, l=L, safety_factor=1.3):
    T_lin = period_analytic(g, l)
    return n_periods * T_lin * safety_factor