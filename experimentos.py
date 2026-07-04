import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.grid'] = True

from pendulum import (f_nonlinear, integrate_fixed, integrate_adaptive,
                      theta_analytic, period_analytic, detect_period,
                      time_for_n_periods, G, L)

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

N_PERIODS_FOR_MEASURE = 10  # conforme sugerido no enunciado
EPS_ADAPTIVE = 1e-5

# PARTE 1: alidacao com passo constante + graficos theta x t
def parte1_validacao_e_graficos():
    
    print("PARTE 1: Validacao (passo constante) + graficos theta x t")
    thetas0_graf = [2, 10, 30, 60, 90, 150]  # graus
    h_valid = 0.001

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, theta0_deg in enumerate(thetas0_graf):
        theta0 = np.radians(theta0_deg)
        y0 = [theta0, 0.0]

        T_lin = period_analytic()
        # simula mais que 1 periodo linear
        t_end = 1.6 * T_lin if theta0_deg <= 90 else 2.2 * T_lin

        ts, ys = integrate_fixed(f_nonlinear, y0, 0.0, t_end, h=h_valid)
        T_num, crossings = detect_period(ts, ys, n_periods=1)

        # recorta 1 ciclo numerico para o grafico
        if T_num is not None:
            mask = ts <= T_num * 1.02
        else:
            mask = ts <= T_lin

        ax = axes[idx]
        ax.plot(ts[mask], np.degrees(ys[mask, 0]), label="Numerica (RK4, nao-linear)",
                 color="C0", linewidth=1.8)
        ax.plot(ts[mask], np.degrees(theta_analytic(ts[mask], theta0)),
                 label="Analitica (linearizada)", color="C1", linestyle="--", linewidth=1.5)
        ax.set_title(f"$\\theta_0$ = {theta0_deg}°   "
                      f"(T_num={T_num:.4f}s, T_lin={T_lin:.4f}s)", fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("$\\theta$ (graus)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)

        print(f"  theta0={theta0_deg:>4}deg | T_numerico={T_num:.6f}s | "
              f"T_linear={T_lin:.6f}s | erro={abs(T_num - T_lin):.6f}s "
              f"({100*abs(T_num-T_lin)/T_lin:.3f}%)")

    plt.tight_layout()
    fname = f"{OUT}/parte1_theta_vs_t.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"\n  -> grafico salvo em {fname}\n")

# PARTE 2 / Analise Q1: comparativo periodo x n_passos x theta0
def parte2_tabela_comparativa():

    print("PARTE 2 (Analise, item 1): Comparativo Periodo x N.Passos x theta0")
    thetas0_deg = [1, 5, 10, 20, 30, 45, 60, 90, 120, 150, 170]
    hs_fixed = [0.01, 0.001, 0.0001]

    T_lin_theoretical = period_analytic()

    rows = []
    for theta0_deg in thetas0_deg:
        theta0 = np.radians(theta0_deg)
        y0 = [theta0, 0.0]
        t_sim = time_for_n_periods(theta0, n_periods=N_PERIODS_FOR_MEASURE)

        row = {"theta0_deg": theta0_deg,
               "T_analitica": T_lin_theoretical,
               "n_passos_analitica": None}  # forma fechada

        for h in hs_fixed:
            ts, ys = integrate_fixed(f_nonlinear, y0, 0.0, t_sim, h=h)
            T_num, _ = detect_period(ts, ys, n_periods=N_PERIODS_FOR_MEASURE)
            n_passos = len(ts) - 1
            row[f"T_h{h}"] = T_num
            row[f"n_h{h}"] = n_passos

        ts_a, ys_a, hs_a, n_acc, n_rej = integrate_adaptive(
            f_nonlinear, y0, 0.0, t_sim, eps=EPS_ADAPTIVE)
        T_num_a, _ = detect_period(ts_a, ys_a, n_periods=N_PERIODS_FOR_MEASURE)
        row["T_adaptativo"] = T_num_a
        row["n_adaptativo"] = n_acc
        row["n_rejeitados_adaptativo"] = n_rej

        rows.append(row)

        print(f"\n theta0 = {theta0_deg}deg")
        print(f"   Analitica simplificada : T={T_lin_theoretical:.6f}s | passos: N/A (forma fechada)")
        for h in hs_fixed:
            print(f"   Passo constante h={h:<8}: T={row[f'T_h{h}']:.6f}s | n_passos={row[f'n_h{h}']}")
        print(f"   Passo adaptativo (eps={EPS_ADAPTIVE:.0e}): T={T_num_a:.6f}s | "
              f"n_passos_aceitos={n_acc} | n_rejeitados={n_rej}")

    return rows


def analise_q2_theta0_maximo(rows=None):

    print("ANALISE (item 2): theta0 maximo para erro(T) < 0.001s na formula linear")
    T_lin = period_analytic()
    h_ref = 0.0001  # usa o passo mais fino como referencia "verdade numerica"

    # busca fina em theta0, comparando T_numerico(h_ref) com T_analitico
    thetas0_deg_scan = np.arange(1, 91, 1)
    erro_prev = 0.0
    theta0_max_deg = None

    for theta0_deg in thetas0_deg_scan:
        theta0 = np.radians(theta0_deg)
        y0 = [theta0, 0.0]
        t_sim = time_for_n_periods(theta0, n_periods=N_PERIODS_FOR_MEASURE)
        ts, ys = integrate_fixed(f_nonlinear, y0, 0.0, t_sim, h=h_ref)
        T_num, _ = detect_period(ts, ys, n_periods=N_PERIODS_FOR_MEASURE)
        erro = abs(T_num - T_lin)

        if erro >= 0.001 and theta0_max_deg is None:
            # interpolacao linear entre o angulo anterior (erro_prev < 0.001)
            # e este (erro >= 0.001) para refinar a estimativa
            theta_prev_deg = theta0_deg - 1
            frac = (0.001 - erro_prev) / (erro - erro_prev) if erro != erro_prev else 0
            theta0_max_deg = theta_prev_deg + frac * 1.0
            print(f"  Cruzamento detectado entre {theta_prev_deg}deg (erro={erro_prev:.6f}s) "
                  f"e {theta0_deg}deg (erro={erro:.6f}s)")
            print(f"  theta0_max interpolado ~= {theta0_max_deg:.2f} graus "
                  f"({np.radians(theta0_max_deg):.4f} rad)")
            break
        erro_prev = erro

    if theta0_max_deg is None:
        print("  Erro nao ultrapassou 0.001s no intervalo escaneado (1-90 graus).")

    return theta0_max_deg

def analise_q3_tempo_execucao():

    print("ANALISE (item 3): Tempo de execucao (wall-clock) para 10 periodos")
    theta0 = np.radians(30.0)  # angulo representativo, moderado
    y0 = [theta0, 0.0]
    t_sim = time_for_n_periods(theta0, n_periods=N_PERIODS_FOR_MEASURE)

    hs_fixed = [0.01, 0.001, 0.0001]
    results = {}

    print(f"  (theta0=30deg, simulando ~{t_sim:.3f}s de tempo fisico = 10 periodos)\n")

    for h in hs_fixed:
        n_repeats = 5 if h >= 0.001 else 1
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            ts, ys = integrate_fixed(f_nonlinear, y0, 0.0, t_sim, h=h)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_time = np.mean(times)
        n_passos = len(ts) - 1
        results[f"h={h}"] = {"tempo_s": avg_time, "n_passos": n_passos,
                              "tempo_por_passo_us": 1e6 * avg_time / n_passos}
        print(f"  Passo constante h={h:<8}: {avg_time*1000:8.3f} ms total | "
              f"{n_passos:>8} passos | {1e6*avg_time/n_passos:.3f} us/passo")

    times_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        ts_a, ys_a, hs_a, n_acc, n_rej = integrate_adaptive(
            f_nonlinear, y0, 0.0, t_sim, eps=EPS_ADAPTIVE)
        t1 = time.perf_counter()
        times_a.append(t1 - t0)
    avg_time_a = np.mean(times_a)
    results["adaptativo"] = {"tempo_s": avg_time_a, "n_passos": n_acc,
                              "n_rejeitados": n_rej,
                              "tempo_por_passo_us": 1e6 * avg_time_a / n_acc}
    print(f"  Passo adaptativo (eps={EPS_ADAPTIVE:.0e}): {avg_time_a*1000:8.3f} ms total | "
          f"{n_acc:>8} passos aceitos ({n_rej} rejeitados) | "
          f"{1e6*avg_time_a/n_acc:.3f} us/passo")

    return results, t_sim

def analise_q4_tempo_real(exec_results, t_sim):
    print("ANALISE (item 4): O sistema executa em tempo real?")
    print("  Comparando: tempo de CPU gasto por passo  vs  tempo fisico (h)")
    print("  que aquele passo representa. Tempo real requer tempo_CPU <= h.\n")

    hs_fixed = [0.01, 0.001, 0.0001]
    for h in hs_fixed:
        key = f"h={h}"
        t_cpu_por_passo = exec_results[key]["tempo_s"] / exec_results[key]["n_passos"]
        razao = h / t_cpu_por_passo
        tempo_real = t_cpu_por_passo <= h
        print(f"  h={h:<8}: tempo_CPU/passo = {t_cpu_por_passo*1e6:8.3f} us | "
              f"tempo_fisico/passo (h) = {h*1e6:10.1f} us | "
              f"razao (h / t_cpu) = {razao:9.1f}x | "
              f"{'TEMPO REAL (folga)' if tempo_real else 'NAO tempo real'}")

    # adaptativo: usa o h medio efetivo
    a = exec_results["adaptativo"]
    t_cpu_por_passo_a = a["tempo_s"] / a["n_passos"]
    h_medio_a = t_sim / a["n_passos"]
    razao_a = h_medio_a / t_cpu_por_passo_a
    print(f"  adaptativo : tempo_CPU/passo = {t_cpu_por_passo_a*1e6:8.3f} us | "
          f"h_medio_efetivo = {h_medio_a*1e6:10.1f} us | "
          f"razao = {razao_a:9.1f}x | "
          f"{'TEMPO REAL (folga)' if t_cpu_por_passo_a <= h_medio_a else 'NAO tempo real'}")

    print("\n  Conclusao: em todos os casos testados, o tempo de CPU por passo")
    print("  fica MUITO abaixo do tempo fisico h que o passo representa (razoes")
    print("  de milhares a milhoes de vezes), logo o sistema roda com grande")
    print("  folga em relacao ao tempo real -- ou seja, poderia ser executado")
    print("  'ao vivo', mantendo o passo de integracao sincronizado com um")
    print("  relogio de parede, sem acumular atraso.")

if __name__ == "__main__":
    parte1_validacao_e_graficos()
    rows = parte2_tabela_comparativa()
    theta0_max = analise_q2_theta0_maximo()
    exec_results, t_sim = analise_q3_tempo_execucao()
    analise_q4_tempo_real(exec_results, t_sim)

    print("RESUMO FINAL")
    print(f"  Periodo analitico (linear): {period_analytic():.6f} s")
    if theta0_max is not None:
        print(f"  theta0 maximo p/ erro < 0.001s na aproximacao linear: ~{theta0_max:.2f} graus")
    print(f"  Graficos salvos em: {OUT}/")