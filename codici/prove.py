import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from smo import train_svr_smo
from dual_solver import train_svr_dual
from utils import predict_values, dual_objective_value
from kernels import linear_kernel

# Paraetri SVR
C = 1.0
epsilon = 0.1
tau = 1e-6  # più severo

#  DIZIONARIO PER SALVARE I LOG
smo_logs = {}

for n_samples in [20, 50, 100, 200]:
    print(f"\n=== ESEMPIO con {n_samples} campioni ===")

    # 1. Dati sintetici 
    X, y = make_regression(n_samples=n_samples, n_features=5, noise=10, random_state=42)
    X = StandardScaler().fit_transform(X)

    # 2. Addestramento con SMO
    start_smo = time.time()
    alpha_smo, alpha_star_smo, b_smo, K_smo, log_smo = train_svr_smo(X, y, C, epsilon, max_iter=2000)
    end_smo = time.time()
    time_smo = end_smo - start_smo

    # 3. Addestramento con dual solver (SLSQP)
    start_slsqp = time.time()
    alpha_dual, alpha_star_dual, b_dual, K_dual, dual_obj_slsqp = train_svr_dual(X, y, C, epsilon)
    end_slsqp = time.time()
    time_slsqp = end_slsqp - start_slsqp

    # 4. Calcolo obiettivi duali e gap
    obj_smo = dual_objective_value(alpha_smo, alpha_star_smo, y, K_smo, epsilon)
    dual_gap = abs(obj_smo - dual_obj_slsqp)
    relative_gap = dual_gap / (abs(dual_obj_slsqp) + 1e-12)

    # 5. Stampa risultati
    print(f"Tempo SMO: {time_smo:.6f} s")
    print(f"Tempo SLSQP: {time_slsqp:.6f} s")
    print(f"Obiettivo SMO:   {obj_smo:.6f}")
    print(f"Obiettivo SLSQP: {dual_obj_slsqp:.6f}")
    print(f"Dual gap assoluto: {dual_gap:.6e}")
    print(f"Dual gap relativo: {relative_gap:.6e}")

    #  Salviamo nel log TUTTO quello che serve per grafici_extra
    smo_logs[n_samples] = {
        "log_smo": log_smo,                     # log di SMO (iterazioni, KKT, dual_obj)
        "dual_obj_slsqp": dual_obj_slsqp        # valore finale trovato da SLSQP
    }

    #  Salvo anche il file per ogni n_samples (opzionale ma comodo)
    np.save(f"smo_log_{n_samples}.npy", smo_logs[n_samples])
    print(f" Log salvato in: smo_log_{n_samples}.npy")

#  A questo punto puoi far girare grafici_extra.py senza runnare tutto di nuovo
