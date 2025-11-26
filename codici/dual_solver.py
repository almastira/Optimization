import numpy as np
from scipy.optimize import minimize
from kernels import linear_kernel
from utils import compute_kernel_matrix

def dual_objective(params, y, K, epsilon):
    """
    Funzione obiettivo duale dello SVR da minimizzare.
    """
    n = len(y)
    alpha = params[:n]
    alpha_star = params[n:]
    alpha_diff = alpha - alpha_star

    term1 = 0.5 * np.dot(alpha_diff, K @ alpha_diff)
    term2 = epsilon * np.sum(alpha + alpha_star)
    term3 = np.dot(y, alpha_diff)

    return term1 + term2 - term3  # minimizzare equivale a massimizzare il duale

def equality_constraint(params):
    """
    Vincolo: somma (alpha_i - alpha_i^*) = 0
    """
    n = len(params) // 2
    alpha = params[:n]
    alpha_star = params[n:]
    return np.sum(alpha - alpha_star)

def train_svr_dual(X, y, C=1.0, epsilon=0.1, verbose=True, log_file=None):
    """
    Risolve la formulazione duale dello SVR con SLSQP.
    Ora salva il valore dell’obiettivo duale a ogni step.
    """
    n = len(y)
    initial_guess = np.zeros(2 * n)

    # Matrice kernel
    K = compute_kernel_matrix(X, linear_kernel)

    # Bounds: 0 ≤ alpha ≤ C
    bounds = [(0, C)] * (2 * n)

    # Vincolo di uguaglianza
    constraint = {'type': 'eq', 'fun': equality_constraint}

    #  Lista per i valori obiettivo
    obj_values = []

    #  Funzione obiettivo con log
    def logging_objective(params, y, K, epsilon):
        val = dual_objective(params, y, K, epsilon)
        obj_values.append(val)
        return val

    #  Ottimzzazione SLSQP
    result = minimize(
        logging_objective,
        initial_guess,
        args=(y, K, epsilon),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint,
        options={'ftol': 1e-9, 'disp': False, 'maxiter': 10000}
    )

    #  Salvo i valori obiettivo su file se richiesto
    if log_file is not None:
        np.save(log_file, np.array(obj_values))

    if verbose:
        if result.success:
            print(f"[SLSQP] Convergenza OK in {result.nit} iterazioni.")
        else:
            print(f"[SLSQP] NON CONVERGE: {result.message}")

    # Estrazione parametri ottimali
    alpha_opt = result.x[:n]
    alpha_star_opt = result.x[n:]
    alpha_diff = alpha_opt - alpha_star_opt

    # Calcolo bias
    support_indices = [
        i for i in range(n)
        if 1e-5 < alpha_opt[i] < C - 1e-5 or 1e-5 < alpha_star_opt[i] < C - 1e-5
    ]

    if support_indices:
        b_values = [
            y[i] - np.sum(alpha_diff * K[:, i])
            for i in support_indices
        ]
        b_opt = np.mean(b_values)
    else:
        b_opt = np.mean([y[i] - np.sum(alpha_diff * K[:, i]) for i in range(n)])

    dual_obj_value = result.fun

    return alpha_opt, alpha_star_opt, b_opt, K, dual_obj_value
