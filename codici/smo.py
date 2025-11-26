import numpy as np

def svr_predict(X, alpha, alpha_star, b, X_train):
    """
    Predice l'output per nuovi dati X usando il modello SVR lineare addestrato con SMO.
    """
    K = X @ X_train.T   # kernel lineare
    return np.sum((alpha - alpha_star) * K, axis=1) + b


def train_svr_smo(X, y, C=0.1, epsilon=0.1, tau=1e-12, max_iter=2000, step_size=0.01):
    """
    Addestra un modello SVR usando l'algoritmo SMO (Sequential Minimal Optimization).
    Ora salva anche il dual objective ad intervalli regolari.
    """
    n_samples = len(X)
    K = X @ X.T  # kernel lineare semplificato

    # Inizializzazione
    alpha = np.zeros(n_samples)
    alpha_star = np.zeros(n_samples)
    b = 0.0

    #  LOG: lista di dizionari per ogni step
    log = []

    for it in range(int(max_iter)):
        for i in range(n_samples):
            f_i = np.sum((alpha - alpha_star) * K[:, i]) + b
            E_i = f_i - y[i]

            E = [(np.sum((alpha - alpha_star) * K[:, k]) + b) - y[k] for k in range(n_samples)]
            j = np.argmax(np.abs(E_i - np.array(E)))
            if i == j:
                continue

            E_j = E[j]

            eta = K[i, i] + K[j, j] - 2 * K[i, j]
            if eta <= 0:
                eta = tau

            delta = step_size * (E_i - E_j) / eta
            delta_star = step_size * (E_j - E_i) / eta

            alpha[i] -= delta
            alpha_star[i] -= delta_star
            alpha[j] += delta
            alpha_star[j] += delta_star

            # clipping tra [0, C]
            alpha = np.clip(alpha, 0, C)
            alpha_star = np.clip(alpha_star, 0, C)

            # aggiornamento del bias
            b1 = y[i] - np.sum((alpha - alpha_star) * K[:, i]) - epsilon
            b2 = y[j] - np.sum((alpha - alpha_star) * K[:, j]) - epsilon
            b = (b1 + b2) / 2.0

        #  Salvo il log ogi 50 iterazioni
        if it % 50 == 0:
            violazione = np.max(np.abs(E))

            # Calcolo dual objective
            dual_obj = 0.5 * np.dot((alpha - alpha_star), K @ (alpha - alpha_star)) \
                       + epsilon * np.sum(alpha + alpha_star) - np.dot(y, (alpha - alpha_star))

            log.append({
                "iter": it,
                "kkt_violation": violazione,
                "dual_obj": dual_obj,  #  aggiunto al log
                "max_alpha": np.max(alpha),
                "max_alpha_star": np.max(alpha_star)
            })

            print(f"[SMO] Iter {it} – KKT: {violazione:.2e} – Dual Obj: {dual_obj:.2f}")

    return alpha, alpha_star, b, K, log
