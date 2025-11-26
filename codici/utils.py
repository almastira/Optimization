
import numpy as np

def compute_kernel_matrix(X, kernel_fn):
    """
    Costruisce la matrice kernel K[i,j] = kernel_fn(X[i], X[j])
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j])
    return K

def predict_values(X_test, X_train, alpha, alpha_star, b, kernel_fn):
    """
    Predizione SVR: f(x) = sum_i (alpha_i - alpha_i^*) K(x_i, x) + b
    """
    n_test = len(X_test) 
    y_pred = np.zeros(n_test)
    for i in range(n_test):
        s = 0
        for j in range(len(X_train)):
            s += (alpha[j] - alpha_star[j]) * kernel_fn(X_train[j], X_test[i])
        y_pred[i] = s + b
    return y_pred

def dual_objective_value(alpha, alpha_star, y, K, epsilon):
    """
    Calcola il valore della funzione duale.
    Utile per confrontare SMO e SLSQP.
    """
    alpha_diff = alpha - alpha_star
    term1 = 0.5 * np.dot(alpha_diff, K @ alpha_diff)
    term2 = epsilon * np.sum(alpha + alpha_star)
    term3 = np.dot(y, alpha_diff)
    return term1 + term2 - term3
