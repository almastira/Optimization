
import numpy as np

def linear_kernel(x1, x2):
    """
    Kernel lieare: prodotto scalare standard tra x1 e x2.
    """
    return float(np.dot(x1.T, x2))
