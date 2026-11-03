import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write coXde here
    #forst we meed tp wrote tje sjaes etc
    n,d = X.shape
    w=np.zeros(d)
    b=0

    for i in range(steps):
        z = X@w + b
        p= _sigmoid(z)

        error = p-y

        dw = (X.T@error)/n
        db=np.mean(error)

        w = w-lr * dw
        b = b - lr *db
    return w,b

    
    