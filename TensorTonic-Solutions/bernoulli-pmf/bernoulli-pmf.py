import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    x = np.asarray(x)
    pmf = np.where(x == 0, 1 - p, np.where(x == 1, p, 0.0))

    return {
        "pmf": pmf,
        "mean": float(p),
        "variance": float(p * (1 - p))
    }