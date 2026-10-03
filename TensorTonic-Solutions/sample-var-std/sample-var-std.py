import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    var = float((1/(len(x) -1)) * (np.sum((x - np.mean(x)) ** 2)))
    print ((np.sum(x - np.mean(x) ** 2)), len(x) -1)
    s = float(np.sqrt(var))
    return {"variance": var, "standard_deviation": s}
    pass