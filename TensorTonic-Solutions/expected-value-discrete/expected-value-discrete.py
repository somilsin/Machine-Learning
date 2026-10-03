import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    xnp = np.array(x)
    pnp = np.array(p)
    ans = np.sum(xnp @ pnp)
    return float(ans)
    pass