from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    mode = Counter(x).most_common(1)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "mode": float(mode[0][0])}

