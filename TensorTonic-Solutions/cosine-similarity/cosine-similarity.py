import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    veca = np.array(a)
    vecb = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float((np.matmul(a,b)) / (np.sqrt(np.sum(veca**2, axis = 0)) * np.sqrt(np.sum(vecb**2, axis = 0))))
    
    pass