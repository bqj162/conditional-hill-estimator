import numpy as np

def inverse_abs(x):
    return 1 / (1 + np.abs(x))

# registry of available functions
TAIL_INDEX_FUNCTIONS = {
    "inverse_abs": inverse_abs
}