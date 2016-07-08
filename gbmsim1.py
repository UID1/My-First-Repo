import numpy as np
import multiprocessing as mp
import math
def simulate_geometric_brownian_motion(p):
    M, I = p
        # time steps, paths
    S0 = 100; r = .05; sigma = .2; T = 1.
        # model parameters
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        paths[t] = paths[t - 1] * np.exp((r - .5 * sigma ** 2) * dt +
                sigma * math.sqrt(dt) * np.random.standard_normal(I))
    return paths

