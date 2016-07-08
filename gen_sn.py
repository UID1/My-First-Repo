def gen_sn(M, I, anti_paths=True, no_match=True):
    """
    Function to generate random numbers for simulation

    Parameters
    ==========
    M : int
        number of time intervals for discretization
    I : int
        number of paths to be simulated
    anti_paths : Boolean
        use of antithetic variables
    mo_math : Boolean
        use of moment matching
    """
    import numpy as np
    import numpy.random as npr

    if anti_paths is True:
        sn = npr.standard_normal((M+1, I/2))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M+1, I))
    if no_match is True:
        sn = (sn - sn.mean())/sn.std()
    return sn

