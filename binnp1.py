def binomial_np(strike):
    """
    Binomial option pricing with NumPy.

    Parameters
    ==========
    strike : float
        strike priceof the European call option
    """
    import numpy as np
    # model and option parameters
    S0 = 100.   #iInitial index level
    T = 1.      # call option maturity
    r = .05     # constant short rate
    vola = .2   # constant volatility factor of diffusion

    # time parameters
    M = 1000    # time steps
    dt = T / M  # length of time interval
    df = np.exp(-r * dt)   # discount factor per time interval

    # binomial parameters
    u = np.exp(vola * np.sqrt(dt)) # up movement
    d = 1 / u   # down movement
    q = (np.exp(r * dt) - d) / (u - d) # martingale probability

    # Index Levels with NumPy
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
        # resize is very similar to
        # meshgrid in Matlab in this case.
        # It tiles a matrix of size (M+1)x(M+1)
        # with the vector.
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Valuation Loop
    pv = np.maximum(S - strike, 0)
    z = 0
    for t in range(M - 1, -1, -1): # backward iteration
        pv[0:M - z,t] = (q * pv[0:M - z, t + 1]
                        + (1 - q) * pv[1:M - z + 1, t + 1]) * df
    return pv[0, 0]

