def bsm_mcs_valuation(strike):
    """
    Dynamic Black_Scholes_Merton Monte Carlo estimator

    Parameters
    ==========
    strike : float
        strike price of the option

    Results
    =======
    value : float
        estimate for present valut of call option
    """
    import numpy as np
    S0 = 100.; T = 1.; r = .05; vola = .2
    M = 50; I = 20000
    dt = T / M
    rand = np.random.standard_normal((M + 1, I))
    S = np.zeros((M + 1, I)); S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t-1] * np.exp((r - .5 * vola ** 2) * dt
                                + vola * np.sqrt(dt) * rand[t])
    value = (np.exp(-r * T)
                    * np.sum(np.maximum(S[-1] - strike, 0)) / I)
    return value

def par_value(n):
    """
    Parallel option valuation

    Parameters
    ==========
    n: int
        number of option valuations/strikes
    """
    import numpy as np
    from ipyparallel import Client
    
    c = Client(profile="default")
    view = c.load_balanced_view()

    strikes = np.linspace(80, 20, n)
    option_values = []
    for strike in strikes:
        values = view.apply_async(bsm_mcs_valuation, strike)
        option_values.append(values)
    c.wait(option_values)
    return strikes, option_values

