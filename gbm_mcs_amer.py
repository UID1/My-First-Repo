def gbm_mcs_amer(K, option='call'):
    """
    Valuation of American option in Black_Scholes-Merton
    by Monte Carlo simulation by LSM algorithm

    Parameters
    ==========
    K : float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put')

    Returns
    =======
    C0 : float
        estimated present value of European call option
    """

    dt = T/M
    df = np.exp(-r*dt)
    # simulation of index levels
    S = np.zeros((M+1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M+1):
        S[t] = S[t-1]*np.exp((r-.5*sigma**2)*dt
                +sigma*np.sqrt(dt)*sn[t])
    # case-based calculation of payoff
    if option == 'call':
        h = np.maximum(S-K, 0)
    else:
        h = np.maximum(K-S, 0)
    # LSM algorithm
    V = np.copy(h)
        # Generate a copy of h. If We had set V = h then changes of
        # h would also affect V and vice verse while a copy ensures
        # that V is a separate copy of h.
    for t in range(M-1, 0, -1):
        reg = np.polyfit(S[t], V[t+1]*df, 7)
        C = np.polyval(reg, S[t])
        V[t] = np.where(C > h[t], V[t+1]*df, h[t])
            # V_t(s) = max{h_t(s),C_t(s)}
    # MCS estimator
    C0 = df/I*np.sum(V[1])
    return C0

