# From page 61 in Python for Finance

tol = 0.5 # tolerance level for moneyness
for option in options_data.index:
# iterating over all option quotes
forward = futures_data[futures_data['MATURITY'] == \
             options_data.loc[option]['MATURITY']]['PRICE'].values[0]
# The outer query: futures_data[ <expression>] returns an object
# of type pandas.core.series.Series. the .values.[0] extracts
# the value from that object.
# We pick the right futures value.
if(forward * (1 - tol) < options_data.loc[option]['STRIKE']
                        < forward * (1 + tol)):
# If the strike price of the option lies within the tolerance
# level of the price of the futures contract we calculate the
# implied volatility, otherwise, we don't bother
    imp_vol = bsm_call_imp_vol(
            V0,  # VSTOXX value, entered previously
            options_data.loc[option]['STRIKE'],
            options_data.loc[option]['TTM'],
            r,   # short rate
            options_data.loc[option]['PRICE'],
            sigma_est=2., # estimate for implied volatility
            it=100)
    options_data['IMP_VOL'].loc[option] = imp_vol







# Page 63

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for maturity in maturities:
    data = plot_data[options_data.MATURITY == maturity]
      # select data for this maturity
    plt.plot(data['STRIKE'], data['IMP_VOL'],
        label=maturity.date(), lw=1.5)
    plt.plot(data['STRIKE'], data['IMP_VOL'], 'r.')
    plt.grid(True)
    plt.xlabel('strike')
    plt.ylabel('implied volatility of volatility')
    plt.legend()
    plt.show()

keep = ['PRICE', 'IMP_VOL']
group_data = plot_data.groupby(['MATURITY', 'STRIKE'])[keep]
group_data


# From page 131

strike = np.linspace(50, 150, 24)
ttm = np.linspace(0.5, 2.5, 24)
strike, ttm = np.meshgrid(strike, ttm)
iv = (strike - 100) ** 2 / (100 * strike) / ttm

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(strike, ttm, iv, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')
fig.colorbar(surf, shrink=0.5, aspect=5)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 60)
ax.scatter(strike, ttm, iv, zdir='z', s=25, c='b', marker='^')
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')

# From page 148
%%time
DAX['Ret_Loop'] = 0.0
for i in range(1, len(DAX)):
  DAX['Ret_Loop'][i] = np.log(DAX['Close'][i] / DAX['Close'][i - 1])

# From page 152
es_url = 'http://www.stoxx.com/download/historical_values/hbrbcpe.txt'
vs_url = 'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
urlretrieve(es_url, './pythonlearning/source/es.txt')
urlretrieve(vs_url, './pythonlearning/source/vs.txt')
!ls -o ./pythonlearning/source/*.txt

lines = open('./pythonlearning/source/es.txt', 'r').readlines()
lines = [line.replace(' ', '') for line in lines]

cols = ['SX5P', 'SX5E', 'SXXP', 'SXXE', 'SXXF', 'SXXA', 'DK5F', 'DKXF']
es = pd.read_csv(es_url, index_col=0, parse_dates=True, sep=';', dayfirst=True, header=none, skiprows=4, names=cols)

vs = pd.read_csv('./pythonlearning/source/vs.txt', index_col=0, header=2, parse_dates=True, sep=',', dayfirst=True)





%time for i in range(rows):
  pointer['Date'] = dt.datetime.now()
  pointer['No1']  = ran_int[i, 0]
  pointer['No2']  = ran_int[i, 1]
  pointer['No3']  = ran_flo[i, 0]
  pointer['No4']  = ran_flo[i, 1]
  pointer.append()
tab.flush()

%time
sarray['Date'] = dt.datetime.now()
sarray['No1'] = ran_int[:, 0]
sarray['No2'] = ran_int[:, 1]
sarray['No3'] = ran_flo[:, 0]
sarray['No4'] = ran_flo[:, 1]




# Page 202

I  = 1000; M = 100; t = 20
# I = number of paths, M = number of time steps, t = number of tasks/simulations
from time import time
times = []
for w in range(1, 6):
  t0 = time()
  pool = mp.Pool(processes=w)
    # the pool of workers
  result = pool.map(simulate_geometric_brownian_motion,
		    t * [(M, I), ])
    # the mapping of the function to the list of parameter tuples
  times.append(time() - t0)
  
  
# The statement t * [(M, I), ] is a matrix product that distributes the
# job to the workers. The expression [(M, I), ] generates a 1x1 matrix
# with a 2-tuple containing M and I. For example setting M = 1 and I = 2
# yields:

# In [N]: [(M, I), ]
# Out[N]: [(1, 2)]

# The 't *' multiplies the matrixed tuple t times, i.e. set t = 5:
  
# In [N+1]: [(M, I), ]
# Out[N+1]: [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]


from math import cos, log
def f_py(I, J):
  res = 0
  for i in range(I):
    for j in range (J):
      res += int(cos(log(1)))
  return res

def f_np(I, J):
  a = np.ones((I, J), dtype=np.float64)
  return int(np.sum(np.cos(np.log(a)))), a



def f_py(I, J):
  res = 0. # we work on a float object
  for i in range(I):
    for j in range (J * I):
      res += 1
  return res

  

%%cython
#
# Nested loop example with cython
#
def f_cy(int I, int J):
  cdef double res = 0
  # double float much slower than in or long
  for i in range(I):
    for j in range (J*I):
      res += 1
  return res


matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1



S0 = 100; r = .05; sigma = .25
T = 2.
I = 10000
M = 50
dt = T / M
S = np.zeros((M+1,I))
S[0] = S0
for t in range(1,M+1):
  S[t] = S[t-1]*np.exp((r-.5*sigma**2)*dt
		      +sigma*np.sqrt(dt)*npr.standard_normal(I))
  

# p260 Heston model
S0 = 100.; r = .05; kappa = 3.; theta = .25
v0 = .1; sigma = .1; rho = .6; T = 1.
corr_mat = array([[1., rho], [rho, 1.]])
cho_mat = np.linalg.cholesky(corr_mat)
cho_mat

# Euler-simulation of stochastic volatility
M = 50
I  = 10000
ran_num = npr.standard_normal((2, M+1, I))
# 2 stochastic variables Z_1 and Z_2,
# M time steps (ignoring first step)
# I simulations.
dt = T / M
v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)
v[0] = v0
vh[0] = v0
for t in range(1, M+1):
  ran = np.dot(cho_mat, ran_num[:, t, :])
  vh[t] = (vh[t-1]+kappa*(theta-np.maximum(vh[t-1], 0))*dt
	  +sigma*np.sqrt(np.maximum(vh[t-1], 0))*np.sqrt(dt)
	  *ran[1])
v = np.maximum(vh, 0)
# Simulation of S(t) using Exact Euler scheme
S = np.zeros_like(ran_num[0])
S[0] = S0
for t in range(1, M+1):
  ran = np.dot(cho_mat, ran_num[:, t, :])
  S[t] = S[t-1]*np.exp((r-.5*v[t])*dt+
		 np.sqrt(v[t])*ran[0]*np.sqrt(dt))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
ax1.hist(S[-1], bins=50)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(v[-1], bins=50)
ax2.set_xlabel('volatility')
ax2.grid(True)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax1.plot(S[:, :10], lw=1.5)
ax1.set_ylabel('index level')
ax1.grid(True)
ax2.plot(v[:, :10], lw=1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')
ax2.grid(True)

def print_statistics(a1, a2):
  """
  Prints selected statistics.
  Parameters
  ==========
  a1, a2 : ndarray objects
  results object from simulation
  """
  
  sta1 = scs.describe(a1)
  sta2 = scs.describe(a2)
  print "%14s %14s %14s" % \
  ('statistic', 'data set 1', 'data set 2')
  print 45 * "-"
  print "%14s %14.3f %14.3f" % ('size', sta1[0], sta2[0])
  print "%14s %14.3f %14.3f" % ('min', sta1[1][0], sta2[1][0])
  print "%14s %14.3f %14.3f" % ('max', sta1[1][1], sta2[1][1])
  print "%14s %14.3f %14.3f" % ('mean', sta1[2], sta2[2])
  print "%14s %14.3f %14.3f" % ('std', np.sqrt(sta1[3]),
  np.sqrt(sta2[3]))
  print "%14s %14.3f %14.3f" % ('skew', sta1[4], sta2[4])
  print "%14s %14.3f %14.3f" % ('kurtosis', sta1[5], sta2[5])

# p263 Euler simulation of jump diffusion processes
S0=100.;r=.05;sigma=.2;lamb=.75;mu=-.6;delta=.25;T=1.
M=50;I=10000
dt = T/M
rj = lamb*(np.exp(mu+.5*delta**2)-1)
S = np.zeros((M+1, I))
S[0] = S0Snatcher (Sega CD)
sn1 = npr.standard_normal((M+1, I))
sn2 = npr.standard_normal((M+1, I))
poi = npr.poisson(lamb*dt, (M+1, I))
for t in range(1, M+1, 1):
  S[t] = S[t-1]*(np.exp((r-rj-.5*sigma**2)*dt
		+sigma*np.sqrt(dt)*sn1[t])
		+(np.exp(mu+delta*sn2[t])-1)
		*1)
  S[t] = np.maximum(S[t], 0)

#p267
S0 = 100.; r=.05;sigma=.25;T=1.;I=50000
def gbm_mcs_stat(K):
  """
  Valuation of European call option in Black-Scholes_Merton
  by Monte Carlo simulation (of index level at maturity)
  
  Parameters
  ==========
  K : float
    (positive) strike price of the option

  Returns
  =======
  C0 : float
    estimated present value of the European call option
  """
  sn = gen_sn(1, I)
  # simulate index level at maturity
  ST = S0*np.exp((r-.5*sigma**2)*T
		 +sigma*np.sqrt(T)*sn[1])
  # calculate payoff at maturity
  hT = np.maximum(ST-K, 0)
  # calculate MCS estimator
  C0 = np.exp(-r*T)/I*np.sum(hT)
  return C0

   
M = 50
def gbm_mcs_dyna(K, option='call'):
  """
  Valuation of European options in Black-Scholes-Scholes_Merton
  by Monte Carlo simulation (of index level paths)
  
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
  # simulation of index level paths
  S = np.zeros((M+1, I))
  S[0] = S0
  sn = gen_sn(M, I)
  for t in range(1, M+1):
    S[t] = S[t-1]*np.exp((r-.5*sigma**2)*dt
			 +sigma*np.sqrt(dt)*sn[t])
  # case based calculation of payoff
  if option == 'call':
    hT = np.maximum(S[-1]-K, 0)
  else:
    hT = np.maximum(K-S[-1], 0)
  # calculation of MCS estimator
  C0 = np.exp(-r*T)/I*np.sum(hT)
  return C0

from bsm_functions import bsm_call_value
stat_res = []
dyna_res = []
anal_res = []
k_list = np.arange(80, 120.1, 5.)
np.random.seed(200000)
for K in k_list:
  stat_res.append(gbm_mcs_stat(K))
  dyna_res.append(gbm_mcs_dyna(K))
  anal_res.append(bsm_call_value(S0, K, T, r, sigma))
stat_res = np.array(stat_res)
dyna_res = np.array(dyna_res)
anal_res = np.array(anal_res)

# Plot of static simulation
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(k_list, anal_res, 'b', label='analytical')
ax1.plot(k_list, stat_res, 'ro', label='static')
ax1.set_ylabel('European call option value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (anal_res - stat_res) / anal_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75, right=125)
ax2.grid(True)


# Plot of dynamic simulation
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(k_list, anal_res, 'b', label='analytical')
ax1.plot(k_list, dyna_res, 'ro', label='dynamic')
ax1.set_ylabel('European call option value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (anal_res - dyna_res) / anal_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75, right=125)
ax2.grid(True)

# GBM valuation of american options
# The valuation function
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

# The Valuation Procedure  
euro_res = []
amer_res = []
k_list = np.arange(80., 120.1, 5.)
for K in k_list:
  euro_res.append(gbm_mcs_dyna(K, 'put'))
  amer_res.append(gbm_mcs_amer(K, 'put'))
euro_res = np.array(euro_res)
amer_res = np.array(amer_res)

# Plot of results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(k_list, euro_res, 'b', label='European put')
ax1.plot(k_list, amer_res, 'ro', label='American put')
ax1.set_ylabel('call option value')
ax1.grid(True)
ax1.legend(loc=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (amer_res - euro_res) / euro_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('early exercise premium in %')
ax2.set_xlim(left=75, right=125)
ax2.grid(True)

#VaR evaluation
S0 = 100.; r=.05; sigma=.25; T=30/365.;I=10000
ST = S=S0*np.exp((r-.5*sigma**2)*T
	       +sigma*np.sqrt(T)*npr.standard_normal(I))
R_gbm=np.sort(ST-S0)
plt.hist(R_gbm, bins=100)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)

percs = [.01, .1, 1., 2.5, 5., 10.]
var = scs.scoreatpercentile(R_gbm, percs)
print "%16s %16s" % ('Confidence Level', 'Value-at-Risk')
print 33*"-"
for pair in zip(percs, var):
  print "%16.2f %16.3f" % (100-pair[0], -pair[1])

#VaR evaluation using jump-diffusion processes
dt = 30./365/M
rj = lamb*(np.exp(mu+.5*delta**2)-1)
S = np.zeros((M+1, I))
S[0]=S0
sn1 = npr.standard_normal((M+1, I))
sn2 = npr.standard_normal((M+1, I))
poi = npr.poisson(lamb*dt, (M+1, I))
for t in range(1, M+1, 1):
  S[t] = S[t-1]*(np.exp((r-rj-.5*sigma**2)*dt
			+sigma*np.sqrt(dt)*sn1[t])
			+(np.exp(mu+delta*sn2[t])-1)
			*poi[t])
  S[t] = np.maximum(S[t], 0)
R_jd = np.sort(S[t]-S0)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
plt.hist(R_jd, bins=100)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)
percs = [.01, .1, 1., 2.5, 5., 10.]
var = scs.scoreatpercentile(R_jd, percs)
print "%16s %16s" % ('Confidence Level', 'Value-at-Risk')
print 33*"-"
for pair in zip(percs, var):
  print "%16.2f %16.3f" % (100-pair[0], -pair[1])

# Let's compare VaR estimation for GBM vs Jump Diffusion
# processes
percs = list(np.arange(.0, 10.1, .1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var = scs.scoreatpercentile(R_jd, percs)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
plt.plot(percs, gbm_var, 'b', lw=1.5, label='Geometric Brownian Motion')
plt.plot(percs, jd_var, 'r', lw=1.5, label='Jump Diffusion')
plt.legend(loc=4)
plt.xlabel('100 - confidence level [%]')
plt.ylabel('Value-at-Risk')
plt.grid(True)
plt.ylim(ymax=.0)


# Credit value adjustments

S0 = 100.;r=.05;sigma=.2;T=1.;I=100000
ST = S0*np.exp((r-.5*sigma**2)*
      +sigma*np.sqrt(T)*npr.standard_normal(I))
L=.5;p=.01
D = npr.poisson(p*T, I)
D = np.where(D>1, 1, D)
np.exp(-r*T)/I*np.sum(ST)

# Value of the CVA
CVaR = np.exp(-r*T)/I*np.sum(ST*L*D)
CVaR

# Credit Value Adjusted value of S0
S0_CVA = np.exp(-r*T)/I*np.sum(ST*(1-L*D))
S0_CVA

# Should be roughly the same as subtracting CVaR
S0_adj = S0 - CVaR
S0_adj

# Plot frequency distribution of losses due to a default
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
plt.hist(L*D*ST, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.grid(True)
plt.ylim(ymax=175)


def print_statistics(array):
  """
  Prints selected statistics.
  
  Parameters
  ==========
  array: ndarray
    object to generate statistics on
  """
  sta = scs.describe(array)
  print "%14s %15s" % ('statistic', 'value')
  print 30 * "-"
  print "%12s %15.5f" % ('size', sta[0])
  print "%12s %15.5f" % ('min', sta[1][0])
  print "%12s %15.5f" % ('max', sta[1][1])
  print "%12s %15.5f" % ('mean', sta[2])
  print "%12s %15.5f" % ('std', np.sqrt(sta[3]))
  print "%12s %15.5f" % ('skew', sta[4])
  print "%12s %15.5f" % ('kurtosis', sta[5])

def normality_tests(arr):
  """
  Tests for normality distribution of given data set.
  
  Parameters
  ==========
  array: ndarray
    object to generate statistics on
  """
  print "Skew of data set %12.3f" % scs.skew(arr)
  print " Skew test p-value %12.3f" % scs.skewtest(arr)[1]
  print "Kurt of data set %12.3f" % scs.kurtosis(arr)
  print " Kurt test p-value %12.3f" % scs.kurtosistest(arr)[1]
  print "Norm test p-value %12.3f" % scs.normaltest(arr)[1]

# Monte Carlo simulation of different portfolio
# returns
%%time
prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean()*weights)*252)
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov()
			*252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

def statistics(weights):
  """
  Returns portfolio statistics.
  
  Parameters
  ==========
  weights : array-like
    weights for different securities in portfolio
    
  Returns
  =======
    pret : float
      expected portfolio return
    pvol : float
      expected portfolio volatility
    pret / pvol : float
      Sharpe ratio for rf=0
  """
  
  weights = np.array(weights)
  pret = np.sum(rets.mean() * weights) * 252
  pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
  return np.array([pret, pvol, pret / pvol])

#p304 finding Markowitz efficient frontier

bnds = tuple((0, 1) for x in weights)
def min_func_port(weights):
    return statistics(weights)[1]
%%time
trets = np.linspace(.0, .25, 50)
tvols = []
for tret in trets:
  cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0]-tret},
	  {'type': 'eq', 'fun': lambda x: np.sum(x)-1})
  res = sco.minimize(min_func_port, noa*[1./noa], method='SLSQP',
		   bounds=bnds, constraints=cons)
  if(np.abs(statistics(res['x'])[0]-tret) < 1e-3):
    tvols.append(res['fun'])
  else:
    tvols.append(NaN)
tvols = np.array(tvols)

# Let's remove these NaN elements as they will be
# a problem later on (not the plots though)

trimnan = ~np.isnan(tvols)
tvols = tvols[trimnan]
trets = trets[trimnan]

# plot the results
plt.figure(figsize=(8,4))
plt.scatter(pvols, prets, c=prets/pvols, marker='o')
  # random portfolio composition
plt.scatter(tvols, trets, c=trets/tvols, marker='x')
  # efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
	 'r*', markersize=15.)
  # portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
	 'y*', markersize=15.)
  # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

# p306
def f(x):
  " Efficient Frontier function (splines approximation). "
  return sci.splev(x, tck, der=0)
def df(x):
  " First derivative of efficient frontier function. "
  return sci.splev(x, tck, der=1)
def equations(p, rf=.01):
  eq1 = rf - p[0]
  eq2 = rf + p[1] * p[2] - f(p[2])
  eq3 = p[1] - df(p[2])
  return eq1, eq2, eq3
  
opt = sco.fsolve(equations, [.01, .5, .15])

# Let's plot everything

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=(prets-.01)/pvols, marker='o')
  # random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
  # efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
  # capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

# Application of pricipal component analysis

# Fetch the DAX30 portofolio
symbols = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE',
'BMW.DE', 'CBK.DE', 'CON.DE', 'DAI.DE', 'DB1.DE',
'DBK.DE', 'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LHA.DE',
'LIN.DE', 'LXS.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE',
'SAP.DE', 'SDF.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE',
'^GDAXI']

%%time
data = pd.DataFrame()
for sym in symbols:
  data[sym] = web3.DataReader(sym, data_source='yahoo')['Close']
data = data.dropna()
# Let's separate the index from the individual assets
dax = pd.DataFrame(data.pop('^GDAXI'))
data[data.columns[:6]].head()

# Let's normalizer our data before extracting the principal components
scale_function = lambda x: (x - x.mean()) / x.std()
# The principal components are eigenvalues
pca = KernelPCA().fit(data.apply(scale_function))

# I couldn't convert dates to num with the dates2num function
# apparently it doesn't take vectors of dates anymore
# My solution:

mpl_dates = []
for t in data.index:
  mpl_dates.append(mpl.dates.date2num(t))
# mpl_dates = np.array(mpl_dates) # don't think this is necessary

lin_reg = np.polyval(np.polyfit(dax['PCA_5'], dax['^GDAXI'],1),
		     dax['PCA_5'])

plt.figure(figsize=(8,4))
plt.scatter(dax['PCA_5'], dax['^GDAXI'], c=mpl_dates)
plt.plot(dax['PCA_5'], lin_reg, 'r', lw=3)
plt.grid(True)
plt.xlabel('PCA_5')
plt.ylabel('^GDAXI')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
	     format=mpl.dates.DateFormatter('%d %b %y'))

cut_date1 = '2011/7/1'
cut_date2 = '2015/11/1'
early_pca = dax[dax.index < cut_date1]['PCA_5']
mid_pca = dax[(dax.index >= cut_date1) &
	      (dax.index < cut_date2)]['PCA_5']
late_pca = dax[dax.index >= cut_date2]['PCA_5']
early_reg = np.polyval(np.polyfit(early_pca,
		dax['^GDAXI'][dax.index < cut_date1], 1),
		early_pca)
mid_reg = np.polyval(np.polyfit(mid_pca,
		dax['^GDAXI'][(dax.index >= cut_date1) &
		(dax.index < cut_date2)], 1), mid_pca)
late_reg = np.polyval(np.polyfit(late_pca,
		dax['^GDAXI'][dax.index >= cut_date2], 1),
		late_pca)

plt.figure(figsize=(8,4))
plt.scatter(dax['PCA_5'], dax['^GDAXI'], c=mpl_dates)
plt.plot(early_pca, early_reg, 'r', lw=3)
plt.plot(mid_pca, mid_reg, 'r', lw=3)
plt.plot(late_pca, late_reg, 'r', lw=3)
plt.grid(True)
plt.xlabel('PCA_5')
plt.ylabel('^GDAXI')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
	     format=mpl.dates.DateFormatter('%d %b %y'))

# The following lines of code requires PyMC3 to be installed
# Follow the installation instructions given on github
# for PyMC3 that currently suggests running (given all
# depip install patsy pandaspendencies resolved):

# pip install git+https://github.com/pymc-devs/pymc3
# pip install patsy pandas

with pm.Model() as model:
  # model specifications in PyMC3
  # are wrapped in a with statement
  # define priors
  tau_0 = 1./20**2
  alpha = pm.Normal('alpha', mu=0, tau = tau_0)
  beta  = pm.Normal('beta', mu=0, tau=tau_0)
  sigma = pm.Uniform('sigma', lower=0, upper=10)
  
  # define linear regression
  y_est = alpha + beta * x
  # define likelihood
  likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
  # inference
  start = pm.find_MAP()
    # find starting value by optimization
  step = pm.NUTS(state=start)
    # instantiate MCMC sampling algorithm
  trace = pm.sample(100, step, start=start, progressbar=True)
  

plt.figure(figsize=(8, 4))
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
plt.grid(True)
plt.xlabel('GDX')
plt.ylabel('GLD')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
format=mpl.dates.DateFormatter('%d %b %y'))

# p319 
with pm.Model() as model:
  alpha = pm.Normal('alpha', mu=0, sd=20)
  beta = pm.Normal('beta', mu=0, sd=20)
  sigma = pm.Uniform('sigma', lower=0, upper=50)
  
  y_est = alpha + beta * data['GDX'].values
  
  likelihood = pm.Normal('GLD', mu=y_est, sd=sigma,
			 observed=data['GLD'].values)
  start = pm.find_MAP()
  step = pm.NUTS(state=start)
  trace = pm.sample(100, step, start=start, progressbar=True)


# Plot the linear regressions from the PyMC3 trace values

plt.figure(figsize=(8, 4))
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
plt.grid(True)
plt.xlabel('GDX')
plt.ylabel('GLD')
for i in range(len(trace)):
  plt.plot(data['GDX'], trace['alpha'][i] + trace['beta'][i] * data
['GDX'])
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
format=mpl.dates.DateFormatter('%d %b %y'))

# p321
# defining the theoretical models

model_randomwalk = pm.Model()
with model_randomwalk:
  # std of random walk best sampled in log space
  sigma_alpha, log_sigma_alpha = \
	  model_randomwalk.TransformedVar('sigma_alpha',
	  pm.Exponential.dist(1. / .02, testval=.1),
	  pm.logtransform)
  sigma_beta, log_sigma_beta = \
	  model_randomwalk.TransformedVar('sigma_beta',
	  pm.Exponential.dist(1. / .02, testval=.1),
	  pm.logtransform)

# The 'TransformedVar' function has been removed, see:
# http://stackoverflow.com/questions/32289834/i-have-been-trying-to-folllow-the-tutorial-on-pymc3-but-when-it-comes-to-the
#
# We use this instead:

model_randomwalk = pm.Model()
with model_randomwalk:
  # std of random walk best sampled in log space
  sigma_alpha = pm.Exponential('sigma_alpha', 1. / .02, testval=.1)
  sigma_beta = pm.Exponential('sigma_beta', 1. / .02, testval=.1)

from pymc3.distributions.timeseries import GaussianRandomWalk
# to make the model simpler, we will apply the same coefficients
# to 50 data points at a time
subsample_alpha = 50
subsample_beta  = 50

with model_randomwalk:
  alpha = GaussianRandomWalk('alpha', sigma_alpha**-2,
			     shape=len(data) / subsample_alpha)
  beta = GaussianRandomWalk('beta', sigma_beta**-2,
			    shape=len(data) / subsample_beta)
  # make coefficients have the same length as prices
  alpha_r = np.repeat(alpha, sfig, ax1 = plt.subplots(figsize=(10, 5))
plt.plot(part_dates, np.mean(trace_rw['alpha'], axis=0),
'b', lw=2.5, label='alpha')
for i in range(45, 55):
  plt.plot(part_dates, trace_rw['alpha'][i], 'b', lw=4, alpha=.1)
plt.xlabel('date')
plt.ylabel('alpha')
plt.axis('tight')
plt.grid(True)
plt.legend(loc=2)
ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b %y') )
ax2 = ax1.twinx()
plt.plot(part_dates, np.mean(trace_rw['beta'], axis=0),
'r', lw=2.5, label='beta')
for i in range(45, 55):
  plt.plot(part_dates, trace_rw['beta'][i], 'r', lw=4, alpha=.1)
plt.ylabel('beta')
plt.legend(loc=4)
fig.autofmt_xdate()
plt.title("Evolution of (mean) alpha and (mean) beta over time (updated estimates over time)")
ubsample_alpha)
  beta_r = np.repeat(beta, subsample_beta)
  
# The GDX.values vector has 2546 datapoints so we remove the
# last 46 points to make it divisible by 50

with model_randomwalk:
  # define regression
  regression = alpha_r + beta_r * data.GDX.values[:2500]
  
  # assume prices are normally distributed
  # the mean comes from the regression
  sd = pm.Uniform('sd', 0, 20)
  likelihood = pm.Normal('GLD', mu=regression, sd=sd,
			 observed=data.GLD.values[:2500])

import scipy.optimize as sco
with model_randomwalk:
  # first optimize random walk
  start = pm.find_MAP(vars=[alpha, beta], fmin=sco.fmin_l_bfgs_b)
  
  # sampling
  step = pm.NUTS(scaling=start)
  trace_rw = pm.sample(100, step, start=start, progressbar=True)

part_dates = np.linspace(min(mpl_dates), max(mpl_dates), 50)

fig, ax1 = plt.subplots(figsize=(10, 5))
plt.plot(part_dates, np.mean(trace_rw['alpha'], axis=0),
'b', lw=2.5, label='alpha')
for i in range(45, 55):
  plt.plot(part_dates, trace_rw['alpha'][i], 'b', lw=4, alpha=.1)
plt.xlabel('date')
plt.ylabel('alpha')
plt.axis('tight')
plt.grid(True)
plt.legend(loc=2)
ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b %y') )
ax2 = ax1.twinx()
plt.plot(part_dates, np.mean(trace_rw['beta'], axis=0),
'r', lw=2.5, label='beta')
for i in range(45, 55):
  plt.plot(part_dates, trace_rw['beta'][i], 'r', lw=4, alpha=.1)
plt.ylabel('beta')
plt.legend(loc=4)
fig.autofmt_xdate()
plt.title("Evolution of (mean) alpha and (mean) beta over time (updated estimates over time)")


plt.figure(figsize=(10, 5))
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
	      format=mpl.dates.DateFormatter('%d %b %y'))
plt.grid(True)
plt.xlabel('GDX')
plt.ylabel('GLD')
x = np.linspace(min(data['GDX']), max(data['GDX']))
for i in range(50):
  alpha_rw = np.mean(trace_rw['alpha'].T[i])
  beta_rw = np.mean(trace_rw['beta'].T[i])
  plt.plot(x, alpha_rw + beta_rw * x,
	   color=plt.cm.jet(256 * i / 50), alpha=.7)
plt.xlim(xmin=min(data['GDX']), xmax=max(data['GDX']))
plt.ylim(ymin=min(np.mean(trace_rw['alpha'], axis=0) + 
		  np.mean(trace_rw['beta'], axis=0)*x[0]),
	 ymax=max(np.mean(trace_rw['alpha'], axis=0) +
		  np.mean(trace_rw['beta'], axis=0)*x[-1]))



class ExampleTwo(object):
  def __init__(self, a, b):
    self.a = a
    self.b = b
class ExampleFour(ExampleTwo):
  def addition(self):
    return self.a + self.b

# p357
import numpy as np
def discount_factor(r, t):
  """
  Function to calculate discount factor
  
  Parameters
  ==========
  r : float
    positive, constant short rate
  t : float, array of floats
    future date(s), in fraction of years;
    e.g. 0.5 means half a year from now
  
  Returns
  =======
  df : float
    discount factor
  """
  df = np.exp(-r * t)
    # use of NumPy universal function for vectorization
  return df

# class definition

class short_rate(object):
  """
  Class to model a constant short rate object.
  
  Parameters
  ==========
  name : string
    name of the object
  rate : float
    positive, constant short rate
  
  Methods
  =======
  get_discount_factors :
    returns discount factors for given list/array
    of dates/times (as year fractions)
  """
  def __init__(self, name, rate):
    self.name = name
    self.rate = rate
  def get_discount_factor(self, time_list):
    " time_list : list/array-like "
    time_list = np.array(time_list)
    return np.exp(-self.rate * time_list)


for r in [0.025, 0.05, 0.1, 0.15]:
  sr.rate = r
  plt.plot(t, sr.get_discount_factor(t),
label='r=%4.2f' % sr.rate, lw=1.5)
plt.xlabel('years')
plt.ylabel('discount factor')
plt.grid(True)
plt.legend(loc=0)

class short_rate(trapi.HasTraits):
  name = trapi.Str
  rate = trapi.Float
  time_list = trapi.Array(dtype=np.float, shape=(5,))
  def get_discount_factors(self)
    return np.exp(-self.rate * self.time_list)

