from scipy.stats import distributions as iid
import numpy as np 


def dgp(n, β, π):
    ## Generate errors 
    u = iid.norm(0, 1)
    v = iid.norm(0, 1)
    Z = iid.norm(0, 1)
    u = u.rvs(n).reshape(-1, 1)
    v = v.rvs(n).reshape(-1,1)

    ## the number of instruments to generate = number of elements in pi
    if type(π) == np.ndarray:
        if π.ndim == 1: 
            π = π.reshape(-1, 1)
    else: 
        π = np.array([π]).reshape(-1,1)
    d = π.size
    Z = Z.rvs(n*d).reshape(n, d)
    x = Z@π + v
    y = β*x + u 
    return (y, x, Z) 


def two_sls(y, x, Z): 
    π = np.linalg.solve(Z.T@Z, Z.T@x)
    xhat = Z@π
    β = np.linalg.solve(xhat.T@xhat, xhat.T@y)
    # calculate variance
    df = Z.shape[0] - Z.shape[1]
    e = y - x@β
    V = compute_variance(e, xhat, df)
    return β, V


def ttest(est, se, val = 0):
    t = (est - val)/se
    return t


def hansen(x, y, Z, b0):
    ybar = y - b0*x
    gammahat = np.linalg.solve(Z.T@Z, Z.T@ybar)
    e = ybar - Z@gammahat
    df = Z.shape[0] - Z.shape[1]
    V = compute_variance(e, Z, df)
    # define f statistic 
    df1 = gammahat.size ## number of restrictions (setting all gamma = 0)
    df2 = df ## N - number of parameters 
    Q = Z@(V/Z.shape[0])@Z.T
    fstat = (Z@gammahat).T@np.linalg.pinv(Q)@(Z@gammahat)
    p = 2*(1 - iid.f.cdf(fstat, df1, df2))
    return p

def compute_variance(e, X, df):
    s2 = e.T@e / df
    V = s2 * np.linalg.inv(X.T@X)
    return V


