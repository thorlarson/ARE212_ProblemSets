from scipy.stats import distributions as iid
import numpy as np 


def dgp(n, β, π):
    ## Generate errors 
    u = iid.norm(0, 1)
    v = iid.norm(0, 1)
    Z = iid.norm(0, 1)
    u = u.rvs(n).reshape(-1, 1)
    v = v.rvs(n).reshape(-1,1)
    Z = Z.rvs(n).reshape(-1,1)

    x = π*Z + v
    y = β*x + u 
    return (y, x, Z) 


def two_sls(y, x, Z): 
    π = np.linalg.solve(Z.T@Z, Z.T@x) 
    xhat = Z@π
    β = np.linalg.solve(xhat.T@xhat, xhat.T@y)
    n = Z.shape[0]

    # calculate variance
    e = y - x@β
    s2 = e.T@e / n 
    V = s2 * np.linalg.inv(xhat.T@xhat)
    return β, V



n = 10000
β = 1 
π = 1
x, y, Z = dgp(n, β, π)

b, V = two_sls(x, y, Z)

print(b)
print(V)