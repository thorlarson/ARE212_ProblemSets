import numpy as np
from scipy.stats import multivariate_normal





if __name__ == "__main__":

    k = 3 # Number of observables in T

    mu = [0]*k
    Sigma=[[1,0.5,0],
        [0.5,2,0],
        [0,0,3]]

    T = multivariate_normal(mu,Sigma)

    u = multivariate_normal(cov=0.2)
    beta = [1/2,1]

    D = np.random.random(size=(3,2)) # Generate random 3x2 matrix
    N=1000 # Sample size
    # Now: Transform rvs into a sample
    T = T.rvs(N)
    u = u.rvs(N) # Replace u with a sample

    X = (T**3)@D  # Note use of ** operator for exponentiation

    y = X@beta + u # Note use of @ operator for matrix multiplication
    print(y.shape)
    
    