import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import inv, sqrtm

def generate_covariance_matrix(n):
    ## creates a random covariance matrix of dimension n
    A = np.random.random(size=(n, n))  
    return A @ A.T

def pt_5(N, k, m):
    np.random.seed(1234)
    '''
        N: number of samples 
        k: number of dependent variables (Y)
        m: number of independent variables (X)
    '''

    ## Generate an Nxk matrix of errors 
    mu = [0]*k
    Sigma = generate_covariance_matrix(k)
    print(Sigma)
    u = multivariate_normal(mu, Sigma)

    ## Generate an Nxm matrix for T
    mu = [0]*m
    Sigma = generate_covariance_matrix(m)
    T = multivariate_normal(mu,Sigma)

    ## Define the mxk beta matrix (lets just make it increasing integers starting at 1)
    beta = np.arange(1, (m*k)+1).reshape(m, k)

    ## Get the sample 
    T = T.rvs(N)
    u = u.rvs(N) 

    # generate D and X
    # D = np.random.random(size=(m, m)) 
    # X = (T**3)@D 
    # for part 6, we want X = T
    X = T
    print(f"True beta: {beta} \n\n")

    y = X@beta + u 

    b = np.linalg.lstsq(T.T@X,T.T@y, rcond=None)[0]
    print(f"Estimated b: {b}")

    e = y - X@b

    TXplus = np.linalg.pinv(T.T@X) # Moore-Penrose pseudo-inverse

    # Covariance matrix of b
    vb = e.var()*TXplus@T.T@T@TXplus.T  # u is known to be homoskedastic

    print(vb)

if __name__ == "__main__":
    pt_5(1000, 3, 2)

    
    