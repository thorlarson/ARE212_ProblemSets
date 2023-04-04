import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import inv, sqrtm
import pandas as pd
import scipy.integrate as integrate

def generate_covariance_matrix(n):
    ## creates a random covariance matrix of dimension n
    A = np.random.random(size=(n, n))  
    return A @ A.T

def pt_5_6(N, k, m, question):
    np.random.seed(1234)
    '''
        N: number of samples 
        k: number of dependent variables (Y)
        m: number of independent variables (X)
    '''
    ## Generate an Nxk matrix of errors 
    mu = [0]*k
    Sigma = generate_covariance_matrix(k)
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
    if question == "five":
        D = np.random.random(size=(m, m)) 
        X = (T**3)@D 
    # for part 6, we want X = T
    else:
        X = T
    print(f"True beta: \n{beta} \n\n")

    y = X@beta + u 

    b = np.linalg.lstsq(T.T@X,T.T@y, rcond=None)[0]
    print(f"Estimated b: \n{b}")
    e = y - X@b
    TXplus = np.linalg.pinv(T.T@X) # Moore-Penrose pseudo-inverse
    # Covariance matrix of b
    vb = e.var()*TXplus@T.T@T@TXplus.T 
    return


def kernel_bias(kernel, density, h): 
    return lambda x: integrate.quad(lambda u: kernel(u)*density(x + h * u), -np.inf, np.inf)[0] - density(x)

