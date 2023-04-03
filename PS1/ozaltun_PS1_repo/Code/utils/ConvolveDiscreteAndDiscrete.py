import numpy as np
from scipy.stats import distributions as iid
import matplotlib.pyplot as plt

class ConvolveDiscreteAndDiscrete():
    """Convolve two discrete random variables s and t.
    So we want to return the properties of k = s + t"""
    def __init__(self,s, t):
        # Constructor
        self.s  = s
        self.t  = t
        self.xk = np.sort(np.unique(np.array([i+j for i in s.xk for j in t.xk])))
        self.pk = np.array([self.pmf(self.xk[i]) for i in range(len(self.xk))])
    def pmf(self,k):
        # Getting the probability mass for a given k (scalar)
        if(k not in self.xk):
            return 0
        f=0
        s = self.s
        t = self.t
        # Lets get the pmf
        for i in range(len(s.xk)):
            # check if k-s.xk is in t.xk
            if(k-s.xk[i] in t.xk):
                f = f + t.pk[np.where(k-s.xk[i] == t.xk)]*s.pk[i]
                f = f.item()
        return f
    def cdf(self,k):
        # The cdf of a rv is a right-continuous function. So we
        # need it to be defined for k not in xk.
        supp = self.xk
        # Find all values that are smaller than k
        neg_supp = supp[supp-k <= 0]
        if(neg_supp.shape[0]==0):
            # If there are none, then the cdf is 0
            F = 0
        else:
            # If there are smaller value, find the closest value
            cdf_value = neg_supp.max()
            # The cdf is the sum of all values smaller
            F = sum(self.pk[np.where(self.xk<=cdf_value)])
            F = F.item()
        return F
    
    def plot_distributions(self, set_figsize=(8,4)):
        # Plot distributions
        x_grid = np.linspace(self.xk.min() - abs(self.xk.min())/2, self.xk.max() + abs(self.xk.max())/2, 300)
        cdf = np.zeros(x_grid.shape)
        for i, value in enumerate(x_grid):
            cdf[i] = self.cdf(value)

        fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=set_figsize)
        ax[0].plot(x_grid, cdf)
        ax[0].set_ylabel("CDF value")
        ax[1].plot(self.xk, self.pk, 'ro', ms=8, mec='r')
        ax[1].vlines(self.xk, 0., self.pk, colors='r', linestyles='-', lw=2)
        ax[1].set_xticks(self.xk)
        ax[1].set_xlabel("Variable value")
        ax[1].set_ylabel("PMF value")
        return fig, ax