from scipy.stats import distributions as iid
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

class ConvolveDiscrete():
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

# class ConvolveDiscrete(iid.rv_discrete):
#     def __init__(self, x, y):
#         self.x = x 
#         self.y = y
#         super().__init__()

#     def _cdf(self, z):
#         F = 0 
#         for val, prob in zip(self.x.xk, self.x.pk):
#             F += self.y.cdf(z - val)* prob
#         return F

#     def _pmf(self, z):
#         f = 0 
#         for val, prob in zip(self.x.xk, self.x.pk):
#             f += self.y.pmf(z - val)* prob
#         return f


class ConvolveContinuous(iid.rv_continuous):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        super().__init__()

    def _cdf(self, z):
        return integrate.quad(lambda x: self.y.cdf(z - x) * self.x.pdf(x), -np.inf, np.inf)

    def _pdf(self, z):
        return integrate.quad(lambda x: self.y.pdf(z - x) * self.x.pdf(x), -np.inf, np.inf)

    def plot_distributions(self,x_range=(-5,5), set_figsize=(8,4)):
        # Plot distributions
        x_grid = np.linspace(x_range[0], x_range[1])
        pdf = np.zeros(x_grid.shape)
        cdf = np.zeros(x_grid.shape)
        for i, value in enumerate(x_grid):
            pdf[i] = self.pdf(value)
            cdf[i] = self.cdf(value)

        fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=set_figsize)
        ax[0].plot(x_grid, cdf)
        ax[0].set_ylabel("CDF value")
        ax[1].plot(x_grid, pdf,color="red")
        ax[1].set_xlabel("Variable value")
        ax[1].set_ylabel("PDF value")

        return fig, ax

class KernelDensityEstimator:
    def __init__(self, X, h, kernel = 'default'):
        self.ktype = kernel
        self.kernel = self.k(self.ktype)
        self.X = X
        self.h = h
        self.fhat = self.kernel_estimator()

    def k(self, kernel): 
        if kernel == "default":
            return lambda u: (np.abs(u) < np.sqrt(3))/(2*np.sqrt(3))
        if kernel == "gaussian":
            return lambda u: np.exp(-(u**2)/2)/np.sqrt(2*np.pi)

    def kernel_estimator(self):
        return lambda x: self.kernel((self.X-x)/self.h).mean()/self.h

    def plot(self, fname, hist_bin_width = 1000, fig_y_limit = 50000):
        # Set a grid to get the KDE estimates
        X = np.linspace(0, fig_y_limit, hist_bin_width)

        # Get KDE estimates
        Y = [self.fhat(x) for x in X]

        fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(10, 5))
        ax.hist(self.X, 
                bins=range(min(self.X), fig_y_limit + hist_bin_width, hist_bin_width),
                density=True)
        ax.plot(X, Y, lw=1.5)
        ax.set_xlim([0, 50000])
        ax.set_xlabel("Household non-durable expenditures in INR")
        ax.set_ylabel("Density")
        ax.legend([f"{self.ktype.title()} kernel. h={round(self.h, 1)}","histogram"])
        # ax.set_title("The histogram and kernel density estimator using Silverman's rule.")
        fig.tight_layout()
        fig.savefig(f"../output/{fname}.png")
        plt.show()

