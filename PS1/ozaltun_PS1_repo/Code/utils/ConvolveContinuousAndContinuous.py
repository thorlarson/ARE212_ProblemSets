import numpy as np
from scipy.stats import distributions as iid
import matplotlib.pyplot as plt

class ConvolveContinuousAndContinuous(iid.rv_continuous):

    """Convolve (add) a continuous rv x and a discrete rv s,
       returning the resulting cdf."""

    def __init__(self,x,y):
        self.x = x
        self.y = y
        super(ConvolveContinuousAndContinuous, self).__init__(name="ConvolveContinuousAndContinuous")
        
    def _pdf(self,z):
        y = self.y
        x = self.x
        # Lets integrate via monte carlo
        y_draws = y.rvs(50000) # Get draws
        f_est   = np.mean(self.x.pdf(z-y_draws)) # 
        return f_est

    def _cdf(self,z):
        y = self.y
        x = self.x
        # Lets integrate via monte carlo
        y_draws = y.rvs(50000) # Get draws
        F_est   = np.mean(self.x.cdf(z-y_draws)) # 
        return F_est
    
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