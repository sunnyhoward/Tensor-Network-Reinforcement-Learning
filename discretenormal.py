import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats
import math
matplotlib.rcParams['mathtext.fontset'] = 'cm'
mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma),'k')
plt.xlim([-3,3])
plt.ylim([0,0.45])
plt.fill_between(np.linspace(-1,1,100),stats.norm.pdf(np.linspace(-1,1,100), mu, sigma))
plt.fill_between(np.linspace(-3,-1,100),stats.norm.pdf(np.linspace(-3,-1,100), mu, sigma),color = 'r')
plt.fill_between(np.linspace(1,3,100),stats.norm.pdf(np.linspace(1,3,100), mu, sigma),color = 'g')
plt.text(-0.30,0.02,r'$\mathcal{X}_{d} = 0$', bbox=dict(facecolor='white', alpha=0.8),fontsize = 11) 
plt.text(-1.95,0.02,r'$\mathcal{X}_{d}=-1$', bbox=dict(facecolor='white', alpha=0.8),fontsize = 11) 
plt.text(1.2,0.02,r'$\mathcal{X}_{d}=1$', bbox=dict(facecolor='white', alpha=0.8),fontsize = 11) 
plt.text(0.8,0.4,r'$\mathcal{X} = \mathcal{N}(\mu = 0,\sigma^{2})$', bbox=dict(facecolor='white', alpha=0.8),fontsize = 14) 

plt.ylabel('P.D.F',fontsize = 13)
plt.xlabel(r'$\mathcal{X}$',fontsize = 13)
plt.show()