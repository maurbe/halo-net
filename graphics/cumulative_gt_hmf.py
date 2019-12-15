import numpy as np, os
import pandas as pd
import matplotlib.ticker as ticker
from analysis.helpers.plotting_help import *

homedir = os.path.dirname(os.getcwd()) + '/'
data = pd.read_csv(homedir + 'source/catalogT/catalogT.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
halo_sizesT = np.log10(halo_sizes)

data = pd.read_csv(homedir + 'source/catalogA/catalogA.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
halo_sizesA = np.log10(halo_sizes)


all_sets = np.hstack((halo_sizesT, halo_sizesA))
x = np.linspace(all_sets.min(), all_sets.max(), 15) - np.log10(64.0/5.66e10)


countsT = plt.hist(np.log10(halo_sizesT), bins=15, histtype='step', log=True, cumulative=-1, color='k')[0]
countsA = plt.hist(np.log10(halo_sizesA), bins=15, histtype='step', log=True, cumulative=-1, color='navy')[0]

plt.figure()
plt.plot(x, np.log10(countsT), 'k-', linewidth=2, color='#1565C0', label='Training')
plt.plot(x, np.log10(countsA), 'k-', linewidth=2, color='darkorange', label='Validation')
plt.legend(loc='upper right', prop={'size': 15}, frameon=False)
plt.xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')
plt.ylabel(r'$\log_{10}$N($>$M)')
plt.savefig('accumulated_hmf.png', dpi=200)
plt.show()
