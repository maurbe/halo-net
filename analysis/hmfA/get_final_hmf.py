import numpy as np, os
import pandas as pd
import matplotlib.ticker as ticker
from analysis.helpers.plotting_help import *

projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
data = pd.read_csv(projectdir + 'source/catalogA/catalogA.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
halo_sizes = np.log10(halo_sizes)
print('Ground truth number of halos:', len(halo_sizes))

ps_pred = np.load('src/corrected_sizes_predictedA.npy')
ps_pred = ps_pred[1:]
print('Number of proto halos before thresholding:', len(ps_pred))
ps_pred = ps_pred[ps_pred>=16*285]
ps_pred = np.log10(ps_pred)
print('Number of proto halos after thresholding:', len(ps_pred))

all_sets = np.hstack((ps_pred, halo_sizes))
no_bins = 20
bins = np.histogram(all_sets, range=(all_sets.min(), all_sets.max()), bins=no_bins)[1]
bins_inset = np.histogram(all_sets - np.log10(64.0/5.66e10),
                          range=((all_sets-np.log10(64.0/5.66e10)).min(), (all_sets-np.log10(64.0/5.66e10)).max()),
                          bins=no_bins)[1]

# create figure
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.0)
x = np.linspace(all_sets.min(), all_sets.max(), no_bins) - np.log10(64.0/5.66e10)

# Inset plot
ax_inset = fig.add_axes([0.2, 0.41, 0.32, 0.2])
counts_log_halo_sizes = ax_inset.hist(halo_sizes, bins=bins, log=True, cumulative=-1, color='k', alpha=0.1)[0]
counts_log_pred = ax_inset.hist(ps_pred, bins=bins, log=True, cumulative=-1, histtype='step', color='darkorange')[0]
ax_inset.cla()
ax_inset.hist(halo_sizes - np.log10(64.0/5.66e10), bins=bins_inset, log=True, cumulative=0, color='k', alpha=0.2)[0]
ax_inset.hist(ps_pred - np.log10(64.0/5.66e10), bins=bins_inset, log=True, cumulative=0, histtype='step', color='darkorange')[0]
ax_inset.set_ylabel('N', fontsize=10)
ax_inset.set_xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$', fontsize=10)
ax_inset.xaxis.labelpad = 3
ax_inset.yaxis.labelpad = -2
for tick in ax_inset.get_xticklabels():
    tick.set_fontsize(7)
for tick in ax_inset.get_yticklabels():
    tick.set_fontsize(7)
start, end = ax_inset.get_xlim()
ax_inset.xaxis.set_ticks(np.linspace(start, end, num=7)[1:-1])
ax_inset.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
ax_inset.text(0.95, 0.95, 'Total: {}'.format(len(halo_sizes)), fontsize=10, color='k', ha='right', va='top', transform = ax_inset.transAxes)
ax_inset.text(0.95, 0.85, 'Total: {}'.format(len(ps_pred)), fontsize=10, color='darkorange', ha='right', va='top', transform = ax_inset.transAxes)


dz = 0.434 * np.sqrt(counts_log_pred)/counts_log_pred
error_high = np.log10(counts_log_pred) + dz
error_low  = np.log10(counts_log_pred) - dz

# Upper plot
ax.plot(x, np.log10(counts_log_halo_sizes), 'k-', label='True Halos')
ax.plot(x, np.log10(counts_log_pred), 'darkorange', label='Predicted Halos')
ax.fill_between(x, error_low, error_high, color='darkorange', alpha=0.4, linewidth=0)
ax.set_ylabel(r'$\log_{10}$N($>$M)')
ax.set_ylim((-0.25, 3.75))
ax.xaxis.set_ticks_position('none')
ax.legend(loc='upper right', prop={'size': 15}, frameon=False)

ax2.plot(x, (counts_log_pred-counts_log_halo_sizes)/counts_log_halo_sizes, color='darkorange', linestyle='-')
ax2.fill_between(x,
                 (counts_log_pred+np.sqrt(10**error_high)-counts_log_halo_sizes)/counts_log_halo_sizes,
                 (counts_log_pred-np.sqrt(10**error_high)-counts_log_halo_sizes)/counts_log_halo_sizes,
                 color='darkorange', alpha=0.4, linewidth=0)
ax2.axhspan(ymin=-0.1, ymax=0.1, color='k', alpha=0.2, linewidth=1, linestyle='--')
ax2.axhline(y=0.0, color='k', alpha=0.2, linewidth=1, linestyle='--')
ax2.set_ylabel(r'$\Delta$ $[\%]$')
ax2.set_xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')
ax2.set_ylim((-1.0, 1.0))
ax2.set_yticks([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
plt.savefig('hmfA.png', dpi=300)
plt.show()
