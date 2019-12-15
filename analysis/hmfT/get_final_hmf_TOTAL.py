import numpy as np, os
import pandas as pd
import matplotlib.ticker as ticker
from analysis.helpers.plotting_help import *
#plt.style.use('/Users/Mauro/Desktop/Biotop2/graphics/mystyle.mplstyle')
font = {'size'   : 12}
matplotlib.rc('font', **font)

projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
data = pd.read_csv(projectdir + 'source/catalogT/catalogT.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
halo_sizes = np.log10(halo_sizes)
print('Ground truth number of halos:', len(halo_sizes))


ps_pred = np.load('src/corrected_sizes_predictedT.npy')
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
fig, ((axT, axA), (ax2T, ax2A)) = plt.subplots(2, 2, figsize=(12, 7), sharex='col', sharey='row', gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.0)
x = np.linspace(all_sets.min(), all_sets.max(), no_bins) - np.log10(64.0/5.66e10)

# Inset plot
ax_insetT = fig.add_axes([0.12, 0.48, 0.2, 0.2])
counts_log_halo_sizes = ax_insetT.hist(halo_sizes, bins=bins, log=True, cumulative=-1, color='k', alpha=0.1)[0]
counts_log_pred = ax_insetT.hist(ps_pred, bins=bins, log=True, cumulative=-1, histtype='step', color='cornflowerblue')[0]
ax_insetT.cla()
ax_insetT.hist(halo_sizes - np.log10(64.0/5.66e10), bins=bins_inset, log=True, cumulative=0, color='k', alpha=0.2)[0]
ax_insetT.hist(ps_pred - np.log10(64.0/5.66e10), bins=bins_inset, log=True, cumulative=0, histtype='step', color='#1565C0')[0]
ax_insetT.set_ylabel('N')
ax_insetT.set_xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')

ax_insetT.xaxis.labelpad = 3
ax_insetT.yaxis.labelpad = -5
for tick in ax_insetT.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax_insetT.get_yticklabels():
    tick.set_fontsize(12)
start, end = ax_insetT.get_xlim()
ax_insetT.xaxis.set_ticks(np.linspace(start, end, num=7)[1:-1])
ax_insetT.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
ax_insetT.text(0.95, 0.95, 'Total: {}'.format(len(halo_sizes)), fontsize=12, color='k', ha='right', va='top', transform = ax_insetT.transAxes)
ax_insetT.text(0.95, 0.85, 'Total: {}'.format(len(ps_pred)), fontsize=12, color='#1565C0', ha='right', va='top', transform = ax_insetT.transAxes)


dz = 0.434 * np.sqrt(counts_log_pred)/counts_log_pred
error_high = np.log10(counts_log_pred) + dz
error_low  = np.log10(counts_log_pred) - dz

# Upper plot
axT.text(0.125, 0.925, 'Training', horizontalalignment='center', fontsize=15,
         verticalalignment='center', transform=axT.transAxes)
axT.plot(x, np.log10(counts_log_halo_sizes), 'k-', label='True Halos')
axT.plot(x, np.log10(counts_log_pred), '#1565C0', label='Predicted Halos')
axT.fill_between(x, error_low, error_high, color='cornflowerblue', alpha=0.4, linewidth=0)
axT.set_ylabel(r'$\log_{10}$N($>$M)')
axT.set_ylim((-0.25, 3.75))
axT.xaxis.set_ticks_position('none')
axT.legend(loc='upper right', prop={'size': 12}, frameon=False)

ax2T.plot(x, (counts_log_pred-counts_log_halo_sizes)/counts_log_halo_sizes, color='#1565C0', linestyle='-')
ax2T.fill_between(x,
                 (counts_log_pred+np.sqrt(10**error_high)-counts_log_halo_sizes)/counts_log_halo_sizes,
                 (counts_log_pred-np.sqrt(10**error_high)-counts_log_halo_sizes)/counts_log_halo_sizes,
                 color='cornflowerblue', alpha=0.4, linewidth=0)
ax2T.axhspan(ymin=-0.1, ymax=0.1, color='k', alpha=0.2, linewidth=1, linestyle='--')
ax2T.axhline(y=0.0, color='k', alpha=0.2, linewidth=1, linestyle='--')
ax2T.set_ylabel(r'$\Delta$ $[\%]$')
ax2T.set_xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')
ax2T.set_ylim((-1.0, 1.0))
ax2T.set_yticks([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
ax2T.set_yticklabels([-100, -50, -10, 10, 50, 100])
ax2T.set_xticks([13, 13.5, 14, 14.5])

# ----------------------------------------------------------------------------------------------------------------------

data = pd.read_csv(projectdir + 'source/catalogA/catalogA.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
halo_sizes = np.log10(halo_sizes)
print('Ground truth number of halos:', len(halo_sizes))

homedir = os.path.dirname(os.getcwd()) + '/'
ps_pred = np.load(homedir + 'hmfA/src/corrected_sizes_predictedA.npy')
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
x = np.linspace(all_sets.min(), all_sets.max(), no_bins) - np.log10(64.0/5.66e10)

# Inset plot
ax_insetA = fig.add_axes([0.585, 0.48, 0.2, 0.2])
counts_log_halo_sizes = ax_insetA.hist(halo_sizes, bins=bins, log=True, cumulative=-1, color='k', alpha=0.1)[0]
counts_log_pred = ax_insetA.hist(ps_pred, bins=bins, log=True, cumulative=-1, histtype='step', color='darkorange')[0]
ax_insetA.cla()
ax_insetA.hist(halo_sizes - np.log10(64.0/5.66e10), bins=bins_inset, log=True, cumulative=0, color='k', alpha=0.2)[0]
ax_insetA.hist(ps_pred - np.log10(64.0/5.66e10), bins=bins_inset, log=True, cumulative=0, histtype='step', color='darkorange')[0]
ax_insetA.set_ylabel('N')
ax_insetA.set_xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')
ax_insetA.xaxis.labelpad = 3
ax_insetA.yaxis.labelpad = -5
for tick in ax_insetA.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax_insetA.get_yticklabels():
    tick.set_fontsize(12)
start, end = ax_insetA.get_xlim()
ax_insetA.xaxis.set_ticks(np.linspace(start, end, num=7)[1:-1])
ax_insetA.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
ax_insetA.text(0.95, 0.95, 'Total: {}'.format(len(halo_sizes)), fontsize=12, color='k', ha='right', va='top', transform = ax_insetA.transAxes)
ax_insetA.text(0.95, 0.85, 'Total: {}'.format(len(ps_pred)), fontsize=12, color='darkorange', ha='right', va='top', transform = ax_insetA.transAxes)


dz = 0.434 * np.sqrt(counts_log_pred)/counts_log_pred
error_high = np.log10(counts_log_pred) + dz
error_low  = np.log10(counts_log_pred) - dz

# Upper plot
axA.text(0.125, 0.925, 'Validation', horizontalalignment='center', fontsize=15,
         verticalalignment='center', transform=axA.transAxes)
axA.plot(x, np.log10(counts_log_halo_sizes), 'k-', label='True Halos')
axA.plot(x, np.log10(counts_log_pred), 'darkorange', label='Predicted Halos')
axA.fill_between(x, error_low, error_high, color='darkorange', alpha=0.4, linewidth=0)
axA.set_ylim((-0.25, 3.75))
axA.xaxis.set_ticks_position('none')
axA.legend(loc='upper right', prop={'size': 12}, frameon=False)

ax2A.plot(x, (counts_log_pred-counts_log_halo_sizes)/counts_log_halo_sizes, color='darkorange', linestyle='-')
ax2A.fill_between(x,
                 (counts_log_pred+np.sqrt(10**error_high)-counts_log_halo_sizes)/counts_log_halo_sizes,
                 (counts_log_pred-np.sqrt(10**error_high)-counts_log_halo_sizes)/counts_log_halo_sizes,
                 color='darkorange', alpha=0.4, linewidth=0)
ax2A.axhspan(ymin=-0.1, ymax=0.1, color='k', alpha=0.2, linewidth=1, linestyle='--')
ax2A.axhline(y=0.0, color='k', alpha=0.2, linewidth=1, linestyle='--')
ax2A.set_xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')
ax2A.set_ylim((-1.0, 1.0))
ax2A.set_yticks([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
ax2A.set_yticklabels([-100, -50, -10, 10, 50, 100])
ax2A.set_xticks([13, 13.5, 14, 14.5])

plt.tight_layout()
plt.savefig('hmf_TOTAL.png', dpi=300)
plt.show()

