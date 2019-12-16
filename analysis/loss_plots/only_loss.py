"""
    Purpose:    Display loss (and other metrics) change over training time.
    Comment:    The selective_r2 has not been normalized in any way.
"""

from analysis.helpers.plotting_help import *
from scipy.signal import savgol_filter
plt.style.use('double_fig.mplstyle')
#plt.rc('xtick', labelsize='large')
#plt.rc('ytick', labelsize='large')

loss_data = np.loadtxt('loss.csv', skiprows=1, delimiter=',')
step = loss_data[:, 1]
loss = loss_data[:, 2]
loss_smoothed = savgol_filter(loss, window_length=31, polyorder=3)

val_loss_data = np.loadtxt('val_loss.csv', skiprows=1, delimiter=',')
val_step = val_loss_data[:, 1]
val_loss = val_loss_data[:, 2]
val_loss_smoothed = savgol_filter(val_loss, window_length=31, polyorder=3)

selective_r2 = np.loadtxt('selective_r2_score.csv', skiprows=1, delimiter=',')
r2_step = selective_r2[:, 1]
r2 = selective_r2[:, 2]
r2_smoothed = savgol_filter(r2, window_length=31, polyorder=3)

val_selective_r2 = np.loadtxt('val_selective_r2_score.csv', skiprows=1, delimiter=',')
val_r2_step = val_selective_r2[:, 1]
val_r2 = val_selective_r2[:, 2]
val_r2_smoothed = savgol_filter(val_r2, window_length=31, polyorder=3)



fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(step, loss, '-', color='#1565C0', alpha=0.2, linewidth=2)
ax1.plot(step, loss_smoothed, '-', color='#1565C0', label='training loss', linewidth=2)
ax1.plot(val_step, val_loss, '-', color='darkorange', alpha=0.2, linewidth=2)
ax1.plot(val_step, val_loss_smoothed, '-', color='darkorange', label='validation loss', linewidth=2)

ax1.annotate('stop training', xy=(2060, 0.5), xycoords='data', fontsize=12,
                xytext=(1600, 0.6), textcoords='data',
                arrowprops=dict(arrowstyle='-|>, head_width=0.15, head_length=0.3', #linestyle="dashed",
                                color="k", connectionstyle="angle,angleA=180,angleB=-90,rad=0"))
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1.set_ylabel(r'$\hat{L}_{1,\mathrm{sel}}$', fontsize=15)
ax1.legend(loc='upper right', prop={'size': 15}, frameon=False)
ax1.set_xlabel(r'Iterations', fontsize=15)

plt.savefig('loss_evolution.png', dpi=250)
plt.show()