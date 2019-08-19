import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

plt.rc('text', usetex=True)
plt.rcParams['axes.unicode_minus']=False    # works and adds minus signs!!!!!!
plt.rcParams["font.family"] = "cmr10"     # Change globally!!!

matplotlib.rcParams.update({'font.size': 10})
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='medium')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'