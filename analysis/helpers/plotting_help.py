import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

""" -------------------------------------------------------------------------------------------------
    Purpose:    Set the plotting parameters.
    Comment:    unicode_minus=False is a workaround for a the missing minus sign in cmr10.
"""

plt.rc('text', usetex=True)
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = 'cmr10'

matplotlib.rcParams.update({'font.size': 10})
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'


""" -------------------------------------------------------------------------------------------------
    Purpose:    Class for mapping 0.0 to the midpoint of a (diverging) matplotlib colormap.
    Comment:    Author is Joe Kington @ http://chris35wills.github.io/matplotlib_diverging_colorbar/
"""

class MidpointNormalize(matplotlib.colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


""" -------------------------------------------------------------------------------------------------
    Purpose:    Adjust the colorbar to what is needed.
    Comment:    Author is Joseph Long @ https://joseph-long.com/writing/colorbars/
"""

def colorbar(mappable, colorbar_label=None):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, label=colorbar_label)