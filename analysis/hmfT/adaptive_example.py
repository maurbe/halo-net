import numpy as np, os
from analysis.helpers.plotting_help import *
font = {'size'   : 15}
matplotlib.rc('font', **font)
from matplotlib.colors import LinearSegmentedColormap
mc = LinearSegmentedColormap.from_list('mycmap', ['#E1D8E1', '#757575', '#424242'], N=3)

import matplotlib.cm as cm
viridis = cm.get_cmap('magma')
values = [viridis(x) for x in np.linspace(0, 1, 100)]
#values[0] = (1, 1, 1, 1)    # set the first value to white
last = values[-1]
for x in range(50):
    values.append(last)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('mycmap', values)

projectdir = os.path.dirname(os.getcwd()) + '/'

"""
# ................
y1, y2, x1, x2 = 100, 135, 45, 75
labels_truth = np.load('/Users/Mauro/Desktop/Biotop2/source/cic_fieldsT/distancemap_rawT.npy')[0, y1:y2, x1:x2]
labels_truth = labels_truth>1

labels = np.load('overplot_src/predicted_labels_for_overplot.npy')[64:-64, 64:-64, 64:-64][0, y1:y2, x1:x2]
plt.figure()
plt.imshow(labels)
plt.contour(labels_truth, levels=[0], colors=['red'])

preds  = np.load(projectdir + 'boxesT/predictionT.npy')[0, y1:y2, x1:x2]
plt.figure()
plt.imshow(preds)
plt.show()

"""
y1, y2, x1, x2 = 100, 127, 43, 70
labels = np.load('overplot_src/predicted_labels_for_overplot.npy')[64:-64, 64:-64, 64:-64][0, y1:y2, x1:x2]
un = np.unique(labels)
labels[labels==un[1]] = 2
labels[labels==un[2]] = 1
preds  = np.load(projectdir + 'boxesT/predictionT.npy')[0, y1:y2, x1:x2]    # 0, 250
new_labels = np.where(preds<0.1, 0, labels)

labels_truth = np.load('/Users/Mauro/Desktop/Biotop2/source/cic_fieldsT/distancemap_rawT.npy')[0, y1:y2, x1:x2]
labels_truth = np.flipud(labels_truth>1.5)

f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
ax0.imshow(preds, cmap=cm, clim=(0, 1), extent=(x1, x2, y2, y1))
ax0.contour(labels_truth, levels=[0], colors=['green'], extent=(x1, x2, y2, y1))
ax1.imshow(labels, cmap=mc, extent=(x1, x2, y2, y1))
ax1.contour(labels_truth, levels=[0], colors=['green'], extent=(x1, x2, y2, y1))
ax2.imshow(new_labels, cmap=mc, extent=(x1, x2, y2, y1))
ax2.contour(labels_truth, levels=[0], colors=['green'], extent=(x1, x2, y2, y1))

ax0.annotate('problematic region', xy=(55, 115), xycoords='data', fontsize=12, color='white',
                xytext=(46, 105), textcoords='data',
                arrowprops=dict(arrowstyle='-|>, head_width=0.15, head_length=0.3',
                                color="white", connectionstyle="angle,angleA=180,angleB=-90,rad=0"))
ax0.set(yticks=[y1, 118, y2], xticks=[x1, 60, x2], xlabel='x [cells]', ylabel='y [cells]')
ax1.set(yticks=[y1, 118, y2], xticks=[x1, 60, x2], xlabel='x [cells]')
ax2.set(yticks=[y1, 118, y2], xticks=[x1, 60, x2], xlabel='x [cells]')
plt.savefig('adaptive_example.png', dpi=250)
plt.show()

