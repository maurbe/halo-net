import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib as mpl
def shiftedColorMap(cmap, min_val, max_val, name='gjvjgb'):
    '''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered.
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.'''
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


density = np.load('cic_fieldsT/cic_densityT.npy')
print density.dtype
"""
density32 = density.astype('float32')
np.save('dens32.npy', density32)
density16 = density.astype('float16')
np.save('dens16.npy', density16)
print np.mean(abs(density-density32))
print np.mean(abs(density-density16))
raise SystemExit
density = nd.gaussian_filter(density, sigma=3.0, mode='wrap')
delta   = (density/density.mean() - 1.0)
"""
"""
vx = np.load('cic_fieldsT/cic_vxT.npy')
vy = np.load('cic_fieldsT/cic_vyT.npy')
vz = np.load('cic_fieldsT/cic_vzT.npy')

dvx_dx = np.gradient(vx)[0]
dvy_dy = np.gradient(vy)[1]
dvz_dz = np.gradient(vz)[2]
divergence = dvx_dx + dvy_dy + dvz_dz
divergence = nd.gaussian_filter(divergence, sigma=0.0, mode='wrap')[50]
"""
dT = np.load('cic_fieldsT/distancemapT.npy')
dT_int64 = dT.astype('int64')
np.save('dT_int64.npy', dT_int64)
dT_int32 = dT.astype('int32')
np.save('dT_int32.npy', dT_int32)
dT_int16 = dT.astype('int16')
np.save('dT_int16.npy', dT_int16)
print dT.dtype

"""
plt.figure()
plt.imshow((delta), cmap=shiftedColorMap(mpl.cm.seismic, min_val=delta.min(), max_val=delta.max()))

plt.figure()
plt.imshow(dT, cmap='jet')

plt.show()
"""