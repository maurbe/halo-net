"""
    Script to inspect dataset if there are (for whatever reason) negatives, NaNs or values larger than 1.
"""
import numpy as np

for i in range(512):
    # print i
    x = np.load('../data/training/x_{}.npy'.format(i))
    y = np.load('../data/training/y_{}.npy'.format(i))

    if np.any(np.isnan(x)):
        print('found NAN in x')
    if np.any(np.isnan(y)):
        print('found NAN in y')

    assert np.alltrue(np.logical_and(x<50, x>-50))
    assert np.alltrue(np.logical_and(y<100, y>-100))

    print(x.dtype)
    print(y.dtype)
