import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Generate some data...
gray_data = np.arange(10000).reshape(100, 100)

masked_data = np.random.random((100,100))
masked_data = np.ma.masked_where(masked_data < 0.9, masked_data)

# Overlay the two images
fig, ax = plt.subplots()
#ax.imshow(gray_data, cmap=cm.gray)
ax.imshow(masked_data, cmap=cm.jet, interpolation='none')
plt.show()