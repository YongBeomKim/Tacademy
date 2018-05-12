import matplotlib.pyplot as plt
import numpy as np
a=[[1,1,1,1,1,1,1,1,1,1,1,1,0.1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,0.2,1,1,1,0.3,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0.1,1,1,1,0.3,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,1,1,1,2,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
a=np.array(a)
print(a.shape)

dx, dy = 0.05, 0.05
x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)
xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
extent = xmin, xmax, ymin, ymax
fig = plt.figure(frameon=False)
im1 = plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest',
                             extent=extent)
plt.show()
