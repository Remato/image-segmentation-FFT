import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.fftpack as fp
import gaussian

sigma = 1.
vmax = .05
Nu, Nv = 100, 100
u = Nu * np.linspace(-vmax, vmax, Nu)
v = Nv * np.linspace(-vmax, vmax, Nv)
U, V = np.meshgrid(v, u)
sigma2 = sigma**2
G = np.exp(-(U*U + V*V) / 2. / sigma2)
G /= sigma2

# Gráfico da gaussiana 2d
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')


ax.set_xticks([])
ax.set_yticks([])
# ax.set_zticks([])
ax.set_xlabel(r'$u$')
ax.set_ylabel(r'$v$')
ax.set_zlabel(r'$G(u,v)$')

ax.plot_surface(U, V, G, cmap='gray', linewidth=0, antialiased=False)
plt.savefig('gaussian-plot-bw.png', bbox_inches='tight')

ax.plot_surface(U, V, G, cmap='plasma', linewidth=0, antialiased=False)
plt.savefig('gaussian-plot-color.png', bbox_inches='tight')

plt.figure()
plt.xticks([])
plt.yticks([])
plt.imshow(G, cmap='gray', interpolation='lanczos')
plt.savefig('gaussian-plot-2d-bw.png', bbox_inches='tight')
plt.imshow(G, cmap='plasma', interpolation='lanczos')
plt.savefig('gaussian-plot-2d-color.png', bbox_inches='tight')
plt.tight_layout()

plt.show()
