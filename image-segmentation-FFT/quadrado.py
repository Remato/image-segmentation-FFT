import numpy as np
import scipy.fftpack as fp
import matplotlib.pyplot as plt

N = 2**10  # número total de pixels em cada direção da imagem
f = np.zeros((N, N))

nx, ny = 10, 5

# um quadrado de largura 2*nx e altura 2*ny centrado na imagem:
f[N//2 - ny : N//2 + ny, N//2 - nx : N//2 + nx] = 1

# mostrando a imagem;
colormap = 'gray'  # mapa de cores

fig = plt.figure(figsize=(12, 5))

ax_img = fig.add_subplot(131)
ax_img.set_title('Imagem original')
ax_img.imshow(f, cmap=colormap)  # cmap = "colormap"

# Calculando a FFT 2D

F = fp.fft2(f)
Fm = np.absolute(F)
Fm /= Fm.max()  # normalizando para obter amplitudes em [0, 1]

# mostrando a  |FFT|
ax_fft = fig.add_subplot(132)
ax_fft.set_title('|FFT|')
ax_fft.imshow(Fm, cmap=colormap)  # vmax é o valor máximo a ser plotado

# fazendo o shift para obter a |FFT| como deve ser
Fm = fp.fftshift(Fm)
ax_fftshift = fig.add_subplot(133)
ax_fftshift.set_title('FFT com shift')
ax_fftshift.imshow(Fm, cmap=colormap, vmax=.2)

plt.tight_layout()
plt.show()
