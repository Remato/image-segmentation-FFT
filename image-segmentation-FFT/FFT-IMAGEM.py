import numpy as np
import scipy.fftpack as fp
import imageio
import matplotlib.pyplot as plt

f = imageio.imread('output_images/saidasegmentada.png', pilmode='L')

# mostrando a imagem
colormap = 'gray'  # mapa de cores

fig = plt.figure(figsize=(12, 5))

ax_img = fig.add_subplot(131)
ax_img.set_title('Imagem original')
ax_img.imshow(f, cmap=colormap)

# Calculando a FFT 2D
F = fp.fft2(f)
Fm = np.absolute(F)
Fm /= Fm.max()  # normalizando para obter amplitudes em [0, 1]
Fm = np.log(Fm)  # logaritmo para conseguir enxargar a FFT!

# mostrando a  |FFT|
ax_fft = fig.add_subplot(132)
ax_fft.set_title('log |FFT|')
ax_fft.imshow(Fm, cmap=colormap)  # vmax é o valor máximo a ser plotado

# fazendo o shift para obter a |FFT| como deve ser
Fm = fp.fftshift(Fm)
ax_fftshift = fig.add_subplot(133)
ax_fftshift.set_title('FFT com shift')
ax_fftshift.imshow(Fm, cmap=colormap, vmax=.2)

plt.tight_layout()
plt.show()
