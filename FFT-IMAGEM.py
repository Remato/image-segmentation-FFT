from imageio.core.functions import imwrite
import numpy as np
import cv2
import scipy.fftpack as fp
import imageio
import matplotlib.pyplot as plt

f = imageio.imread('output_images/saidasegmentada.png', pilmode='L')

# mostrando a imagem
colormap = 'gray'  # mapa de cores

fig = plt.figure(figsize=(12, 5))


ax_img = fig.add_subplot(131)
ax_img.set_title('Imagem Segmentada')
ax_img.imshow(f, cmap=colormap)


# Calculando a FFT 2D
F = fp.fft2(f)
Fm = np.absolute(F)
Fm /= Fm.max()  # normalizando para obter amplitudes em [0, 1]
Fm = np.log(Fm)  # logaritmo para conseguir enxergar a FFT!

# mostrando a  |FFT|
ax_fft = fig.add_subplot(132)
ax_fft.set_title('log |FFT|')
ax_fft.imshow(Fm, cmap=colormap)  # vmax é o valor máximo a ser plotado


# fazendo o shift para obter a |FFT| como deve ser
Fs = fp.fftshift(Fm)
ax_fftshift = fig.add_subplot(133)
ax_fftshift.set_title('FFT com shift')
ax_fftshift.imshow(Fs, cmap=colormap, vmax=.2)


plt.tight_layout()
plt.show()

# voltando para  a imagem
convolve = F*Fs #chave de descriptografia
im_out = fp.ifft2(convolve).real
imageio.imwrite('output_images/inversa.png', im_out)

inversa = cv2.imread('output_images/inversa.png')
segmentada = cv2.imread('output_images/saidasegmentada.png')

histInversa = cv2.calcHist([inversa], [0, 1], None, [8, 8], [0, 256, 0, 256])
histInversa = cv2.normalize(histInversa, histInversa).flatten()

histSegmentada = cv2.calcHist([segmentada], [0, 1], None, [8, 8], [0, 256, 0, 256])
histSegmentada = cv2.normalize(histSegmentada, histSegmentada).flatten()

d = cv2.compareHist(histInversa, histSegmentada, cv2.HISTCMP_BHATTACHARYYA)

# plt.hist(histInversa)
# plt.hist(histSegmentada)
# plt.show()

print(d)





