import numpy as np
import scipy.fftpack as fp
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import gaussian

#image = misc.face(gray=True)  # carrega a imagem do guaxinim (racoon)
# ao invés disso, se quiser importar sua própria imagem, use:
import imageio
image = imageio.imread('output_images/saidasegmentada.png', pilmode='L')

fig = plt.figure(figsize=(8,8))

# IMAGEM ORIGINAL
ax_image = fig.add_subplot(221)
ax_image.set_title('Imagem original')
ax_image.imshow(image, cmap='gray')

# log(|FFT|)
ax_fft = fig.add_subplot(222)
ax_fft.set_title('log(|FFT|)')
F = fp.fft2(image)  # FFT completa
Fm = fp.fftshift(F)
Fm = np.absolute(Fm)
Fm = np.log10(Fm)
ax_fft.imshow(Fm, cmap='gray')

# EIXO PARA TRANSFORMADA FILTRADA
ax_gauss = fig.add_subplot(224)
ax_gauss.set_title('log(|FFT|) * G(u, v)')

# EIXO PARA IMAGEM DESFOCADA
ax_blur = fig.add_subplot(223)
ax_blur.set_title('Imagem filtrada')

# FUNÇÃO DE ATUALIZAÇÃO DOS GRÁFICOS da FFT
def blur(sigma):
    G = gaussian.gaussian(image, sigma)  # gaussiana
    FF = F * G  # apicando o filtro gaussiano
    FFm = np.absolute(FF)
    FFm = fp.fftshift(FFm)
    FFm = np.log10(FFm+1e-8)
    ax_gauss.imshow(FFm, cmap='gray')  # gráfico da transformada filtrada

    image_blurred = fp.ifft2(FF)  # transformada inversa
    image_blurred = np.absolute(image_blurred)
    ax_blur.imshow(image_blurred, cmap='gray')  # figura desfocada

# EIXO PARA O SLIDER
ax_sigma = plt.axes([.1, .01, .8, .025])
ax_sigma.set_title('Desfoque gaussiano')
sigma_slide = widgets.Slider(ax_sigma, valmin=.1, valmax=10, valinit=10, label=f'$\sigma$ : ')
sigma_slide.on_changed(blur)

# inicializando as figuras com sigma = 10
blur(10)

# mostrando a janela gráfica
plt.show()
