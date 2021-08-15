import numpy as np
import scipy.fftpack as fp


def gaussian(S, sigma=1, vmax=.05):
    '''
        Retorna uma gaussiana de média zero e
        desvio padrão sigma
        G(u, v) = 1/(2 pi sigma^2) exp(-(u^2+v^2)/(2 sigma^2))
        no intervalo [-vmax, vmax]x[-vmax, vmax]
        deslocada com fftshift
        O array retornado terá tem a mesma dimensão da imagem (S)
    '''

    Nu, Nv = S.shape
    u = Nu * np.linspace(-vmax, vmax, Nu)
    v = Nv * np.linspace(-vmax, vmax, Nv)
    U, V = np.meshgrid(v, u)

    sigma2 = sigma**2
    G = np.exp(-(U*U + V*V) / 2. / sigma2)
    G = fp.fftshift(G)  # faz o deslocamento

    return G / sigma2
