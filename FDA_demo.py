import numpy as np
import sys
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc

# para nao resumir a saida
np.set_printoptions(threshold=sys.maxsize)

im_src = Image.open("input_images/source.png").convert('RGB')
im_trg = Image.open("output_images/target.png").convert('RGB')

# redimensiona a imagem para um padrao de 1024x512 px
im_src = im_src.resize( (1024,512), Image.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

# salva a imagem original no novo tamanho
im_src.save("input_images/source.png")

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))


src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))
scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('output_images/resultado.png')


# matriz de pixels da saida da imagem original
original = scipy.misc.imread('input_images/source.png', mode='F', flatten=False)

f = open('keys/original.txt', 'w')

for i in range(len(original)):
  for j in range(len(original[i])):
    f.write(str(original[i][j]))
    f.write('\n')
f.close()

# matriz de pixels da saida da imagem segmentada
saida = scipy.misc.imread('output_images/resultado.png', mode='F', flatten=False)

f = open('keys/modificada.txt', 'w')

for i in range(len(saida)):
  for j in range(len(saida[i])):
    f.write(str(saida[i][j]))
    f.write('\n')
f.close()


# for i in range(len(saida)):
#   for j in range(len(saida[i])):
#     print(saida[i][j], file='saida2.txt')
