#!/bin/bash

python3.6 segmentation.py input_images/source.png output_images/saidasegmentada.png
python3.6 FFT-IMAGEM.py 

echo "Atividade Concluída com Sucesso"


