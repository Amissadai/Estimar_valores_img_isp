from function_correction import bilinearDemosaicing
from function_correction import automaticWhiteBalance
from function_correction import adjust_gamma
from function_correction import denoise
from function_correction import calculate_ssim
from function_correction import colorFilterArray
import matplotlib.pyplot as plt
import cv2
import os


data_folder = 'data'
canon_folder = os.path.join(data_folder, 'canon')
raw_folder = os.path.join(data_folder, 'raw')

img_raw = []
img_ref = []

# Lista todos os arquivos na pasta "canon"
canon_files = os.listdir(canon_folder)

# Lista todos os arquivos na pasta "raw"
raw_files = os.listdir(raw_folder)

# Ler e processar as imagens na pasta "canon"
for file_name in canon_files:
    file_path_canon = os.path.join(canon_folder, file_name)
    image_ref = plt.imread(file_path_canon)
    img_ref.append((file_name, image_ref))  # Save the image name along with the image

# Ler e processar as imagens na pasta "raw"
for file_name in raw_files:
    file_path = os.path.join(raw_folder, file_name)
    image_raw = plt.imread(file_path)
    image_raw = colorFilterArray(image_raw)
    img_raw.append((file_name, image_raw))  # Save the image name along with the image

k = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
w = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
wb = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3,1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
kernel_size = [3, 5, 7, 9, 11, 13, 15]

results = []


# Iterar sobre as imagens raw e as imagens de referência correspondentes
for i, img in enumerate(img_raw):
    img_raw_data = img[1]  # Dados da imagem raw
    ref_img_data = img_ref[i][1]  # Dados da imagem de referência correspondente
    
    max_ssim = 0.0
    max_ssim_params = {}
    max_ssim_image_name = ""

    for j in k:
        print('Valor de k: ', j)
        for x in w:
            print('Valor de w: ', x)
            for z in wb:
                for y in gamma:
                    for d in kernel_size:
                        imgFinal = bilinearDemosaicing(img_raw_data, j, x)
                        imgFinalWhiteBalance = automaticWhiteBalance(imgFinal, z)
                        img_gamma = adjust_gamma(imgFinalWhiteBalance, y)
                        img_denoise = denoise(img_gamma, d)
                        img_resul = cv2.resize(img_denoise, (448, 448))
                        img_resul = cv2.cvtColor(img_resul, cv2.COLOR_BGR2RGB)

                        # Calcular o SSIM apenas entre a imagem raw e sua respectiva imagem de referência
                        ssim_score = calculate_ssim(ref_img_data, img_resul)
                        print("SSIM Score:", ssim_score)

                        # Salvar o máximo SSIM e os parâmetros correspondentes
                        if ssim_score > max_ssim:
                            max_ssim = ssim_score
                            max_ssim_params = {
                                'k': j,
                                'w': x,
                                'wb': z,
                                'gamma': y,
                                'kernel_size': d
                            }
                            max_ssim_image_name = img[0]  # Nome da imagem
    # Armazenar os resultados da imagem atual na lista
    results.append({
          'max_ssim': max_ssim,
          'max_ssim_params': max_ssim_params,
          'max_ssim_image_name': max_ssim_image_name
    })

# Salvar o resultado em um arquivo de texto
with open('resultado_ssim.txt', 'w') as file:
    for result in results:
        file.write(f"Nome: {result['max_ssim_image_name']}, Max SSIM: {result['max_ssim']}, Parameters: {result['max_ssim_params']}\n")