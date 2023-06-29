import cv2
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def colorFilterArray (data):
    shape = data.shape
    img = np.zeros([(shape[0]), (shape[1]), 3], dtype=np.double)
    
    for i in range(0, shape[0], 2):
        for j in range(0, shape[1], 2):
            img [i+1, j+1, 0]  = np.double( data[i+1, j+1])    #B
            img [i+1, j, 1]     =  np.double(data[i+1, j]) #G
            img [i, j+1, 1] =  np.double(data[i, j+1])     #G
            img [i, j, 2]   =  np.double(data[i, j])   #R /65535.0 valor utilizado para normalização entre 0 e 1

    img = img*(1/65535.0)
    return img

def bilinearDemosaicing(img, k, w):
    imgFinal = img.copy()
    shape = img.shape

    for i in range(1, shape[0]-2):
        for j in range(1, shape[1]-2):
            if (img[i, j, 1] == 0):
                imgFinal[i, j, 1] = (img[i, j+1, 1] + img[i, j-1, 1] + img[i+1, j, 1] + img[i-1, j, 1])*k

            if (img[i, j, 2] == 0):
                if (img[i, j+1, 2] != 0 or img[i, j-1, 2] != 0):
                    imgFinal[i, j, 2] = (img[i, j+1, 2] + img[i, j-1, 2])*w
                elif(img[i+1, j, 2] != 0 or img[i-1, j, 2] != 0):
                    imgFinal[i, j, 2] = (img[i+1, j, 2] + img[i-1, j, 2])*w
                else:
                    imgFinal[i, j, 2] = (img[i-1, j+1, 2] + img[i+1, j-1, 2] + img[i-1, j-1, 2] + img[i+1, j+1, 2])*k
		    
            if (img[i, j, 0] == 0):
                if (img[i, j-1, 0] != 0 or img[i, j+1, 0] != 0):
                    imgFinal[i, j, 0] = (img[i, j+1, 0] + img[i, j-1, 0])*w
                elif(img[i+1, j, 0] != 0 or img[i-1, j, 2] != 0):
                    imgFinal[i, j, 0] = (img[i+1, j, 0] + img[i-1, j, 0])*w
                else:
                    imgFinal[i, j, 0] = (img[i-1, j+1, 0] + img[i+1, j-1, 0] + img[i-1, j-1, 0] + img[i+1, j+1, 0])*k
    return imgFinal[2:shape[0]-2, 2:shape[1]-2, :]

def balance_channel(channel, cutoff):
    low = np.percentile(channel, cutoff)
    high = np.percentile(channel, 100 - cutoff)
    new_channel = np.uint8(np.clip((channel-low)*255.0/(high - low), 0, 255))
    return new_channel

def automaticWhiteBalance(img, cutoff):
    b = balance_channel(img[:,:,0], cutoff)
    g = balance_channel(img[:,:,1], cutoff)
    r = balance_channel(img[:,:,2], cutoff)
    
    return cv2.merge((b,g,r))

def denoise(image, kernel_size):
   # Aplica o filtro de média
    denoised_image = cv2.blur(image, (kernel_size, kernel_size))
    return denoised_image

def calculate_ssim(image1, image2):
  if image1.shape != image2.shape:
    raise ValueError("As imagens devem ter o mesmo tamanho.")
    
  # calcular o SSIM
  ssim_value = ssim(image1, image2, data_range=image2.max() - image2.min(), multichannel=True)
    
  return ssim_value

