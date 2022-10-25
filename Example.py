#%% libraries
import numpy as np
import cv2 as cv
import pandas as pd
from unet import unet
from proceso import (imagenProceso, removerAreas, aumentoTam, cuadrarRect,
                     dimRec)


#%% Load Image example
from glob import glob
list_imgs = glob('./imgs/*.jpg')
img_path = list_imgs[np.random.choice(list(range(len(list_imgs))))]

img_array = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
#%%
#imagen = cv.imdecode(np.frombuffer(img, np.uint8), cv.IMREAD_GRAYSCALE)
## Convertir la lista en array
#img_array = np.array(imagen)
# Creación del modelo
modelo = unet()
# Cargar los pesos pre-entrenados del modelo
modelo.load_weights('models/pesosBalanceBlancos.h5')
# Procesar la imagen-array
img_process = imagenProceso(img_array)
# Pasar la imagen procesada a la etapa de inferencia
prediccion = modelo.predict(img_process)
# Limitar la predicción
aux = prediccion < 1.0
prediccion[aux] = 0
# Pasar de un tensor-imagen a una imagen que se pueda mostrar
prediccion = prediccion[0, :, :, 0]
# Eliminar areas pequeñas de la imagen
img_areas_remove = removerAreas(prediccion)
# Redondear los valores del preproces anterior
img_round = np.round(aumentoTam(img_areas_remove, img_array.shape))
# Calcular el rectángulo que encierra la predicción
mask_rectangle = cuadrarRect(img_round)
# cinfigurar el rectangulo como una imagen
final_image = dimRec(mask_rectangle, img_array) # nos interesa esta
# Multiplicar el rectángulo con la imagen original
# ee = np.multiply(mask_rectangle, img_array) / 255.0
# %%

