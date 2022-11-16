import streamlit as st

from PIL import Image,ImageOps
import numpy as np
import cv2 as cv
import pandas as pd
from unet import unet
from proceso import (imagenProceso, removerAreas, aumentoTam, cuadrarRect,
                     dimRec)
from model_seg_unet import (upsample_simple, upsample_conv, create_model)


def saludo():
    # Título de la App
    st.header("Anest App")
    # Descripción del aplicativo
    texto = """ Esta aplicación permite extraer el recuadro de ultrasonido y, luego, 
    genera la segmentación automática del nervio en dicha imagen.  
    """
    st.write(texto)


def camara():
    # Cargar imagen o tomar foto
    uploaded_file = st.file_uploader("Cargar o Tomar la Foto")

    if uploaded_file is not None:
        # Extraer la imagen en formato Bytes
        st.image(uploaded_file.getvalue())
        
        # Decodificar la imagen para ser  leida como una lista
        imagen = cv.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8) , cv.IMREAD_GRAYSCALE)
        
        # Convertir la lista en array
        img_array = np.array(imagen)
        img_color =  cv.cvtColor(img_array,cv.COLOR_GRAY2RGB)
        
        # Creación del modelo para extraer US
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
        # configurar el rectangulo como una imagen
        final_image = dimRec(mask_rectangle, img_array)
        # Multiplicar el rectángulo con la imagen original
        ee = np.multiply(mask_rectangle, img_array) / 255.0
        # Mostrar la imagen
        st.image(ee)
        st.subheader("Imagen a descargar o compartir")
        # Mostrar el resultado Final
        st.image(final_image)
        
        # =======================================================================
        
        # Se usa otro modelo para segmentar nervios
        modelo_seg = create_model()
        # Cargar los pesos pre-entrenados del modelo
        modelo_seg.load_weights('models/model_seg_w/model_Unet_wei.h5fd')
        
        img2pred = cv.resize(np.array(final_image), (256,256))
        img2pred = img2pred[np.newaxis,...,np.newaxis]
        img2pred = img2pred/255
        
        st.write("img2predi "+str(type(img2pred)))
        st.write("img2predi "+str(np.shape(img2pred)))
        
        st.write("img2predi Unique "+str(np.unique(img2pred)))
        st.write("img2predi max "+str(np.max(img2pred)))
        st.write("img2predi min "+str(np.min(img2pred)))
        
        mask_est = modelo_seg.predict(img2pred)
        st.subheader("Segmentación.")
        st.image(mask_est)

        
    
        

        


saludo()
camara()
