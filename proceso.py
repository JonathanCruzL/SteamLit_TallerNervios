import numpy as np
import cv2 as cv


# Método Balance de Blancos

def Limitar(im):
    Mayores = im > 255
    im[Mayores] = 255
    Menores = im < 0
    im[Menores] = 0
    return im


def imadjust(imagen, low_in, hig_in, low_out, hig_out, gamma):
    imagen_mod = np.zeros(imagen.shape, dtype=np.uint8)

    imagen_mod = low_out + (hig_out - low_out) * ((imagen - low_in) / (hig_in - low_in)) ** gamma

    imagen_mod = np.round(255 * imagen_mod)

    imagen_mod = Limitar(imagen_mod)

    imagen_mod = np.uint8(imagen_mod)

    return imagen_mod


def white_balance(Imagen):
    Canal = Imagen.astype('float') / 255.0

    Min_Canal = np.percentile(Canal, 1)
    Max_Canal = np.percentile(Canal, 99)

    canal_balance = imadjust(Canal, Min_Canal, Max_Canal, 0.0, 1.0, 1)

    return canal_balance


def manipulacionDatos_prediccion(imagen):
    print('-' * 60)
    print('Loading and preprocessing train data...')
    print('-' * 60)

    imagen = imagen.reshape((1,
                             imagen.shape[0],
                             imagen.shape[1], 1))

    imagen = imagen / 255.

    return imagen


def imagenProceso(imagen):
    balance = white_balance(imagen)
    resized_imagen = cv.resize(balance, (320, 180), interpolation=cv.INTER_AREA)
    resized_imagen = np.array(resized_imagen)
    imagenTensor = manipulacionDatos_prediccion(resized_imagen)
    return imagenTensor


def aumentoTam(imagen, tamNuevo):
    resized_imagen = cv.resize(imagen,
                               (tamNuevo[1], tamNuevo[0]),
                               interpolation=cv.INTER_AREA)
    # resized_imagen = Limitar(resized_imagen)
    return resized_imagen


def removerAreas(imagen, min_size=500):
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(np.uint8(imagen))

    sizes = stats[:, -1]

    sizes = sizes[1:]

    nb_blobs -= 1

    im_result = np.zeros(imagen.shape)

    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 1.0
    return im_result


def cuadrarRect(mascara):
    img = np.array(mascara, dtype=np.uint8)

    imgRGB = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)

    a, ctrs = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # print(a)
    boxes = []
    for ctr in a:
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append([x, y, w, h])

    for box in boxes:
        top_left = (box[0], box[1])
        bottom_right = (box[0] + box[2], box[1] + box[3])
        # para que quede solo el recuadro, cambiar el -1 por un 2
        cv.rectangle(imgRGB, top_left, bottom_right, (255, 255, 255), -1)

    return imgRGB[:, :, 0] / 255.0


def dimRec(mascara, imagen):
    cnts, _ = cv.findContours(np.uint8(mascara), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = imagen[cnts[0][0][0][1]: cnts[0][1][0][1], cnts[0][0][0][0]: cnts[0][2][0][0]]
    return mask
