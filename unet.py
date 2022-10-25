from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Metrica de evalaución
smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Función de costo
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet():
    ## Etapa de contracción
    entrada = Input((180, 320, 1))

    # Bloque 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(entrada)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=2, padding='same')(conv1)
    drop1 = Dropout(0.2)(pool1)
    # Bloque 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=2, padding='same')(conv2)
    drop2 = Dropout(0.2)(pool2)
    # Bloque 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(5, 2), padding='same')(conv3)
    drop3 = Dropout(0.2)(pool3)
    # Bloque 4
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(3, 2), padding='same')(conv4)
    drop4 = Dropout(0.2)(pool4)

    ## Enlace o puente
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    ## Etapa de expansión
    # Bloque 5
    exp1 = Conv2DTranspose(256, (2, 2), strides=(3, 2), padding='same',
                           activation='relu')(conv5)
    cont1 = concatenate([exp1, conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(cont1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    # Bloque 6
    exp2 = Conv2DTranspose(128, (2, 2), strides=(5, 2), padding='same',
                           activation='relu')(conv6)
    cont2 = concatenate([exp2, conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(cont2)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    # Bloque 7
    exp3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                           activation='relu')(conv7)
    cont3 = concatenate([exp3, conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(cont3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    # Bloque 8
    exp4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',
                           activation='relu')(conv8)
    cont4 = concatenate([exp4, conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(cont4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    ## Etapa de organización. Practicamente no cambia y es obligatoria en casi
    ## todas las redes FCN y por supuesto va siempre al final
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    ## Juntamos el modelo. Le estamos diciendo desde donde inicia el modelo
    ## y hasta donde va.
    model = Model(inputs=[entrada], outputs=[conv10])

    ## Compilamos el modelo
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=dice_coef_loss,
                  metrics=[dice_coef])

    # model.compile(optimizer='adam',
    #              loss='binary_crossentropy',
    #              metrics=['accuracy'])

    return model