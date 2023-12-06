import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

model = tf.keras.models.load_model('modelo_entrenado.h5') 
def cargar_y_preprocesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)  #
    img = cv2.resize(img, (48, 48))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=(0, -1))  
    return img
ruta_nueva_imagen = 'imagen2.jpeg' 
imagen_nueva = cargar_y_preprocesar_imagen(ruta_nueva_imagen)
prediccion = model.predict(imagen_nueva)
clase_predicha = np.argmax(prediccion)
correspondencia_emociones = {
    0: "Enojo",
    1: "Asco",
    2: "Miedo",
    3: "Feliz",
    4: "Triste",
    5: "Sorpresa",
    6: "Neutral"
}
emocion_predicha = correspondencia_emociones[clase_predicha]
print(f"La expresión facial predicha es: {emocion_predicha}")
