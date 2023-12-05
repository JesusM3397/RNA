import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_entrenado.h5')  # Reemplaza con la ruta correcta

# Función para cargar y preprocesar una imagen nueva
def cargar_y_preprocesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)  # Leer la imagen en escala de grises
    img = cv2.resize(img, (48, 48))  # Redimensionar la imagen a 48x48 píxeles
    img = img / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
    img = np.expand_dims(img, axis=(0, -1))  # Agregar dimensiones para adaptarse al formato del modelo
    return img

# Ruta de la nueva imagen que deseas clasificar
ruta_nueva_imagen = 'imagen6.jpeg'  # Reemplaza con la ruta correcta

# Cargar y preprocesar la nueva imagen
imagen_nueva = cargar_y_preprocesar_imagen(ruta_nueva_imagen)

# Hacer la predicción
prediccion = model.predict(imagen_nueva)

# Decodificar la salida para obtener la clase predicha
clase_predicha = np.argmax(prediccion)

# Definir la correspondencia de clases con emociones
correspondencia_emociones = {
    0: "Enojo",
    1: "Asco",
    2: "Miedo",
    3: "Feliz",
    4: "Triste",
    5: "Sorpresa",
    6: "Neutral"
}

# Imprimir la clase predicha
emocion_predicha = correspondencia_emociones[clase_predicha]
print(f"La expresión facial predicha es: {emocion_predicha}")
