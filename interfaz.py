import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir rutas de los datos
train_dir = "train/"
validation_dir = "validation/"

# Funciones del entrenador
def entrenar_modelo():
    # Configurar generadores de imágenes
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Configurar generadores de datos
    batch_size = 1
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    print("Número de muestras de entrenamiento:", train_generator.n)
    print("Número de pasos por época:", train_generator.n // train_generator.batch_size)

    # Crear la arquitectura del modelo
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(7, activation="softmax"),  # Capa de salida para 7 emociones
        ]
    )

    # Compilar el modelo
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Entrenar el modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size
    )

    # Guardar el modelo entrenado
    model.save("modelo_entrenado.h5")
    print("Modelo guardado correctamente.")

# Funciones del clasificador
def cargar_y_preprocesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def detectar_expresion(ruta_imagen):
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('modelo_entrenado.h5')

    # Cargar y preprocesar la nueva imagen
    imagen_nueva = cargar_y_preprocesar_imagen(ruta_imagen)

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

# Funciones para la interfaz gráfica
def ejecutar_entrenador():
    entrenar_modelo()
    print("Modelo entrenado correctamente.")

def seleccionar_imagen():
    ruta_imagen = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")])
    if ruta_imagen:
        # Mostrar vista previa de la imagen
        img = Image.open(ruta_imagen)
        img = img.resize((200, 200), Image.LANCZOS)  # Cambiado a LANCZOS
        img = ImageTk.PhotoImage(img)
        vista_previa.config(image=img)
        vista_previa.image = img

        # Detectar expresión facial
        detectar_expresion(ruta_imagen)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Detector de Expresión Facial")

# Botón para ejecutar el entrenador
btn_entrenar = tk.Button(root, text="Entrenar Modelo", command=ejecutar_entrenador)
btn_entrenar.pack(pady=10)

# Botón para seleccionar imagen
btn_seleccionar = tk.Button(root, text="Seleccionar Imagen", command=seleccionar_imagen)
btn_seleccionar.pack(pady=10)

# Vista previa de la imagen
vista_previa = tk.Label(root)
vista_previa.pack(pady=10)

# Ejecutar la interfaz
root.mainloop()
