import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import ttk

train_dir = "train/"
validation_dir = "validation/"


def entrenar_modelo():
    
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

   
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

    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    
    barra_progreso = ttk.Progressbar(root, length=400, mode="determinate")
    barra_progreso.pack(pady=10)

    
    def actualizar_barra_progreso(epoch, logs):
        porcentaje_progreso = (epoch + 1) / 15 * 100
        barra_progreso["value"] = porcentaje_progreso
        root.update_idletasks()

    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=actualizar_barra_progreso)]
    )

    
    model.save("modelo_entrenado.h5")
    print("Modelo guardado correctamente.")


def cargar_y_preprocesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def detectar_expresion(ruta_imagen):
    
    model = tf.keras.models.load_model('modelo_entrenado.h5')

   
    imagen_nueva = cargar_y_preprocesar_imagen(ruta_imagen)

    
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

   
    etiqueta_emocion.config(text=f"Emoción predicha: {emocion_predicha}")


def ejecutar_entrenador():
    entrenar_modelo()
    print("Modelo entrenado correctamente.")

def seleccionar_imagen():
    ruta_imagen = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")])
    if ruta_imagen:
        
        img = Image.open(ruta_imagen)
        img = img.resize((200, 200), Image.LANCZOS)  
        img = ImageTk.PhotoImage(img)
        vista_previa.config(image=img)
        vista_previa.image = img

        
        detectar_expresion(ruta_imagen)


root = tk.Tk()
root.title("Detector de Expresión Facial")


ancho_ventana = 800
alto_ventana = 600
posicion_x = (root.winfo_screenwidth() // 2) - (ancho_ventana // 2)
posicion_y = (root.winfo_screenheight() // 2) - (alto_ventana // 2)
root.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")


btn_entrenar = tk.Button(root, text="Entrenar Modelo", command=ejecutar_entrenador)
btn_entrenar.pack(pady=10)


barra_progreso = ttk.Progressbar(root, length=400, mode="determinate")
barra_progreso.pack(pady=10)


btn_seleccionar = tk.Button(root, text="Seleccionar Imagen", command=seleccionar_imagen)
btn_seleccionar.pack(pady=10)


vista_previa = tk.Label(root)
vista_previa.pack(pady=10)


etiqueta_emocion = tk.Label(root, text="Emoción predicha: ")
etiqueta_emocion.pack(pady=10)


root.mainloop()
