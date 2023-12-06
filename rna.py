import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
train_dir = "train/"
validation_dir = "validation/"

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


batch_size = (
    1  
)
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

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

model.save("modelo_entrenado.h5")
print("Modelo guardado correctamente.")
