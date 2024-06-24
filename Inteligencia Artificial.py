import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Cargar y preprocesar datos
# El dataset CIFAR-10 contiene 60,000 imágenes de 10 clases diferentes
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar las imágenes a valores entre 0 y 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definir los nombres de las clases para facilitar la visualización
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Visualizar algunas imágenes de entrenamiento
# Se muestran 25 imágenes del conjunto de entrenamiento con sus respectivas etiquetas
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Definir el modelo de red neuronal convolucional
# La arquitectura del modelo consiste en varias capas convolucionales y de pooling
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compilar el modelo
# Se utiliza el optimizador Adam y la pérdida de entropía cruzada categórica
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenar el modelo
# El modelo se entrena durante 10 épocas, utilizando el conjunto de datos de entrenamiento y validación
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluar el modelo
# Se evalúa el rendimiento del modelo en el conjunto de datos de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Guardar el modelo entrenado en un archivo
model.save('cifar10_model.h5')

# Visualizar precisión y pérdida durante el entrenamiento
# Se generan gráficos para mostrar la evolución de la precisión y la pérdida en el conjunto de entrenamiento y validación
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1.5])
plt.legend(loc='upper right')

plt.show()
