import matplotlib.pyplot as plt
import numpy as np
import cv2
import pathlib
import os
import PIL
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = pathlib.Path('./datasets/flower_photos/flower_photos')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f'Total images: {image_count}')
roses = list(data_dir.glob('roses/*'))
rs = PIL.Image.open(str(roses[1]))
rs.show()
rs.show()
tulips = list(data_dir.glob('tulips/*'))
tp=PIL.Image.open(str(tulips[0]))
tp.show()
flowers_images_dict = {
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'roses': list(data_dir.glob('roses/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}   
flowers_labels_dict = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4,
}
img= cv2.imread(str(flowers_images_dict['daisy'][0]))
cv2.resize(img, (180, 180)).shape
X,y=[],[]
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        img = cv2.resize(img, (180, 180))
        X.append(img)
        y.append(flowers_labels_dict[flower_name])
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
num_classes = 5
model = Sequential([
    layers.Input(shape=(180, 180, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10)
model.evaluate(X_test_scaled, y_test)
# Overfitting mitigation through data augmentation
predctions = model.predict(X_test_scaled)
print(predctions[:5])
score = tf.nn.softmax(predctions[0])
data_augmentation = keras.Sequential([
    layers.RandomZoom(0.1),
])
plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))
plt.show()
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])
model = Sequential([
    layers.Input(shape=(180, 180, 3)),
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.evaluate(X_test_scaled, y_test)
