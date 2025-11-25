import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
def plot_sample(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[index])
    plt.show()
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plot_sample(x_train,y_train,0)
y_train = y_train.reshape(-1,)
# Normalize pixel values and convert labels to categorical
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# Build the ANN model
ann=models.Sequential(
    [
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
ann.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, epochs=5)
y_pred = ann.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
# Evaluate the ANN model
print("ANN Classification Report:\n", classification_report(np.argmax(y_test, axis=1), y_pred_classes))
print("ANN Confusion Matrix:\n", confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes))
# Build the CNN model
cnn = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train, y_train, epochs=5) 
# Evaluate the CNN model
y_pred_cnn = cnn.evaluate(x_test,y_test)    
y_test=y_test.reshape(-1,)
plot_sample(x_test,y_test,1) 
y_pred_cnn = cnn.predict(x_test)
y_pred_classes_cnn = [np.argmax(element) for element in y_pred_cnn]
plot_sample(x_test,y_test,1)
