import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

reviews=[
    'nice food',
    'amazing restaurant',
    'too good',
    'just loved it!',
    'will go again',
    'horrible food',
    'never go there',
    'poor service',
    'poor quality',
    'needs improvement'
]
sentiment=np.array([1,1,1,1,1,0,0,0,0,0])
print(one_hot('amazing restaurant',30))
vocab_size=30
encoded_reviews=[one_hot(d,vocab_size)for d in reviews]
print("Encoded reviews \n",encoded_reviews)
max_length=4
padded_reviews=pad_sequences(encoded_reviews,maxlen=max_length,padding='post')
print("Padded reviews \n",padded_reviews)
embedded_vector_size=4
model=Sequential()
model.add(Embedding(vocab_size,embedded_vector_size,input_length=max_length,name='embedding'))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
X=padded_reviews
y=sentiment
model.summary()
model.fit(X,y,epochs=50,verbose=0)
loss,accuracy=model.evaluate(X,y)
print('Accuracy',accuracy)
