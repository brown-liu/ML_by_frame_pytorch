
import keras
#1 x 28 x 28
from keras.layers.convolutional import Convolution2D
#convolution
model2.add(Convolution2D(25,3,3),input_shape=(1,28,28))
#Max pooling
model2.add(MaxPooling2D((2,2)))
#convolution
model2.add(Convolution2D(50,3,3))
#Max pooling
model2.add(MaxPooling2D((2,2)))
#flatten