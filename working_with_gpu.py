pip install virtualenv
python3 -m plaidml-venv source plaidml-venv/bin/activate
pip install -U plaidml-keras
plaidml-setup
import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ['KERAS_BACKEND']="plaidml.keras.backend"
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#normalizing the data so that all values remain in the same interval
x_train = x_train.astype('float32').reshape(60000, 28, 28,1)/255
x_test = x_test.astype('float32').reshape(10000, 28, 28,1)/255
model = keras.Sequential()
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10)
score=model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test Accuracy: ', score[1])
