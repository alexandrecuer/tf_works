from __future__ import print_function
import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
mnist = tf.keras.datasets.mnist

batch_size = 128
num_classes = 10
# pour augmenter la précision du modèle, passer epochs de 5 à 10
epochs = 5

# input image dimensions - all images in mnist are 28 pixels by 28 pixels
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plt.figure()
#plt.imshow(x_test[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# configure le modèle pour l'entrainement
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['acc'])


class AccuracyHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.acc = []
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        #print("loss is {}".format(logs.get('loss')))
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        print("acc is {}".format(logs.get('acc')))
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)

# sauver le modèle comme un fichier hd5 pour réutiliusation ultérieure à des fins de prédicitions
model.save('hand_digit.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, epochs+1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
