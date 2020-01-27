import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random

mnist = tf.keras.datasets.mnist

img_x, img_y = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

loaded_model = tf.keras.models.load_model('hand_digit.h5')
print(loaded_model.layers[0].input_shape)
loaded_model.summary()

# on tire un index aléatoirement dans la base de test
# l'idée sera ensuite d'utiliser des scans de mon écriture
index = random.randrange(0,x_test.shape[0])
print(index)
img = x_test[index].reshape(1, img_x, img_y, 1)
result=loaded_model.predict_classes(img)
print("je suis une machine entrainée et j'ai reconnu le chiffre {}".format(result[0]))

plt.figure()
plt.imshow(x_test[index])
plt.show()
