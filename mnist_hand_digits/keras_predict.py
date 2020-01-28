import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

loaded_model = tf.keras.models.load_model('hand_digit_basic.h5')
focus_shape=loaded_model.layers[0].input_shape
#loaded_model.summary()

# on tire un index aléatoirement dans la base de test
# l'idée sera ensuite d'utiliser des scans de mon écriture
for i in range(10):
    index = random.randrange(0,x_test.shape[0])
    print(index)
    if len(focus_shape)==3:
        img = x_test[index].reshape(1,focus_shape[1],focus_shape[2])
    else:
        img = x_test[index].reshape(1,focus_shape[1],focus_shape[2],focus_shape[3])
    result=loaded_model.predict_classes(img)
    print("je suis une machine entrainée et j'ai reconnu le chiffre {}".format(result[0]))

    plt.figure()
    plt.imshow(x_test[index])
    plt.show()
