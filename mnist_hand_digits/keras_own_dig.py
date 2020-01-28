import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import cv2

mnist = tf.keras.datasets.mnist
contrast_improve = 0
path = "savine"
extension = ".png"

img_x, img_y = 28, 28

loaded_model = tf.keras.models.load_model('hand_digit.h5')
print(loaded_model.layers[0].input_shape)

for r,d,f in os.walk(path):
    for file in f:
        if extension in file :
            # try to use pillow (PIL) not working - convert("L") is for greyscale
            #img = Image.open(path+"/"file).convert("L")
            #img = np.resize(img, (28,28,1))
            # import in greyscale
            gray = cv2.imread(path+"/"+file,0)

            if contrast_improve:
                adjusted = gray
                adjusted = cv2.equalizeHist(adjusted)
                contrast=2.5
                #for j in [-60,-70,-80,-90,-100]:
                for j in [-20,-25,-30]:
                    adjusted = cv2.addWeighted(adjusted, contrast, np.zeros(adjusted.shape, adjusted.dtype),0,j)
                    #contrast = contrast + 0.25
                gray = adjusted

            # reverse and resize
            gray = cv2.resize(255-gray, (28,28))
            if contrast_improve:
                plt.title("image finale {} et j={}".format(file,j))
            else:
                plt.title("image finale {} - contraste inchangé".format(file))
            # output the image
            plt.imshow(gray)
            plt.show()

            im2arr = np.array(gray)
            im2arr = im2arr.reshape(1,28,28,1)
            # are these 2 below lines necessary
            im2arr = im2arr.astype('float32')
            im2arr /= 255
            y_pred = loaded_model.predict_classes(im2arr)
            print("pour le scan {} la prédiction est  {}".format(file,y_pred))
