import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import cv2

mnist = tf.keras.datasets.mnist
contrast_improve = 0

img_x, img_y = 28, 28

loaded_model = tf.keras.models.load_model('hand_digit.h5')
print(loaded_model.layers[0].input_shape)

#for i in range(10):
for i in [6]:
  # import in greyscale
  gray = cv2.imread("samples/alex/savine/"+str(i)+".png",0)

  if contrast_improve:
    adjusted = gray
    adjusted = cv2.equalizeHist(adjusted)
    #for j in [-60,-70,-80,-90,-100]:
    contrast=1.5
    for j in [-20,-25,-30]:
      adjusted = cv2.addWeighted(adjusted, contrast, np.zeros(adjusted.shape, adjusted.dtype),0,j)
      #contrast = contrast + 0.25
    # reverse
    gray = adjusted
  gray = cv2.resize(255-gray, (28,28))
  if contrast_improve:
    plt.title("image finale pour i={} et j={}".format(i,j))
  else:
    plt.title("image finale pour i={} - contraste inchangé".format(i))
  plt.imshow(gray)
  plt.show()

  #img = Image.open("samples/alex/"+str(i)+".png").convert("L")
  #img = np.resize(img, (28,28,1))
  im2arr = np.array(gray)
  im2arr = im2arr.reshape(1,28,28,1)
  y_pred = loaded_model.predict_classes(im2arr)
  print("pour le scan {} la prédiction est  {}".format(i,y_pred))
