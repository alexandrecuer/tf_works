the mnist set was originally created for the american post office

when testing, make sure you have written american numbers

basic preprocessing before testing :
- greyscale
- white digits on a black background

https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4

# how to use

run `python3 keras_cnn.py` to train, test and save the model as an hd5 reusable file

run `python3 keras_predict.py` to predict on the test dataset

run `python3 keras_own_dig.py` to predict on your personal handwritten digits

# nota

the model works fine with those [digits](savine) but not with others

tips on how to adjust contrast and brightness on openCV

https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
