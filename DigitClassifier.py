#Imports all required modules
from keras.models import load_model
from PIL import Image, ImageFilter
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot

#Load pre-trained model
model = load_model('digit_classifier.h5')

#Here's some code that could be used in the future to convert images to MNIST format
'''
img = Image.open('newsample5.png').convert('L')
width = float(img.size[0])
height = float(img.size[1])
convertedImage = Image.new('L', (28, 28), (255))  

if width > height:  
    nheight = int(round((20.0 / width * height), 0))  
    if (nheight == 0):
        nheight = 1
    img = img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wtop = int(round(((28 - nheight) / 2), 0))  
    convertedImage.paste(img, (4, wtop))
else:
    # Height is bigger. Heigth becomes 20 pixels.
    nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
    if (nwidth == 0):  # rare case but minimum is 1 pixel
        nwidth = 1
        # resize and sharpenpython
    img = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
    convertedImage.paste(img, (wleft, 4))  # paste resized image on white canvas

array = np.array(convertedImage)
array = array.astype('float32')
array = array.reshape(1, 28, 28, 1)
array /= 255
'''


#Loads and converts image to MNIST format
img = load_img('sample9.png', color_mode = "grayscale", target_size=(28,28))
img = img_to_array(img)
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')
img = img/255.0

#Displays converted image for debugging purposes
matplotlib.pyplot.imshow(img.reshape(28,28), cmap = matplotlib.cm.Greys_r)
matplotlib.pyplot.show()

#Makes and prints prediction
pred = model.predict(img)
print(pred.argmax())

