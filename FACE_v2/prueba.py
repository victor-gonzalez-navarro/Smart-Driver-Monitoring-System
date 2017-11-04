
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy
import PIL
from PIL import Image
import Image




#imageFile = "imagenTest.png"
#im1 = Image.open("imagenTest.png")
#im2 = im1.resize((54, 47), Image.BILINEAR)


#ext = ".png"
#im2.save("NEAREST" + ext)


img = Image.open("imagenTest.png")

img = img.resize((5, 59), Image.BILINEAR)
ext = ".png"
img.save("NEAREST" + ext)