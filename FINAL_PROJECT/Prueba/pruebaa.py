from PIL import Image
import numpy as np
import dlib
import cv2
import time
#import winsound
import pygame
from PIL import Image
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
import Image
import scipy.misc
import scipy.ndimage
import numpy as np

#test_image = "imagenTest.png"
#original = Image.open(test_image)
#original.show()

#width, height = original.size   # Get dimensions
#left = width/4
#top = height/4
#right = 3 * width/4
#bottom = 3 * height/4
#cropped_example = original.crop((left, top, right, bottom))

#cropped_example.show()




#test_image = "imagenTest.png"

#original = Image.open(test_image)
#original.show()

#width, height = original.size   # Get dimensions
#left = width/4
#top = height/4
#right = 3 * width/4
#bottom = 3 * height/4
#cropped_example = original.crop((left, top, right, bottom))

#cropped_example.show()



test_image = "imagenTest.png"
original = cv2.imread(test_image)
im = scipy.misc.imresize(original, (6,6), interp='bilinear', mode=None)
