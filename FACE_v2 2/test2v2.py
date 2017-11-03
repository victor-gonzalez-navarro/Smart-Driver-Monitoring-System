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

# Cambiarlo tanto en train como en test
emotions = ["CR","Jobs", "Messi"]  # Emotion list
data = {}

def get_landmarks(image):
    detections = detector(image, 1)
    #for k, d in enumerate(detections):  # For all detected face instances individually


    xlist = []
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            xlist.append(float(image[i,j]))


    landmarks_vectorised = xlist




    data['landmarks_vectorised'] = landmarks_vectorised
    #if len(detections) < 1:
    #    data['landmarks_vestorised'] = "error"


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/Victor/Dektop/SDMS/FACE_v2/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
clf = joblib.load("/Users/Victor/Dektop/SDMS/FACE_v2/Filenames2/filenameNew.pkl")

test_data = []




img = Image.open("imagenTest2.png")
img = img.resize((200, 200), Image.BILINEAR)
image = np.asarray(img)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe_image = clahe.apply(gray)


get_landmarks(clahe_image)
if data['landmarks_vectorised'] == "error":
    print("no face detected on this one")
else:
    test_data.append(data['landmarks_vectorised'])


##################################################################################################################
# R E D U C I R     C A R A C T E R I S T I C A S
pca = joblib.load("/Users/Victor/Dektop/SDMS/FACE_v2/pca2.pkl")
test_data = pca.transform(test_data)

##################################################################################################################

npar_train = np.array(test_data)
prediction_probs = clf.predict_proba(npar_train)
prediction = clf.predict(npar_train)
print('Probabilities',prediction_probs)
print "\nBienvenido al coche ", emotions[numpy.argmax(prediction_probs)]



