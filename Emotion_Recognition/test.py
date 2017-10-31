import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.externals import joblib

data = {}

def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(int(math.atan((y - ymean) / (x - xmean)) * 360 / math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/Victor/Dektop/SDMS/Emotion_Recognition/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
clf = joblib.load("/Users/Victor/Dektop/SDMS/Emotion_Recognition/Filenames/filename.pkl")

test_data = []
image = cv2.imread("imagenTest.png")  # open image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
clahe_image = clahe.apply(gray)
get_landmarks(clahe_image)
if data['landmarks_vectorised'] == "error":
    print("no face detected on this one")
else:
    test_data.append(data['landmarks_vectorised'])  # append image array to training data list
npar_train = np.array(test_data)
prediction_probs = clf.predict_proba(npar_train)
prediction = clf.predict(npar_train)
print('Probabilities',prediction_probs)
print('Class',prediction)

