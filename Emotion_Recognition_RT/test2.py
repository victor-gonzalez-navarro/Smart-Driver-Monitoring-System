#####################################---LEGALITY---#############################################
# @copyright All rights reserved to: Universitat Politecnica de Catalunya (UPC)
#
# Principals Authors: Victor Gonzalez, Alex Guerrero, Alejandro Gonzalez
################################################################################################



####################################---FUNCTIONS---#############################################
# Function 1
def to_help_opencv(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# Function 2. Shape is an object containing the 68 (x,y)-coordinates
def shape_to_coords(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    # it is like a vector of 68 positions with 2 coordinates for each position
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# Function 3: Euclidian distance
def eucli(vec):
    res = math.sqrt(vec[0]**2 + vec[1]**2)
    return res


# Function 4: Calculate the EAR and increment the counter if it's blinking. In addition, measure how many frames lasts the blink
def processing1(shape, counter, eyelid_down, consecutive_frames):

    # Calculate the Eye Aspect Ratio of the left eye
    num = (eucli(10*shape[37]-10*shape[41]) + eucli(10*shape[38]-10*shape[40]))
    den = (2*(eucli(10*shape[36]-10*shape[39])))
    left_ear = num/float(den)

    # Calculate the Eye Aspect Ratio of the right eye
    num = (eucli(10 * shape[43] - 10 * shape[47]) + eucli(10 * shape[44] - 10 * shape[46]))
    den = (2 * (eucli(10 * shape[42] - 10 * shape[45])))
    right_ear = num / float(den)

    ear = (left_ear + right_ear) / 2

    if ear < 0.22 and eyelid_down == False:
        eyelid_down = True
        counter += 1
    elif ear < 0.22 and eyelid_down == True:
        consecutive_frames += 1
    elif ear > 0.22 and eyelid_down == True:
        eyelid_down = False
        consecutive_frames = 0

    return counter, eyelid_down, consecutive_frames

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

#Detect emotions and notify the emotion
def detect_emotion(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/Victor/Dektop/SDMS/Emotion_Recognition_RT/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file

    clf = joblib.load("/Users/Victor/Dektop/SDMS/Emotion_Recognition_RT/Filenames/filename5.pkl")

    test_data = []
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    clahe_image = clahe.apply(gray)
    get_landmarks(clahe_image)
    if data['landmarks_vectorised'] == "error":
        print("no face detected on this one")
    else:
        test_data.append(data['landmarks_vectorised'])  # append image array to training data list
    npar_train = np.array(test_data)
    prediction_probs = clf.predict_proba(npar_train)
    print('Probabilities', prediction_probs)


##################################---MAIN PROGRAM---############################################
# Import the required libraries
import numpy as np
import dlib
import cv2
import math
import time
import pygame
from sklearn.externals import joblib

initime = time.time()
# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/Victor/Dektop/SDMS/Emotion_Recognition_RT/shape_predictor_68_face_landmarks.dat")

data = {}

# Create an object to capture the video
# If the parameter is equal to '0' it uses the camera of the computer,
#  if the parameter is equal to '0' it uses the external webcam
cap = cv2.VideoCapture(0)


# variables for processing 1
counter = 0
eyelid_down = False
l = 0
v = np.zeros(4)
mean = 0.0
wmax = 0.0
MAX_CONSECUTIVE_FRAMES = 10
consecutive_frames = 0
consecutive_frames_emotion = 0

# For the resize image:
cf = 4
numi = 0

while(True):
    # Capture frame-by-frame. The ret parameter is useful
    # when reading a file that can have an end. Now we read
    # from the webcam, so we don't have this problem
    ret, image = cap.read()

    # resize to be able to run in real-time
    imagemod = cv2.resize(image, (0,0), fx=1/float(cf), fy=1/float(cf))
    # RGB -> B&W
    gray = cv2.cvtColor(imagemod, cv2.COLOR_BGR2GRAY)

    # detector of the faces in the grayscale frame
    rects = detector(gray, 1)

    # loop over the face detections
    # Check who is closer to the camera and only do the detection for that face

    wmax = 0
    for (i, rect) in enumerate(rects):

        actualtime = int(time.time() - initime)

        # determine the facial landmarks for the face region by entering the B&W image and the detection of a face
        shape = predictor(gray, rect)
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = shape_to_coords(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        (x, y, w, h) = to_help_opencv(rect)

        # draw the face bounding box
        # parameters: image, one vertex, opposite vertex, color, thickness
        cv2.rectangle(image, (cf*x, cf*y), (cf*x + cf*w, cf*y + cf*h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (cf*x - 10, cf*y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks and draw a point on the image
        for (x, y) in shape:
            cv2.circle(image, (cf*x, cf*y), 2, (0, 0, 255), -1)

        # Processing 1
        counter, eyelid_down, consecutive_frames = processing1(shape, counter, eyelid_down, consecutive_frames)

        numi = numi +1
        if (numi == 100):
            detect_emotion(imagemod)
            numi = 0

    cv2.putText(image, "UPC-Lear Corporation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),2)
    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
