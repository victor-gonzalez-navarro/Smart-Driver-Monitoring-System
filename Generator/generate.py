#####################################---LEGALITY---#############################################
# @copyright All rights reserved to: Universitat Politecnica de Catalunya (UPC)
#
# Principals Authors: Victor Gonzalez, Alex Guerrero, Alejandro Gonzalez
# Advisor: Dr Sergi Bermejo
################################################################################################



##################################################################################################################################
                # F U N C T I O N S
##################################################################################################################################

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


data = {}
def get_landmarks_not(image):

    xlist = []

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            xlist.append(float(image[i,j]))


    landmarks_vectorised = xlist

    data['landmarks_vectorised'] = landmarks_vectorised






# Percentage of eye closure: proportion of time for which the eyelid remains closed
# more than 70-80% within a predefined time period
# PERCLOS = (Number of frames that the eye is more than 70% closed in one minute / Number of frames in one minute)x100




##################################################################################################################################
                # M A I N   P R O G R A M
##################################################################################################################################

# Import the required libraries
import time
import pygame
import cv2
import math
import numpy as np
import dlib
from sklearn.externals import joblib
import numpy
import scipy.ndimage



##################################################################################################################################
                # M A I N   V A R I A B L E S
##################################################################################################################################

# ---------------------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# test_data = []
# ----------------------------------------------------------------------------------------


# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1) in case we want to use an external camera


# variables for processing 1
counter = 0
eyelid_down = False
l = 0
v = np.zeros(4)
mean = 0.0
wmax = 0.0
MAX_CONSECUTIVE_FRAMES = 10
consecutive_frames = 0


# Eye Aspect Ratio
ear = 0
# Blinking too much time alert variables
MAX_CONSECUTIVE_FRAMES = 10
consecutive_frames = 0
consecutive_frames_emotion = 0



# For the resize image:
##########################################
# Si lo cambio, cambia h1 e w1!
cf = 2
##########################################

# variables for face rec
numa = 0
prediction_probs = 2








##################################################################################################################################
                # F O R    E V E R Y    F R A M E      F R O M      T H E     V I D E O
##################################################################################################################################

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




    ##################################################################################################################################
    # F O R    E V E R Y    F A C E      D E T E C T E D
    ##################################################################################################################################
    # loop over the face detections
    for (i, rect) in enumerate(rects):

        # determine the facial landmarks for the face region by entering the B&W image and the detection of a face
        shape = predictor(gray, rect)
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = shape_to_coords(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        (x1, y1, w1, h1) = to_help_opencv(rect)
        y1 = y1 - 55
        h1 = h1 + 70
        #print(h1)

        if (y1 < 0):
            y1=1
        if(h1<0):
            h1 = 1

        # draw the face bounding box
        # parameters: image, one vertex, opposite vertex, color, thickness
        cv2.rectangle(image, (cf*x1, cf*y1), (cf*x1 + cf*w1, cf*y1 + cf*h1), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i+1), (cf*x1 - 10, cf*y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks and draw a point on the image
        for (x, y) in shape:
            cv2.circle(image, (cf*x, cf*y), 2, (0, 0, 255), -1)




        ##################################################################################################################################
        # F A C E    R E C O G N I T I O N
        ##################################################################################################################################
        numa = numa +1
        if (numa % 1 == 0):
            test_data = []


            grayl = gray[y1:y1+h1,x1:x1+w1]


            k = "../Victor/fiile_" + str(numa) + ".png"
            cv2.imwrite(k, grayl)


            pixelx = 250 #266
            pixely = 250 #266

            imq = scipy.misc.imresize(grayl, (pixelx, pixely), interp='bilinear', mode=None)
            clahe_image = clahe.apply(imq)


            get_landmarks_not(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                test_data.append(data['landmarks_vectorised'])


            #------------------------------------------------------------------------------------------------------------




    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
