#####################################---LEGALITY---#############################################
# @copyright All rights reserved to: Universitat Politecnica de Catalunya (UPC)
#
# Principals Authors: Victor Gonzalez, Alex Guerrero, Alejandro Gonzalez
# Advisor: Sergi Bermejo
################################################################################################



##################################################################################################################################
                # M A I N   F U N C T I O N S
###################################################################################################################################
#  Function 1
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

# Function 5: getLandmarks
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
            #landmarks_vectorised.append(w)
            #landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(int(math.atan((y) / (x+0.0001)) * 360 / (2*math.pi)))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

# Function 6: Emotion
def emotion(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Or set this to whatever you named the downloaded file
    clf = joblib.load('correcto.pkl')

    test_data = []
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
    # print('Probabilities',prediction_probs)
    # print('Class',prediction)
    if prediction == 0 and prediction_probs[0][0] > 0.8:
        # print('El sujeto esta enfadado')
        num=1
    elif prediction == 1:
        # print('El sujeto esta feliz')
        num=2
    elif prediction == 2:
        # print('El sujeto esta sorprendido')
        num=3
    else:
        # print('El sujeto esta neutral')
        num=4

    return num

##################################################################################################################################
                # M A I N   P R O G R A M
################################################################################################################################### Import the required libraries
import numpy as np
import dlib
import cv2
import math
import time
from sklearn.externals import joblib



initime = time.time()
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
num=4

# For the resize image:
cf = 4




##################################################################################################################################
                # F O R    E V E R Y    F R A M E      F R O M      T H E     V I D E O
##################################################################################################################################
while(True):
    # Capture frame-by-frame. The ret parameter is useful
    # when reading a file that can have an end. Now we read
    # from the webcam, so we don't have this problem
    ret, image = cap.read()
    actualtime = int(time.time() - initime)
    # resize to be able to run in real-time
    imagemod = cv2.resize(image, (0,0), fx=1/float(cf), fy=1/float(cf))
    # RGB -> B&W
    gray = cv2.cvtColor(imagemod, cv2.COLOR_BGR2GRAY)

    # detector of the faces in the grayscale frame
    rects = detector(gray, 1)

    if (actualtime % 5 == 0):
        ret, image = cap.read()
        num = emotion(image)

    if num == 1:
        # print('El sujeto esta enfadado')
        cv2.putText(image, "El sujeto esta enfadado", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)
    elif num == 2:
        # print('El sujeto esta feliz')
        cv2.putText(image, "El sujeto esta feliz", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)
    elif num == 3:
        # print('El sujeto esta sorprendido')
        cv2.putText(image, "El sujeto esta sorprendido", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)
    else:
        # print('El sujeto esta neutral')
        cv2.putText(image, "El sujeto esta neutral", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)



    cv2.putText(image, "UPC-Lear Corporation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),2)
    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
