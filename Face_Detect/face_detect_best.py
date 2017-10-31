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


# Function 4: Processing 1
def processing1(shape, counter, abajo):
    num = (eucli(10*shape[37]-10*shape[41]) + eucli(10*shape[38]-10*shape[40]))
    den = (2*(eucli(10*shape[36]-10*shape[39])))
    res = num/float(den)

    if res < 0.22 and abajo == 0:
        abajo = 1
        counter = counter + 1

    if res > 0.22 and abajo == 1:
        abajo = 0

    cv2.putText(image, "Counter #{}".format(counter), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255),2)
    return counter, abajo



##################################---MAIN PROGRAM---############################################
# Import the required libraries
import numpy as np
import dlib
import cv2
import math
import time

initime = time.time()
# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/Victor/Dektop/SDMS/Face_Detect/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# variables for processing 1
counter = 0
abajo = 0
l = 0
z = np.zeros(4)
med = 0.0
first = True
wmax = 0.0

# For the resize image:
cf = 4

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



    for (i, rect) in enumerate(rects):


        '''# loop over the face detections'''
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

        actualtime = int(time.time()-initime)
        counter, abajo = processing1(shape, counter, abajo)
        cv2.putText(image, "Time {}".format(actualtime), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        counter2=counter
        if (actualtime % 10 == 0) and first==True:
            for t in [2,1,0]:
                z[t+1]=z[t]
            z[0]= counter
            counter = 0
            first = False

        elif actualtime % 10 != 0:
            first = True

        med = (z[0]+z[1]+z[2]+z[3])/4
        cv2.putText(image, "BL {}".format(med), (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    cv2.putText(image, "UPC-Lear Corporation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),2)
    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
