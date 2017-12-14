#####################################---LEGALITY---#############################################
# @copyright All rights reserved to: Universitat Politecnica de Catalunya (UPC)
#
# Principals Authors: Victor Gonzalez, Alex Guerrero, Alejandro Gonzalez
# Advisor: Dr Sergi Bermejo
################################################################################################


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


# Function 3: Euclidian distance
# def eucli(vec):
#     res = math.sqrt(vec[0]**2 + vec[1]**2)
#     return res


# Function 4: Calculate the EAR and increment the counter if it's blinking. In addition, measure how many frames lasts the blink
def blinking_and_alert(shape, blinking_counter, eyelid_down, consecutive_frames):

    # Calculate the Eye Aspect Ratio of the left eye
    left_num = (np.linalg.norm(10*shape[37]-10*shape[41]) + np.linalg.norm(10*shape[38]-10*shape[40]))
    left_den = (2*(np.linalg.norm(10*shape[36]-10*shape[39])))
    left_ear = left_num/float(left_den)

    # Calculate the Eye Aspect Ratio of the right eye
    right_num = (np.linalg.norm(10 * shape[43] - 10 * shape[47]) + np.linalg.norm(10 * shape[44] - 10 * shape[46]))
    right_den = (2 * (np.linalg.norm(10 * shape[42] - 10 * shape[45])))
    right_ear = right_num / float(right_den)

    #Calculate the Eye Aspect Ratio
    ear = (left_ear + right_ear) / 2

    if ear < 0.22 and not eyelid_down:
        eyelid_down = True
        blinking_counter += 1
    elif ear < 0.22 and eyelid_down:
        consecutive_frames += 1
    elif ear > 0.22 and eyelid_down:
        eyelid_down = False
        consecutive_frames = 0

    # cv2.putText(image, "Blinking Counter #{}".format(blinking_counter), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255),2)
    return ear,blinking_counter, eyelid_down, consecutive_frames


# Function 5: Face Recognition
# Cambiarlo tanto en train como en test
emotions = ["Alex","AlexTel", "Victor"]  # Emotion list
data = {}
def get_landmarks_not(image):

    xlist = []

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            xlist.append(float(image[i,j]))


    landmarks_vectorised = xlist

    data['landmarks_vectorised'] = landmarks_vectorised

def face_recognition(clahe_image, test_data):

    get_landmarks_not(clahe_image)

    if data['landmarks_vectorised'] == "error":
        print("no face detected on this one")
    else:
        test_data.append(data['landmarks_vectorised'])

    # R E D U C I R     C A R A C T E R I S T I C A S   C O N   P C A
    pca = joblib.load('pca.pkl')
    test_data = pca.transform(test_data)

    npar_train = np.array(test_data)
    prediction_probs = clf.predict_proba(npar_train)
    prediction = clf.predict(npar_train)

    return prediction_probs, prediction, test_data


def yawning_detection(shape):

    global yawnStartTime, isYawning, yawnCounter

    #Calculate the yawning aspect ratio
    num = (np.linalg.norm(10 * shape[50] - 10 * shape[58]) + np.linalg.norm(10 * shape[52] - 10 * shape[56]))
    den = (2 * (np.linalg.norm(10 * shape[48] - 10 * shape[54])))
    yar = num / float(den)

    #Decide whether it is yawning or not
    if yar > 0.67:
        if not isYawning:
            isYawning = True
            yawnStartTime = time.time()
    else:
        if isYawning:
            isYawning = False
            if (time.time() - yawnStartTime) >= AVERAGE_YAWN_TIME:
                yawnCounter += 1
                print(yawnCounter)

    # cv2.putText(image, "Yawn Counter #{}".format(yawnCounter), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 2,16)



# Percentage of eye closure: proportion of time for which the eyelid remains closed
# more than 70-80% within a predefined time period
# PERCLOS = (Number of frames that the eye is more than 70% closed in one minute / Number of frames in one minute)x100
def perclos(ear,fps,actualtime):

    global counterPerclos,firstTime,perClos

    # When the eye is almost closed -> ear = 0.2
    # When the eye is closed -> ear = 0.12

    if ear < 0.23:
        counterPerclos += 1
    if (actualtime % 10 == 0) and firstTime:
        # Total number of frames
        total_frames = 10.0*fps
        perClos = (counterPerclos / total_frames)
        firstTime = False

        # print"Counter Perclos %s" %counterPerclos
        # print "Total FPS %s" %total_fps
        # print "PERCLOS %.2f" %perClos

    if (actualtime % 10) == 1:
        counterPerclos = 0
        firstTime = True

    # cv2.putText(image, "PERCLOS #{}".format(perClos), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)




##################################################################################################################################
                # M A I N   P R O G R A M
##################################################################################################################################





########################################################################################################################
                # M A I N   V A R I A B L E S
########################################################################################################################

# ---------------------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clf = joblib.load('pfilenameNew.pkl')
# test_data = []
# ----------------------------------------------------------------------------------------


global initime
initime = time.time()
# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1) in case we want to use an external camera


# Variables for Blinking and Alert---------------------

blinking_counter = 0
eyelid_down = False
l = 0
v = np.zeros(4)
mean = 0.0
wmax = 0.0
ear = 0 #Eye Aspect Ratio
# Blinking too much time alert variables
MAX_CONSECUTIVE_FRAMES = 10
consecutive_frames = 0
#consecutive_frames_emotion = 0



# For the resize image:
##########################################
# Si lo cambio, cambia h1 e w1!
cf = 2
##########################################

# Variables for face rec----------------
numa = 0
prediction_probs = 2


# Yawn variables----------------------------
global yawnStartTime
yawnStartTime = 0

# Flag for testing the start time of the yawn
global isYawning
isYawning = False

# Yawn Counter
global yawnCounter
yawnCounter = 0

# Average yawn time
AVERAGE_YAWN_TIME = 2.6

#PERCLOS variables-----------------------------

#Counter of frames of the entire proportion of time
global counterPerclosTotal
counterPerclosTotal = 0

#Counter of frames when the eyes is closed
global counterPerclos
counterPerclos = 0

global perClos
perClos =0.0

global frameCounter
frameCounter = 0

global tick
tick = 0

global firstTime
firstTime = True

fps = 0
frameant = 0

v2 = np.array([9,9,9,9,9,9,9,9,9,9])


########################################################################################################################
                # F O R    E V E R Y    F R A M E      F R O M      T H E     V I D E O
########################################################################################################################

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




    ####################################################################################################################
    # F O R    E V E R Y    F A C E      D E T E C T E D
    ####################################################################################################################
    # loop over the face detections

    frameact = 0
    vari = 0
    for (i, rect) in enumerate(rects):

        vari = 1

        # Determine the facial landmarks for the face region by entering the B&W image and the detection of a face
        shape = predictor(gray, rect)
        # Convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = shape_to_coords(shape)

        # Convert dlib's rectangle to a OpenCV-style bounding box
        (x1, y1, w1, h1) = to_help_opencv(rect)
        y1 = y1 - 55
        h1 = h1 + 70
        # print(h1)

        if y1 < 0:
            y1=1
        if h1 < 0:
            h1 = 1

        # draw the face bounding box
        # parameters: image, one vertex, opposite vertex, color, thickness
        cv2.rectangle(image, (cf*x1, cf*y1), (cf*x1 + cf*w1, cf*y1 + cf*h1), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i+1), (cf*x1 - 10, cf*y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks and draw a point on the image
        for (x, y) in shape:
            cv2.circle(image, (cf*x, cf*y), 2, (0, 0, 255), -1)


        ################################################################################################################
        # B L I N K I N G     C O U N T E R
        ################################################################################################################
        # blinking_and_alert
        actualtime = int(time.time()-initime)
        ear,blinking_counter, eyelid_down, consecutive_frames = blinking_and_alert(shape, blinking_counter, eyelid_down, consecutive_frames)



        ################################################################################################################
        # D R O W S I N E S S     D E T E C T I O N
        ################################################################################################################
        # Warn the driver if it is staying too much time with the eyelid down
        if consecutive_frames > MAX_CONSECUTIVE_FRAMES:
            pygame.init()
            alarma = pygame.mixer.Sound('guau.wav')
            alarma.play()



        ################################################################################################################
        # B L I N K I N G     F R E Q U E N C Y  (not used yet)
        ################################################################################################################

        cv2.putText(image, "Time {}".format(actualtime), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if (actualtime % 10 == 0) and first:
            for t in [2,1,0]:
                v[t+1]=v[t]

            v[0] = blinking_counter
            blinking_counter = 0
            mean = (v[0]+v[1]+v[2]+v[3])/4
            first = False

        elif (actualtime % 10) != 0:
            first = True


        ################################################################################################################
        # Y A W N    D E T E C T I O N
        ################################################################################################################

        yawning_detection(shape)


        ################################################################################################################
        # P E R C L O S   D E T E C T I O N
        ################################################################################################################
        # Calculate the frames per second. It will be used to calculate the total number of frames of the interval where
        # PERCLOS is calculated
        frameCounter += 1
        if (actualtime - tick) >= 1:
            tick += 1
            fps = frameCounter
            frameCounter = 0

        perclos(ear, fps, actualtime)


        ################################################################################################################
        # F A C E    R E C O G N I T I O N
        ################################################################################################################
        numa = numa +1
        frameact = 1
        #if (numa % 5 == 0):
        if((frameant == 0 and frameact ==1 )or numa % 10 == 0):
            test_data = []
            # Set a rectangle instead of a square in order to get better the face
            grayl = gray[y1:y1+h1,x1:x1+w1]

            # k = "../Fotos/file_" + str(numa) + ".png"
            # cv2.imwrite(k, grayl)

            pixelx = 250 #266
            pixely = 250 #266

            # Get an image of only the face inside the rectangle
            imq = scipy.misc.imresize(grayl, (pixelx, pixely), interp='bilinear', mode=None)
            clahe_image = clahe.apply(imq)

            prediction_probs, prediction,test_data = face_recognition(clahe_image, test_data)

            # print(numpy.argmax(prediction_probs))
            # for t in [8, 7, 6, 5, 4, 3, 2, 1, 0]:
            #     v2[t + 1] = v2[t]
            # v2[0] = numpy.argmax(prediction_probs)
            # print(v2)
            # counts = np.bincount(v2)
            # print np.argmax(counts)
            # print("-----------")

        #if(v2[9] != 9):
            #cv2.putText(image, "Welcome to the car #{}".format(emotions[counts]), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

        cv2.putText(image, "Welcome to the car #{}".format(emotions[numpy.argmax(prediction_probs)]), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

            #------------------------------------------------------------------------------------------------------------



    frameant = frameact

    cv2.putText(image, "UPC-Lear Corporation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),2)
    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
