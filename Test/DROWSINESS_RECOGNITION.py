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

# Take a bounding predicted by dlib and convert it to the format (x, y, w, h) as we would normally do with OpenCV
def to_help_opencv(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# Shape is an object containing the 68 (x,y)-coordinates
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



# Calculate the EAR and increment the counter if it's blinking. In addition, measure how many frames lasts the blink
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
    if ear < 0.2 and not eyelid_down:
        eyelid_down = True
        blinking_counter += 1
    elif ear < 0.2 and eyelid_down:
        consecutive_frames += 1
    elif ear > 0.2 and eyelid_down:
        eyelid_down = False
        consecutive_frames = 0

    return ear,blinking_counter, eyelid_down, consecutive_frames


def yawning_detection(shape):

    global yawn_start_time, is_yawning, yawn_counter

    #Calculate the yawning aspect ratio
    num = (np.linalg.norm(10 * shape[50] - 10 * shape[58]) + np.linalg.norm(10 * shape[52] - 10 * shape[56]))
    den = (2 * (np.linalg.norm(10 * shape[48] - 10 * shape[54])))
    yar = num / float(den)

    #Decide whether it is yawning or not
    if yar > 0.6:
        if not is_yawning:
            is_yawning = True
            yawn_start_time = time.time()
    else:
        if is_yawning:
            is_yawning = False
            if (time.time() - yawn_start_time) >= AVERAGE_YAWN_TIME:
                yawn_counter += 1


# Percentage of eye closure: proportion of time for which the eyelid remains closed
# more than 70-80% within a predefined time period
# PERCLOS = (Number of frames that the eye is more than 70% closed in one minute / Number of frames in one minute)x100
def perclos(ear,fps,actualtime):

    global counter_perclos,first_time,perClos

    # When the eye is almost closed -> ear = 0.23
    # When the eye is closed -> ear = 0.12

    if ear < 0.27:
        counter_perclos += 1
    if (actualtime % 60 == 0) and first_time:
        # Total number of frames
        total_frames = 60.0*fps
        perClos = (counter_perclos / total_frames)
        first_time = False

        # print"Counter Perclos %s" %counterPerclos
        # print "Total FPS %s" %total_fps
        # print "PERCLOS %.2f" %perClos

    if (actualtime % 60) == 1:
        counter_perclos = 0
        first_time = True


def warn_with_alarm():
    pygame.init()
    alarma = pygame.mixer.Sound('Dir/alarmav2.wav')
    alarma.play()


##################################################################################################################################
                # M A I N   P R O G R A M
##################################################################################################################################





########################################################################################################################
                # M A I N   V A R I A B L E S
########################################################################################################################


global initime
initime = time.time()
# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Dir/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1) in case we want to use an external camera


# Variables for Blinking and Alert---------------------

blinking_counter = 0
eyelid_down = False
l = 0
v = np.zeros(4)
blinking_frequency = 0.0
wmax = 0.0
ear = 0 #Eye Aspect Ratio
# Blinking too much time alert variables
MAX_CONSECUTIVE_FRAMES = 13
consecutive_frames = 0
#consecutive_frames_emotion = 0



# For the resize image:
##########################################
# Si lo cambio, cambia h1 e w1!
cf = 2

##########################################


# Yawn variables----------------------------
global yawn_start_time
yawn_start_time = 0

# Flag for testing the start time of the yawn
global is_yawning
is_yawning = False

# Yawn Counter
global yawn_counter
yawn_counter = 0

# Average yawn time
AVERAGE_YAWN_TIME = 2.0

#PERCLOS variables-----------------------------

#Counter of frames of the entire proportion of time
global counter_perclos_total
counter_perclos_total = 0

#Counter of frames when the eyes is closed
global counter_perclos
counter_perclos = 0

global perClos
perClos =0.0

global frame_counter
frame_counter = 0

global tick
tick = 0

global first_time
first_time = True

fps = 0

# Drowsiness decision
drowsiness_state = 0

ALERT = 0
SOME_SIGNS_OF_SLEEPINESS = 1
SLEEPY = 2



########################################################################################################################
                # F O R    E V E R Y    F R A M E      F R O M      T H E     V I D E O
########################################################################################################################

while(True):
    # ret is used for videos with an end, we don't use it
    # Capture frame-by-frame
    ret, image = cap.read()

    # resize to be able to run in real-time
    imagemod = cv2.resize(image, (0,0), fx=1/float(cf), fy=1/float(cf))
    gray = cv2.cvtColor(imagemod, cv2.COLOR_BGR2GRAY)

    # detector of the faces in the grayscale frame
    rects = detector(gray, 1)

    ####################################################################################################################
    # F O R    E V E R Y    F A C E      D E T E C T E D
    ####################################################################################################################
    # loop over the face detections
    for (i, rect) in enumerate(rects):

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

        if x1 < 0:
            x1=1
        if w1 < 0:
            w1 = 1

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
        cv2.putText(image, "Time {}".format(actualtime), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ear,blinking_counter, eyelid_down, consecutive_frames = blinking_and_alert(shape, blinking_counter, eyelid_down, consecutive_frames)


        ################################################################################################################
        # R E S E T   V A R I A B L E S
        ################################################################################################################

        if (actualtime % 60) == 0:
            blinking_counter = 0
            yawn_counter = 0


        ################################################################################################################
        # Y A W N    D E T E C T I O N
        ################################################################################################################

        yawning_detection(shape)


        ################################################################################################################
        # P E R C L O S   D E T E C T I O N
        ################################################################################################################
        # Calculate the frames per second. It will be used to calculate the total number of frames of the interval where
        # PERCLOS is calculated
        frame_counter += 1
        if (actualtime - tick) >= 1:
            tick += 1
            fps = frame_counter
            frame_counter = 0

        perclos(ear, fps, actualtime)

        ################################################################################################################
        # D R O W S I N E S S   D E C I S I O N
        ################################################################################################################


        # Si estas en un estado en el que te encuentras normal estaras siempre en ALERT.
        # En el caso de que pestanyees mucho, o bosteces significa que hay algun signo de que comienzas a tener suenyo y
        # pasas a SOME_SIGNS_OF_SLEEPINESS. Ademas, esta incluido el perClos en esta condicion, ya que habia casos en que
        # se detectaba un perclos sin llegar a ser preocupante, pero relativamente alto, y como todavia estaba en la primera
        # fase, no hacia nada con el. De manera que si nota que aumenta un poco el perclos ira a medir este directamente en la siguiente fase.
        # De esta manera estaria midiendo el perclos y el blinking counter a la vez, que es lo que no queriamos porque cuando se acerca al threshold
        # y se mantiene en esa zona, los valores van variando y cuenta muchos blinks fantasma. Sin embargo, ya viene bien que lo haga de esta manera, ya que
        # si ocurre este caso significa que esta empezando a estar drowsy y cambiara de estado, que es el objetivo.


        if drowsiness_state == ALERT:

            cv2.putText(image, "PERCLOS #{}".format(perClos), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)
            cv2.putText(image, "Blinking Counter #{}".format(blinking_counter), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)
            cv2.putText(image, "Yawn Counter #{}".format(yawn_counter), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)
            cv2.putText(image, "Drowsiness State: ALERT", (20, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if (blinking_counter > 30) or (yawn_counter >= 1) or (perClos > 0.4):
                drowsiness_state = SOME_SIGNS_OF_SLEEPINESS

            elif consecutive_frames > MAX_CONSECUTIVE_FRAMES:
                drowsiness_state = SLEEPY

        elif drowsiness_state == SOME_SIGNS_OF_SLEEPINESS:

            cv2.putText(image, "Drowsiness State: SOME SIGNS OF SLEEPINESS", (20, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "PERCLOS #{}".format(perClos), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)

            if (perClos > 0.5) or (consecutive_frames > MAX_CONSECUTIVE_FRAMES):
                drowsiness_state = SLEEPY

            elif perClos < 0.15:
                drowsiness_state = ALERT
                yawn_counter = 0
                blinking_counter = 0
                perClos = 0

        elif drowsiness_state == SLEEPY:

            warn_with_alarm()
            drowsiness_state = ALERT
            yawn_counter = 0
            blinking_counter = 0
            perClos = 0


    cv2.putText(image, "UPC-Lear Corporation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),2)
    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
