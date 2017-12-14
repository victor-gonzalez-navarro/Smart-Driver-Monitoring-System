
# Import the required libraries
import time
import cv2
import numpy as np
import dlib
from sklearn.externals import joblib
import numpy
import scipy.ndimage
import serial
import time


########################################################################################################################
                # F U N C T I O N S
########################################################################################################################

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

# ----------------------------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------


def face_recognition(clahe_image, test_data):

    get_landmarks_not(clahe_image)

    if data['landmarks_vectorised'] == "error":
        print("no face detected on this one")
    else:
        test_data.append(data['landmarks_vectorised'])

    # R E D U C I R     C A R A C T E R I S T I C A S   C O N   P C A
    pca = joblib.load('dir/pca.pkl')
    test_data = pca.transform(test_data)

    npar_train = np.array(test_data)
    prediction_probs = clf.predict_proba(npar_train)
    prediction = clf.predict(npar_train)

    return prediction_probs, prediction, test_data

# ----------------------------------------------------------------------------------------------------------------------






########################################################################################################################
                # M A I N   P R O G R A M
########################################################################################################################


########################################################################################################################
                # M A I N   V A R I A B L E S
########################################################################################################################

# ---------------------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clf = joblib.load('dir/pfilenameNew.pkl')
# ----------------------------------------------------------------------------------------


global initime
initime = time.time()
# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dir/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)





# For the resize image:
##########################################
# Si lo cambio, cambia h1 e w1!
cfx = 2
cfy = 2
##########################################

# Variables for face rec----------------
numa = 0
prediction_probs = 2
frameant = 0


########################################################################################################################
                # F O R    E V E R Y    F R A M E      F R O M      T H E     V I D E O
########################################################################################################################

#ser = serial.Serial('/dev/cu.usbmodem1421', 9600)
#Sentado = 0
#time.sleep(2)

while(True):

    # -----------------------------------------------------------------
    #ser.reset_input_buffer()
    # -----------------------------------------------------------------


    # Capture frame-by-frame. The ret parameter is useful
    # when reading a file that can have an end. Now we read
    # from the webcam, so we don't have this problem
    ret, image = cap.read()

    # ---------------------------------------------------------------------------------------
    # Load two images
    img1 = image
    img2 = cv2.imread('dir/small.png')
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[5:5+rows, 5:5+cols] = dst
    image = img1
    # ---------------------------------------------------------------------------------------


    # resize to be able to run in real-time
    imagemod = cv2.resize(image, (0,0), fx=1/float(cfx), fy=1/float(cfy))
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
            y1 = 1
        if h1 < 0:
            h1 = 1
        if x1 < 0:
            x1 = 1
        if w1 < 0:
            w1 = 1

        # draw the face bounding box
        # parameters: image, one vertex, opposite vertex, color, thickness
        cv2.rectangle(image, (cfx*x1, cfy*y1), (cfx*x1 + cfx*w1, cfy*y1 + cfy*h1), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i+1), (cfx*x1 - 10, cfy*y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks and draw a point on the image
        for (x, y) in shape:
            cv2.circle(image, (cfx*x, cfy*y), 2, (0, 0, 255), -1)




        ################################################################################################################
        # F A C E    R E C O G N I T I O N
        ################################################################################################################
        numa = numa + 1
        print(numa)
        frameact = 1

        if((frameant == 0 and frameact ==1 )or (numa % 10 == 0)):
            test_data = []
            # Set a rectangle instead of a square in order to get better the face
            grayl = gray[y1:y1+h1,x1:x1+w1]

            pixelx = 250
            pixely = 250

            # Get an image of only the face inside the rectangle
            imq = scipy.misc.imresize(grayl, (pixelx, pixely), interp='bilinear', mode=None)
            clahe_image = clahe.apply(imq)

            prediction_probs, prediction,test_data = face_recognition(clahe_image, test_data)

            #cv2.putText(image, "New Information", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(image, "Welcome to the car", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 0, 50), 2)
        cv2.putText(image, "#{}".format(emotions[numpy.argmax(prediction_probs)]), (50, 270), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 100, 255), 2)
        #cv2.putText(image, "#Victor".format(emotions[numpy.argmax(prediction_probs)]), (50, 270), cv2.FONT_HERSHEY_COMPLEX, 2, (50, 0, 50), 2)


            #-----------------------------------------------------------------------------------------------------------


    frameant = frameact

    # cv2.putText(image, "UPC-Lear Corporation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),2)
    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)

    # -----------------------------------------------------------------
    #while (not ser.inWaiting() > 0):
     #   ()
    # ser.reset_input_buffer()
    #Seat = ser.readline()  # // Read the PulseSensor's value. // Assign this value to the "Signal" variable.
    # Seat = int(Seat)
    #print(Seat)
    #caca = ser.readline()
    #caca = int(caca)
    # caca = ser.in_waiting
    # print(caca)
    # -----------------------------------------------------------------



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
