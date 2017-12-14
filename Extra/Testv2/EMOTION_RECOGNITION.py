
import numpy as np
import dlib
import cv2
import math
import time
from sklearn.externals import joblib

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
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(int(math.atan((y) / (x+0.0001)) * 360 / (2*math.pi)))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

# Function 6: Emotion
def emotion(image, clf):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clf = joblib.load('pfilenameNew2.pkl')

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
    if prediction == 0 and prediction_probs[0][0] > 0.7:
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


clf = joblib.load('pfilenameNew2.pkl')

initime = time.time()
# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1) in case we want to use an external camera


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

    # ---------------------------------------------------------------------------------------
    # Load two images
    img1 = image
    img2 = cv2.imread('small.png')
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
    img1[5:5 + rows, 5:5 + cols] = dst
    image = img1
    # ---------------------------------------------------------------------------------------


    actualtime = int(time.time() - initime)

    cv2.putText(image, "The driver is ", (30, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, 16)

    if (actualtime % 5 == 0):
        # ret, image = cap.read()
        num = emotion(image, clf)

    if num == 1:
        # print('El sujeto esta enfadado')
        cv2.putText(image, "Angry", (20, 270), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2, 16)
    elif num == 2:
        # print('El sujeto esta feliz')
        cv2.putText(image, "Happy", (20, 270), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2, 16)
    elif num == 3:
        # print('El sujeto esta sorprendido')
        cv2.putText(image, "Surprised", (20, 270), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2, 16)
    else:
        # print('El sujeto esta neutral')
        cv2.putText(image, "Neutral", (20, 270), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2, 16)



    # Display the resulting frame and press q to exit
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
