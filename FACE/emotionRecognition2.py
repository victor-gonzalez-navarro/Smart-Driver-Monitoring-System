import cv2
import glob
import random
import math
import numpy as np
import dlib
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from PIL import Image
import numpy

# Cambiarlo tanto en train como en test
emotions = ["Messi","s20", "s21", "s22", "s23", "s24", "s25", "s26"]  # Emotion list

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
#clf = SVC(kernel='linear', probability=True, tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel
#clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
#clf = SVC(C=5.0, kernel='rbf', degree=5, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
#clf = SVC(C=1.0, kernel='rbf', degree=5, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
#clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
clf = SVC(kernel='linear', probability=True, tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel


data = {}  # Make dictionary for all values


# data['landmarks_vectorised'] = []

def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset2//%s//*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually

        ##################################################################################################################

        xlist = []
        for i in range(1, image.shape[0]):
            for j in range(1, image.shape[1]):
                xlist.append(float(image[i,j]))


        landmarks_vectorised = xlist


        ##################################################################################################################


        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))



        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))


    ##################################################################################################################
    # R E D U C I R     C A R A C T E R I S T I C A S
    pca = PCA(n_components=150).fit(training_data)

    joblib.dump(pca, 'pca.pkl')

    training_data = pca.transform(training_data)
    prediction_data = pca.transform(prediction_data)

    ##################################################################################################################

    return training_data, training_labels, prediction_data, prediction_labels







accur_lin = []
for i in range(0, 10):
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()


    npar_train = np.array(training_data) # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" % i)  # train SVM
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print "linear: ", pred_lin
    accur_lin.append(pred_lin)  # Store accuracy in a list

print("Mean value lin svm: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs

# http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/

#########################################################################################
joblib.dump(clf,'filenameNew.pkl')