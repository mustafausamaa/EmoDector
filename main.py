
# viola jones libraries
import Questions.question as q
import Text.nlp_emotion as nlp
import time
import threading
import json
import math
# Audio libraries
# Import necessary libraries
import os
import pickle
from cmath import sqrt
from math import ceil

# import skimage as sk
import cv2 as cv

import librosa.core
import pywt
import librosa.core
# Emotion detection libraries
import matplotlib.pyplot as plt
import numpy as np
# this a custom module found the commonfunctions.
import pywt
import skimage.io as io
import xmltodict
from cv2 import (VideoCapture, destroyWindow, groupRectangles, imshow, imwrite,
                 namedWindow, rectangle, resize, waitKey)
from numba import jit, njit
# from commonfunctions import *
from PIL import Image as im
from python_speech_features import mfcc
from scipy import ndimage
from scipy import signal as sig
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import wiener
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy
from skimage.morphology import dilation
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from deepface import DeepFace
# from Rect import *
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


# Facial emotion detection
############################################################################################################################
# Variance
# var=mean^2 -1/N(sum(X^2))
def calculateVariance(window, sqrdWindow):
    # Calculate mean value of the image and square it
    # sqrdMean=(np.mean(window))**2
    mean = ((window[1][1]+window[-1][-1]-window[1]
            [-1]-window[-1][1])/np.size(window))

    # Sum of pixels in integral of the squared image
    # sqrdSum=np.sum(sqrdWindow)
    sqrdSum = (sqrdWindow[1][1]+sqrdWindow[-1][-1] -
               sqrdWindow[1][-1]-sqrdWindow[-1][1])

    # Number of pixels in the window
    N = np.size(window)

    # Calculated variance
    variance = (sqrdSum/N)-(mean*mean)

    # Normalize the window and return it
    # normalizedWindow=window*variance

    return variance


def integrateImage(img):

    rows = img.shape[0]
    cols = img.shape[1]
    integralImage = np.zeros((rows+1, cols+1))
    integralImage[1:, 1:] = np.cumsum(img, 1)
    integralImage = np.cumsum(integralImage, 0)

    # return outputImage[i][j]
    return integralImage


class Rect:
    def __init__(self, inputText, y1=0, x1=0, width=0, height=0, weight=0.0, x2=0, y2=0):
        splitInput = inputText.split(" ")

        self.x1 = int(splitInput[0])-1
        self.y1 = int(splitInput[1])-1

        self.width = int(splitInput[2])
        self.height = int(splitInput[3])

        self.weight = float(splitInput[4])

        self.y2 = self.y1+self.height
        self.x2 = self.x1+self.width

    def calculateArea(self, window, scale):
        # self.printRect()
       # print("window values: ",window[self.y2][self.x2],window[self.y1][self.x1],window[self.y1][self.x2],window[self.y2][self.x1])
        scaledX1 = self.x1*scale
        scaledY1 = self.y1*scale
        scaledY2 = self.y2*scale
        scaledX2 = self.x2*scale
        area = (window[scaledY2][scaledX2]+window[scaledY1][scaledX1] -
                window[scaledY1][scaledX2]-window[scaledY2][scaledX1])
       # print("weight*area/size: ",(self.weight*area)/(np.size(window)))
        # return (self.weight*area)/((window[0][0]+window[-1][-1]-window[0][-1]-window[-1][-0]))
        return (self.weight*area)/np.size(window)

    def printRect(self):
        print('Rect')
        print('x1: ', self.x1)
        print('x2: ', self.x2)
        print('y1: ', self.y1)
        print('y2: ', self.y2)
        print('width: ', self.width)
        print('height: ', self.height)
        print('weight: ', self.weight)


# A feature is equivalent to tree node
# Each node has a value(sum of the pixels inside the rectangles).
# Either a left or a right value is returned based on the comparison with the threshold.
class Feature:
    def __init__(self, leftValue, rightValue, threshold, rects):
        self.leftValue = leftValue
        self.rightValue = rightValue
        self.threshold = threshold
        self.rects = rects

    # Calculate initial value of the feature(two or three rectangle value)
    def calculateValue(self, window, scale):
        initialValue = 0
        for i in range(0, len(self.rects)):
            initialValue += self.rects[i].calculateArea(window, scale)
        return initialValue

    # weak classifier
    def classify(self, window, variance, scale):

        value = self.calculateValue(window, scale)
        if (value > self.threshold*variance):
            # print('feature right value after weak classifier:',self.rightValue)
            return self.rightValue
        else:
            # print('feature left value after weak classifier:',self.leftValue)
            return self.leftValue

    def printFeature(self):
        print('Feature: ')
        print('Left value: ', self.leftValue)
        print('Right value: ', self.rightValue)
        print('Threshold: ', self.threshold)
        for i in range(0, len(self.rects)):
            self.rects[i].printRect()


# Stages of the cascaded classifier
# We have 25 stages numbered from 0 to 24
class Stage:
    def __init__(self, features, threshold):
        self.features = features
        self.threshold = threshold

    # Validate if a window should pass a stage

    def validateWindow(self, window, variance, scale):
        sum = 0
        for i in range(0, len(self.features)):
            sum += self.features[i].classify(window, variance, scale)
        if (sum > self.threshold):
            return True, sum
        else:
            return False, sum

    def printStage(self):
        print('Stage threshold', self.threshold)
        for i in range(0, len(self.features)):
            self.features[i].printFeature()

# Read haar cascade xml file and convert to json


def xmlToJson(xmlFile):
    with open(xmlFile) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    json_data = json.loads(json.dumps(data_dict))
    return json_data


# Call xmlToJson function
json_data = xmlToJson('Facial/haarcascade_frontalface_default.xml')
textRect = json_data['opencv_storage']['haarcascade_frontalface_default']['stages']["_"][0]['trees']['_'][0]['_']['threshold']
# print(textRect)


# Stages allocation

# number of stages
numberOfStages = len(json_data['opencv_storage']
                     ['haarcascade_frontalface_default']['stages']['_'])
# print(numberOfStages)

# init list of stages
stagesList = []

# list of features per stage
featuresList = []

# unMapped rects list
textRects = []

# list of rects for each feature
rectsList = []


for i in range(0, numberOfStages):
    # allocating stage threshold
    stageThreshold = float(
        json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['stage_threshold'])

    # allocate mumber of features
    numberOfFeatures = len(
        json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['trees']['_'])

    # allocate each feature
    for j in range(0, numberOfFeatures):
        # allocate the feature's left value
        leftValue = float(json_data['opencv_storage']['haarcascade_frontalface_default']
                          ['stages']['_'][i]['trees']['_'][j]['_']['left_val'])

        # allocate the feature's right value
        rightValue = float(json_data['opencv_storage']['haarcascade_frontalface_default']
                           ['stages']['_'][i]['trees']['_'][j]['_']['right_val'])

        # allocate the feature's threshold
        featureThreshold = float(json_data['opencv_storage']['haarcascade_frontalface_default']
                                 ['stages']['_'][i]['trees']['_'][j]['_']['threshold'])

        # allocate rects as its xml format
        textRects = json_data['opencv_storage']['haarcascade_frontalface_default'][
            'stages']['_'][i]['trees']['_'][j]['_']['feature']['rects']['_']

        # map rects' xml format to Rect class's format
        for k in range(0, len(textRects)):
            # print(textRects[k])
            rectsList.append(Rect(textRects[k]))

        # add each feature to features list in the stage
        featuresList.append(Feature(leftValue, rightValue,
                            featureThreshold, rectsList))

        # clear list of rects for each feature
        rectsList = []

    # append stage to the cascade stages list
    stagesList.append(Stage(featuresList, stageThreshold))

    # clear list of features per stage
    featuresList = []


# Capture photo using a webcam, returns the path of the photo


def capture_photo():
    namedWindow("preview")
    vc = VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        imshow("preview", frame)

        rval, frame = vc.read()
        key = waitKey(20)
        if key == 27:  # exit on ESC
            break

    # image path
    imwrite("captured_image.png", frame)
    vc.release()
    destroyWindow("preview")

    return "captured_image.png"


# Show the figures / plots inside the notebook

def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def detect_face(img_path, frame_index):

    # Define the edge detection filter
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # scale value
    scale = 6

    # input image
    finalImage = io.imread(img_path, as_gray=False)
    finalImage = cv.resize(finalImage, (384, 288))
    img = io.imread(img_path, as_gray=True)

    # resizing
    img = cv.resize(img, (384, 288))

    # cropping
    # box = (384,288)
    # img = img.crop(box)
    facesList = []

    # show_images([img])

    # Convert the window to grayscale
    gray = img

    # Apply the Sobel filter to detect edges
    # img_edges_x = cv2.filter2D(gray, -1, sobel_x)
    # img_edges_y = cv2.filter2D(gray, -1, sobel_y)
    # img_edges = cv2.magnitude(img_edges_x, img_edges_y)
    # #img_edges = np.sqrt(np.square(img_edges_x) + np.square(img_edges_y))

    # print('img edges mean: ',img_edges.mean())

    # while(facesList==[]):
    for l in range(0, 2):

        #print('scale', scale)

        # show_images([img])

        # calculate integral img
        integralImage = integrateImage(img)

        # calculate integral of the img squared
        sqrdIntegralImage = integrateImage(np.square(img))

        # width and height of the sliding window(24x24)
        windowWidth = int(np.ceil(24*scale))
        windowHeight = int(np.ceil(24*scale))

        # windows inits
        currentWindow = np.zeros((windowHeight, windowWidth))
        currentWindowSqrd = np.zeros((windowHeight, windowWidth))
        varNormalizedWindow = np.zeros((windowHeight, windowWidth))

        # boolean to check if a window passes the stage

        # list of passed windows
        rectsList = []
        rectangleIndicesList = []
        facesList = []

        # window score variable
        score = 0

        # stage threshold values list
        thresh_list = []

        # Main loop(sliding window and detection)
        #print('step size: ', int(0.04*windowWidth))
        for i in range(1, integralImage.shape[0]-windowHeight+1, 1):
            for j in range(1, integralImage.shape[1]-windowWidth+1, int(0.04*windowWidth)):

                # Convert the window to grayscale
                gray = img[i-1:i+windowHeight, j-1:j+windowWidth]

                # Apply the Sobel filter to detect edges
                edges_x = cv.filter2D(gray, -1, sobel_x)
                edges_y = cv.filter2D(gray, -1, sobel_y)
                edges = cv.magnitude(edges_x, edges_y)
                # edges = np.sqrt(np.square(edges_x) + np.square(edges_y))

                # current window of the .integral image
                currentWindow = integralImage[i -
                                              1:i+windowHeight, j-1:j+windowWidth]

                # current window of integral image squared
                currentWindowSqrd = sqrdIntegralImage[i -
                                                      1:i+windowHeight, j-1:j+windowWidth]
                # variance normalization of the image
                # varNormalizedWindow=varianceNormalize(currentWindow,currentWindowSqrd)
                variance = calculateVariance(currentWindow, currentWindowSqrd)

                # if edges.mean() > img_edges.mean():
                # print(edges.mean())
                # Loop on the cascaded classifiers
                for k in range(0, len(stagesList)):
                    # return a boolean based on the window's validation with a stage
                    passed, sum = stagesList[k].validateWindow(
                        currentWindow, np.sqrt(variance), scale)

                # stagesList[k].printStage()
                    if (not passed):
                        break
                    else:
                        score += 1

                # print(score)
                if (score == 25):
                    # print(edges.mean())
                    # thresh_list.append(sum)
                    # facesList.append(img[i:i+windowHeight,j:j+windowWidth])
                    rect = [i, j, i+windowHeight, j+windowWidth]
                    # print(rect)
                    rectsList.append(rect)
                    rectsList.append(rect)
                    # finalImage=cv.rectangle(finalImage,(j,i),(j+windowWidth,i+windowHeight),(0,0,255),1)
                score = 0

        # generated grouped rectangels' list
        groupedRectsList = cv.groupRectangles(rectsList, 1)

        if (rectsList != []):

            # # #without multiple face comparison
            # for i in range(len(groupedRectsList[0])):
            #     #draw each grouped rectangle on the image
            #     facesList.append(finalImage[groupedRectsList[0][i][0]:groupedRectsList[0][i][2], groupedRectsList[0][i][1]:groupedRectsList[0][i][3]])
            #     finalImage = cv.rectangle(finalImage, (groupedRectsList[0][i][1], groupedRectsList[0][i][0]), (groupedRectsList[0][i][3], groupedRectsList[0][i][2]), (0, 0, 255), 2)

            # with multiple face comparison
            for i in range(len(groupedRectsList[0])):
                gray = cv.cvtColor(finalImage[groupedRectsList[0][i][0]:groupedRectsList[0][i]
                                              [2], groupedRectsList[0][i][1]:groupedRectsList[0][i][3]], cv.COLOR_BGR2GRAY)

                # Apply the Canny edge detection algorithm
                edges = cv.Canny(gray, 50, 150)

                # Compute the edge density
                area = edges.size
                perimeter = cv.countNonZero(edges) * 2
                edge_density = perimeter / area
                thresh_list.append(edge_density)

            facesList.append(finalImage[groupedRectsList[0][thresh_list.index(max(thresh_list))][0]:groupedRectsList[0]
                             [thresh_list.index(max(thresh_list))][2], groupedRectsList[0][thresh_list.index(max(thresh_list))][1]:groupedRectsList[0][thresh_list.index(max(thresh_list))][3]])
            finalImage = cv.rectangle(finalImage, (groupedRectsList[0][thresh_list.index(max(thresh_list))][1], groupedRectsList[0][thresh_list.index(max(thresh_list))][0]), (
                groupedRectsList[0][thresh_list.index(max(thresh_list))][3], groupedRectsList[0][thresh_list.index(max(thresh_list))][2]), (0, 0, 255), 2)

        if (facesList == []):
            # show_images([finalImage])
            scale = int(np.ceil(scale*(1.25)))
            # empty the lists for the next scale value
            rectsList = []
            groupedRectsList = []
            # scale-=1
        else:
            break

    # Show the image with the detected face
    # show_images([finalImage])

    # print(thresh_list.index(max(thresh_list)))

    if (facesList == []):

        scale = 6
        windowWidth = int(np.ceil(24*scale))
        windowHeight = int(np.ceil(24*scale))

        window_x = (384/2)-windowWidth/2
        window_y = (288/2)-windowHeight/2
        window_x2 = window_x + windowWidth
        window_y2 = window_y + windowHeight

        path = 'Facial/images/captured_frame'+str(frame_index)+'.jpg'
        cv.imwrite(path,
                   finalImage[int(window_y):int(window_y2), int(window_x):int(window_x2)])
        return finalImage[int(window_y):int(window_y2), int(window_x):int(window_x2)], path, finalImage

    else:
        # without multiple face elimination
        #
        # cv.imwrite("captured_face.png", facesList[0])
        # return facesList[0], "captured_face.png"

        # with multiple face elimination
        #
        # print('detected!')

        path = 'Facial/images/captured_frame'+str(frame_index)+'.jpg'
        cv.imwrite(path,
                   facesList[thresh_list.index(max(thresh_list))])
        return facesList[thresh_list.index(max(thresh_list))], path, finalImage

    # Open the video file


def video_to_frame(path):
    # 'WhatsApp Video 2023-05-22 at 11.26.43 PM.mp4'
    video = cv.VideoCapture(path)
    # Get the total number of frames in the video
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    start_frame = 0
    quart_frame = total_frames//4
    middle_frame = total_frames // 2
    third_quart_frame = int(total_frames*0.75)
    end_frame = total_frames - 1
    # Initialize variables
    count = 0
    success = True
    index = 0
    # path list
    frame_paths = []

    # Loop through the selected frames
    for frame in [start_frame, middle_frame, end_frame]:
        # Set the frame position
        video.set(cv.CAP_PROP_POS_FRAMES, frame)

        # Read the frame
        success, image = video.read()

        # If successful, save the frame as an image
        if success:
            flipped_image = cv.flip(image, 0)
            frame_path = 'Facial/images/frame'+str(index)+'.jpg'
            # print(frame_path)
            cv.imwrite(frame_path, flipped_image)
            index += 1
            frame_paths.append(frame_path)
    return frame_paths


def detect_faces(frame_paths):

    # To be allocated
    detected_faces = []
    detected_faces_paths = []

    # Return it as a matrix and save it on a different path than the input
    for i in range(0, 3):

        detected_face, detected_image_path, final_image = detect_face(
            frame_paths[i], i)
        detected_faces.append(detected_face)
        detected_faces_paths.append(detected_image_path)

    return detected_faces, detected_faces_paths


def show_detected_faces(detected_faces):

    for i in range(len(detected_faces)):
        show_images([detected_faces[i]])


# Emotion detection functions

# Preprocessing functions
def preprocess(img):
    img = cv.resize(img, (100, 100))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(5, 5))
    final_img = clahe.apply(gray)
    return final_img


def pad_img(img, dec_lo):
    img = img.copy().astype(np.double)
    pad_width = len(dec_lo)//2
    img = pywt.pad(img, pad_width, 'periodic')

    return img, pad_width


def correct_dst_dim(swt_img):
    dst_dim = swt_img.shape[0]
    if dst_dim % 2:
        dst_dim -= 1
    while pywt.swt_max_level(dst_dim) < 4:
        dst_dim -= 2
    swt_img = cv.resize(swt_img, (dst_dim, dst_dim))
    return swt_img

# Feature extraction


def detectemotion(roi_gray):
    wavelet = pywt.Wavelet('bior1.3')
    swt_img, pad_width = pad_img(roi_gray, wavelet.dec_lo)
    # we need to pad the image to make sure the dimensions are correct when we do the SWT

    # we remove 2 pixels from the image to make sure that max_level is bigger than 4
    # so that we can do the SWT decomposition with 4 levels
    swt_img = correct_dst_dim(swt_img)

    coeffsAll = pywt.swt2(swt_img, wavelet, level=4, start_level=0)

    dCoeff = []
    aCoeff = []

    for id, coeffs2 in enumerate(coeffsAll):

        LL, (HL, LH, HH) = coeffs2

        LL = wiener(LL, mysize=(3, 3))
        LH = wiener(LH, mysize=(3, 3))
        HL = wiener(HL, mysize=(3, 3))
        HH = wiener(HH, mysize=(3, 3))

        # we need to take the absolute value of the coefficients to make sure that we don't have negative values
        LL = np.abs(LL)
        LH = np.abs(LH)
        HL = np.abs(HL)
        HH = np.abs(HH)

        # LL -= np.min(LL)
        # LH -= np.min(LH)
        # HL -= np.min(HL)
        # HH -= np.min(HH)

        # we need to normalize the coefficients to make sure that we don't have values bigger than 255
        LL = (255)*(LL/np.max(LL))
        LH = (255)*(LH/np.max(LH))
        HL = (255)*(HL/np.max(HL))
        HH = (255)*(HH/np.max(HH))

        # we need to convert the coefficients to integers
        LL = np.fix(LL)
        LH = np.fix(LH)
        HL = np.fix(HL)
        HH = np.fix(HH)

        # remove the padding that we added before the SWT decomposition
        start = pad_width+pad_width//2
        end = -pad_width-pad_width//2-1
        LL = LL[start:end, start:end]
        LH = LH[start:end, start:end]
        HL = HL[start:end, start:end]
        HH = HH[start:end, start:end]

        coeffsAll[id] = (LL, LH, HL, HH)
        dCoeff.append((LH, HL, HH))
        aCoeff.append(LL)

    # %% [markdown]
    # 5) Apply wiener filter to all subbands by
    # equation 6 to reduce the high frequency
    # components effect.

    # %%
    coeffsAll = np.array(coeffsAll)
    dCoeff = np.array(dCoeff)
    aCoeff = np.array(aCoeff)
    coeffsAll = coeffsAll[::-1]
    dCoeff = dCoeff[::-1]
    aCoeff = aCoeff[::-1]

    # 6) Estimate the local energy in each
    # coefficient of detailed subbands by
    # equation 8.

    E = dCoeff**2

    # 7) For each subband, calculate the mean
    # local energy wavelet subband and
    # maximum local energy wavelet subband
    # using equation 9 & 10.
    filter = np.ones((3, 3))
    LEmax = E
    LEmu = E
    for i in range(4):
        for j in range(3):
            LEmu[i, j] = ndimage.uniform_filter(LEmu[i, j], size=3)
            # To get the maxima of the local energy, we dilate the local energy
            LEmax[i, j] = dilation(LEmax[i, j], filter)

    LEmax = (255)*(LEmax/np.max(LEmax))
    LEmu = (255)*(LEmu/np.max(LEmu))

    # 8) Evaluate pixel level fusion to the local
    # energy wavelet subbands to obtain a
    # combination set of subbands Skj(a,b)
    # (k=1,2…..,6 & j=1,2,3,4) by equations
    # 11-14.

    # %%
    # Equations from 12 to 14
    u = np.zeros((4, 1))
    v = np.zeros((4, 1))
    w = np.zeros((4, 1))
    for j in range(4):
        u[j] = np.sum(0.5*(1+np.corrcoef(coeffsAll[j][0], coeffsAll[j][1])))
        v[j] = np.sum(0.5*(1+np.corrcoef(coeffsAll[j][0], coeffsAll[j][2])))
        w[j] = np.sum(0.5*(1+np.corrcoef(coeffsAll[j][0], coeffsAll[j][3])))

    # %%
    # Equation 11
    Skj = np.zeros((6, 4, E.shape[2], E.shape[3]))
    for j in range(4):
        Skj[0, j] = u[j]*LEmu[j, 0]+v[j]*LEmu[j, 1]
        Skj[1, j] = u[j]*LEmu[j, 0]+w[j]*LEmu[j, 2]
        Skj[2, j] = v[j]*LEmu[j, 1]+w[j]*LEmu[j, 2]
        Skj[3, j] = u[j]*LEmax[j, 0]+v[j]*LEmax[j, 1]
        Skj[4, j] = u[j]*LEmax[j, 0]+w[j]*LEmax[j, 2]
        Skj[5, j] = v[j]*LEmax[j, 1]+w[j]*LEmax[j, 2]

    Skj = (255)*(Skj/np.max(Skj))

    # 9) Calculate the entropy value of all
    # subbands using equation 15. The
    # selections of subbands are made by the
    # consideration of maximal entropy
    # values. Se(a,b) (e=1,2….,15)

    # shannon entropy from equation 15
    H = np.zeros((6, 4))
    for k in range(6):
        for j in range(4):
            H[k, j] = shannon_entropy(Skj[k, j])

    H = H.flatten()
    H_id = H.argsort()[::-1]

    Se = Skj.reshape((24, E.shape[2], E.shape[3]))
    Se = Se[H_id[:15]]

    Se = (255)*(Se/np.max(Se))
    # 10)Apply 8x8 block dct to all selected
    # subbands and the dc coefficient obtained
    # in each block alone is retained so that the
    # size of each subband is changed to
    # Sr(m,n). Here mxn = d<<axb, d is the
    # number of dc coefficients obtained in
    # each subband.

    Sr = np.zeros((15, E.shape[2], E.shape[3]))
    for i in range(15):
        Sr[i] = dct(Se[i], norm='ortho')

    # 11) The statistical parameters such as mean,
    # standard deviation, covariance, median,
    # energy, skewness and kurtosis are
    # estimated from each subband and is
    # switched to one dimensional vector with
    # size of 1x105 (seven parameters from 15
    # subbands 7x15=105) which forms
    # feature vector Fimg

    Fimg = np.zeros((15, 7))
    for r in range(15):
        Fimg[r, 0] = np.mean(Sr[r])
        Fimg[r, 1] = np.std(Sr[r])
        Fimg[r, 2] = np.var(Sr[r])
        Fimg[r, 3] = np.median(Sr[r])
        Fimg[r, 4] = np.sum(np.square(Sr[r]))
        Fimg[r, 5] = skew(Sr[r], axis=None)
        Fimg[r, 6] = kurtosis(Sr[r], axis=None)

    return Fimg.flatten()

# Function to load the pickled machine learning model


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_emotion(detected_faces_paths):
    # Emotion detection
    class_names = ['happy', 'anger', 'sad', 'fear', 'surprise']

    # Load the pickled models
    pca = load_model('Facial/Emotion recognition models/pca_model.pkl')
    model = load_model('Facial/Emotion recognition models/model85.pkl')

    predictions = []
    for i in range(len(detected_faces_paths)):
        # print(detected_faces_paths[i])
        # read the image containing the detected face
        input_image = cv.imread(
            detected_faces_paths[i])

        # preprocess
        preprocessed_img = preprocess(input_image)
        # show_images([preprocessed_img])

        # Feature extraction
        features = detectemotion(preprocessed_img)

        # normalization
        for i in range(len(features)):
            features[i] = (features[i] - np.mean(features)) / np.std(features)

        # print(features.shape)
        # reducing the features dimensions using PCA

        features = pca.transform([features.flatten()])

        # features=features.reshape(1, -1)

        # Predict the emotion
        prediction = model.predict(np.array(features))
        print(class_names[(int)(model.predict(features))])
        predictions.append(class_names[(int)(model.predict(features))])
    return predictions
###############################################################################################################################


# Sound emotion detection
###############################################################################################################################

# Function to extract feature extraction from the audio sample
def ex_features(X, sample_rate: float) -> np.ndarray:

    # Extract Short-Time Fourier Transform
    # Represents the frequency content of the input signal over time
    # Take the absolute value of the complex STFT coefficients
    stft = np.abs(librosa.stft(X))

    # Estimate the pitch of the stft extracted
    # Fmin and Fmax are the pitch frequencies to search for
    # returns 2 lists pitches and magnitudes.
    # pitches contains the pitch frequencies estimated for each time frame.
    # magnitudes contains the corresponding magnitudes (or strengths) of those pitch estimates.
    pitches, magnitudes = librosa.piptrack(
        y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)

    # Extract the pitch estimates with the highest magnitudes for each time frame
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    # Estimate the tuning offset of the pitch
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    # Mean
    pitchmean = np.mean(pitch)
    # Standard deviation
    pitchstd = np.std(pitch)
    # Max
    pitchmax = np.max(pitch)
    # Min
    pitchmin = np.min(pitch)

    # Spectral centroid of the audio signal
    # Spectral centroid is a measure of the "center of mass" of the power spectrum
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    # Normalize to ensure that the values sum to 1
    cent = cent / np.sum(cent)
    # Mean
    meancent = np.mean(cent)
    # Std
    stdcent = np.std(cent)
    # Max
    maxcent = np.max(cent)

    # Calculate the spectral flatness
    # Spectral flatness is a measure of how "flat" or "peaked" is the power spectrum of the signal
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # Mel-frequency cepstral coefficients (MFCCs) of the audio signal
    # Calculate mean, std and max
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # Calculate chroma features of the audio signal from the Short-time Fourier
    # Represents the distribution of pitch classes (the twelve notes in an octave)
    # chroma list contains the mean value of each of the twelve pitch classes
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)

    # Calculate the Mel-scaled spectrogram feature of the audio signal
    mel = np.mean(librosa.feature.melspectrogram(
        y=X, sr=sample_rate).T, axis=0)

    # calculates the spectral contrast feature of the audio signal from the Short-time Fourier Transform (STFT) coefficients
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)

    # This code block computes the zero-crossing rate feature of the audio signal X using the librosa.feature.zero_crossing_rate function. The zero-c
    # rossing rate is a measure of the number of times that the audio signal crosses the zero-axis per second, and is often used as an indicator of
    # the amount of high-frequency content or noisiness in the signal.

    # Calculates the zero-crossing rate of the audio signal
    # A measure of the number of times that the audio signal crosses the zero-axis per second.
    # used to obtain the amount of high-frequency content or noisiness in the signal.
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    # Calculate magnitude features
    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # RMS of the magnitudes
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    # Concatenate all the single valued features
    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    # Concatenate the single values to the other feature lists
    ext_features = np.concatenate(
        (ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features


# Function to normalizate the audio samples
def normalize_sample(sample, min_list, max_list):

    for i in range(len(sample)):
        sample[i] = sample[i]-min_list[i]/max_list[i]-min_list[i]

    return sample


# Function to read the normalization info of the used dataset
def read_normalization_file():
    with open("Audio/normalization.txt", "r") as file:
        lines = file.readlines()

        min_list = []
        max_list = []

        # Remove leading/trailing whitespaces and cast lines to float32
        # float_list = [float(line) for line in lines]

        for line in lines:
            # Split the line into separate values based on spaces
            values = line.strip().split()
        # print(values[0],values[1])
            min_list.append(float(values[0]))
            max_list.append(float(values[1]))

        return min_list, max_list
###############################################################################################################################


def facial_emotion_thread():
    # File path
    path = 'Facial/WhatsApp Video 2023-05-22 at 10.40.19 PM.mp4'

    # # Extract frames from the video
    frame_paths = video_to_frame(path)
    # print(frame_paths)

    detected_faces, detected_faces_paths = detect_faces(frame_paths)

    # Emotion detection
    predictions = []
    for frame in detected_faces_paths:
        objs = DeepFace.analyze(img_path=frame, enforce_detection=False,
                                actions=['emotion'])

        pred = objs[0]['dominant_emotion']
        if(pred == "sad" or pred == "neutral"):
            pred = "sad"
        elif(pred == "angry" or pred == "disgust"):
            pred = "angry"
        elif(pred == "happy"):
            pred = "happy"
        elif(pred == "fear"):
            pred = "fear"
        elif(pred == "surprise"):
            pred = "surprise"
        predictions.append(pred)

    return predictions


# import nlp_emotion script

def audio_emotion_thread():
    # Audio file path
    audio_sample_path = "Audio/audio_files/Recording(6).wav"

    # Load the audio file
    audio, sr = librosa.load(audio_sample_path, sr=22050)

    # Read normalization file
    min_list, max_list = read_normalization_file()

    # Feature extraction
    features = ex_features(audio, sr)

    # Feature normalization
    features_norm = normalize_sample(features, min_list, max_list)
    features_norm = features_norm.reshape(1, -1)

    # Model path
    model_path = 'Audio/sound_models/svm_model81.6_s+n+c.pkl'

    # Load the machine learning mode
    model = load_model(model_path)

    # Predict
    prediction = model.predict(features_norm)

    print(prediction)
    return prediction[0]


def text_emotion_thread():
    # prediction = nlp.pred("I am afraid that I will fail in the exam")
    prediction = nlp.pred("I am very happy today")
    return prediction


def main():
    # Record the starting time
    start_time = time.time()

    # Create threads to execute the functions
    threads = []

    # Create a shared data structure to store the predictions
    results = []

    # Create a lock to synchronize access to the shared data structure
    lock = threading.Lock()

    def execute_function(func):
        result = func()

        # Acquire the lock to update the shared data structure
        lock.acquire()
        results.append(result)
        lock.release()

    # Create threads for each function
    thread1 = threading.Thread(
        target=execute_function, args=(audio_emotion_thread,))
    thread2 = threading.Thread(
        target=execute_function, args=(facial_emotion_thread,))
    thread3 = threading.Thread(
        target=execute_function, args=(text_emotion_thread,))

    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    # Start the threads
    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for both threads to finish
    for thread in threads:
        thread.join()
    print(results)

    # Record the ending time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    # take the mode of the results
    # results contains the predictions of the three threads  2 items and a list of 3 items
    final_result = []
    for item in results:
        if(type(item) != list):
            final_result.append(item)
        else:
            for i in item:
                final_result.append(i)

    # take the majority vote of the final result
    final_result = max(set(final_result), key=final_result.count)
    print("Final vote is ", final_result)

    print(q.get_emotion_question(final_result))


if __name__ == "__main__":
    main()
