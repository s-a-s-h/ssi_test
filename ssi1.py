#model
import os
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
#import matplotlib.pyplot as plt
import pandas
import sklearn.linear_model

#direc = 'neg-samples'


def get_landmarks(direc):
	directory = os.listdir(direc)
	i = 0
	shapes =[]
	for filename in directory:
		i += 1
		if i > 10:
			break
        #print(filename)
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor() #shape_predictor_68_face_landmarks.dat

        # load the input image, resize it, and convert it to grayscale
		image_path = direc + "/" + filename
		image = cv2.imread(image_path)
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
		rects = detector(gray, 1)

        # loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, the
			# convert the facial landmark (x, y)-coordinates to a NumPy
        	# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
        	# convert dlib's rectangle to a OpenCV-style bounding box
        	# [i.e., (x, y, w, h)], then draw the face bounding box
			#(x, y, w, h) = face_utils.rect_to_bb(rect)
			# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        	# show the face number
			#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        	# loop over the (x, y)-coordinates for the facial landmarks
        	# and draw them on the image
			#cv2.imshow("Output", image)
			#cv2.waitKey(0)
			data = []
			for (x, y) in shape:
			# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
				data.append(x)
				data.append(y)
			# here is where you label/label

			shapes.append(data)
	df = pandas.DataFrame(shapes)
	df.to_csv('./output.csv')



get_landmarks('neg-samples')


    #model = sklearn.linear_model.LogisticRegression(penalty='none')
    #model.reshape(1, -1)
    #model.fit(xs, ys)
    #print(model)
    # show the output image with the face detections + facial landmarks
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    ##plt.scatter(shape[0], shape[1]) #broke python
    #print("this is the shape {}".format(shape))

    # df = pandas.DataFrame(shape)
    # df.to_csv('./output.csv')
