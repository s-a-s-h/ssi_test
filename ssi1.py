#model
import time
start_time = time.time()
import os
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
#import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
#import sklearn.multiclass
#from sklearn.multiclass import OneVsRestClassifier
import csv
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#direc = 'neg-samples'
#class sklearn.multiclass.OneVsRestClassifier(estimator, *, n_jobs=None)
def get_landmarks(direc, output, name):
	directory = os.listdir(direc)
	z = 0
	shapes =[]
	for filename in directory:
		z = z+1
		#print(z)
		#if z > 100:
			#break
		#print(filename)
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #shape_predictor_68_face_landmarks.dat

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
			#print(shape)
        	# convert dlib's rectangle to a OpenCV-style bounding box
        	# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        	# show the face number
			cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        	# loop over the (x, y)-coordinates for the facial landmarks
        	# and draw them on the image

			data = []
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
				data.append(x)
				data.append(y)
			#cv2.imshow("Output", image)
			#cv2.waitKey(0)
			# here is where you label/label

			shapes.append(data)
	df = pd.DataFrame(shapes)
	df['category'] = name

	df.to_csv(output) #'./output.csv'
	# do i return output or df?
	return output



def train_val_test_split(dataset):
	df = pd.read_csv(dataset)
	df = df.drop(df.index[0])
	#randomize

  # Returns a tuple of 3 sub-datasets.
  # The first 80% train, next 10% val, last 10% test.
  # do i split it w a csv model, do i use numpy?
  # what is the argument that i feed in, is it output.csv is it
	return np.array_split(df, [int(df.shape[0] * 0.8), int(df.shape[0] * 0.9)])


def log_regression(neg_train, neg_val, neg_test, wh_train, wh_val, wh_test, yn_train, yn_val, yn_test):
	#vstack the trains
	# do i need to randomize?
	train = np.vstack((neg_train, np.vstack((wh_train, yn_train))))
	y_train = train[:, -1] #get_dummies
	train = train[:, :-1]
	#vstack the vals
	val = np.vstack((neg_val, np.vstack((wh_val, yn_val))))
	y_val = val[:, -1]
	val = val[:, :-1]
	#vstack the tests
	test = np.vstack((neg_test, np.vstack((wh_test, yn_test))))
	y_test = test[:, -1]
	test = test[:, :-1]

	model = sklearn.linear_model.LogisticRegression(max_iter = 500, multi_class = 'ovr') #multiclass.OneVsRestClassifer(SVC(kernal = 'linear'))
	model.fit(train, y_train)

	#LinearSVC(random_state=0)

	yhat_train = model.predict(train)
	#print(yhat_train)
	f1_train = sklearn.metrics.f1_score(y_train, yhat_train, average='macro')
	recall_train = sklearn.metrics.recall_score(y_train, yhat_train, average='macro')
	precision_train = sklearn.metrics.precision_score(y_train, yhat_train, average='macro')
	print("this is the f1 score for train{}".format(f1_train))
	print("this is the recall score for train{}".format(recall_train))
	print("this is the precision score for train{}".format(precision_train))

	#yhat_val = model.predict(val)
	#f1_val = sklearn.metrics.f1_score(y_val, yhat_val, average='macro')
	yhat_test = model.predict(test)
	f1_test = sklearn.metrics.f1_score(y_test, yhat_test, average='macro')
	recall_test = sklearn.metrics.recall_score(y_test, yhat_test, average='macro')
	precision_test = sklearn.metrics.precision_score(y_test, yhat_test, average='macro')
	print("this is the f1 score for test{}".format(f1_test))
	print("this is the recall score for test{}".format(recall_test))
	print("this is the precision score for test{}".format(precision_test))
	print(model.coef_)
	print('this is classes')
	print(model.classes_)
	print('this is intercept')
	print(model.intercept_)
	print('this is n inter')
	print(model.n_iter_)
	print("--- %s seconds ---" % (time.time() - start_time))
	#print(pandas.train.columns)

	# find a better penalty
	# predict f1_score
	#use it on test set only once
	# penalty none was taking too long



neg = get_landmarks('neg-samples', './negoutput.csv', 0)
wh = get_landmarks('wh-samples', './whoutput.csv', 1)
yn = get_landmarks('yes_no-samples', './ynoutput.csv', 2)
neg_train, neg_val, neg_test = train_val_test_split(neg)
wh_train, wh_val, wh_test = train_val_test_split(wh)
yn_train, yn_val, yn_test = train_val_test_split(yn)
log_regression(neg_train, neg_val, neg_test, wh_train, wh_val, wh_test, yn_train, yn_val, yn_test)
print('this is iter 500')





#print("this is it {}".format(neg_train[0]))


# test with changing the size to 224 pixels


    #model = sklearn.linear_model.LogisticRegression(penalty='none')
    #model.reshape(1, -1)
    #model.fit(xs, ys)
    #print(model)
    # show the output image with the face detections + facial landmarks
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    ##plt.scatter(shape[0], shape[1]) #broke python
    #print("this is the shape {}".format(shape))
