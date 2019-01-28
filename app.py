#Importing necessary libraries
import cv2
import os 
import numpy
import sys
import string
sys.path.append(os.path.abspath('./model'))

#Loading our trained model
from load import *
global graph,model
model, graph = init()

#Alphabets dictionary
dic = list(string.ascii_lowercase)
img = cv2.imread('open_cv_0.png',cv2.IMREAD_GRAYSCALE)

# Our predict function 
def predict_(imgData):
	x = imgData
	x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	x = cv2.resize(x, (28, 28))
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(dic[np.argmax(out,axis=1)[0]])
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	

#using webcam for input image.
cap = cv2.VideoCapture(0)
cv2.startWindowThread()

while True:
	_,frame = cap.read()
	# Our region of interest
	x_left,x_right = (0,400)
	y_left,y_right = (0,400)
	roi = frame[y_left:y_right,x_left:x_right]

	# Adding a border to our frame
	cv2.rectangle(frame,(x_left,y_left),(x_right,y_right),(0,255,0),0)
	cv2.imshow("test",frame)

	# Now our skin extraction code
	# Define range of skin in HSV
	lower_skin = np.array([0,10,60],dtype=np.uint8 )
	upper_skin = np.array([20,150,180],dtype=np.uint8 )
	# extract skin
	roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
	# Define the kernel - the size of the window used to fill the dark 
	# spots of to remove the white noise
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	# extrapolate the hand to fill dark spots within
	# mask = cv2.erode(mask, kernel, iterations = 2)
	mask = cv2.dilate(mask, kernel, iterations = 2)

	# blur the image
	mask = cv2.GaussianBlur(mask,(3, 3), 0) 
	skin = cv2.bitwise_and(roi, roi, mask = mask)
	# cv2.imshow('mask',mask)
	predict_(skin)

	cv2.imshow('skin',skin)
	# Terminates the program on pressing 'g'
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
