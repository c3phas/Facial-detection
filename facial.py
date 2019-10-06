#!/usr/bin/python3
import cv2

#This are the classifiers that helps to detect diff parts of the body

#hade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
hade =cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml') 


def detectFace(image):
	#convert the image to a gray scale(cv2 works with gray images)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#use our classifier to detect the region of interest
	faces = hade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5)
	font = cv2.FONT_HERSHEY_SIMPLEX
	return faces,font

def drawFaceRegion(face_area,Text_font):
	
	for (x,y,w,h) in faces:
		#draw a rectangle around the face region
		COLOR = (0,255,0)
		STROKE = 2
		cv2.rectangle(image,(x,y),(x+h,y+h),COLOR,STROKE)
		#write some text on the image detected
		cv2.putText(image,'Cephas',(x,y),font,1,(0,0,255),4)



#define the windows name

cv2.namedWindow('faces found',cv2.WINDOW_NORMAL)
video_capture = cv2.VideoCapture(0)
#passing 0 to videoCapture allows you to use webcam,pass a filename to get the image from a file eg video

#keep the webcam on 

while True:
	ret,image = video_capture.read()
	if not ret:
		break
	faces,font = detectFace(image)
	drawFaceRegion(faces,font)#takes the image as arg

	#TODO
	#implement a machine learning algorithm for the image face recog
	
	cv2.resize(image,(1280,800))	
	cv2.imshow('faces found',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# exit the loop and release all the resources
video_capture.release()
cv2.destroyAllWindows()
