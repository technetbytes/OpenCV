import numpy as np
import cv2

#open data folder path
folder_path = "/home/saqib/anaconda3/share/OpenCV/haarcascades/"

#set cascade classifier xml files
frontalface_classifier = 'haarcascade_frontalface_default.xml'
eye_classifier = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(folder_path + frontalface_classifier)
eye_cascade = cv2.CascadeClassifier(folder_path + eye_classifier)

input_name = "close_up.jpg"
output_name = "classifier_output.jpg"

#read image
img = cv2.imread(input_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#apply both classifier
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,155,0),12)

#write image
cv2.imwrite(output_name,img)