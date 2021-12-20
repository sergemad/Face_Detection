import streamlit as st
import streamlit.components.v1 as components
import cv2 as cv
import logging as log
import datetime as dt
from time import sleep

cascPath ="haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv.VideoCapture(0)
anterior = 0

st.title("Face detection app")

imageLocation = st.empty()

if st.button("activate camera"):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera')
            sleep(5)
            pass

        # Capture frame by frame 

        ret, frame = video_capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= 1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame,(x, y), (x+w, y+h), (0,225,225), 2)
        
        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(anterior) + " at " + str(dt.datetime.now()))
        
        imageLocation.image(frame,"RGB")

        if cv.waitKey(1) == ord('q'):
            break

        #st.image(frame,"BGR")
    
video_capture.release()
cv.destroyAllWindows()

