import cv2 as cv
import time
import pandas as pd
from datetime import datetime


firstFrame = None
capture = cv.VideoCapture(r"C:\Users\sahil\A_MEPy\videos\Cars2.mp4")

while True:

    isTrue, frame = capture.read()
    status = 0  # Initially there is no object
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    # Store the first frame of the video
    if firstFrame is None:
        firstFrame = gray
        continue

    # Calculate difference between 1st frame and subsequent frames
    changeInFrame = cv.absdiff(firstFrame, gray)
    changeInFrame = cv.GaussianBlur(changeInFrame, (5, 5), 0)
    # Apply threshold value of 30 pixels (<30 = black, >30 = white)
    threshDelta = cv.threshold(changeInFrame, 30, 255, cv.THRESH_BINARY)[1]
    threshDelta = cv.dilate(threshDelta, None, iterations=3)

    # Find contour of moving object
    contours, hierarchies = cv.findContours(threshDelta.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Check the area of each of the contours detected
        if cv.contourArea(contour) < 700:
            continue
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), thickness=2)

    cv.imshow("Frame", frame)
    cv.imshow("Capturing", gray)
    cv.imshow("Delta", changeInFrame)
    cv.imshow("Threshold", threshDelta)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

