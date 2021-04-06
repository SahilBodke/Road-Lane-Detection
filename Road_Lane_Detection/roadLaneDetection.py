import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def drawLines(img, lines):
    copyImg = np.copy(img)
    blank = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(blank, (x1, y1), (x2, y2), (0,255,0), thickness=17)

    # Merge the two images
    img = cv.addWeighted(img, 0.8, blank, 1, 0.0)
    return img

def regionOfInterest(img, vertices):
    mask = np.zeros_like(img)    # Completely black image
    # noOfChannels = img.shape[2]
    matchMaskColor = 255
    cv.fillConvexPoly(mask, vertices, matchMaskColor)  # Fill colors only in the region of interest
    maskedImg = cv.bitwise_and(img, mask)
    return maskedImg

def process(img):

    height = img.shape[0]
    width = img.shape[1]
    regionOfInterestVertices = [
        (0, width),
        (width/4, height/2),
        (width/5, height/2),
        (width/2, height/2),
        (width, height)
    ]

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 1)
    gray = cv.dilate(gray, (3,3), iterations=2)
    canny = cv.Canny(gray, 100, 200)

    croppedRoi = regionOfInterest(canny, np.array([regionOfInterestVertices], np.int32))

    lines = cv.HoughLinesP(croppedRoi, 6, np.pi/60, 160, lines = np.array([]), minLineLength=40, maxLineGap=25)
    laneDetectedImg = drawLines(img, lines)
    return laneDetectedImg


capture = cv.VideoCapture(r"C:\Users\sahil\A_MEPy\videos\Road_Video.mp4")

while True:

    isTrue, frame = capture.read()  # Captures frame by frame
    frame = process(frame)
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()