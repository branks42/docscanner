from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# Load the image and compute the ratio of the old height to the new height, clone and resize
image = cv2.imread(args['image'])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# Convert image to grayscale, blur it and find edges
gray = cv2.cvToColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Show original image and edge detected image
print("Edge Detection")
cv2.imshow('Image', image)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find the contours in the edged image, keeponly the largest ones and initialize screen contour
cnts = cv2.findContours(edged.copy(), cv2RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# Loop over the contours
for c in cnts:
	#Approximate the contours
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# If approximate contour has four points, assume screen is found
	if len(approx) == 4:
		screenCnt = approx
		break

# Show the contour (outline) of paper
print("Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply four point transform to get top-down view of image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Convert warped image to grayscale, threshold to give black and white paper look
warped = cv2.cvToColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method='gaussian')
warped = (warped > T).astype('uint8') * 255

# Show original and scanned imaged
print("Apply perspective transform")
cv2.imshow('Original', imutils.resize(orig, height=650))
cv2.imshow('Scanned', imutils.resize(warped, height=650))
cv2.waitKey(0)