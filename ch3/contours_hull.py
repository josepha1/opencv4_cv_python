import cv2
import numpy as np

# Load in a downsampled image of the lineup of cucumbers
img = cv2.pyrDown(cv2.imread(filename = "/Users/joseph.allen11/Documents/Projects/opencv4_cv_python/ch3/hammer.jpg", flags = cv2.IMREAD_UNCHANGED))
# Create a binary thresholded image
ret, thresh = cv2.threshold(src = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY), thresh = 127, maxval = 255, 
                            type = cv2.THRESH_BINARY)

contours, hierarch = cv2.findContours(image = thresh, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

black = np.zeros_like(a = img)
for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(curve = cnt, closed = True)
    approx = cv2.approxPolyDP(curve = cnt, epsilon = epsilon, closed = True)
    hull = cv2.convexHull(points = cnt)
    cv2.drawContours(image = black, contours = [cnt], contourIdx = -1, color = (0, 255, 0), thickness = 2)
    cv2.drawContours(image = black, contours = [approx], contourIdx = -1, color = (255, 255, 0), thickness = 2)
    cv2.drawContours(image = black, contours = [hull], contourIdx = -1, color = (0, 0, 255), thickness = 2)

cv2.imshow(winname = "Hull", mat = black)
cv2.waitKey()
cv2.destroyAllWindows()
