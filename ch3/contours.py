import cv2 
import numpy as np 

img = np.zeros(shape = (200, 200), dtype = np.uint8)
img[50:150, 50:150] = 255

ret, thresh = cv2.threshold(src = img, thresh = 127, maxval = 255, type = 0)
contours, hierarchy = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, 
                                       method = cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(src = img, code = cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(image = color, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2)
cv2.imshow(winname = "Contours", mat = color)
cv2.waitKey()
cv2.destroyAllWindows()