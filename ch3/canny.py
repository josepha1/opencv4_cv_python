"""
Author: Joseph Allen
Date: December 1, 2024
Description: This file is a simple implementation of the Canny Edge Detection Algorithm.
Canny Algorithm:
    1. Denoise the image with a Gaussian filter.
    2. Calculate the gradients.
    3. Apply non-maximum suppression (NMS) on the edges.  
    4. Apply a double threshold to all the detected edges to eliminate any false positives.
    5. Analyze all the edges and their connection to each other to keep the real edges and discard the weak ones.  
"""


import cv2 
import numpy as np

# Load the image in grayscale from the file
img = cv2.imread('statue_small.jpg', cv2.IMREAD_GRAYSCALE)

# Use the Canny edge detector to find edges in the image
# threshold1 and threshold2 are the lower and upper thresholds for the hysteresis process
cv2.imwrite('canny.jpg', cv2.Canny(image=img, threshold1=200, threshold2=300))

# Display the Canny edge image
cv2.imshow("canny", cv2.imread("canny.jpg"))

# Wait for a key press to close the windows
cv2.waitKey()
cv2.destroyAllWindows()