import cv2 
import numpy as np

# Load the image from the specified path in unchanged format (including alpha channel if present)
img = cv2.imread(filename = "ch3/lines.jpg", flags = cv2.IMREAD_UNCHANGED)

# Convert the image from BGR color space to grayscale
# This is required for edge detection
gray = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
# threshold1: first threshold for the hysteresis procedure
# threshold2: second threshold for the hysteresis procedure
edges = cv2.Canny(image = gray, threshold1 = 50, threshold2 = 120)

# Set minimum line length - lines shorter than this are rejected
minLineLength = 20
# Set maximum gap between line segments - gaps larger than this are split into separate lines
maxLineGap = 5

# Apply probabilistic Hough transform to detect lines
# rho: distance resolution of the accumulator in pixels (1 pixel)
# theta: angle resolution of the accumulator in radians (1 degree)
# threshold: minimum number of votes needed to accept a line
# Returns an array of lines where each line is represented by (x1,y1,x2,y2) coordinates
lines = cv2.HoughLinesP(image = edges, rho = 1, theta = np.pi/180.0, threshold = 50, 
                       minLineLength = minLineLength, maxLineGap = maxLineGap)

# Draw detected lines on the original image
# Iterate through each line detected by Hough transform
for line in lines:
    for x1, y1, x2, y2 in line:
        # Draw a green line (0,255,0) with thickness 2 pixels
        cv2.line(img = img, pt1 = (x1, y1), pt2 = (x2, y2), color = (0, 255, 0), thickness = 2)

# Display the edge detection result    
cv2.imshow(winname = "Edges", mat = edges)
# Display the original image with detected lines drawn
cv2.imshow(winname = "Lines", mat = img)

# Wait for a key press
cv2.waitKey()
# Close all OpenCV windows
cv2.destroyAllWindows()