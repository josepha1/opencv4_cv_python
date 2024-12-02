import cv2 
import numpy as np

# Load the image of planets from file
planets = cv2.imread(filename = "ch3/planet_glow.jpg")

# Convert the image to grayscale since HoughCircles requires a single channel image
gray_img = cv2.cvtColor(src = planets, code = cv2.COLOR_BGR2GRAY)

# Apply median blur to reduce noise while preserving edges
# Kernel size of 5 means a 5x5 pixel window is used for filtering
gray_img = cv2.medianBlur(gray_img, 5)

# Detect circles using Hough Circle Transform
# Parameters:
# - cv2.HOUGH_GRADIENT: Detection method
# - dp=1: Inverse ratio of accumulator resolution to image resolution
# - minDist=120: Minimum distance between circle centers
# - param1=100: Upper threshold for Canny edge detector
# - param2=30: Threshold for center detection (lower means more false circles)
# - minRadius=0, maxRadius=0: Min/max circle radius (0 means no limits)
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 120,
                           param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)

# Convert circle parameters to integers
# np.around rounds the floating point values
# np.uint16 converts to 16-bit unsigned integers
circles = np.uint16(np.around(circles))

# Draw detected circles on the original image
for i in circles[0, :]:
    # Draw the outer circle in green
    # i[0], i[1] are x,y coordinates of center
    # i[2] is the radius
    # (0, 255, 0) is BGR color for green
    # 2 is the thickness of the circle line
    cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
    
    # Draw a small red circle at the center
    # 2 is the radius of center point
    # (0, 0, 255) is BGR color for red
    # 3 is the thickness
    cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

# Save the result to a new image file
cv2.imwrite("Planets_Circles.jpg", planets)

# Display the result in a window
cv2.imshow(winname = "HoughCircles", mat = planets)

# Wait for a key press
cv2.waitKey()

# Clean up all OpenCV windows
cv2.destroyAllWindows()