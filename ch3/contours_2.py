import cv2 
import numpy as np 

img = cv2.pyrDown(cv2.imread(filename = "hammer.jpg", flags = cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(src = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY), thresh = 127, maxval = 255, 
                            type = cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image = thresh, mode = cv2.RETR_EXTERNAL,
                                       method = cv2.CHAIN_APPROX_SIMPLE)

if contours:
    print(f"Found {len(contours)} contours.")
else:
    print("No contours found")

for c in contours:
    # Find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img = img, pt1 = (x, y), pt2 = (x+w, y+h), color = (0, 255, 0), thickness = 2)
    
    # Find minimum area rectangle
    rect = cv2.minAreaRect(c)
    # Calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # Normalize the coordinates to integers
    box = np.int64(box)
    # Draw the contours
    cv2.drawContours(image = img, contours = [box], contourIdx = 0, color = (0, 0, 255), thickness = 3)
    
    # Calculate the center and the radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # Cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # Draw the circle
    image = cv2.circle(img = img, center = center, radius = radius, color = (0, 255, 0), thickness = 2)
    
cv2.drawContours(image = img, contours = contours, contourIdx = -1, color = (255, 0, 0), thickness = 1)
cv2.imshow(winname = "Contours", mat = img)
cv2.waitKey()
cv2.destroyAllWindows()