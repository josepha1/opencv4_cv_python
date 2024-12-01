import cv2
import numpy as np
from scipy import ndimage

# Define a 3x3 kernel for basic edge detection
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

# Define a 5x5 kernel for a more complex edge detection that includes more surrounding  pixels
# This kernel is often used for detecting finer edges as it takes a larger neighborhood into account
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])

# Load an image in grayscale
img = cv2.imread("statue_small.jpg", 0)

print(f"Shape of 3x3 Kernel: {kernel_3x3.shape}")
print(f"Shape of 5x5 Kernel: {kernel_5x5.shape}")
print(f"Shape of input image: {img.shape}")

# Apply the 3x3 kernel to the image using convolution to highlight edges
k3 = ndimage.convolve(img, kernel_3x3)
# Apply the 5x5 kernel to the image using convolution to highlight finer edges
k5 = ndimage.convolve(img, kernel_5x5)

# Apply Gaussian blurring to smooth the image, which helps in reducing image noise and details
blurred = cv2.GaussianBlur(img, (17,17), 0)

# Subtract the blurred image from the original image to get a high pass filtered image
# This process enhances edges by subtracting the low-frequency areas (smoothed by Gaussian blurring)
g_hpf = img - blurred 

# Display the original and processed images in separate windows to compare effects
cv2.imshow("original image", img)
cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("blurred", blurred)
cv2.imshow("g_hpf", g_hpf)

# Wait indefinitely until a key is pressed to close all windows.  
cv2.waitKey()
cv2.destroyAllWindows()