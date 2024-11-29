import cv2

# Initialize a boolean variable to indicate whether 
# the window has been clicked or not.
clicked = False  

def onMouse(event: int, x: int, y: int, flags: int, param) -> None:
    """
    Handle mouse events in the OpenCV window.

    This function sets the global variable `clicked` to True when the left mouse button is released,
    allowing the program to detect this user interaction.

    Args:
        event (int): The type of the mouse event (e.g., left button up, left button down).
        x (int): The x-coordinate of the event.
        y (int): The y-coordinate of the event.
        flags (int): Any relevant flags passed by OpenCV. This parameter is not used in this function.
        param: Additional parameters passed by OpenCV. This parameter is not used in this function.
    """
    global clicked 
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True
    
# Initialize a camera capture object to capture video from the first connected camera.
cameraCapture = cv2.VideoCapture(0)
if not cameraCapture.isOpened():
    raise IOError("Cannot open webcam")

# Create an OpenCV window to display the video frames.
cv2.namedWindow('MyWindow')

# Set the mouse callback function which will be called on a mouse event within the 'MyWindow'.
cv2.setMouseCallback('MyWindow', onMouse)

print(f"Showing camera feed. Click window or press any key to stop.")
# Read the first frame from the video.
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    # Display the current frame in the 'MyWindow'.
    cv2.imshow('MyWindow', frame)
    # Read the next frame from the camera.
    success, frame = cameraCapture.read()
    
# Clean up: close the window and release the camera.
cv2.destroyWindow('MyWindow')
cameraCapture.release()
