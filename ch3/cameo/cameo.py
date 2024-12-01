import cv2 
import filters
from managers import WindowManager, CaptureManager 

class Cameo(object):
    """
    A simple application class for managing video capture, processing frames, and interacting
    with the user through key presses.

    This class integrates the WindowManager for displaying video and handling user input, and
    the CaptureManager for managing the video capture process.
    """
    def __init__(self):
        """
        Initialize the Cameo application with the necessary managers for window and capture management.
        """
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.BGRPortraCurveFilter()
        
    def run(self) -> None:
        """
        Start the main loop of the application.

        This method continuously captures frames from the video source, processes them (currently no
        processing is done, but this is where you would insert frame filtering or detection logic),
        and handles user inputs until the window is closed.
        """
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()   # Start capturing a frame
            frame = self._captureManager.frame  # Retrieve the current frame
            if frame is not None:
                filters.strokeEdges(src = frame, dst = frame)
                self._curveFilter.apply(src = frame, dst = frame)
            self._captureManager.exitFrame()    # Finish capturing the frame and handle any outputs
            self._windowManager.processEvents() # Handle any window events, such as key presses
            
    def onKeypress(self, keycode: int) -> None:
        """
        Respond to key presses captured by the WindowManager.

        Args:
            keycode (int): The code of the pressed key, which determines the action to take.

        Actions:
            space (32): Take a screenshot.
            tab (9): Start/Stop recording a video.
            escape (27): Quit the application.
        """
        if keycode == 32:  # space key
            self._captureManager.writeImage("screenshot.png")  # Save the current frame as an image
        elif keycode == 9:  # tab key
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("screencast.avi")  # Start recording a video
            else:
                self._captureManager.stopWritingVideo()  # Stop recording
        elif keycode == 27:  # escape key
            self._windowManager.destroyWindow()  # Close the application window
            
if __name__ == "__main__":
    Cameo().run()   # Start the application