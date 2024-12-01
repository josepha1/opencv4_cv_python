import cv2 
import numpy as np
import time 

class CaptureManager(object):
    """
    Manage the capture process from a video source, allowing for frame capture, preview management, and recording.
    
    This class encapsulates the operations related to video capture including managing preview windows, mirroring previews,
    capturing frames, saving snapshots, and recording video clips.

    Attributes:
        previewWindowManager (Any): Optional manager for any preview windows.
        shouldMirrorPreview (bool): Flag to indicate if the preview should be mirrored.
        _capture (cv2.VideoCapture): The video capture object.
        _channel (int): The channel of the video capture device.
        _enteredFrame (bool): Flag to check if the frame has been entered.
        _frame (np.ndarray): The current video frame.
        _imageFilename (str): Filename for saving snapshots.
        _videoFilename (str): Filename for saving video.
        _videoEncoding (tuple): Video encoding format.
        _videoWriter (cv2.VideoWriter): Video writer object for recording.
        _startTime (float): Start time of the recording.  
        _framesElapsed (int): Count of frames that have elapsed.
        _fpsEstimate (float) Estimated frame per second in the video.
    """
    def __init__(self, capture: cv2.VideoCapture, previewWindowManager = None,
                 shouldMirrorPreview: bool = False):
        """
        Initializes the CaptureManagers object with the video capture device, optional preview window manager,
        and a flag to determine if the preview should be mirrored.

        Args:
            capture (cv2.VideoCapture): The video capture device.
            previewWindowManager (Any, optional): Manger for any preview windows. Defaults to None.
            shouldMirrorPreview (bool, optional): If set to True, the preview display will be mirrored. Defaults to False.
        """
        self.previewWindowManager = previewWindowManager 
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture 
        self._channel = 0
        self._enteredFrame = False 
        self._frame = None
        self._imageFilename = None 
        self._videoFilename = None 
        self._videoEncoding = None 
        self._videoWriter = None 
        self._startTime = None 
        self._framesElapsed = 0
        self._fpsEstimate = None 
        
    @property 
    def channel(self) -> int:
        """
        Get the current channel of the video capture device
        
        Returns:
            int: The current channel number.
        """
        return self._channel  
    
    @channel.setter 
    def channel(self, value: int):
        """
        Sets the channel of the video capture device to a new value.

        Args:
            value (int): The new channel numbers
        """
        if self._channel != value:
            self._channel = value 
            self._frame = None 
            
    @property 
    def frame(self) -> np.ndarray:
        """
        Retrieves the current frame from the capture device.  
        
        Returns:
            np.ndarray: The current video frame.
        """
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(self._frame, self.channel)
            
        return self._frame 
    
    @property 
    def isWritingImage(self) -> bool:
        """
        Check if an image file is currently being written
        
        Returns:
            bool: True if an image file is being written, otherwise False
        """
        return self._imageFilename is not None 
    
    @property 
    def isWritingVideo(self) -> bool:
        """
        Check if a video file is currently being recorded
        
        Returns:
            bool: True if a video file is being recorded, otherwise False
        """
        return self._videoFilename is not None
    
    def enterFrame(self):
        """
        Enter a new frame to be captured.
        
        This method must be called before attempting to retrieve the next frame with the 'frame' property.
        It checks to ensure that the last frame was properly exited, grabs the next frame if available,
        and raises an AssertionError if the previous frame was not properly exited.
        """
        # Assert that we did not already enter a frame without exiting.
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        # Attempt to grab the next frame if the capture device is initialized.
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()
            
    def exitFrame(self):
        """
        Process and release the current frame.
        
        This method draws the frame to the window if a window manager is present, writes the frame
        to an image file if a snapshot is requested, records the frame to a video if recording is active,
        and then releases the frame to process the next one.  It also calculates and updates the frame
        rate estimate.
        """
        # Check if the frame was grabbed successfully. If not, reset the entered frame flag and return.
        if self.frame is None:
            self._enteredFrame = False 
            return 
        
        # Calculate and update the frame rate estimate.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime 
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1
        
        # Display the frame in the window, mirroring it if necessary.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame)
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)
                
        # Save the frame as an image file if requested.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None 
            
        # Record the frame to a video file if recording is active.
        self._writeVideoFrame()
        
        # Release the current frame to prepare for the next one.
        self._frame = None 
        self._enteredFrame = False  
        
    def writeImage(self, filename: str) -> None:
        """
        Write the next exited frame to an image file.

        Args:
            filename (str): Filename of the image.
        """
        self._imageFilename = filename 
        
    def startWritingVideo(self, filename: str, encoding = cv2.VideoWriter_fourcc('M','J','P','G')):
        """
        Start writing exited frames to a video file.

        Args:
            filename (str): Filename of the video.
            encoding (_type_, optional): _description_. Defaults to cv2.VideoWriter_fourcc('M','J','P','G').
        """
        self._videoFilename = filename 
        self._videoEncoding = encoding 
        
    def stopWritingVideo(self) -> None:
        """
        Stop writing exited frames to a video file.
        """
        self._videoFilename = None 
        self._videoEncoding = None 
        self._videoWriter = None 
        
    def _writeVideoFrame(self):
        """
        Writes the current frame to the video file, initializing the video writer if necessary.

        This method checks if the video writer has been initialized and if not, initializes it with
        the estimated frame rate if the actual frame rate is not known. It skips writing the initial frames
        if the frame rate is still being estimated to ensure the stability of the video output.
        """
        if not self.isWritingVideo:
            return 
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the 
                    # estimate is more stable.
                    return 
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding, fps, size
            )
        self._videoWriter.write(self._frame)
        
class WindowManager(object):
    """
    Manages a named window in OpenCV, handling the creation, display, and destruction of the window.
    It processes keyboard and mouse events by invoking callback functions when these events occur.  

    Attributes:
        windowName (str): The name of the window as displayed in the title.
        keypressCallback (Callable[int], None], optional): A function to call when a key event occurs.
            This function should take a single integer parameter (the keycode of the pressed key).
        mouseCallback (Callable[[int, int, int, int, Any], None], optional): A function to call when a mouse event occurs.
    """
    def __init__(self, windowName: str, keypressCallback = None, mouseCallback = None):
        """
        Initializes the WindowManager with a name and an optional key press callback function

        Args:
            windowName (str): The name of the window to be created and managed.
            keypressCallback (Callable[[int], None], optional): A callback function that is called on a key press. 
                If None, no callback is executed.  Defaults to None.
            mouseCallback (Callable[[int, int, int, int, Any], None], optional): A callback function that is called on mouse events.
        """
        self.keypressCallback = keypressCallback
        self.mouseCallback = mouseCallback
        self._windowName = windowName 
        self._isWindowCreated = False 
        
    @property 
    def isWindowCreated(self) -> bool:
        """
        Returns whether the window has been created.  
        
        Returns:    
            bool: True if the window is created, False otherwise.  
        """
        return self._isWindowCreated
    
    def createWindow(self) -> None:
        """
        Creates a window using the stored window name if it hasn't been created already and sets the mouse callback
        if one is provided.
        
        This method also sets `_isWindowCreated` to True to indicate that the window now exists.
        """
        if not self._isWindowCreated:   # Ensure that we don't recreate the window if it already exists.
            cv2.namedWindow(self._windowName)
            if self.mouseCallback is not None:
                cv2.setMouseCallback(self._windowName, self.mouseCallback)
            self._isWindowCreated = True 
        
    def show(self, frame) -> None:
        """
        Displays an image or video frame in the window.

        Args:
            frame (np.ndarray): The frame (image or video frame) to be displayed.
        """
        if self._isWindowCreated:
            cv2.imshow(self._windowName, frame)
        
    def destroyWindow(self) -> None:
        """
        Destroys the window and updates the state to reflect that the window no longer exists.
        
        This method should be called to properly close the window and release resources.  
        """
        if self._isWindowCreated:
            cv2.destroyWindow(self._windowName)
            self._isWindowCreated = False
        
    def processEvents(self) -> None:
        """
        Processes any events, such as keyboard inputs, that occur in the window.
        
        If a keypress callback is set, it will call this function with the keycode of any key that is pressed.
        The method handles a single keypress event per invocation.
        """
        keycode = cv2.waitKey(1) # Wait for 1 ms for a key press
        if self.keypressCallback is not None and keycode != -1:
            # Call the callback function if a key was pressed and a callback exists.
            self.keypressCallback(keycode)