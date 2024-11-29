import cv2  

# Construct a VideoCapture objecgt by passing the 
# camera's device index instead of a video's
# filename
cameraCapture = cv2.VideoCapture(0)

# Retrieve the frames per second (fps) from the camera.
# This value is used to configure the output video so that its playback
# speed matches the original video's speed.
fps = 30    # An assumption

# Retrieve the width and height of the video frames.
# These values are used to configure the output video to match the
# resolution of the original video.
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Initialize a VideoWriter object to write frames to a video file.
# 'MyOutputVid2.avi' specifies the output file name.
# The `cv2.VideoWriter_fourcc('I', '4', '2', '0')` specifies the codec
# used to compress frames. This particular codec ('I420') is widely supported and provides good quality.
# `fps` and `size` are passed to ensure the output video has the same frame rate
# and frame size as the input video.
videoWriter = cv2.VideoWriter(
    'MyOutputVid2.avi', cv2.VideoWriter_fourcc('I','4','2','0'),
    fps, size)

# Read the first frame from the camera
success, frame = cameraCapture.read()

# Get the number of frame remaining
numFramesRemaining = 10 * fps - 1   # 10 seconds of frames

# Loop over all frames in the video.
# The loop continues as long as `videoCapture.read()` successfully reads a frame
# and the number of frames is greater than 0
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
    
# Cleanup: Release the cameraCapture and videoWriter objects to free up their resources
cameraCapture.release()
videoWriter.release()