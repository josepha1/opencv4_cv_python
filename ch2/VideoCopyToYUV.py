import cv2

# Initialize a VideoCapture object to read from a video file.
# The parameter 'MyInputVid.avi' specifies the file name of the input video.
videoCapture = cv2.VideoCapture('MyInputVid.avi')

# Retrieve the frames per second (fps) from the video file.
# This value is used to configure the output video so that its playback
# speed matches the original video's speed.
fps = videoCapture.get(cv2.CAP_PROP_FPS)

# Retrieve the width and height of the video frames.
# These values are used to configure the output video to match the
# resolution of the original video.
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Initialize a VideoWriter object to write frames to a video file.
# 'MyOutputVid.avi' specifies the output file name.
# The `cv2.VideoWriter_fourcc('I', '4', '2', '0')` specifies the codec
# used to compress frames. This particular codec ('I420') is widely supported and provides good quality.
# `fps` and `size` are passed to ensure the output video has the same frame rate
# and frame size as the input video.
videoWriter = cv2.VideoWriter(
    'MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'),
    fps, size)

# Read the first frame from the video.
success, frame = videoCapture.read()

# Loop over all frames in the video.
# The loop continues as long as `videoCapture.read()` successfully reads a frame.
while success:
    # Write the current frame to the output video file.
    videoWriter.write(frame)
    # Read the next frame from the video.
    success, frame = videoCapture.read()

# Clean up: release the VideoCapture and VideoWriter objects to free their resources.
videoCapture.release()
videoWriter.release()
