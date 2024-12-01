"""
Author: Joseph Allen
Date: November 29, 2024
Explanation of the Code:
    - Blurring: Median blurring is applied to reduce image noise which is crucial for improving the accuracy of edge detection. The kernel size for blurring is adjustable.
    - Grayscale Conversion: The image is converted to grayscale because edge detection typically operates on single-channel images.
    - Edge Detection: The Laplacian operator is used for edge detection, capable of calculating second-order derivatives.
    - Edge Enhancement: After detecting edges, the code calculates an inverse alpha map from the grayscale edge image, which is used to modulate the intensity of the color channels. 
                        This step enhances the visibility of the edges.
    - Channel Processing: The source image's channels are individually adjusted based on the edge map and then recombined.
"""


import cv2 
import numpy as np
import utils 

class VConvolutionFilter(object):
    """
    

    Args:
        object (_type_): _description_
    """
    def __init__(self, kernel):
        self._kernel = kernel 
    def apply(self, src, dst):
        """
        Apply the filter with a BGR or gray source/destination.

        Args:
            src (_type_): _description_
            dst (_type_): _description_
        """
        cv2.filter2D(src, -1, self._kernel, dst)
        
class SharpenFilter(VConvolutionFilter):
    """
    A sharpen filter with a 1-pixel radius.

    Args:
        VConvolutionFilter (_type_): _description_
    """
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)
        
class FindEdgesFilter(VConvolutionFilter):
    """
    An edge-finding filter with a 1-pixel radius.

    Args:
        VConvolutionFilter (_type_): _description_
    """
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)
        
class BlurFilter(VConvolutionFilter):
    """
    A blur filter with a 2-pixel radius.

    Args:
        VConvolutionFilter (_type_): _description_
    """
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)
        
class EmbossFilter(VConvolutionFilter):
    """
    An emboss filter with a 1-pixel radius

    Args:
        VConvolutionFilter (_type_): _description_
    """
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)
        
class BGRFuncFilter(object):
    """
    A filter that applies different functions to each of BGR.

    Args:
        object (_type_): _description_
    """
    def __init__(self, vFunc = None, bFunc = None, gFunc = None, rFunc = None, dtype = np.uint8):
        length = np.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), length)
        
    def apply(self, src, dst):
        """
        Apply the filter with a BGR source/destination.

        Args:
            src (_type_): _description_
            dst (_type_): _description_
        """
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)
        
class BGRCurveFilter(BGRFuncFilter):
    """
    A filter that applies different curves to each of BGR.

    Args:
        BGRFuncFilter (_type_): _description_
    """
    def __init__(self, vPoints = None, bPoints = None,
                 gPoints = None, rPoints = None, dtype = np.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)
        
class BGRPortraCurveFilter(BGRCurveFilter):
    """
    A filter that applies Portra-like curves to BGR.

    Args:
        BGRCurveFilter (_type_): _description_
    """
    def __init__(self, dtype = np.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints = [(0, 0), (23, 20), (157, 173), (255, 255)],
            bPoints = [(0, 0), (41, 46), (231, 228), (255, 255)],
            gPoints = [(0, 0), (52, 47), (189, 196), (255, 255)],
            rPoints = [(0, 0), (69, 69), (213, 218), (255, 255)],
            dtype = dtype
        )

def strokeEdges(src: np.ndarray, dst: np.ndarray, blurKsize: int = 7, edgeKsize: int = 5) -> None:
    """
    Apply an edge detection filter to the input image `src` and store the result in `dst`.
    This function first blurs the image to reduce noise and then applies a Laplacian filter to detect edges.

    Args:
        src (numpy.ndarray): The source image on which edge detections is to be applied.
        dst (numpy.ndarray): The destination image where the result will be stored.
        blurKsize (int, optional): The size of the kernel used for median blurring.  If it's less than 3,
                                   blurring is skipped to avoid errors. Defaults to 7.
        edgeKsize (int, optional): The aperature size used for the Laplacian operator. Defaults to 5.
    """# If the blur kernel size is greater than or equal to 3, apply median blurring to reduce noise.
    if blurKsize >= 3:
        # Median blurring to reduce noise and details in the image
        blurredSrc = cv2.medianBlur(src, blurKsize)
        # Convert the blurred image to grayscale for edge detection
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        # Convert the original image to grayscale if blurring is not applied.
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
    # Apply the Laplacian operator with the specified kernel size to detect edges.  
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
    
    # Calculate the inverse of the normalized grayscale image to highlight edges
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    
    # Split the source image into its individual color channels
    channels = cv2.split(src)
    
    # Multiply each channel by the normalized inverse alpha to enhance edges
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
        
    # Merge the processed channels back into the destination image.  
    cv2.merge(channels, dst)
    
    """
    Note: We allow kernel sizes to be specified as arguments for `strokeEdges`
    """