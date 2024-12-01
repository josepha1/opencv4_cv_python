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

class VConvolutionFilter:
    """
    A generic convolution filter class using a kernel for image filtering.

    Attributes:
        kernel (np.ndarray): The convolution kernel.
    """
    def __init__(self, kernel: np.ndarray):
        self._kernel = kernel

    def apply(self, src: np.ndarray, dst: np.ndarray) -> None:
        """
        Apply the convolution filter to an image.

        Args:
            src (np.ndarray): The source image.
            dst (np.ndarray): The destination image.
        """
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
    """
    A sharpen filter using a specific sharpening kernel.
    """
    def __init__(self):
        super().__init__(kernel=np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]]))

class FindEdgesFilter(VConvolutionFilter):
    """
    An edge detection filter using a Laplacian-like kernel.
    """
    def __init__(self):
        super().__init__(kernel=np.array([[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]]))

class BlurFilter(VConvolutionFilter):
    """
    A simple blur filter using an averaging kernel.
    """
    def __init__(self):
        super().__init__(kernel=np.full((5, 5), 0.04))

class EmbossFilter(VConvolutionFilter):
    """
    An emboss filter that simulates an embossed look by using a gradient kernel.
    """
    def __init__(self):
        super().__init__(kernel=np.array([[-2, -1, 0],
                                          [-1, 1, 1],
                                          [0, 1, 2]]))

class BGRFuncFilter:
    """
    A filter that applies different functions to each channel of a BGR image.

    Attributes:
        bFunc, gFunc, rFunc (callable): Functions to be applied to the B, G, R channels respectively.
    """
    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src: np.ndarray, dst: np.ndarray) -> None:
        """
        Apply the configured functions to the BGR channels of the source image and store in dst.

        Args:
            src (np.ndarray): The source BGR image.
            dst (np.ndarray): The destination BGR image, modified in place.
        """
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

class BGRCurveFilter(BGRFuncFilter):
    """
    A filter that applies different curves to each channel of a BGR image based on specified control points.
    """
    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=np.uint8):
        super().__init__(vFunc=utils.createCurveFunc(vPoints),
                         bFunc=utils.createCurveFunc(bPoints),
                         gFunc=utils.createCurveFunc(gPoints),
                         rFunc=utils.createCurveFunc(rPoints), dtype=dtype)

class BGRPortraCurveFilter(BGRCurveFilter):
    """
    A filter that applies specific curves to each channel of a BGR image to emulate the look of Portra film.
    """
    def __init__(self, dtype=np.uint8):
        super().__init__(vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
                         bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
                         gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
                         rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)], dtype=dtype)

def strokeEdges(src: np.ndarray, dst: np.ndarray, blurKsize: int = 7, edgeKsize: int = 5) -> None:
    """
    Apply an edge detection filter to the input image `src` and store the result in `dst`.
    This function first blurs the image to reduce noise and then applies a Laplacian filter to detect edges.

    Args:
        src (np.ndarray): The source image on which edge detections is to be applied.
        dst (np.ndarray): The destination image where the result will be stored.
        blurKsize (int, optional): The size of the kernel used for median blurring.  If it's less than 3,
                                   blurring is skipped to avoid errors. Defaults to 7.
        edgeKsize (int, optional): The aperture size used for the Laplacian operator. Defaults to 5.
    """
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    
    channels = cv2.split(src)
    
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
        
    cv2.merge(channels, dst)