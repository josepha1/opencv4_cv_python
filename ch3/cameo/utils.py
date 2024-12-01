import cv2 
import numpy as np
import scipy.interpolate 

def createCurveFunc(points: list) -> callable:
    """
    Create a function from a list of points suitable for interpolation.

    Args:
        points (list): List of tuples (x, y) representing points.
        
    Returns:
        callable: An interpolated function derived from the points.
    """
    if points is None:
        return None 
    numPoints = len(points)
    if numPoints < 2:
        return None 
    xs, ys = zip(*points)
    if numPoints < 3:
        kind = 'linear'
    elif numPoints < 4:
        kind = 'quadratic'
    else:
        kind = 'cubic'
        
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error = False)

def createLookupArray(func: callable, length:int = 256) -> np.ndarray:
    """
    Create a lookup array for a function over the range of integer inputs.

    Args:
        func (callable): Function to be applied.
        length (int): Length of the lookup array. Defaults to 256

    Returns:
        np.ndarray: Array containing the lookup values
    """
    if func is None:
        return None 
    lookupArray = np.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookupArray

def applyLookupArray(lookupArray: np.ndarray, src: np.ndarray, dst: np.ndarray) -> None:
    """
    Apply a lookup array to a source image to create a destination image.

    Args:
        lookupArray (np.ndarray): The lookup array.
        src (np.ndarray): The source image.
        dst (np.ndarray): The destination image, modified in place.
    """
    if lookupArray is None:
        return 
    dst[:] = lookupArray[src]

def createCompositeFunc(func0: callable, func1: callable) -> callable:
    """
    Combine two functions into a single composite function.

    Args:
        func0 (callable): First function.
        func1 (callable): Second function.

    Returns:
        callable: A composite function that applies func1 and then func0.
    """
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))