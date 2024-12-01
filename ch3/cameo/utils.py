import cv2 
import numpy as np
import scipy.interpolate 

def createCurveFunc(points):
    """
    Return a function derived from control points.

    Args:
        points (numpy.ndarray): A Numpy array of (x,y) pairs.
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

def createLookupArray(func, length:int = 256):
    """
    Return a lookup for whole-number inputs to a function
    
    The lookup values are clamped to [0, length - 1]

    Args:
        func (_type_): _description_
        length (int, optional): _description_. Defaults to 256.
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

def applyLookupArray(lookupArray, src, dst):
    """
    Map a source to a destination using a lookup.

    Args:
        lookupArray (_type_): _description_
        src (_type_): _description_
        dst (_type_): _description_
    """
    if lookupArray is None:
        return 
    dst[:] = lookupArray[src]

def createCompositeFunc(func0, func1):
    """
    Return a composite of two functions

    Args:
        func0 (_type_): _description_
        func1 (_type_): _description_
    """
    if func0 is None:
        return func1 
    if func1 is None:
        return func0 
    return lambda x: func0(func1(x))