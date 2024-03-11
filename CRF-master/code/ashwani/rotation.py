# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:48:02 2018

@author: ashwa
"""
import numpy as np
import math
from skimage.transform import rotate
import matplotlib.pyplot as plt
#Rotate X by alpha degrees (angle) in a counterclockwise direction around its center point.
#This may enlarge the image.
#So trim the result back to the original size, around its center point.
def rotate_image(X, alpha):
    """
    Rotate image X by alpha degrees in a counterclockwise direction.
    The image is enlarged to fit the entire rotated image and then trimmed back to the original size.
    """
    # Rotate the image
    Y = rotate(X, alpha, resize=True, mode='edge', preserve_range=True)
    
    # Calculate dimensions to trim the rotated image back to the original size
    lenx1, lenx2 = X.shape
    leny1, leny2 = Y.shape
    fromx = math.floor((leny1 + 1 - lenx1) / 2)
    fromy = math.floor((leny2 + 1 - lenx2) / 2)
    
    # Trim the result back to the original size, around its center point
    Y_trimmed = Y[fromx:fromx + lenx1, fromy:fromy + lenx2]
    
    return Y_trimmed

#x=np.arange(10).reshape(2,5)
#print (x)
#print (rotate(x,30))
    

    