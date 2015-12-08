''''
Preprocess the data, divide training into train, validation sets, using cross-validation.
Load training data, training labels, test data, test labels
'''
import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy import misc
from sklearn.feature_extraction import image
from skimage import io
from skimage.feature import hog
from skimage import data, color, exposure



pic = misc.imread("um_000000.png")
img = img_as_float(pic [::2, ::2])
