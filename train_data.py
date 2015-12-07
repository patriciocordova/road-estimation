import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy import misc
from sklearn.feature_extraction import image
from feature_extraction import *
