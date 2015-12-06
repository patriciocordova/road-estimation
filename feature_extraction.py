import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy import misc
from sklearn.feature_extraction import image

pic = misc.imread("um_000000.png")
img = img_as_float(pic [::2, ::2])
#print img.shape

segments_slic = slic(img, n_segments=300, compactness=20, sigma=1)
plt.imshow(mark_boundaries(img, segments_slic))
plt.show()
