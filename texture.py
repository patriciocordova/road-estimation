import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from skimage.segmentation import slic
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.util import img_as_float
from scipy import misc
from skimage import color

xs = []
ys = []

def getSuperPixelLocations(superpixels,grayscale):
    locations = []
    numSuperpixels = np.max(superpixels)+1
    
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels == i)
        glcm = greycomatrix([grayscale[indices]], [1], [0], 256, symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        xs.append(dissimilarity)
        ys.append(correlation)
        
        grayscale[indices] = ((dissimilarity*correlation)*255/8)
    max_xs = np.max(xs);
    min_xs = np.min(xs);
    max_ys = np.min(ys);
    min_ys = np.min(xs);
    factor = 5;
    mi = 15
    ma = 7

    xs_range = (max_xs - min_xs)/mi;
    ys_range = (max_xs - min_xs)/ma;
    ranges = []
    cont = 0;
    increase = 255/(mi*ma) + 30

    c_min_xs = min_xs
    for i in range(1,mi):
        c_max_xs = c_min_xs + xs_range
        c_min_ys = min_ys
        for j in range(1,ma):
            c_max_ys = c_min_ys + ys_range
            print "****"
            print str(c_min_xs) + " " + str(c_max_xs)
            print str(c_min_ys) + " " + str(c_max_ys)
            print "****"
            for i in xrange(0,numSuperpixels):
                if xs[i] >= c_min_xs and xs[i] <= c_max_xs and ys[i] >= c_min_ys and ys[i] <= c_max_ys:
                    indices = np.where(superpixels == i)
                    print xs[i]
                    print ys[i]
                    print cont
                    grayscale[indices] = cont + increase
                    print "----"
            c_min_ys = c_max_ys
        cont = cont + increase
        c_min_xs = c_max_xs
    return grayscale

# open the camera image
image = data.camera()

grayscale = np.around(color.rgb2gray(mpimg.imread('um_000000.png')) * 255,0);
segments_slic = slic(grayscale, n_segments=300, compactness=20, sigma=1)
grayscale = getSuperPixelLocations(segments_slic,grayscale)
plt.imshow(grayscale,cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
plt.show()

fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(1, 1, 1)
ax.plot(xs, ys, 'go', label='distribution')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLVM Correlation')
ax.legend()
plt.show()

#plt.show()
#print segments_slic[0]
#print pic[0]
    

'''
# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Grass')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLVM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
'''