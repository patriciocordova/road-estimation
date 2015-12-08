from feature import Feature
import scipy.io, sys
import glob
import argparse

training_images = glob.glob("../data_road/training/image_2/*.png")
label_images = glob.glob("../data_road/training/gt_image_2/*road*.png")
num_files = len(training_images)

for i in xrange(0,num_files):
	feature = Feature()
	feature.create(training_images[i],label_images[i])
	scipy.io.savemat(str(i)+"", {'features':feature.getFeaturesVectors()}, oned_as='column')
	scipy.io.savemat(str(i)+"", {'features':feature.getFeaturesVectors()}, oned_as='column')
	scipy.io.savemat(str(i)+"", {'features':feature.getFeaturesVectors()}, oned_as='column')
	scipy.io.savemat(str(i)+"", {'features':feature.getFeaturesVectors()}, oned_as='column')
	scipy.io.savemat(str(i)+"", {'features':feature.getFeaturesVectors()}, oned_as='column')
	# DEBUG
	print training_images[i] + " " +label_images[i]
	break