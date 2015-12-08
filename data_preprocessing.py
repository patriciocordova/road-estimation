from feature import Feature
from feature_extraction import *
import scipy.io, sys
import glob
import argparse
import random
import os

image_filenames = glob.glob("../data_road/training/image_2/*.png")
label_image_filenames = glob.glob("../data_road/training/gt_image_2/*road*.png")
num_images = len(image_filenames)

TRAINING=0
VALIDATION=1
TESTING=2

# define data split
num_train = int(num_images * 0.6)
num_test = int(num_images * 0.3)
num_valid = num_images - num_train - num_test

# define data split label 0 - train, 1 - validation, 2 - test
file_labels = np.zeros(num_images)
for i in xrange(0,num_test):
    file_labels[i] = 2 
for i in xrange(num_test,(num_valid+num_test)):
    file_labels[i] = 1 

random.seed(42)
random.shuffle(file_labels)

train_labels = []
train_data = []
test_labels = []
test_data = []
valid_labels = []
valid_data = []
valid_files = []
test_files = []
valid_files_count = 0
test_files_count = 0
test_superpixels = []
train_superpixels = []
validation_original_image = []
test_original_image = []
test_pixels_labels = []
valid_pixels_labels = []
valid_superpixels = []

for i in xrange(0,num_images):
	feature = Feature()
	feature.create(image_filenames[i],label_image_filenames[i])
	labels = feature.getSuperpixelLabels()
	feature_vectors = feature.getFeaturesVectors()
	h, img_filename = os.path.split(image_filenames[i])
	h, label_img_filename = os.path.split(label_image_filenames[i])

	folder = "extracted/" + img_filename + "-" + label_img_filename + "/"
	if not os.path.exists(folder):
		os.makedirs(folder)

	scipy.io.savemat(folder+"superpixel", {'features':feature.getSuperpixelImage()}, oned_as='column')
	scipy.io.savemat(folder+"superpixel_labels", {'features':feature.getSuperpixelLabels()}, oned_as='column')
	scipy.io.savemat(folder+"location", {'features':feature.getSuperpixelsLocation()}, oned_as='column')
	scipy.io.savemat(folder+"color", {'features':feature.getSuperpixelsColor()}, oned_as='column')
	scipy.io.savemat(folder+"hog", {'features':feature.getSuperpixelsHog()}, oned_as='column')
	scipy.io.savemat(folder+"size", {'features':feature.getSuperpixelsSize()}, oned_as='column')
	scipy.io.savemat(folder+"texture", {'features':feature.getSuperpixelsTexture()}, oned_as='column')
	scipy.io.savemat(folder+"features_combined", {'features':feature.getFeaturesVectors()}, oned_as='column')

	if file_labels[i] != TESTING:
		# store data
		if file_labels[i] == TRAINING:
			train_superpixels.append(feature.getSuperpixelImage())
			train_labels = np.append(train_labels, labels, 0)
			if train_data==[]:
				train_data = feature_vectors
			else:
				train_data = np.vstack((train_data,feature_vectors))
		else:
			valid_superpixels.append(feature.getSuperpixelImage())
			#validation_original_image.append(images[i])
			valid_pixels_labels.append(getPixelLabel(feature.getLabelImage()))
			valid_files_count += 1
			valid_labels = np.append(valid_labels, labels, 0)
			if valid_data==[]:
				valid_data = feature_vectors
			else:
				valid_data = np.vstack((valid_data,feature_vectors))
	else:
		test_files_count += 1
		test_superpixels.append(feature.getSuperpixelImage())
		#testOriginalImage.append(images[i])
		#test_files = sp.getSuperValidFiles(fe.getSuperpixelImage(), test_files_count, test_files)
		test_pixels_labels.append(getPixelLabel(feature.getLabelImage()))
		test_files_count += 1
		test_labels = np.append(test_labels, labels, 0)
		if test_data==[]:
			test_data = feature_vectors
		else:
			test_data = np.vstack((test_data,feature_vectors))
	sys.stdout.write('\r')
	sys.stdout.write(image_filenames[i] + " " +label_image_filenames[i] + '\nprogress %2.2f%%' %(100.0*i/num_images))
	sys.stdout.flush()

scipy.io.savemat("train_matrices", {'train_data':train_data, 'valid_data':valid_data, 'train_labels':train_labels, 'valid_labels':valid_labels, 'file_labels':file_labels, 'image_filenames':image_filenames, 'label_image_filenames':label_image_filenames,'valid_pixels_labels':valid_pixels_labels, ''' 'valid_files':valid_files,'valid_files_count':valid_files_count, ''' 'valid_superpixels':valid_superpixels,'test_files_count':test_files_count, ''' 'validationOriginalImage':validationOriginalImage, ''' 'train_superpixels':train_superpixels}, oned_as='column')
scipy.io.savemat("train_data", {'train_data':train_data, 'train_labels':train_labels}, oned_as='column')
scipy.io.savemat("valid_data", {'valid_data':valid_data, 'valid_labels':valid_labels}, oned_as='column')
scipy.io.savemat("test_data",{'test_data':test_data,'test_label':test_labels},oned_as='column')