import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy import misc
from feature import *
from data_read import *
from skimage.segmentation import mark_boundaries

from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class road_estimation:
	def __init__(self, model_selection):
		self._train_data, self._train_targets, self._valid_data, \
		self._valid_targets, self._test_data, self._test_targets = data_load()
	    
		self._model_selection = model_selection
		self._classifier = []



	def train(self):
		if self._model_selection == "svm":
			# selected the svc in svm
			self._classifier = svm.SVC()
		elif self._model_selection == "nb":
			self._classifier = GaussianNB()
		elif self._model_selection == "knn":
			# parameter n_jobs can be set to -1 to enable parallel calculating
			self._classifier = KNeighborsClassifier(n_neighbors=7)
		elif self._model_selection == "ada":
			# Bunch of parameters, n_estimators, learning_rate 
			self._classifier = AdaBoostClassifier()
		elif self._model_selection == "rf":
			# many parameters including n_jobs
			self._classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
		elif self._model_selection == "qda":
			#complicated array like parameters, perhaps leave it default
			self._classifier = QuadraticDiscriminantAnalysis()
		else:
			print "Please refer to one classifier"





		self._classifier.fit(self._train_data, self._train_targets)
		# predict on valid data
		prediction_valid = self._classifier.predict(self._valid_data)
		# print validation result for selected model.
		print("Classification report for classifier %s on valid_data:\n%s\n"
	      % (self._model_selection, metrics.classification_report(self._valid_targets, prediction_valid)))

	def test(self):
		# predict on test data
		prediction_test = self._classifier.predict(self.test_data)
		# print test result for selected model.
		print("Classification report for classifier %s on test_data:\n%s\n"
	      % (self._model_selection, metrics.classification_report(self._test_targets, prediction_test)))


	def showPredictionImage(self):
		f = Feature()
		f.loadImage("um_000000.png")
		f.extractFeatures()
		fea_matrix = f.getFeaturesVectors()

		predict = self._classifier.predict(fea_matrix)
		image = np.copy(f.image)

		num_superpixels = np.max(f.superpixel) + 1
		for i in xrange(0,num_superpixels):
			indices = np.where(f.superpixel==i)
			if predict[i] == 1:
		    	
				image[indices[0],indices[1],0] = 1
				image[indices[0],indices[1],1] = 1
				image[indices[0],indices[1],2] = 0
		plt.imshow(image)
		plt.show()
		# show prediction image with superpixels
		plt.imshow(mark_boundaries(image, superpixels))
		plt.show()

if __name__ == '__main__':
	road_est = road_estimation("svm")
	road_est.train()
	road_est.showPredictionImage()

            






