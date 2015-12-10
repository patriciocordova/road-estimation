import scipy.io
# load data from files
def data_load():
	train_data = scipy.io.loadmat("train_data.mat")
	test_data = scipy.io.loadmat("test_data.mat")
	valid_data = scipy.io.loadmat("valid_data.mat")
	return train_data["train_data"], train_data["train_labels"].ravel(), valid_data["valid_data"], \
	valid_data["valid_labels"].ravel(), test_data["test_data"], test_data["test_label"].ravel()
	# print train_data["train_data"].shape
	# print test_data["test_data"].shape
	# print valid_data["valid_data"].shape


