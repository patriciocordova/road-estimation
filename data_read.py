import scipy.io

train_data = scipy.io.loadmat("train_data.mat")
test_data = scipy.io.loadmat("test_data.mat")
print train_data["train_data"].shape
print test_data["test_data"].shape