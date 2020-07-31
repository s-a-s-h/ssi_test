# jk this is log reg



# def train_val_test_split(dataset):
#   # Returns a tuple of 3 sub-datasets.
#   # The first 80% train, next 10% val, last 10% test.
#   return np.split(dataset, [int(dataset.shape[0] * 0.8), int(dataset.shape[0] * 0.9)])
#
# X_train, X_val, X_test = train_val_test_split(X)
# y_train, y_val, y_test = train_val_test_split(y)
from sklearn.multiclass import OneVsRestClassifier
