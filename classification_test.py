from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import *
from opencv_preprocess import *

training_dict, test_dict = labels_dict()
import_list, filename_to_id_dict = create_import_list(test_dict)