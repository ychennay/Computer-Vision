from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
import random, numpy
from itertools import count
import os
from opencv_preprocess import *
from files_preprocess import ImageUploadScheme

attempt = ImageUploadScheme('/Users/yuchen/Desktop/Image Repo/')
attempt.find_all_subdirectories()
attempt.rename_images()
attempt.create_import_list()
attempt.generate_cluster_labels(method='SIFT') #available methods are SIFT, SURF, and HOGS, although HOGS will need more fine-tuning and customization


# training the model
linearSVC = OneVsRestClassifier(LinearSVC(random_state=0)).fit(attempt.cluster_center_list,attempt.label_cluster_list)
kMeans = KMeans(n_clusters=25).fit(attempt.cluster_center_list,attempt.label_cluster_list)
RBKSVM = OneVsRestClassifier(SVC()).fit(attempt.cluster_center_list,attempt.label_cluster_list)
mlPerceptron = OneVsRestClassifier(MLPClassifier()).fit(attempt.cluster_center_list, attempt.label_cluster_list)

#####TRAINING

#Linear SVM
print("Support Vector Machine")
print(float(sum(linearSVC.predict(attempt.cluster_center_list) == attempt.label_cluster_list)) / len(attempt.label_cluster_list))
#K Means
print("K-Means")
print(float(sum(kMeans.predict(attempt.cluster_center_list) == attempt.label_cluster_list)) / len(attempt.label_cluster_list))
#Radial Basis Kernel SVM
print("Radial Basis Kernel SVM")
print(float(sum(RBKSVM.predict(attempt.cluster_center_list) == attempt.label_cluster_list)) / len(attempt.label_cluster_list))
#Multi-layer Perceptron
print("Multilayer Perceptron")
print(float(sum(mlPerceptron.predict(attempt.cluster_center_list) == attempt.label_cluster_list)) / len(attempt.label_cluster_list))

####### CROSS VALIDATION
kappa = make_scorer(cohen_kappa_score)

np.mean(cross_val_score(mlPerceptron, attempt.cluster_center_list, attempt.label_cluster_list, cv=10, scoring='accuracy'))
np.mean(cross_val_score(RBKSVM, attempt.cluster_center_list, attempt.label_cluster_list, cv=10, scoring='accuracy'))
np.mean(cross_val_score(linearSVC, attempt.cluster_center_list, attempt.label_cluster_list, cv=10, scoring='accuracy'))
np.mean(cross_val_score(linearSVC, attempt.cluster_center_list, attempt.label_cluster_list, cv=20, scoring='accuracy'))

1.0 / len(attempt.import_folders)

