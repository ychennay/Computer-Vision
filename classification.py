from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from opencv_preprocess import *

#TRAINING
os.chdir('/Users/yuchen/PycharmProjects/Artmonious/data/')
training_dict, test_dict = labels_dict()
import_list, filename_to_id_dict = create_import_list(training_dict)

def generate_cluster_labels(import_list, filename_to_id_dict):
    label_cluster_list = []
    cluster_center_list = []
    for index, image in enumerate(import_list):
        image_model = ImageModel(image)
        image_model.gray_scale()
        image_model.process_SIFT(image_model.gray_scale_image)
        kmeans = KMeans(n_clusters=6, random_state=0)\
            .fit(image_model.SIFT_descriptors)
        for cluster_center in kmeans.cluster_centers_:
            label_cluster_list.append(filename_to_id_dict[image])
            cluster_center_list.append(cluster_center)
        print("On {0}, class label {1}.".format(image, filename_to_id_dict[image]))
        print("Length of label_cluster_list: {0}".format(len(label_cluster_list)))
    return label_cluster_list, cluster_center_list

label_cluster_list, cluster_center_list = generate_cluster_labels(import_list, filename_to_id_dict)

linearSVC = OneVsRestClassifier(LinearSVC(random_state=0)).fit(cluster_center_list,label_cluster_list)
kMeans = KMeans(n_clusters=25).fit(cluster_center_list,label_cluster_list)
RBKSVM = OneVsRestClassifier(SVC()).fit(cluster_center_list,label_cluster_list)

#calculate accuracy

#Linear SVM
float(sum(linearSVC.predict(cluster_center_list) == label_cluster_list)) / len(label_cluster_list)
#K Means
float(sum(kMeans.predict(cluster_center_list) == label_cluster_list)) / len(label_cluster_list)
#Radial Basis Kernel SVM
float(sum(RBKSVM.predict(cluster_center_list) == label_cluster_list)) / len(label_cluster_list)

##########################

#TEST
import_list, filename_to_id_dict = create_import_list(test_dict)
label_cluster_list, cluster_center_list = generate_cluster_labels(import_list, filename_to_id_dict)

#calculate accuracy
#Linear SVM
float(sum(linearSVC.predict(cluster_center_list) == label_cluster_list)) / len(label_cluster_list)
#K Means
float(sum(kMeans.predict(cluster_center_list) == label_cluster_list)) / len(label_cluster_list)
#Radial Basis Kernel SVM
float(sum(RBKSVM.predict(cluster_center_list) == label_cluster_list)) / len(label_cluster_list)
