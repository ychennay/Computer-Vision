from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import
from opencv_preprocess import *

os.chdir('/Users/yuchen/PycharmProjects/Artmonious/data/')
training_dict, test_dict = labels_dict()
import_list, filename_to_id_dict = create_import_list(training_dict)


def generate_cluster_labels():
    label_cluster_list = []
    cluster_center_list = []
    for index, image in enumerate(import_list):
        image_model = ImageModel(image)
        image_model.gray_scale()
        image_model.process_SIFT(image_model.gray_scale_image)
        kmeans = KMeans(n_clusters=20, random_state=0)\
            .fit(image_model.SIFT_descriptors, filename_to_id_dict[image])
        for cluster_center in kmeans.cluster_centers_:
            label_cluster_list.append(filename_to_id_dict[image])
            cluster_center_list.append(cluster_center)
        print("On {0}, class label {1}.".format(image, filename_to_id_dict[image]))
        print("Length of label_cluster_list: {0}".format(len(label_cluster_list)))

    return label_cluster_list, cluster_center_list