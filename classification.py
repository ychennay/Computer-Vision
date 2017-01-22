import cv2, numpy as np, os, pandas as pd
from opencv_preprocess import ImageModel, create_import_list, labels_dict
from sklearn.cluster import KMeans

os.chdir('/Users/yuchen/PycharmProjects/Artmonious/data/')
training_dict, test_dict = labels_dict()
import_list, filename_to_id_dict = create_import_list(training_dict)


label_cluster_dict = {}
for index, image in enumerate(import_list):
    image_model = ImageModel(image)
    image_model.gray_scale()
    image_model.process_SIFT(image_model.gray_scale_image)
    kmeans = KMeans(n_clusters=20, random_state=0).fit(image_model.SIFT_descriptors)
    zip(kmeans.cluster_centers_ = filename_to_id_dict[image]
