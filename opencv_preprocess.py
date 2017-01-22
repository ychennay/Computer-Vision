import cv2, numpy as np, os, pandas as pd
os.chdir('/Users/yuchen/PycharmProjects/Artmonious/data/')

def labels_dict():
    df = pd.read_csv("labeled_data.csv")
    train_df = df[df['type'] == 'train']
    test_df = df[df['type'] == 'test']

    training_image_ids_labels = zip(train_df['img_id'].values,train_df['group_id'].values)
    test_image_ids_labels = zip(test_df['img_id'].values,test_df['group_id'].values)

    training_dict = {}
    test_dict = {}

    for id, label in training_image_ids_labels:
        training_dict[id] = label


    for id, label in test_image_ids_labels:
        test_dict[id] = label

    return training_dict, test_dict

def create_import_list(dict):

    """

    :param dict:
    :return:
    import_list: a list of filenames of training or test sets to import
    filename_to_id_dict: a dictionary of filenames with their appropriate label
    """

    import_list = []
    os.chdir('/Users/yuchen/PycharmProjects/Artmonious/data/labeled_data')
    for file in os.listdir(os.getcwd()):
        if file.split(".")[1] not in ['png', 'jpeg', 'jpg']:
            continue
        else:
            import_list.append(file)
            for id in import_list:
                if int(id[5:].split(".")[0]) not in dict.keys():
                    import_list.remove(id)

    filename_to_id_dict = {}
    for file in import_list:
        filename_to_id_dict[file] = training_dict[int(file[5:].split(".")[0])]

    return import_list, filename_to_id_dict


class ImageModel(object):

    path = '/Users/yuchen/PycharmProjects/Artmonious/data/labeled_data/'

    def __init__(self, filename):

        self.filepath = ImageModel.path + filename
        self.filename = filename
        if os.getcwd() != ImageModel.path:
            os.chdir(ImageModel.path)

        if os.path.exists(ImageModel.path + filename):
            self.original_image = cv2.imread(filename)
            print("Image {0} imported with shape {1}".format(filename, self.original_image.shape))
        else:
            print("File {0} not found in {1}.".format(filename, ImageModel.path))
            return
        self.original_shape = self.original_image.shape

    def gray_scale(self):
        self.gray_scale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return self.gray_scale_image

    def process_HOG(self, image_object):
        self.HOG = cv2.HOGDescriptor().compute(image_object)
        return self.HOG

    def process_SIFT(self, image_object):
        self.SIFT_keypoints, self.SIFT_descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(image_object, None)
        self.SIFT_image = cv2.drawKeypoints(image_object, self.SIFT_keypoints, self.gray_scale_image,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.SIFT_image
    def process_SURF(self, image_object, threshold=0):
        self.SURF_keypoints, self.SURF_descriptors = \
            cv2.xfeatures2d.SURF_create(hessianThreshold=threshold).detectAndCompute(image_object, None)
        return self.SURF_keypoints, self.SURF_descriptors

    def process_FAST(self, image_object):
        self.FAST_keypoints = cv2.FastFeatureDetector_create().detect(image_object, None)
        self.ORB_image = cv2.drawKeypoints(image_object, self.FAST_keypoints, self.gray_scale_image,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.FAST_keypoints

    def process_ORB(self, image_object, keypoints_threshold=500):
        self.ORB_keypoints = cv2.ORB_create(keypoints_threshold).detect(image_object, None)
        self.ORB_image = cv2.drawKeypoints(image_object, self.ORB_keypoints, self.gray_scale_image,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.ORB_image

    def display_image(self, image_object, title="Image"):
        cv2.imshow(title, image_object)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def resize_image(self, image_object, width, height):
        print("New dimensions {0} x {1}".format(width, height))
        return cv2.resize(image_object, (width, height), interpolation= cv2.INTER_AREA)

    def save_image(self, filename, image_object):
        cv2.imwrite(filename, image_object)
        print("Image saved as {0}".format(filename))

    def display_details(self):
        print("Original object shape: {0}".format(self.original_shape.shape))
        print("Gray scale shape: {0}".format(self.gray_scale_image.shape))
        print("SIFT shape: {0}".format(self.SIFT_image.shape))

class MutableLabel(object):
    def __init__(self,label_name):
        self.label_name = label_name