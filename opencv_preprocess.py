import cv2, numpy as np, os
name = ' 7.jpg'

class ImageModel(object):

    path = '/Users/yuchen/PycharmProjects/Artmonious/data/train1/'

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
        self.SIFT_keypoints = cv2.xfeatures2d.SIFT_create().detect(image_object, None)
        self.SIFT_image = cv2.drawKeypoints(image_object, self.SIFT_keypoints, a.gray_scale_image,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.SIFT_image

    def process_SURF(self, image_object, threshold=0):
        self.SURF_keypoints, self.SURF_descriptors = \
            cv2.xfeatures2d.SURF_create(hessianThreshold=threshold).detectAndCompute(image_object, None)
        return self.SURF_keypoints, self.SURF_descriptors

    def process_FAST(self, image_object):
        self.FAST_keypoints = cv2.FastFeatureDetector_create().detect(image_object, None)
        self.ORB_image = cv2.drawKeypoints(image_object, self.FAST_keypoints, a.gray_scale_image,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.FAST_keypoints

    def process_ORB(self, image_object, keypoints_threshold=500):
        self.ORB_keypoints = cv2.ORB_create(keypoints_threshold).detect(image_object, None)
        self.ORB_image = cv2.drawKeypoints(image_object, self.ORB_keypoints, a.gray_scale_image,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.ORB_image

    def display_image(self, image_object, title="Image"):
        cv2.imshow(title, image_object)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def display_details(self):
        print("Original object shape: {0}".format(self.original_shape.shape))
        print("Gray scale shape: {0}".format(self.gray_scale_image.shape))
        print("SIFT shape: {0}".format(self.SIFT_image.shape))





a = ImageModel(' 7.jpg')
a.gray_scale()
a.process_HOG(a.gray_scale_image)
a.process_SIFT(a.gray_scale_image)
a.display_image(a.SIFT_image)
a.process_ORB(a.gray_scale_image)
a.display_image(a.ORB_image)

sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(a.gray_scale_image, None)
new_image = cv2.drawKeypoints(a.gray_scale_image, keypoints, a.original_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Name", a.gray_scale_image)
cv2.waitKey()
cv2.destroyAllWindows()