class ImageUploadScheme(object):

    """
    This class handles all file manipulations and uploading. It ends by generating the cluster labels of the images according
    to either SIFT, SURF, or HOG techniques.
    """

    ACCEPTED_FILETYPES = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']

    _instances = count(1)
    filenames_to_dict = {}
    label = 1
    import_folders = []
    import_list = []
    label_cluster_list = []
    cluster_center_list = []
    missing_files = []

    def __init__(self, path):

        """
        Sets initial parameters. The self.instance_number attribute is intended to ensure that no more than one instance of
        this class will be renaming image filenames in one session.
        :param path: path of the directory where the subdirectories (folders) contain images belonging to distinct classes
        """
        self.path = path
        self.instance_number  = self._instances.next()

    def find_all_subdirectories(self):

        """
        :return: True if the operation is successful and self.import_folders is populated with a list of folder names
        """

        os.chdir(self.path)
        self.import_folders = []
        for x in os.walk(self.path):
            self.import_folders.append(x[0].split("/")[-1])
        self.import_folders = self.import_folders[1:]
        print(self.import_folders)
        return True

    def rename_images(self):

        """

        :return: True if self.filenames_to_dict is correctly populated.
        _NUMBERS is a final list of integer strings used to detect if the filenames have already been altered. If they
        have, then the method skips over them and does not attempt to rename them.
        """
        _NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        for folder in self.import_folders:
            os.chdir(self.path + '/' + folder)
            print("Currently in folder " + os.getcwd() + " with classification label {0}".format(self.label))
            files_to_import = os.listdir(os.getcwd())

            if self.instance_number == 1:
                #following lines of code only for when you first run the script
                for file in files_to_import:
                     if file.split(".")[1] in self.ACCEPTED_FILETYPES and file.split(".")[0][0] not in _NUMBERS:
                         # if the file name ends in an accepted filetype and the beginning of the file is not already a number...
                         os.rename(file, str(self.label) + file)

            for file in files_to_import:
                if file.split(".")[1] in self.ACCEPTED_FILETYPES:
                    self.filenames_to_dict[str(self.label) + file] = self.label
                    continue
                else:
                    files_to_import.remove(file)

            self.label += 1
        return True

    def create_import_list(self):

        """

        :return: True if self.import is correctly populated with a list of files to import through OpenCV. It skips all
         files that are not in self.ACCEPTED_FILETYPES
        """
        for i in range(len(self.import_folders)):
            os.chdir(path + "/" + self.import_folders[i])
            for file in filter(lambda x: x.split('.')[1] in self.ACCEPTED_FILETYPES, os.listdir(os.getcwd())):
                self.import_list.append(file)
        print(self.import_list)
        return True

    def revert_image_name(self):

        """
        Method to erase the appended number from the filenames of images.
        :return: True if filenames are corrected reverted
        """

        for folder in self.import_folders:
            os.chdir(self.path + '/' + folder)
            print("Currently in folder " + os.getcwd() + " with classification label {0}".format(self.label))
            files_to_import = os.listdir(os.getcwd())

            if self.instance_number == 1:
                #following lines of code only for when you first run the script
                for file in files_to_import:
                     if file.split(".")[1] == 'jpg':
                         os.rename(file, file[1:])
        return True

    def generate_cluster_labels(self, method):

        """
        WARNING: at this, this method will throw an error regarding array shape and format. If this occurs, reset the files by
        pulling them again from the github repo, and reinitialize this class. Sometimes lingering instance attributes are
        contributing to misshapen cluster arrays.

        :return: label_cluster are the y_true values
        cluster_center are the y_hat values
        """
        self.method = method
        for index, image in enumerate(self.import_list):
            os.chdir(self.path + "/" + self.import_folders[int(image[0]) - 1] + "/")
            ImageModel.path = os.getcwd()
            print("--------------------------" + os.getcwd() + "-------------------------------")
            print(image)
            image_model = ImageModel(image)
            # image is a filepath
            image_model.gray_scale()

            image_model.process_SIFT(image_model.gray_scale_image)
            if self.method == 'SIFT':
                image_model.process_SIFT(image_model.gray_scale_image)
                self.kmeans = KMeans(n_clusters=6, random_state=0).fit(image_model.SIFT_descriptors)
            elif self.method == 'HOGS':
                image_model.process_HOG(image_model.gray_scale_image)
                self.kmeans = KMeans(n_clusters=6, random_state=0).fit(image_model.HOG)
            elif self.method == 'SURF':
                image_model.process_SURF(image_model.gray_scale_image)
                self.kmeans = KMeans(n_clusters=6, random_state=0).fit(image_model.SURF_descriptors)

            for cluster_center in self.kmeans.cluster_centers_:
                try:
                    self.label_cluster_list.append(self.filenames_to_dict[image])
                    self.cluster_center_list.append(cluster_center)
                except KeyError:
                    self.missing_files.append(image)
                    print("Error... image in path {0}".format(image))
                    continue
            print("On {0}.".format(image))
            print("Length of label_cluster_list: {0}".format(len(self.label_cluster_list)))
        return True
