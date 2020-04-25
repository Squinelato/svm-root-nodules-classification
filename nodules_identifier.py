"""this module aims in load, processes and train SVM algoritm in root nodules dataset
"""
import os
import sys
import glob
import json
import getopt
from datetime import datetime as dt

import cv2
import joblib
import numpy as np
import pandas as pd
from mahotas.features import haralick
from skimage.measure import moments_hu

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, train_test_split



class Identifier:
    """this class is designed to load and preocess imagens and train SVM classifiers
    """

    def __init__(self):
        self.features = None
        self.labels = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.cross_val = StratifiedKFold(n_splits=5)
        self.optimized_classifier = None
        self.bbox_list = list()
        self.y_predict = list()
        self.image = None
        self.csv_file = None
        self.json_file = None


    @staticmethod
    def load_images_from_folder(folder):
        """this method can load all the imagens from a given folder

        Arguments:
            folder {str} -- the path of the folder

        Returns:
            list -- a list with the loaded images
        """
        images = list()
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        return images

    def read_image(self, image_path):
        """this method read an input image turn it to a RGB representation and return its shape

        Arguments:
            image_path {Str} -- the input path of the given image

        Returns:
            list -- a list with the height, width and the number of color channels of the image
        """
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image = original_image

        return original_image.shape

    def read_csv_file(self, csv_path):
        """this method read a csv file and get the position of each bounding box
        of the image, besides its label, and save it on the class objects

        Arguments:
            csv_path {str} -- the input path of the csv file
        """
        self.csv_file = pd.read_csv(csv_path)
        for minr_norm, minc_norm, maxr_norm, maxc_norm in zip(self.csv_file.minr_norm,
                                                              self.csv_file.minc_norm,
                                                              self.csv_file.maxr_norm,
                                                              self.csv_file.maxc_norm):

            self.bbox_list.append([minr_norm, minc_norm, maxr_norm, maxc_norm])

    @staticmethod
    def extract_features(image):
        """this method applies the hu moments and the haralick method on a given image

        Arguments:
            image {ndarray} -- the image whose the features will be extracted

        Returns:
            numpy.array -- an array with the extracted features
        """
        return np.r_[moments_hu(image), haralick(image).flatten()]

    def save_extracted_features(self, bboxes_path):
        """this method aggregate all the extracted features from all the images and saves
        this data in the disk as a numpy file

        Arguments:
            bboxes_path {str} -- the path of the dataset folder
        """
        positive_instance = self.load_images_from_folder(bboxes_path + 'nodules/')
        negative_instance = self.load_images_from_folder(bboxes_path + 'non-nodules/')

        positive_features = np.array(list(map(self.extract_features, positive_instance)))
        negative_features = np.array(list(map(self.extract_features, negative_instance)))

        features = np.r_[positive_features, negative_features]
        labels = np.r_[np.ones(len(positive_instance)), np.zeros(len(negative_instance))]

        np.save('features/features.npy', features)
        np.save('features/labels.npy', labels)

    def load_features(self):
        """this method loads the extracted features and its labels
        """
        self.features = np.load('features/features.npy')
        self.labels = np.load('features/labels.npy')

    def split_dataset(self):
        """this method split the loaded features into a training and a test
        dataset, in a stratified way
        """
        x_train, x_test, y_train, y_test = train_test_split(self.features,
                                                            self.labels,
                                                            stratify=self.labels,
                                                            test_size=0.2)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def reset_attributes(self):
        """this method reset all the attributes of the Maker class
        """
        self.image = None
        self.y_predict = list()
        self.bbox_list = list()

    def normalize(self):
        """this method normalizes a given dataset
        """
        self.scaler.fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def pipeline(self):
        """this method executes a pipeline with the load_features, split_dataset
        and the normalize method
        """
        print('Loading features')
        self.load_features()
        print('Splitting the dataset')
        self.split_dataset()
        print('Normalizing the dataset')
        self.normalize()

    def training(self, kernel, scoring='f1_weighted'):
        """this method train and fine-tune a SVM classfier with the given kernel

        Arguments:
            kernel {str} -- the desired kernel to train the SVM

        Keyword Arguments:
            scoring {str} -- the scoring method to evaluate the trainig (default: {'f1_weighted'})
        """
        if kernel != 'linear':
            default_classifier = SVC(class_weight='balanced',
                                     decision_function_shape='ovo',
                                     cache_size=4000)

            default_params = {'C': np.reciprocal(np.arange(1, 10).astype(np.float)),
                              'kernel': [kernel], 'gamma': ['scale'],
                              'coef0': np.arange(0, 10, 0.1), 'degree': range(1, 10)}
        else:
            number_instances = self.x_train.shape[0]
            default_classifier = SGDClassifier(loss='hinge',
                                               class_weight='balanced',
                                               max_iter=np.ceil(10**6 / number_instances),
                                               shuffle=True)

            default_params = {'alpha': 10.0**-np.arange(1, 7),
                              'l1_ratio': np.arange(0.00, 1.001, 0.001)}

        grid_search = GridSearchCV(default_classifier,
                                   param_grid=default_params,
                                   cv=self.cross_val, scoring=scoring,
                                   verbose=3, n_jobs=4)

        grid_search.fit(self.x_train, self.y_train)
        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        now = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
        joblib.dump(grid_search.best_estimator_, 'classifiers/{}_{}.plk'.format(kernel, now))

    def load_optimized_classifier(self, classifier_path):
        """this method can load a trained SVM from the disk

        Arguments:
            classifier_path {str} -- the path of the classifier
        """
        self.optimized_classifier = joblib.load(classifier_path)
        print('Parameters: {}'.format(self.optimized_classifier.get_params()))

    def get_metrics(self, classifier):
        """this method calculate the balanced accuracy and the F1 score for
        a given classifier

        Arguments:
            classifier {sklearn.classifier} -- the classifier to get the metrics

        Returns:
            list -- the mean and the standard deviation of the balanced accuracy and the F1 score
        """
        scores = cross_validate(classifier, self.x_train, self.y_train,
                                cv=self.cross_val, n_jobs=-1,
                                scoring=['balanced_accuracy', 'f1_weighted'])

        return (scores['test_balanced_accuracy'].mean(),
                scores['test_balanced_accuracy'].std(),
                scores['test_f1_weighted'].mean(),
                scores['test_f1_weighted'].std())

    def calculate_metrics(self):
        """this method uses the get_metrics method in order to evaluate all the classifiers
        on a given folder and then compile these informations in a CSV file
        """
        classifiers = [(clf, joblib.load('./classifiers/' + clf))
                       for clf in os.listdir('./classifiers') if '.plk' in clf]

        headers = ['file_name', 'day', 'hour', 'balanced_accuracy', 'std', 'f1_weighted', 'std']

        with open('scores/results.csv', 'w') as file:

            file.writelines(','.join(headers) + '\n')
            for name, classifier in classifiers:
                data = [str(metric) for metric in self.get_metrics(classifier)]
                name = name.replace('.plk', '')
                file.writelines(','.join(name.split('_')) + ',' + ','.join(data) + '\n')

    def csv_predict(self, image_path, csv_path, output_directory):
        """this method uses the trained SVM classifier to predict if a given bbox
        is a nodule or a false nodule, then save thos label on the given CSV file

        Arguments:
            image_path {str} -- the path of the image
            csv_path {str} -- the path of the CSV file
            output_directory {str} -- the path of the output directory
        """
        image_height, image_width = self.read_image(image_path)
        self.read_csv_file(csv_path)

        for bbox in self.bbox_list:

            minr_norm, minc_norm, maxr_norm, maxc_norm = bbox

            minr, maxr = int(minr_norm*image_height), int(maxr_norm*image_height)
            minc, maxc = int(minc_norm*image_width), int(maxc_norm*image_width)

            features = self.extract_features(self.image[minr:maxr, minc:maxc])
            features = self.scaler.transform([features])

            self.y_predict.append(int(self.optimized_classifier.predict(features)[0]))

        self.csv_file['y_label'] = self.y_predict
        self.csv_file.to_csv(output_directory + csv_path.split('/')[-1], index=False)

    def json_predict(self, image_path, json_path, output_directory):
        """this method uses the trained SVM classifier to predict if a given bbox
        is a nodule or a false nodule, then save thos label on the given JSON file

        Arguments:
            image_path {str} -- the path of the image
            json_path {str} -- the path of the JSON file
            output_directory {str} -- the path of the output directory
        """
        image_height, image_width = self.read_image(image_path)
        with open(json_path, 'r') as f_json:
            self.json_file = json.load(f_json)

        for i, bbox in enumerate(self.json_file['bboxes']):

            minr_norm, minc_norm, maxr_norm, maxc_norm = (bbox['rendering']['minr'],
                                                          bbox['rendering']['minc'],
                                                          bbox['rendering']['maxr'],
                                                          bbox['rendering']['maxc'])

            minr, maxr = int(minr_norm*image_height), int(maxr_norm*image_height)
            minc, maxc = int(minc_norm*image_width), int(maxc_norm*image_width)

            if minc < 0:
                minc = 0
            elif maxc > image_width:
                maxc = image_width
            elif minr < 0:
                minr = 0
            elif maxr > image_height:
                maxr = image_height

            features = self.extract_features(self.image[minr:maxr, minc:maxc])
            features = self.scaler.transform([features])

            if not int(self.optimized_classifier.predict(features)[0]):
                del self.json_file['bboxes'][i]

        with open(output_directory + json_path.split('/')[-1], 'w') as f_json:
            json.dump(self.json_file, f_json)

    def predict(self, image_path, metadata_path, output_directory, file_type):
        """this method only select the proper function due the chosen file type

        Arguments:
            image_path {str} -- the path of the image
            metadata_path {str} -- the path of the CSV or JSON file
            file_type {str} -- the type (CSV|JSON) of the metadata file
            output_directory {str} -- the path of the output directory
        """
        if file_type == 'csv':
            self.csv_predict(image_path, metadata_path, output_directory)
        else:
            self.json_predict(image_path, metadata_path, output_directory)


if __name__ == "__main__":

    try:
        OPTS, ARGS = getopt.getopt(sys.argv[1:], 'i:m:o:t:f:h', ['input_image_directory',
                                                               'input_metadata_directory',
                                                               'output_directory'
                                                               'meta_type',
                                                               'image_type'
                                                               'help'])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(1)

    IDENTIFIER = Identifier()
    IDENTIFIER.pipeline()
    IDENTIFIER.load_optimized_classifier('classifiers/poly_2020-04-21_21:18:09.plk')
    METADATA_PATH = list()
    IMAGES_PATH = list()
    META_TYPE = None
    IMAGE_TYPE = None

    for opt, arg in OPTS:

        if opt in ('-h', '--help'):
            print('nodules_Identifer.py -i --input_image_directory -t --meta_type [CSV|JSON] '
                  '-f --image_type -m --input_metadata_directory -h --help')
            sys.exit(2)

        elif opt in ('-t', '--meta_type'):
            if arg.lower() == 'csv':
                META_TYPE = 'csv'
            elif arg.lower() == 'json':
                META_TYPE = 'json'
            else:
                print('Invalid metadata format')
                sys.exit(3)

        elif opt in ('-f', '--image_type'):
            if arg.lower() == 'png':
                IMAGE_TYPE = 'png'
            elif arg.lower() == 'jpg':
                IMAGE_TYPE = 'jpg'
            else:
                print('Invalid image format')
                sys.exit(4)

    if not META_TYPE:
        print('missing --meta_type parameter (CSV|JSON)')
        sys.exit(4)
    elif not IMAGE_TYPE:
        print('missing --image_type parameter (PNG|JPG)')
        sys.exit(5)

    for opt, arg in OPTS:

        if opt in ('-i', '--input_image_directory'):

            if os.path.isdir(arg):
                print(IMAGE_TYPE)
                for image_file in glob.glob('./{}/*.{}'.format(arg, IMAGE_TYPE)):
                    IMAGES_PATH.append(image_file)
            else:
                print('the input directory {} does not exist\n'.format(arg))
                sys.exit(6)

        elif opt in ('-m', '--input_metadata_directory'):
            if os.path.isdir(arg):
                for meta_file in glob.glob('./{}/*.{}'.format(arg, META_TYPE)):
                    METADATA_PATH.append(meta_file)
            else:
                print('the input directory {} does not exist\n'.format(arg))
                sys.exit(7)

    if IMAGES_PATH == list():
        print('\nNo image loaded. Check the input image folder\n')
        sys.exit(8)
    elif METADATA_PATH == list():
        print('\nNo CSV or JSON loaded. Check the input metadate folder\n')
        sys.exit(9)

    METADATA_PATH.sort()
    IMAGES_PATH.sort()

    for opt, arg in OPTS:

        if opt in ('-o', '--output_directory'):
            if not os.path.exists(arg):
                os.mkdir(arg)

            for imag, meta in zip(IMAGES_PATH, METADATA_PATH): 

                jpg_name = imag.split('/')[-1].split('.')[-2]
                meta_name = meta.split('/')[-1].split('.')[-2]

                if jpg_name == meta_name:
                    print(jpg_name, meta_name)
                    IDENTIFIER.predict(imag, meta, arg, META_TYPE)
                    IDENTIFIER.reset_attributes()
