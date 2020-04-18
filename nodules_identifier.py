import os
import cv2
import numpy as np
from skimage.measure import moments_hu
from mahotas.features import haralick


class Identifier:

    def __init__(self):
        self.features = None
        self.labels = None

    @staticmethod
    def load_images_from_folder(folder):
        images = list()
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        return images

    @staticmethod
    def extract_features(image):
        return np.r_[moments_hu(image), haralick(image).flatten()]

    def save_extracted_features(self, bboxes_path):
        positive_instance = self.load_images_from_folder(bboxes_path + 'nodules/')
        negative_instance = self.load_images_from_folder(bboxes_path + 'non-nodules/')

        positive_features = np.array(list(map(self.extract_features, positive_instance)))
        negative_features = np.array(list(map(self.extract_features, negative_instance)))

        features = np.r_[positive_features, negative_features]
        labels = np.r_[np.ones(len(positive_instance)), np.zeros(len(negative_instance))]

        np.save('features/features.npy', features)
        np.save('features/labels.npy', labels)

    def load_features(self):
        self.features = np.load('features/features.npy')
        self.labels = np.load('features/labels.npy')

