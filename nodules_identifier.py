import os
import cv2
import numpy as np
from skimage.measure import moments_hu
from mahotas.features import haralick


class Identifier:

    def __init__(self):
        self.features = None
        self.labels = None
        self.X_train = None  
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cross_val = StratifiedKFold(n_splits=5)
        self.default_classifier = SVC(class_weight='balanced', 
                                      decision_function_shape='ovo', 
                                      cache_size=4000)
        self.default_params = {
                               'C': reciprocal(1, 1000),
                               'gamma': ['scale'],
                               'coef0': np.arange(0, 10, 0.001),
                               'degree': range(1, 10)
                              }
        self.optimized_classifier = None

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
        
    def split_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, 
                                                            self.labels, 
                                                            stratify=self.labels, 
                                                            test_size=0.2)
        self.X_train = X_train  
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def normalize(self):
        scaler = RobustScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
    def training(self, kernel, n_iter, scoring='f1_weighted'):
        self.default_params['kernel'] = [kernel]
        
        ran_search = RandomizedSearchCV(self.default_classifier, 
                                        param_distributions=self.default_params,  
                                        cv=self.cross_val, scoring=scoring, 
                                        n_iter=n_iter, verbose=3, n_jobs=4)
        
        ran_search.fit(self.X_train, self.y_train)
        print('Best score: {}'.format(ran_search.best_score_))
        print('Best parameters: {}'.format(ran_search.best_params_))
        
        joblib.dump(ran_search.best_estimator_, 'classifiers/{}_{}.plk'.format(kernel, n_iter))
        
    def load_optimized_classifier(self, classifier_path):
        self.optimized_classifier = joblib.load(classifier_path)
        print('Parameters: {}'.format(self.optimized_classifier.get_params()))
