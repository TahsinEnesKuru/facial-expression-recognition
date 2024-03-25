import cv2
import numpy as np

import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class SIFT:
    def extract_sift_features(self,images):
        sift = cv2.SIFT_create()
        descriptors_list = []
        for img in images:
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                descriptors_list.append(des)
        return descriptors_list

    def match_features(self,query_features, train_features, ratio=0.75):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        good_matches = []
        for q_des in query_features:
            if len(q_des) > 0:  # make sure that query descriptor is not empty
                all_matches = bf.knnMatch(q_des, np.vstack(train_features), k=2)
                # Apply ratio test
                good = []
                for m, n in all_matches:
                    if m.distance < ratio * n.distance:
                        good.append([m])
                good_matches.append(len(good))
            else:
                good_matches.append(0)
        return good_matches

    def classify_expression(self,test_img, reference_features):
        # Extract features from the test image
        test_features = self.extract_sift_features([test_img])
        
        # Match test image features to reference features and count good matches
        matches = {expression: self.match_features(test_features, ref_des) for expression, ref_des in reference_features.items()}
        
        # Find the best match (expression with the highest match count)
        best_match = max(matches, key=lambda x: sum(matches[x]))
        
        return best_match, matches

