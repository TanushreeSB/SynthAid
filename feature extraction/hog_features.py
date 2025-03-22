import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True)
    return features

def load_and_extract_features(folder):
    features_list
