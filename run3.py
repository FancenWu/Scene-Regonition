from sklearn.svm import LinearSVC
import os
import time
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

s_time = time.time()
class_paths = glob.glob('dataset/training/*')
class_names = [name[17:] for name in class_paths]


# Function to load the training images (files and their corresponding classes)
def load_train_images(num_per_class):
    train_images = []
    train_labels = []
    for class_name in class_names:
        image_paths = glob.glob('dataset/training/' + class_name + '/*')
        image_paths = image_paths[:num_per_class]
        for image_path in image_paths:
            train_images.append(cv.imread(image_path, 0))
            train_labels.append(class_name)
    return train_images, train_labels


# Function to load the testing dataset(files and file names)
def load_test_images(number_of_images):
    test_images = []
    file_names = []
    image_paths = glob.glob('dataset/testing/*')
    image_paths = image_paths[:100]
    for image_path in image_paths:
        test_images.append(cv.imread(image_path, 0))
        file_names.append(image_path[16:])
    return test_images, file_names


# Function to generate denseSIFT features for an image
def gen_denseSIFT_features(img):
    step_size = 3
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    # kp is a list of keypoints
    kp = [cv.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
          for x in range(0, gray.shape[1], step_size)]
    # desc is the denseSIFT descriptors
    desc = sift.compute(gray, kp)

    img = cv.drawKeypoints(gray, kp, img)
    plt.imshow(cv.drawKeypoints(gray, kp, img))
    plt.show()
    return kp, desc


# training_data, training_labels = load_train_images(100)
# testing_data, testing_filenames = load_test_images(100)
bedroom = cv.imread('dataset/training/bedroom/0.jpg')
gen_denseSIFT_features(bedroom)
e_time = time.time()
print("Processing " + str(e_time - s_time) + 's')
