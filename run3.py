import os
import time
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

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
    image_paths = image_paths[:number_of_images]
    for image_path in image_paths:
        test_images.append(cv.imread(image_path, 0))
        file_names.append(image_path[16:])
    return test_images, file_names


# Function to generate denseSIFT features for an image.
# Parameter: img is the path of an image file
def gen_denseSIFT_features(img):
    step_size = 2
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    # kp is a list of keypoints obtained by scanning pixels
    kp = [cv.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
          for x in range(0, img.shape[1], step_size)]
    # desc is the denseSIFT descriptors
    desc = sift.compute(img, kp)

    # img = cv.drawKeypoints(gray, kp, img)
    # plt.imshow(cv.drawKeypoints(gray, kp, img))
    # plt.show()
    return desc


# Return all descriptors from a set of images
def get_all_descriptors(data):
    desc_list = []
    for i in range(len(data)):
        desc = gen_denseSIFT_features(data[i])
        desc_list.append(desc)
    result = np.asarray(desc_list)
    return result


# Build the codebook
# descs is set of descriptors from images; k is the number of clusters
def gen_codebook(desc_list, k):
    features = np.vstack((desc for desc in desc_list))
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    print("Successfully build codebook")
    return codebook


# Build the histogram with spatial pyramid matching using different levels
# img is input image file; level ranging from 0 to 2; A codebook with k vocabularies
def build_spatial_pyramid(img, level, codebook, k):
    height, width = img.shape
    step_size = 2  # It equals to the step_size in function gen_denseSIFT_features
    pyramid = []
    # Build histograms for each level
    for l in range(level + 1):
        if l == 0:
            desc0 = gen_denseSIFT_features(img)
            predict0 = codebook.predict(desc0)
            hist0 = np.bincount(predict0, minlength=k).ravel()
            pyramid.append(hist0*0.25)
        if l == 1:
            a, b = 0, 0
            for y in range(1, 3):
                b = 0
                for x in range(1, 3):
                    desc1 = gen_denseSIFT_features(img[a: a + height//2, b: b + width//2])
                    predict1 = codebook.predict(desc1)
                    hist1 = np.bincount(predict1, minlength=k).ravel()
                    pyramid.append(hist1*0.25)
                    b = b + width//2
                a = a + height//2
        if l == 2:
            a, b = 0, 0
            for y in range(1, 5):
                b = 0
                for x in range(1, 5):
                    desc2 = gen_denseSIFT_features(img[a: a + height//4, b: b + width//4])
                    predict2 = codebook.predict(desc2)
                    hist2 = np.bincount(predict2, minlength=k).ravel()
                    pyramid.append(hist2*0.5)
                    b = b + width//4
                a = a + height//4
    pyramid = np.asarray(pyramid).ravel()
    return pyramid


# Get pyramid for each images
def get_pyramid(data, level, codebook, k):
    result = []
    for i in range(len(data)):
        pyramid = build_spatial_pyramid(data[i], level, codebook, k)
        result.append(pyramid)
        result = np.asarray(result)
        return result


training_data, training_labels = load_train_images(100)
testing_data, testing_filenames = load_test_images(100)
k = 200
codebook = gen_codebook(get_all_descriptors(training_data), k)
training_hist = get_pyramid(training_data, 2, codebook, k)
testing_hist = get_pyramid(testing_data, 2, codebook, k)


# Train a linear SVM
clf = LinearSVC(random_state=0)
clf.fit(training_hist, training_labels)
predict = clf.predict(testing_hist)


e_time = time.time()
print("Processing " + str(e_time - s_time) + 's')


