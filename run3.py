import os
import time
import cv2 as cv
import numpy as np
import glob
from natsort import natsorted
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

# Get the class names of training data
s_time = time.time()
class_paths = glob.glob('dataset/training/*')
class_names = [name[17:] for name in class_paths]

# Sort the test data in ascending order based on file name
test_files = glob.glob('dataset/testing/*')
test_files = natsorted(test_files)


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
    print("Success to load training images")
    return train_images, train_labels


# Function to load the testing dataset(files and file names)
def load_test_images(number_of_images):
    test_images = []
    file_names = []
    image_paths = test_files
    image_paths = image_paths[:number_of_images]
    for image_path in image_paths:
        test_images.append(cv.imread(image_path, 0))
        file_names.append(image_path[16:])
    print("Success to load test images")
    return test_images, file_names


# Function to generate denseSIFT features for an image.
def gen_denseSIFT_features(img):
    step_size = 10
    rows, cols = img.shape
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    # kp = []
    # kp is a list of keypoints obtained by scanning pixels
    kp = [cv.KeyPoint(x, y, step_size) for y in range(0, rows, step_size)
           for x in range(0, cols, step_size)]

    # for x in range(step_size, cols, step_size):
    #     for y in range(step_size, rows, step_size):
    #         kp.append(cv.KeyPoint(x, y, bin_size))

    descriptors = sift.compute(img, kp)[1]
    return descriptors


# Build the codebook
# desc_list is set of descriptors; k is the number of clusters
def gen_codebook(desc_list, k):
    print("Building the Codebook, it will take some time")
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(desc_list)
    # codebook = kmeans.cluster_centers_
    c_time = time.time()
    print("Successfully build codebook, it costs(s): ", c_time - s_time)
    return kmeans


# Build the histogram with spatial pyramid matching using different levels
# img is input image file; level ranging from 0 to 2; A codebook with k vocabularies
def build_spatial_pyramid(img, level, codebook, k):
    height, width = img.shape
    pyramid = []
    # Build histograms for each level
    for l in range(level + 1):
        if l == 0:
            desc0 = gen_denseSIFT_features(img)
            predict0 = codebook.predict(desc0)
            hist0 = np.bincount(predict0, minlength=k).reshape(1, -1).ravel()
            pyramid.append(hist0 * 0.25)
        if l == 1:
            a, b = 0, 0
            for y in range(1, (2 ** l) + 1):
                b = 0
                for x in range(1, (2 ** l) + 1):
                    desc1 = gen_denseSIFT_features(
                        img[a: a + height // (2 ** l), b: b + width // (2 ** l)])
                    predict1 = codebook.predict(desc1)
                    hist1 = np.bincount(predict1, minlength=k).reshape(1, -1).ravel()
                    pyramid.append(hist1 * 0.25)
                    b = b + width // (2 ** l)
                a = a + height // (2 ** l)
        if l == 2:
            a, b = 0, 0
            for y in range(1, (2 ** l) + 1):
                b = 0
                for x in range(1, (2 ** l) + 1):
                    desc2 = gen_denseSIFT_features(
                        img[a: a + height // (2 ** l), b: b + width // (2 ** l)])
                    predict2 = codebook.predict(desc2)
                    hist2 = np.bincount(predict2, minlength=k).reshape(1, -1).ravel()
                    pyramid.append(hist2 * 0.5)
                    b = b + width // (2 ** l)
                a = a + height // (2 ** l)
    pyramid = np.array(pyramid).ravel()
    # print("Success to build the spatial pyramid")

    # normalize the histogram
    # dev = np.std(pyramid)
    # pyramid -= np.mean(pyramid)
    # pyramid /= dev

    [pyramid] = normalize([pyramid], norm="l1")
    return pyramid


# Get pyramid for each image and append them to a 2D array
def get_pyramid(data, level, codebook, k):
    result = []
    for i in range(len(data)):
        pyramid = build_spatial_pyramid(data[i], level, codebook, k)
        # print(pyramid.shape)
        pyramid_size = pyramid.shape[0]
        result = np.array(result)
        result = np.append(result, pyramid)
    result = result.reshape(len(data), pyramid_size)
    return result


# Extract denseSIFT descriptors from training data
def extract_descriptors(data):
    train_desc = [gen_denseSIFT_features(img) for img in data]
    train_desc_list = []
    for i in range(len(train_desc)):
        for j in range(train_desc[i].shape[0]):
            train_desc_list.append(train_desc[i][j, :])
    train_desc_list = np.array(train_desc_list)
    return train_desc_list


training_data, training_labels = load_train_images(100)
testing_data, testing_filenames = load_test_images(2985)

k = 200
print("Codebook size is: ", k)
all_train_desc = extract_descriptors(training_data)
print("The shape of descriptors: ", all_train_desc.shape)
codebook = gen_codebook(all_train_desc, k)

training_hist = get_pyramid(training_data, 2, codebook, k)
testing_hist = get_pyramid(testing_data, 2, codebook, k)
training_labels = np.asarray(training_labels)
print(training_hist.shape)
print(testing_hist.shape)
print(training_labels.shape)

# Train a linear SVM
clf = LinearSVC(loss="hinge", random_state=0, max_iter=2000)
clf.fit(training_hist, training_labels)
predict = clf.predict(testing_hist)

# write the predication to run3.txt file
file = open("run3_2.txt", "w")
for i in range(len(predict)):
    file.write(testing_filenames[i] + " " + predict[i] + "\n")
file.close()

e_time = time.time()
print("Processing " + str(e_time - s_time) + 's')
