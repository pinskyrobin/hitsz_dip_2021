import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_histogram(image):
    rows, cols = image.shape[:2]
    histogram = np.zeros([256], dtype=int)
    for r in range(rows):
        for c in range(cols):
            histogram[image[r][c]] += 1
    return histogram


if __name__ == '__main__':
    # STEP 1: read the image and get the gray level matrix of the image
    original_image = cv2.imread('DIP_Project1/Q1_images/face.png', cv2.IMREAD_GRAYSCALE)

    # STEP 2: calculate the gray histogram of the original image
    # STEP 2.1: count how many pixels does every grey level have
    origin_grey2pixel = {}
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            k = original_image[i][j]
            if k in origin_grey2pixel:
                origin_grey2pixel[k] += 1
            else:
                origin_grey2pixel[k] = 1

    # STEP 2.2: sort by grey level from low to high
    grey_level_list = sorted(origin_grey2pixel)

    # STEP 2.3: construct a dict where original dict is sorted
    sorted_grey2pixel = {}

    for j in range(len(grey_level_list)):
        sorted_grey2pixel[grey_level_list[j]] = origin_grey2pixel[grey_level_list[j]]

    # STEP 3: histogram equalization
    # STEP 3.1: construct the probability distribution mapping dict
    probability_dict = {}
    pixels = original_image.shape[0] * original_image.shape[1]

    for grey_level in sorted_grey2pixel.keys():
        probability_dict[grey_level] = sorted_grey2pixel[grey_level] / pixels

    # STEP 3.2: calculate CDF
    cumulative_probability = 0
    for grey_level in probability_dict.keys():
        cumulative_probability += probability_dict[grey_level]
        probability_dict[grey_level] = max(sorted_grey2pixel) * cumulative_probability

    # STEP 3.3: get the gray level matrix of the processed image
    processed_image = np.zeros(shape=(original_image.shape[0], original_image.shape[1]), dtype=int)
    for row in range(original_image.shape[0]):
        for col in range(original_image.shape[1]):
            processed_image[row][col] = probability_dict[original_image[row][col]]

    # STEP 4: get two grey level histograms and draw them
    origin_histogram = get_histogram(original_image)
    processed_histogram = get_histogram(processed_image)

    x = np.arange(256)
    plt.figure(num=1)
    plt.subplot(1, 2, 1)
    plt.plot(x, origin_histogram, 'r', linewidth=2, c='black')
    plt.title("Before processing")
    plt.ylabel("number of pixels")
    plt.subplot(1, 2, 2)
    plt.plot(x, processed_histogram, 'r', linewidth=2, c='black')
    plt.title("After processing")
    plt.ylabel("number of pixels")
    plt.savefig("Q1_face_histogram.png")

    plt.figure(num=2)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap=plt.cm.gray)
    plt.title('Before processing')
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap=plt.cm.gray)
    plt.title('After processing')
    plt.savefig("Q1_face_res.png")
