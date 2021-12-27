import cv2 as cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def load(txt_dir):
    label = []
    fd = open(txt_dir, 'r')
    line = fd.readline()[:-1]
    while line:
        pic_name, pic_label = line.split(' ', 1)
        label.append(int(pic_label))
        line = fd.readline()[:-1]
    return label


def FER():
    # get lists that contains labels of every image
    train_lable = load("dataset/train_labels.txt")
    test_lable = load("dataset/test_labels.txt")

    # read images
    train_images = []
    test_images = []
    for i in range(926):
        train_images.append(cv.imread('dataset/train/' + str(i + 1) + '.png', 0))
    for i in range(927, 1237):
        test_images.append(cv.imread('dataset/test/' + str(i) + '.png', 0))


    # size of the checking window
    winSize = (24, 24)
    # size of the block
    blockSize = (24, 24)
    # step of the block
    blockStride = (8, 8)
    # size of the cell
    cellSize = (8, 8)
    # number of histogram bins
    nbins = 9

    # do feature extraction
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # step of the window
    winStride = (8, 8)
    padding = (8, 8)

    # texture feature extraction
    train_hog = []
    test_hog = []
    for i in range(310):
        test_hog.append(hog.compute(test_images[i], winStride, padding).reshape((-1,)))
    for i in range(926):
        train_hog.append(hog.compute(train_images[i], winStride, padding).reshape((-1,)))

    # dimension reduction using LDA
    lda = LinearDiscriminantAnalysis(n_components=6)
    lda.fit(train_hog, train_lable)
    train_lda = lda.transform(train_hog)
    test_lda = lda.transform(test_hog)

    # training the KNN-classifier
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_lda, train_lable)

    # predict by using KNN-classifier
    predict_lable = neigh.predict(test_lda)

    # calculate the accuracy of the algorithm
    cnt = 0
    for i in range(310):
        if predict_lable[i] == test_lable[i]:
            cnt = cnt + 1

    print("Done! accuracy=" + str(cnt / 310))


if __name__ == '__main__':
    FER()
