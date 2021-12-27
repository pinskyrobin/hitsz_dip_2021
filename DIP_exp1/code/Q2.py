import cv2
import numpy as np


def fill(src):
    # Read image
    image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    row, col = image.shape

    # binarization
    # Set values equal to or above 220 to 255.
    # Set values below 220 to 0.

    binarization_img = image.copy()

    for i in range(row):
        for j in range(col):
            if image[i][j] > 220:
                binarization_img[i][j] = 255
            else:
                binarization_img[i][j] = 0

    # Construct a mask matrix
    floodfill_img = binarization_img.copy()

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            floodfill_img[i][j] = 0

    # make a floodfill!
    while True:
        temp = floodfill_img.copy()
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                if floodfill_img[i - 1, j - 1] or floodfill_img[i - 1, j] or \
                        floodfill_img[i - 1, j + 1] or floodfill_img[i, j - 1] or \
                        floodfill_img[i, j] or floodfill_img[i, j + 1] or \
                        floodfill_img[i + 1, j - 1] or floodfill_img[i + 1, j] or \
                        floodfill_img[i + 1, j + 1]:
                    temp[i, j] = 255
        temp = temp & binarization_img
        difference = cv2.subtract(temp, floodfill_img)
        if not np.any(difference):
            break
        floodfill_img = temp

    # Invert floodfilled image
    inverted_floodfill_img = floodfill_img.copy()

    for i in range(row):
        for j in range(col):
            if floodfill_img[i][j] == 0:
                inverted_floodfill_img[i][j] = 255
            else:
                inverted_floodfill_img[i][j] = 0

    return inverted_floodfill_img


if __name__ == '__main__':
    im_out = fill("DIP_Project1/Q2_images/rye_catcher_c_1.png")
    cv2.imwrite("Q2_res_c.png", im_out)
    im_out = fill("DIP_Project1/Q2_images/rye_catcher_e_1.png")
    cv2.imwrite("Q2_res_e.png", im_out)
