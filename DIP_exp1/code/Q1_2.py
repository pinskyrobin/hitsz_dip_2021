import cv2
import numpy as np


def median_filter(src, dst):

    # construct the grey level matrix
    ori_image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(ori_image, (440, 280))
    row, col = image.shape

    # do nothing when encountering the boundary of the image
    edge = 1
    new_arr = np.zeros((row, col), dtype="uint8")
    for i in range(row):
        for j in range(col):
            # if the pixel isn't at the edge
            # then do nothing
            if i <= edge - 1 or i >= row - 1 - edge or j <= edge - 1 or j >= col - edge - 1:
                new_arr[i, j] = image[i, j]
            # or take the medium grey level as this pixel
            else:
                new_arr[i, j] = np.median(image[i - edge:i + edge + 1, j - edge:j + edge + 1])
    cv2.imwrite(dst, new_arr)


if __name__ == '__main__':
    src_pic = "DIP_Project1/Q1_images/hit.png"
    dst_dir = "Q1_hit_res.png"
    median_filter(src_pic, dst_dir)
