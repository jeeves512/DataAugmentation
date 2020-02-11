import cv2
import numpy as np


def flip_image(img):
    return cv2.flip(img, 1)  # 1 for flipping vertical


def dark_image(img, low=0.2, high=0.75):
    img = img.astype(np.float32)  # convert to float
    r_num = np.random.uniform(low, high)  # generate random numbers between a range
    img[:, :, :] *= r_num
    np.clip(img, 0, 255)  # values below 0 become - and values above 255 becomes 255
    return img.astype(np.uint8)


def add_random_shadow(img, w_low=0.6, w_high=0.85):
    cols, rows = (img.shape[0], img.shape[1])

    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * bottom_y
        top_y_right = top_y - np.random.random_sample() * top_y

    poly = np.asarray([[[top_y, 0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right, 0]]], dtype=np.int32)

    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight

    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)


image = cv2.imread('road.JPG')
cv2.imshow('shadow', add_random_shadow(image))
cv2.imshow('dark', dark_image(image))
cv2.imshow('flipped image', flip_image(image))

cv2.waitKey(0)
