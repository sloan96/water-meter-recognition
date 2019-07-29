# -*- coding: utf-8
import os
import cv2
import numpy as np
import random


def jpeg_compression(img, quality=15):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, img_encode = cv2.imencode(".jpg", img, params)
    # data_encode = np.array(img_encode)
    # print(len(img_encode), "bytes")
    x = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
    return x


def threshold_img(img):
    threshold = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    threshold = cv2.GaussianBlur(threshold, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold = cv2.erode(threshold, kernel2)
    threshold = cv2.dilate(threshold, kernel2)
    threshold = cv2.erode(threshold, kernel2)
    # cv2.imshow("erode", _imgGray)
    threshold = cv2.dilate(threshold, kernel2)

    return threshold


def get_digits_area(threshold):
    # threshold = threshold_img(img)

    # cv2.imshow('threshold', threshold)
    y = cv2.bitwise_not(threshold)
    # cv2.imshow('y', y)
    edges = y  # cv2.Canny(y, 50, 150)
    # cv2.imshow('edges', edges)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    c_max = []

    bounding_boxes = []
    filtered_contours = []
    # print(len(contours))
    for contour in contours:
        bounds = cv2.boundingRect(contour)
        # print(bounds)
        # if 20 <= bounds[3] <= 90 and bounds[3] > bounds[2] > 5:
        # if 35 <= bounds[3] <= 70 and bounds[2] > 110:
        if 35 <= bounds[3] <= 70 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
            bounding_boxes.append(bounds)
            filtered_contours.append(contour)
            return threshold[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
            # print(bounds)
            # img2 = np.copy(img)
            #
            # pt1 = (bounds[0], bounds[1])
            # pt2 = (bounds[0] + bounds[2], bounds[1] + bounds[3])
            # cv2.rectangle(img2, pt1, pt2, (0, 255, 0))
            # cv2.imshow('x', img2)
            # cv2.waitKey(-1)

    return threshold[30:80, 28:150]
    # return None


class WaterWaterData:
    def __init__(self, dirpath="D:\\support\\watermeter\\image2", whole_label=False):
        self.file_list = []
        for (root, dirs, files) in os.walk(dirpath, False):
            for filename in files:
                file_path = os.path.join(root, filename)
                if whole_label:
                    label = filename[2:7]
                else:
                    label = filename[:5]
                self.file_list.append([file_path, label])
                # print(file_path, label)
        random.shuffle(self.file_list)
        self.index = 0

    def keras_label_generator(self, batch_size=50, use_threshold=True):
        # img = get_number_collection_img2()
        channel = 1
        # height = 120
        height = 32
        width = 120
        char_num = 5
        classes = 10
        # w = 28
        # h = 28 / 2
        x_train = np.zeros([batch_size, height, width, channel])
        y_train = np.zeros([batch_size, char_num, classes])

        while True:
            # for i in range(batch_size):
            i = 0
            while i < batch_size:
                file_path, captcha_str = self.file_list[self.index]
                self.index += 1
                if self.index >= len(self.file_list):
                    self.index = 0
                    random.shuffle(self.file_list)
                image = cv2.imread(file_path)
                if use_threshold:
                    # print(file_path)
                    try:
                        if not file_path.endswith(".jpg"):
                            image = jpeg_compression(image)
                        image = threshold_img(image)
                        image = get_digits_area(image)
                    except:
                        print(file_path)
                        continue
                elif channel == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if image is None:
                    continue
                image = cv2.resize(image, (width, height))

                X = np.zeros([1, height, width, channel])
                Y = np.zeros([1, char_num, classes])
                # print(Y.shape)
                # print(y_train.shape)
                # print(i)
                X[0] = np.reshape(image, [height, width, channel]) / 255.0

                for j, ch in enumerate(captcha_str):
                    Y[0, j, ord(ch) - ord('0')] = 1
                # Y = np.reshape(Y, (1, char_num * classes))
                x_train[i] = X
                y_train[i] = Y
                i += 1

            y_train2 = np.reshape(y_train, (batch_size, char_num * classes))
            yield x_train, y_train2

    def keras_whole_label_generator(self, batch_size=50, use_threshold=True):
        # img = get_number_collection_img2()
        channel = 1
        height = 120
        width = 160
        char_num = 5
        classes = 10
        # w = 28
        # h = 28 / 2
        x_train = np.zeros([batch_size, height, width, channel])
        y_train = np.zeros([batch_size, char_num, classes])

        while True:
            # for i in range(batch_size):
            i = 0
            while i < batch_size:
                file_path, captcha_str = self.file_list[self.index]
                self.index += 1
                if self.index >= len(self.file_list):
                    self.index = 0
                    random.shuffle(self.file_list)
                image = cv2.imread(file_path)
                if use_threshold:
                    # print(file_path)
                    try:
                        image = jpeg_compression(image)
                        image = threshold_img(image)
                        # image = get_digits_area(image)
                    except:
                        print(file_path)
                        continue
                elif channel == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if image is None:
                    continue
                image = cv2.resize(image, (width, height))

                X = np.zeros([1, height, width, channel])
                Y = np.zeros([1, char_num, classes])
                # print(Y.shape)
                # print(y_train.shape)
                # print(i)
                X[0] = np.reshape(image, [height, width, channel]) / 255.0

                for j, ch in enumerate(captcha_str):
                    Y[0, j, ord(ch) - ord('0')] = 1
                # Y = np.reshape(Y, (1, char_num * classes))
                x_train[i] = X
                y_train[i] = Y
                i += 1

            y_train2 = np.reshape(y_train, (batch_size, char_num * classes))
            yield x_train, y_train2
