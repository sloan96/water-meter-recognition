# -*- coding:utf-8 -*-
# import tensorflow as tf
import numpy as np
import os
import cv2

import math
import random
from svm_train import svm_config, svm_train, svm_save, svm_load


def rotate_img(img, angle, scale=1):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.bitwise_not(img)
    rotated = cv2.warpAffine(rotated, matrix, (w, h))
    rotated = cv2.bitwise_not(rotated)
    return rotated


def detect_skew(img):
    _imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # _imgGray = cv2.GaussianBlur(_imgGray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    _imgGray = cv2.erode(_imgGray, kernel)
    _imgGray = cv2.dilate(_imgGray, kernel)
    edges = cv2.Canny(_imgGray, 50, 150)

    # print(edges)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 55)  # 这里对最后一个参数使用了经验型的值
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 65)  # 这里对最后一个参数使用了经验型的值
    if lines is None:
        return 0.0
    theta_min = 60.0 * np.pi / 180.0
    theta_max = 120.0 * np.pi / 180.0
    theta_avr = 0.0
    theta_deg = 0.0

    filtered_lines = []

    for x in lines:
        for line in x:
            if len(line) < 2:
                continue
            rho = line[0]
            theta = line[1]
            if theta_min <= theta <= theta_max and rho > 10:
                filtered_lines.append(line)
                theta_avr += theta

    if len(filtered_lines) > 0:
        theta_avr /= len(filtered_lines)
        theta_deg = (theta_avr / np.pi * 180.0) - 90
    #     print("detectSkew: %.1f deg" % theta_deg)
    # else:
    #     print("failed to detect skew")

    # print("lines number: " + str(len(lines)))
    # print("filteredLines number: " + str(len(filtered_lines)))
    # result = img.copy()
    # # result = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # draw_lines(result, filtered_lines)
    #
    # cv2.imshow('OK', img)
    # cv2.imshow('Canny', edges)
    # cv2.imshow('Result', result)
    # cv2.waitKey(-1)
    return theta_deg


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
    threshold = cv2.erode(threshold, kernel2)
    threshold = cv2.dilate(threshold, kernel2)
    threshold = cv2.erode(threshold, kernel2)
    # cv2.imshow("erode", _imgGray)
    threshold = cv2.dilate(threshold, kernel2)

    return threshold


def get_negative_sample(threshold, bounds, box=(128, 48)):
    box_h, box_w = box
    height, width = threshold.shape[:2]
    x, y, w, h = bounds
    ym = h//2 + y

    if ym < height//2:
        ny = random.randint(y + h//2 + 5, height-box_w)
    else:
        ny = random.randint(0, y + h//2 - 5 - box_w)
    nx = random.randint(0, width-box_h)
    return threshold[ny:ny + box_w, nx:nx + box_h]


def get_positive_sample(threshold, bounds, box=(128, 48)):
    box_h, box_w = box
    return threshold[bounds[1]:bounds[1] + box_w, bounds[0]:bounds[0] + box_h]


def get_digits_rect(threshold):
    # threshold = threshold_img(img)

    # cv2.imshow('threshold', threshold)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # y = cv2.dilate(threshold, kernel3)
    # y = cv2.dilate(y, kernel2)
    y = cv2.bitwise_not(threshold)

    # cv2.imshow('y', y)
    edges = y  # cv2.Canny(y, 50, 150)
    # edges = cv2.Canny(y, 50, 150)
    # cv2.imshow('edges', edges)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    c_max = []
    # print("*"*50)
    # bounding_boxes = []
    # filtered_contours = []
    # print(len(contours))
    for contour in contours:
        bounds = cv2.boundingRect(contour)
        # print(bounds)
        # if 20 <= bounds[3] <= 90 and bounds[3] > bounds[2] > 5:
        # if 35 <= bounds[3] <= 70 and bounds[2] > 110:
        if 35 <= bounds[3] <= 80 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
        # if 28 <= bounds[3] <= 50 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
        # if 10 <= bounds[3] and bounds[2] > 10 and bounds[0] > 10 and bounds[1] > 10:
        # if 12 <= bounds[3] <= 18 and 16 > bounds[2] > 8 and bounds[0] > 10 and bounds[1] > 10:
        #     bounding_boxes.append(bounds)
        #     filtered_contours.append(contour)
            # return threshold[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
            # print(bounds)
            return bounds
            # return threshold[bounds[1]:bounds[1] + 48, bounds[0]:bounds[0] + 128]
            # print(bounds)
            # # img2 = np.copy(img)
            # img2 = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            #
            # pt1 = (bounds[0], bounds[1])
            # pt2 = (bounds[0] + bounds[2], bounds[1] + bounds[3])
            # cv2.rectangle(img2, pt1, pt2, (0, 255, 0))
            # cv2.imshow('x', img2)
            # cv2.waitKey(-1)

    # return threshold[30:80, 28:150]
    return None


def get_digits_area(threshold):
    # threshold = threshold_img(img)

    # cv2.imshow('threshold', threshold)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # y = cv2.dilate(threshold, kernel3)
    # y = cv2.dilate(y, kernel2)
    y = cv2.bitwise_not(threshold)

    # cv2.imshow('y', y)
    edges = y  # cv2.Canny(y, 50, 150)
    # edges = cv2.Canny(y, 50, 150)
    # cv2.imshow('edges', edges)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    c_max = []
    # print("*"*50)
    # bounding_boxes = []
    # filtered_contours = []
    # print(len(contours))
    for contour in contours:
        bounds = cv2.boundingRect(contour)
        # print(bounds)
        # if 20 <= bounds[3] <= 90 and bounds[3] > bounds[2] > 5:
        # if 35 <= bounds[3] <= 70 and bounds[2] > 110:
        if 35 <= bounds[3] <= 80 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
        # if 28 <= bounds[3] <= 50 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
        # if 10 <= bounds[3] and bounds[2] > 10 and bounds[0] > 10 and bounds[1] > 10:
        # if 12 <= bounds[3] <= 18 and 16 > bounds[2] > 8 and bounds[0] > 10 and bounds[1] > 10:
        #     bounding_boxes.append(bounds)
        #     filtered_contours.append(contour)
            # return threshold[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
            # print(bounds)
            return get_negative_sample(threshold, bounds), get_positive_sample(threshold, bounds)
            # return threshold[bounds[1]:bounds[1] + 48, bounds[0]:bounds[0] + 128]
            # print(bounds)
            # # img2 = np.copy(img)
            # img2 = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            #
            # pt1 = (bounds[0], bounds[1])
            # pt2 = (bounds[0] + bounds[2], bounds[1] + bounds[3])
            # cv2.rectangle(img2, pt1, pt2, (0, 255, 0))
            # cv2.imshow('x', img2)
            # cv2.waitKey(-1)

    # return threshold[30:80, 28:150]
    return None, None


def pre_precess_img(image):
    theta_deg = detect_skew(image)
    image = jpeg_compression(image)
    image = threshold_img(image)
    image = rotate_img(image, theta_deg)
    ns, ps = get_digits_area(image)
    return ns, ps


def get_img_hog(img, debug=False):
    img = np.sqrt(img / float(np.max(img)))
    if debug:
        cv2.imshow("norm img", img)

    height, width = img.shape[:2]
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    if debug:
        print(gradient_magnitude.shape, gradient_angle.shape)

    cell_size = 8
    bin_size = 8
    angle_unit = 360 / bin_size
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))
    if debug:
        print(cell_gradient_vector.shape)

    def cell_gradient(cell_magnitude, cell_angle):
        orientation_centers = [0] * bin_size
        for k in range(cell_magnitude.shape[0]):
            for l in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[k][l]
                gradient_angle = cell_angle[k][l]
                min_angle = int(gradient_angle / angle_unit) % 8
                max_angle = (min_angle + 1) % bin_size
                mod = gradient_angle % angle_unit
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
        return orientation_centers

    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            # print(cell_angle.max())

            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

    if debug:
        hog_image = np.zeros([height, width])
        cell_gradient = cell_gradient_vector
        cell_width = cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap


        # plt.imshow(hog_image, cmap=plt.cm.gray)
        # plt.show()
        cv2.imshow("hog_image", hog_image)

    # fifth part
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)

    if debug:
        print("np.array(hog_vector).shape", np.array(hog_vector).shape)
    return np.array(hog_vector)


def img_hog(img_path, debug=False):
    if debug:
        print(img_path)
    img = cv2.imread(img_path)

    ns, ps = pre_precess_img(img)
    if ns is None or ps is None:
        return None, None
    if debug:
        cv2.imshow('pre_precess_img', img)
        cv2.imshow('negative sample', ns)
        cv2.imshow('positive sample', ps)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hog_vector_ns = get_img_hog(ns, debug=debug)
    if debug:
        cv2.waitKey()
    hog_vector_ps = get_img_hog(ps, debug=debug)
    # print(np.array(hog_vector).shape)

    if debug:
        cv2.waitKey()

    return hog_vector_ns, hog_vector_ps


def main():
    # dirpath = "D:\\support\\watermeter\\image111\\"
    # dirpath = "Z:\\work\\WMImages\\image3\\"
    dirpath = "../../data/image-process-2-2"

    for (root, dirs, files) in os.walk(dirpath, False):
        for filename in files:
            if not filename.endswith(".bmp"):
                continue
            file_path = os.path.join(root, filename)
            img_hog(file_path, debug=True)

    cv2.destroyAllWindows()


def generate_hog_data(batch_size=1000):
    dirpath = "../../data/image-process-2-2"
    features = np.zeros((batch_size, 75, 32), np.float32)
    labels = np.zeros(batch_size, np.float32)
    pos_count = 0

    target_shape = (75, 32)
    for (root, dirs, files) in os.walk(dirpath, False):
        for filename in files:
            if not filename.endswith(".bmp"):
                continue
            file_path = os.path.join(root, filename)
            # img = cv2.imread(file_path)
            # ns, ps = pre_precess_img(img)
            ns, ps = img_hog(file_path)
            if ns is None or ps is None\
                    or target_shape != ns.shape or target_shape != ps.shape:
                continue

            # cv2.imshow('pre_precess_img', img)
            # cv2.imshow('negative sample', ns)
            # cv2.imshow('positive sample', ps)
            # cv2.waitKey()
            print(ns.shape)
            features[pos_count] = ns
            labels[pos_count] = -1
            pos_count += 1
            features[pos_count] = ps
            labels[pos_count] = 1
            pos_count += 1
            # features.append(ns)
            # labels.append(-1)
            # pos_count += 1
            # features.append(ps)
            # labels.append(1)
            # pos_count += 1
            print(pos_count*100 // batch_size, "% (", pos_count, "//", batch_size, ")")
            if batch_size <= pos_count:
                break
    features = np.reshape(features, (batch_size, 75*32))
    return features, labels


# 获取svm参数
def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


# hog训练
def hog_train():
    # features = []
    # labels = []

    svm = svm_config()

    # get hog features
    features, labels = generate_hog_data(batch_size=10000)
    print(features.shape, labels.shape)
    # svm training
    print('svm training...')
    svm_train(svm, features, labels)
    print('svm training complete...')

    svm_save(svm, "../../models/HOGSVM_Digits.bin")


def hog_test2():
    dirpath = "../../data/image-process-2-2"
    pos_count = 0
    svm = svm_load("../../models/HOGSVM_Digits.bin")
    target_shape = (75, 32)

    features = np.zeros((1, 2400), np.float32)
    for (root, dirs, files) in os.walk(dirpath, False):
        for filename in files:
            if not filename.endswith(".bmp"):
                continue
            file_path = os.path.join(root, filename)

            ns, ps = img_hog(file_path)
            if ns is None or ps is None \
                    or target_shape != ns.shape or target_shape != ps.shape:
                continue
            features[0] = np.reshape(ns, (2400, ))
            features = np.reshape(features, (1, 2400))
            r, predicted = svm.predict(features)
            print("r=", r)
            print("predicted=", predicted)

            features[0] = np.reshape(ps, (2400,))
            features = np.reshape(features, (1, 2400))
            r, predicted = svm.predict(features)
            print("r=", r)
            print("predicted=", predicted)

            img = cv2.imread(file_path)
            cv2.imshow('pre_precess_img', img)
            cv2.waitKey()


def get_img_hog_batch(img, debug=False):

    img = np.sqrt(img / float(np.max(img)))
    if debug:
        cv2.imshow("norm img", img)

    height, width = img.shape[:2]
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    if debug:
        cv2.imshow("gradient_values_x", gradient_values_x)
        cv2.imshow("gradient_values_y", gradient_values_y)
        cv2.imshow("gradient_magnitude img", gradient_magnitude)
        cv2.imshow("gradient_angle img", gradient_angle)
        print(gradient_magnitude.shape, gradient_angle.shape)

    cell_size = 8
    bin_size = 8
    angle_unit = 360 / bin_size
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))
    if debug:
        print(cell_gradient_vector.shape)

    def cell_gradient(cell_magnitude, cell_angle):
        orientation_centers = [0] * bin_size
        for k in range(cell_magnitude.shape[0]):
            for l in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[k][l]
                gradient_angle = cell_angle[k][l]
                # print("gradient_strength.shape:", gradient_strength)
                # print("gradient_angle.shape:", gradient_angle)
                # print("gradient_angle / angle_unit:", gradient_angle / angle_unit)
                min_angle = int(gradient_angle / angle_unit) % 8
                max_angle = (min_angle + 1) % bin_size
                mod = gradient_angle % angle_unit
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))

        return orientation_centers

    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            # print(cell_angle.max())

            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
            if i == 8 and j == 15:
                print(cell_magnitude)
                print("=" * 50)
                print(cell_angle)
                print("=" * 50)
                print(cell_gradient_vector[i][j])

    for y in range(cell_gradient_vector.shape[0]):
        for x in range(cell_gradient_vector.shape[1]):
            for c in range(cell_gradient_vector.shape[2]):
                print(y,x,c,cell_gradient_vector[y,x,c])

    if debug:
        hog_image = np.zeros([height, width])
        cell_gradient = cell_gradient_vector
        cell_width = cell_size / 2
        max_mag = np.array(cell_gradient).max()
        print(max_mag)
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                # print(cell_grad.shape)
                # print(cell_grad)
                cell_grad /= max_mag
                angle = 0
                angle_gap = angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap

        # plt.imshow(hog_image, cmap=plt.cm.gray)
        # plt.show()
        cv2.imshow("hog_image", hog_image)

    cy, cx, cch = cell_gradient_vector.shape[:3]
    box = (6, 16)
    box_h, box_w = box
    move_y = cy - box_h
    move_x = cx - box_w
    block_length = cch * 4  # 32
    feature_length = (box_h - 1) * (box_w - 1) * block_length  # 2400
    features = np.zeros((move_y * move_x, feature_length), np.float32)
    for y in range(0, move_y):
        for x in range(0, move_x):
            sub = cell_gradient_vector[y:y+box_h, x:x+box_w, :]
            # print("sub.shape", sub.shape)
            # fifth part
            hog_vector = []
            for i in range(sub.shape[0] - 1):
                for j in range(sub.shape[1] - 1):
                    block_vector = []
                    block_vector.extend(sub[i][j])
                    block_vector.extend(sub[i][j + 1])
                    block_vector.extend(sub[i + 1][j])
                    block_vector.extend(sub[i + 1][j + 1])
                    mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                    magnitude = mag(block_vector)
                    if magnitude != 0:
                        normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                        block_vector = normalize(block_vector, magnitude)
                    hog_vector.append(block_vector)
            # if debug:
            #     print("np.array(hog_vector).shape", np.array(hog_vector).shape)
            hog_vector = np.array(hog_vector)
            hog_vector = np.reshape(hog_vector, 2400)
            features[x+y*move_x] = hog_vector

    # print("features.shape", features.shape)
    return features


def hog_predict(image, svm, box_shape=(48, 128), step=(8, 8)):
    theta_deg = detect_skew(image)
    # image = jpeg_compression(image)
    image = threshold_img(image)
    image = rotate_img(image, theta_deg)
    cv2.imshow('pre_precess_img', image)

    height, width = image.shape[:2]
    box_y, box_x = box_shape
    step_y, step_x = step

    move_width = (width-box_x)//step_x
    move_height = (height-box_y)//step_y
    features = get_img_hog_batch(image, debug=True)
    # features = np.zeros((move_width*move_height, 2400), np.float32)
    # for y in range(0, move_height):
    #     for x in range(0, move_width):
    #         tx = x * step_x
    #         ty = y * step_y
    #         hog = get_img_hog(image[ty:ty+box_y, tx:tx+box_x], debug=True)
    #         features[y * move_width + x] = np.reshape(hog, 2400)
    #
    # print(features.shape)

    r, predicted = svm.predict(features)
    print("r=", r)
    _positon = np.argmax(predicted)
    predicted = np.reshape(predicted, (move_height, move_width))
    print("predicted.shape=", predicted.shape)
    print("_positon=", _positon)
    ty = (_positon//move_width)
    tx = (_positon - move_width*ty)*step_x
    ty = ty * step_y
    return tx, ty, box_x, box_y



def hog_test():
    # dirpath = "Z:\\work\\WMImages\\image3\\"
    # dirpath = "D:\\support\\watermeter\\image111\\"
    # dirpath = "D:\\support\\watermeter\\image99\\"
    # dirpath = "I:\\image\\"
    dirpath = "../../data/image-process-2-2"

    pos_count = 0
    svm = svm_load("../../models/HOGSVM_Digits.bin")
    target_shape = (48, 128)

    features = np.zeros((1, 2400), np.float32)
    for (root, dirs, files) in os.walk(dirpath, False):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            image = cv2.imread(file_path)

            cv2.imshow('source', image)

            # theta_deg = detect_skew(image)
            # image = jpeg_compression(image)
            # image = threshold_img(image)
            # image = rotate_img(image, theta_deg)
            # x = get_img_hog_batch(image, debug=True)
            # print(x.shape)

            x, y, w, h = hog_predict(image, svm, box_shape=target_shape)
            ns, ps = pre_precess_img(img)
            theta_deg = detect_skew(image)
            image = rotate_img(image, theta_deg)

            # image = jpeg_compression(image)
            print(x, y, w, h)
            pt1 = (x, y)
            pt2 = (x+w, y+h)
            cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
            cv2.imshow('digit', image)
            cv2.waitKey()

            # return


def predict_digits_rect(image):
    theta_deg = detect_skew(image)
    image = jpeg_compression(image)
    image = threshold_img(image)
    image = rotate_img(image, theta_deg)
    digits_rect = get_digits_rect(image)
    return digits_rect


def save_list(filename, lst):
    with open(filename, 'wt') as f:
        for i in lst:
            print(i, file=f)


def draw_rect(img, rect, color):
    x, y, w, h = rect
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, color, 1)


def calc_intersection(rect1, rect2):
    x11, y11, w1, h1 = rect1
    x12 = x11 + w1
    y12 = y11 + h1
    x21, y21, w2, h2 = rect2
    x22 = x21 + w2
    y22 = y21 + h2
    # print((x11 < x21 and x12 < x21), (x21 < x11 and x22 < x11),
    #         (y11 < y21 and y12 < y21), (y21 < y11 and y22 < y11))
    if (x11 < x21 and x12 < x21) or (x21 < x11 and x22 < x11) or \
            (y11 < y21 and y12 < y21) or (y21 < y11 and y22 < y11):
        return 0
    # print(x12, x22, x11, x22, min(x12, x22), max(x11, x22))
    iw = max(x11, x21) - min(x12, x22)
    # iw = min(x12, x22) - max(x11, x22)
    ih = max(y11, y21) - min(y12, y22)
    # ih = min(y12, y22) - max(y11, y22)
    intersection = iw*ih
    # print(iw, ih)
    # print(intersection, (w1 * h1 + w2 * h2 - intersection))
    iou = intersection * 1.0 / (w1 * h1 + w2 * h2 - intersection)
    return iou


def hog_label():
    # dirpath = "Z:\\work\\WMImages\\image3\\"
    # dirpath = "D:\\support\\watermeter\\image111\\"
    # dirpath = "D:\\support\\watermeter\\exception\\"
    # dirpath = "D:\\support\\watermeter\\112\\"
    # dirpath = "I:\\image\\"
    dirpath = "../../data/image-process-2-2"

    sample_list_easy = []
    sample_list_hard = []
    sample_list_sum = []

    pos_count = 0
    svm = svm_load("../../models/HOGSVM_Digits.bin")
    target_shape = (48, 128)

    features = np.zeros((1, 2400), np.float32)
    for (root, dirs, files) in os.walk(dirpath, False):
        for filename in files:
            # if filename != "60914_2018-11-23-9-19-58_003565861.9557.bmp":
            # if filename != "80811_2018-11-23-9-19-58_017841873.4968.bmp":
            #     continue
            file_path = os.path.join(root, filename)
            print(file_path)
            image = cv2.imread(file_path)

            cv2.imshow('source', image)

            # theta_deg = detect_skew(image)
            # image = jpeg_compression(image)
            # image = threshold_img(image)
            # image = rotate_img(image, theta_deg)
            # x = get_img_hog_batch(image, debug=True)
            # print(x.shape)

            img_bak = np.copy(image)
            hog_rect = hog_predict(img_bak, svm, box_shape=target_shape)
            img_bak = np.copy(image)
            digits_rect = predict_digits_rect(img_bak)
            theta_deg = detect_skew(image)
            image = rotate_img(image, theta_deg)

            # image = jpeg_compression(image)
            draw_rect(image, hog_rect, (255, 0, 0))

            if digits_rect is not None:
                draw_rect(image, digits_rect, (0, 255, 0))
                iou = calc_intersection(digits_rect, hog_rect)
                # print("iou =", iou)
            info = "e{0}:h{1}".format(len(sample_list_easy), len(sample_list_hard))
            cv2.putText(image, info, (0, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            info = "{}".format(len(sample_list_sum))
            cv2.putText(image, info, (0, 45),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            cv2.imshow('digit', image)

            while True:
                ch = cv2.waitKey()
                if ch == ord('q'):
                    return
                elif ch == ord('j'):
                    sample_list_sum.append(file_path)
                    sample_list_easy.append(file_path)
                    break
                elif ch == ord('k'):
                    sample_list_sum.append(file_path)
                    sample_list_hard.append(file_path)
                    break
                elif ch == ord('x'):
                    break

            # return
    save_list("sample_list_sum.txt", sample_list_sum)
    save_list("sample_list_easy.txt", sample_list_easy)
    save_list("sample_list_hard.txt", sample_list_hard)


def hog_predict_rect(image, svm, box_shape=(48, 128), step=(8, 8)):
    height, width = image.shape[:2]
    box_y, box_x = box_shape
    step_y, step_x = step

    move_width = (width-box_x)//step_x
    move_height = (height-box_y)//step_y
    features = get_img_hog_batch(image)
    # features = np.zeros((move_width*move_height, 2400), np.float32)
    # for y in range(0, move_height):
    #     for x in range(0, move_width):
    #         tx = x * step_x
    #         ty = y * step_y
    #         hog = get_img_hog(image[ty:ty+box_y, tx:tx+box_x], debug=True)
    #         features[y * move_width + x] = np.reshape(hog, 2400)
    #
    # print(features.shape)

    r, predicted = svm.predict(features)
    # print("r=", r)
    _positon = np.argmax(predicted)
    predicted = np.reshape(predicted, (move_height, move_width))
    # print("predicted.shape=", predicted.shape)
    # print("_positon=", _positon)
    ty = (_positon//move_width)
    tx = (_positon - move_width*ty)*step_x
    ty = ty * step_y
    # return image[ty:ty+box_y, tx:tx+box_x]
    return tx, ty, box_x, box_y


def hog_predict2(image, svm, box_shape=(48, 128), step=(8, 8)):
    img_bak = np.copy(image)
    hog_rect = hog_predict_rect(img_bak, svm, box_shape=box_shape, step=step)
    img_bak = np.copy(image)
    digits_rect = get_digits_rect(image)

    if digits_rect is not None:
        # draw_rect(image, digits_rect, (0, 255, 0))
        iou = calc_intersection(digits_rect, hog_rect)
        # print("iou =", iou)
        if iou > 0.45:
            ret_rect = digits_rect
        else:
            ret_rect = hog_rect
    else:
        ret_rect = hog_rect

    x, y, w, h = ret_rect
    return image[y:y + h, x:x + w]


def digits_rect_detect(image, svm, box_shape=(48, 128), step=(8, 8)):
    image = np.copy(image)
    theta_deg = detect_skew(image)
    image = jpeg_compression(image)
    image = threshold_img(image)
    image = rotate_img(image, theta_deg)
    img_bak = np.copy(image)
    hog_rect = hog_predict_rect(img_bak, svm, box_shape=box_shape, step=step)
    img_bak = np.copy(image)
    digits_rect = get_digits_rect(image)
    if digits_rect is not None:
        # draw_rect(image, digits_rect, (0, 255, 0))
        iou = calc_intersection(digits_rect, hog_rect)
        # print("iou =", iou)
        if iou > 0.45:
            ret_rect = digits_rect
        else:
            ret_rect = hog_rect
    else:
        ret_rect = hog_rect

    return ret_rect


if __name__ == "__main__":
    # main()
    hog_train()
    # hog_test()
    # hog_label()
