# -*- coding: utf-8 -*-
from socket import *
from time import ctime
import struct
import threading
import numpy as np

import os
import random
import sys
import cv2

import tensorflow as tf
from svm_train import svm_load
from hog_svm_detect_digit import hog_predict2


def draw_lines(img, lines):
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)


class WatermeterPredictor:
    OP_CODE_TRANSFER_FILE_BYTE = 1
    is_stop = True

    def __init__(self):
        self.HOST = ""
        self.PORT = 21327
        self.BUFSIZ = 1024 * 8
        self.ADDR = (self.HOST, self.PORT)

        self.OP_CODE_TRANSFER_FILE_BYTE = 1
        self.OP_CODE_TRANSFER_FILE_STR = 2

        self.T_TransferFileInfo = 'IIIII'
        self.T_TransferFileInfo_size = struct.calcsize(self.T_TransferFileInfo)

        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.characters = range(10)
        self.width = 120
        self.height = 32
        self.mutex = threading.Lock()
        print("init...")
        with tf.gfile.FastGFile(os.path.join(self.current_dir, "../../models/frozen_model.pb"), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        print("init ok")

        self.svm = svm_load(os.path.join(self.current_dir, "../../models/HOGSVM_Digits.bin"))
        self.target_shape = (48, 128)

    @staticmethod
    def rotate_img(img, angle, scale=1):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.bitwise_not(img)
        rotated = cv2.warpAffine(rotated, matrix, (w, h))
        rotated = cv2.bitwise_not(rotated)
        return rotated

    @staticmethod
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
            print("detectSkew: %.1f deg" % theta_deg)
        else:
            print("failed to detect skew")

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

    @staticmethod
    def jpeg_compression(img, quality=15):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, img_encode = cv2.imencode(".jpg", img, params)
        # data_encode = np.array(img_encode)
        # print(len(img_encode), "bytes")
        x = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
        return x

    @staticmethod
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

    @staticmethod
    def get_digits_area(threshold):
        # threshold = threshold_img(img)

        # cv2.imshow('threshold', threshold)
        y = cv2.bitwise_not(threshold)

        # cv2.imshow('y', y)
        edges = y  # cv2.Canny(y, 50, 150)
        # cv2.imshow('edges', edges)
        _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        c_max = []
        print("*"*50)
        bounding_boxes = []
        filtered_contours = []
        # print(len(contours))
        for contour in contours:
            bounds = cv2.boundingRect(contour)
            # print(bounds)
            # if 20 <= bounds[3] <= 90 and bounds[3] > bounds[2] > 5:
            # if 35 <= bounds[3] <= 70 and bounds[2] > 110:
            if 35 <= bounds[3] <= 80 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
            # if 28 <= bounds[3] <= 50 and bounds[2] > 100 and bounds[0] > 10 and bounds[1] > 10:
            # if 10 <= bounds[3] and bounds[2] > 10 and bounds[0] > 10 and bounds[1] > 10:
                bounding_boxes.append(bounds)
                filtered_contours.append(contour)
                return threshold[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
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

    def pre_precess_img(self, image, width, height):
        print("detect_skew")
        theta_deg = self.detect_skew(image)
        print("jpeg_compression")
        image = self.jpeg_compression(image)
        # cv2.imshow('jpeg_compression', image)
        print("threshold_img")
        image = self.threshold_img(image)
        # cv2.imshow('threshold_img', image)
        print("rotate_img")
        image = self.rotate_img(image, theta_deg)
        # cv2.imshow('detect_skew', image)
        # cv2.imshow('hog_predict2', image)
        # image2 = self.get_digits_area(image)
        # if image2 is None:
        #     image2 = hog_predict2(image, self.svm, box_shape=self.target_shape)
        # image = cv2.resize(image2, (width, height))
        print("hog_predict2")
        image = hog_predict2(image, self.svm, box_shape=self.target_shape)
        print("resize")
        image = cv2.resize(image, (width, height))
        return image

    def predict_image(self, image, sess):
        image = self.pre_precess_img(image, self.width, self.height)
        if image is None:
            return None
        # cv2.imshow("image", image)
        out_tensor = sess.graph.get_tensor_by_name('out:0')
        x = sess.graph.get_tensor_by_name('x:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        test_x = np.reshape(image, [self.height, self.width, 1]) / 255.0

        if self.mutex.acquire(15):
            pre_list = sess.run(out_tensor, feed_dict={x: [test_x], keep_prob: 1})
            self.mutex.release()
        else:
            return ''

        s = ''
        for i in pre_list:
            s = ''
            for j in i:
                s += str(self.characters[j])
        return s

    def deal_cmd01(self, tcpCliSock, data, sess):
        dwOpCode, dwPackNum, dwPackSum, dwFileSize, dwCrc32 = struct.unpack(self.T_TransferFileInfo,
                                                                            data[:self.T_TransferFileInfo_size])
        print("dwOpCode:{},dwPackNum:{},dwPackSum:{},dwFileSize:{},dwCrc32:{}".
              format(dwOpCode, dwPackNum, dwPackSum, dwFileSize, dwCrc32))
        tcpCliSock.send(bytes("OK", 'utf-8'))
        print("Ready for receive image")
        rest = dwFileSize

        img_data = []
        while rest > 0:
            tcpCliSock.settimeout(30.0)
            data = tcpCliSock.recv(self.BUFSIZ)
            tcpCliSock.settimeout(None)
            if not data:
                break
            # print(type(data))
            # print(len(data))
            img_data += bytes(data)
            rest -= len(data)

        img_data =np.asarray(bytearray(img_data), dtype="uint8")
        print("FINISH")
        tcpCliSock.send(bytes("FINISH", 'utf-8'))
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        print("imdecode ok")
        predict = self.predict_image(img, sess)

        if predict is None or len(predict) < 5:
            predict = "ERROR"
        print("predict: " + predict)
        # 发回去
        send_data = struct.pack('6s', bytes(predict, 'ascii'))
        tcpCliSock.send(send_data)

    def deal_data(self, tcpCliSock, data, sess):
        dwOpCode = struct.unpack('I', data[:struct.calcsize('I')])
        dwOpCode = dwOpCode[0]
        print('dwOpCode:{0}'.format(dwOpCode))
        # content = str(len(data))

        if dwOpCode == self.OP_CODE_TRANSFER_FILE_BYTE:
            self.deal_cmd01(tcpCliSock, data, sess)
        else:
            tcpCliSock.send(bytes("ER", 'utf-8'))

    def start_socket_server(self):
        tcpSerSock = socket(AF_INET, SOCK_STREAM)
        tcpSerSock.bind(self.ADDR)
        tcpSerSock.listen(5)
        is_stop = True
        with tf.Session() as sess:
            while is_stop:
                print("waiting for connection...")
                tcpCliSock, addr = tcpSerSock.accept()
                print("connected from :", addr)

                while True:
                    try:
                        data = tcpCliSock.recv(self.BUFSIZ)
                        if not data:
                            break
                        self.deal_data(tcpCliSock, data, sess)
                    except Exception as err:
                        print(err)
                        break
                tcpCliSock.close()
            tcpSerSock.close()


def test_model(imgpath):
    image = cv2.imread(imgpath)
    img_src = np.copy(image)
    cv2.imshow("img_src", img_src)
    wp = WatermeterPredictor()
    with tf.Session() as sess:
        predict = wp.predict_image(image, sess)
        print(predict)
    cv2.waitKey(-1)


def test_model_dir(dir_path):
    file_list = []
    for (root, dirs, files) in os.walk(dir_path, False):
        for filename in files:
            if not filename.endswith(".bmp") and not filename.endswith(".jpg") and not filename.endswith(".png"):
                continue
            file_path = os.path.join(root, filename)
            file_list.append(file_path)
    random.shuffle(file_list)

    wp = WatermeterPredictor()
    with tf.Session() as sess:
        for file_path in file_list:
            print(file_path)
            image = cv2.imread(file_path)
            img_src = np.copy(image)

            predict = wp.predict_image(image, sess)
            if predict is None:
                predict = "Error"

            cv2.putText(img_src, predict, (30, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            cv2.imshow("img_src", img_src)
            cv2.waitKey(-1)


def start_server():
    wp = WatermeterPredictor()
    wp.start_socket_server()


if __name__ == '__main__':
    # test_model("D:\\support\\watermeter\\iii\\2018-11-27 11-20-12_000000078.0045.bmp")
    # test_model_dir("D:\\support\\watermeter\\image115")
    # print(len(sys.argv))
    if len(sys.argv) > 1:
        test_model_dir(sys.argv[1])
    else:
        start_server()

