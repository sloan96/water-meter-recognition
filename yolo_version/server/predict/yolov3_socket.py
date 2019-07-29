# -*- coding: utf-8 -*-
# @Time    : 19-1-27 下午1:53
# @Author  : Sloan
# @Email   : 630298149@qq.com
# @File    : yolov3.py
# @Software: PyCharm
from ctypes import *
import random
import cv2
import time
import sys,os
import numpy as np
from socket import *
import struct
import threading



def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class yolov3_wm:
    OP_CODE_TRANSFER_FILE_BYTE = 1
    is_stop = True
    def __init__(self, use_gpu = False):
        try: 
            self.HOST = ""
            self.PORT = 21328
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
            print("start init...")

            lib_path = "./lib/libdarknet.so" if use_gpu else "./lib/libdarknet_cpu.so"
            print("start CDLL...")
            lib = CDLL(lib_path, RTLD_GLOBAL)
            lib.network_width.argtypes = [c_void_p]
            lib.network_width.restype = c_int
            lib.network_height.argtypes = [c_void_p]
            lib.network_height.restype = c_int

            print("start network_predict...")
            predict = lib.network_predict
            predict.argtypes = [c_void_p, POINTER(c_float)]
            predict.restype = POINTER(c_float)

            print("start cuda_set_device...")
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]

            print("start make_image...")
            make_image = lib.make_image
            make_image.argtypes = [c_int, c_int, c_int]
            make_image.restype = IMAGE

            print("start get_network_boxes...")
            self.get_network_boxes = lib.get_network_boxes
            self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
            self.get_network_boxes.restype = POINTER(DETECTION)

            print("start make_network_boxes...")
            make_network_boxes = lib.make_network_boxes
            make_network_boxes.argtypes = [c_void_p]
            make_network_boxes.restype = POINTER(DETECTION)

            print("start free_detections...")
            self.free_detections = lib.free_detections
            self.free_detections.argtypes = [POINTER(DETECTION), c_int]

            print("start free_ptrs...")
            free_ptrs = lib.free_ptrs
            free_ptrs.argtypes = [POINTER(c_void_p), c_int]

            print("start network_predict...")
            network_predict = lib.network_predict
            network_predict.argtypes = [c_void_p, POINTER(c_float)]

            print("start reset_rnn...")
            reset_rnn = lib.reset_rnn
            reset_rnn.argtypes = [c_void_p]

            print("start load_network...")
            self.load_net = lib.load_network
            self.load_net.argtypes = [c_char_p, c_char_p, c_int]
            self.load_net.restype = c_void_p

            print("start do_nms_obj...")
            self.do_nms_obj = lib.do_nms_obj
            self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

            print("start do_nms_sort...")
            do_nms_sort = lib.do_nms_sort
            do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

            print("start free_image...")
            self.free_image = lib.free_image
            self.free_image.argtypes = [IMAGE]

            print("start letterbox_image...")
            letterbox_image = lib.letterbox_image
            letterbox_image.argtypes = [IMAGE, c_int, c_int]
            letterbox_image.restype = IMAGE

            print("start get_metadata...")
            self.load_meta = lib.get_metadata
            lib.get_metadata.argtypes = [c_char_p]
            lib.get_metadata.restype = METADATA

            print("start load_image_color...")
            self.load_image = lib.load_image_color
            self.load_image.argtypes = [c_char_p, c_int, c_int]
            self.load_image.restype = IMAGE

            print("start rgbgr_image...")
            rgbgr_image = lib.rgbgr_image
            rgbgr_image.argtypes = [IMAGE]

            print("start network_predict_image...")
            self.predict_image = lib.network_predict_image
            self.predict_image.argtypes = [c_void_p, IMAGE]
            self.predict_image.restype = POINTER(c_float)

            cfg_file = './configuration_file/yolov3-voc.cfg'
            weight_file = './configuration_file/yolov3-voc.weights'
            meta_file = './configuration_file/voc.data'
            #python3 need trans
            cfg_file = bytes(cfg_file, 'ascii')
            weight_file = bytes(weight_file, 'ascii')
            meta_file = bytes(meta_file, 'ascii')
            print("start load yolo net ...")
            self.net = self.load_net(cfg_file, weight_file, 0)
            print("start load yolo meta data ...")
            self.meta = self.load_meta(meta_file)
            print("init ok")
        except Exception as e:
            print("init error: " + str(e))
            print(traceback.format_exc())

    def classify(self,net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def b2str(self,byte):
        #python3 need change code
        return str(byte, encoding="utf-8")

    def detect(self,net, meta, image, thresh=.5, hier_thresh=.5, nms=0.45):
        im = self.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def value_func(self,np_boxes_A, np_boxes_B, choose=1):
        # 4种评估方式
        if choose == 1:
            return (np_boxes_A[3] - np_boxes_A[1]) > (np_boxes_B[3] - np_boxes_B[1])
        elif choose == 2:
            A_ratio = (np_boxes_A[3] - np_boxes_A[1]) / (
                        (np_boxes_B[3] - np_boxes_B[1]) + (np_boxes_A[3] - np_boxes_A[1]))
            return A_ratio + np_boxes_A[-1] > (1 - A_ratio) + np_boxes_B[-1]
        elif choose == 3:
            height_ratio = 0.65
            A_ratio = (np_boxes_A[3] - np_boxes_A[1]) / (
                        (np_boxes_B[3] - np_boxes_B[1]) + (np_boxes_A[3] - np_boxes_A[1]))
            return A_ratio * height_ratio + np_boxes_A[-1] * (1 - height_ratio) > (1 - A_ratio) * height_ratio + \
                   np_boxes_B[-1] * (1 - height_ratio)
        else:
            return np_boxes_A[-1] > np_boxes_B[-1]

    def check_boxes(self,boxes, value_method=2):
        if len(boxes) <= 5:
            return boxes
        else:
            # 找到xmin差距最小的一组，移除得分较低的框
            record = -1
            while (len(boxes) - 5):
                np_boxes = np.array([nums[1:] for nums in boxes[:]])
                min_value = 160
                min_idx = -1
                for idx in range(len(np_boxes) - 1):
                    current_min_value = np.min(np.abs(np_boxes[idx + 1:, 0] - np_boxes[idx][0]))
                    if current_min_value < min_value:
                        min_value = current_min_value
                        min_idx = idx
                if record == -1:
                    if self.value_func(np_boxes[min_idx], np_boxes[min_idx + 1], value_method):
                        record = int(np_boxes[min_idx][1] < np_boxes[min_idx + 1][1])
                        del boxes[min_idx + 1]
                    else:
                        record = int(np_boxes[min_idx][1] > np_boxes[min_idx + 1][1])
                        del boxes[min_idx]
                else:
                    remove_idx = min_idx * int(int(np_boxes[min_idx][1] > np_boxes[min_idx + 1][1]) == record) + \
                                 (min_idx + 1) * (
                                             1 - int(int(np_boxes[min_idx][1] > np_boxes[min_idx + 1][1]) == record))
                    del boxes[remove_idx]
            return boxes
    def back_recog_result(self,in_img="temp.jpg"):
        # python3.5

        is_check = True
        value_method = 3

        in_img = bytes(in_img,'ascii')
        start_time = time.time()
        r = self.detect(self.net, self.meta, in_img)
        end_time = time.time()
        # print("cost time:%g s" % (end_time - start_time))
        # print(r)

        back_box = []
        box = [0] * 4
        for i in range(len(r)):
            point = r[i][2]
            box[0] = int(point[0] - point[2] / 2)
            box[1] = int(point[1] - point[3] / 2)
            box[2] = int(point[2])
            box[3] = int(point[3])
            label = self.b2str(r[i][0])
            pre_score = r[i][1]
            if label!="wm":
                back_box.append([label, box[0], box[1], box[0] + box[2], box[1] + box[3], pre_score])
        #from small to big sorted by xmin
        sorted_boxes = sorted(back_box, key=lambda x: x[1])
        if is_check:
            sorted_boxes = self.check_boxes(sorted_boxes,value_method)
        detect_number = [nums[0] for nums in sorted_boxes[:]]
        # detect_number = list(map(int, detect_number))
        detect_number = ''.join(detect_number)
        # print(detect_number,type(detect_number))
        return detect_number
    def deal_cmd01(self, tcpCliSock, data):
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

        cv2.imwrite("temp.jpg",img)
        print("save temp.jpg")
        predict = self.back_recog_result()
        print("predict: " + predict)
        if predict is None or len(predict) != 5:
            predict = "ERROR"

        # 发回去
        send_data = struct.pack('6s', bytes(predict, 'ascii'))
        tcpCliSock.send(send_data)
    def deal_data(self, tcpCliSock, data):
        dwOpCode = struct.unpack('I', data[:struct.calcsize('I')])
        dwOpCode = dwOpCode[0]
        print('dwOpCode:{0}'.format(dwOpCode))
        # content = str(len(data))

        if dwOpCode == self.OP_CODE_TRANSFER_FILE_BYTE:
            self.deal_cmd01(tcpCliSock, data)
        else:
            tcpCliSock.send(bytes("ER", 'utf-8'))
    def start_socket_server(self):
        tcpSerSock = socket(AF_INET, SOCK_STREAM)
        tcpSerSock.bind(self.ADDR)
        tcpSerSock.listen(5)
        is_stop = True

        while is_stop:
            print("waiting for connection...")
            tcpCliSock, addr = tcpSerSock.accept()
            print("connected from :", addr)

            while True:
                try:
                    data = tcpCliSock.recv(self.BUFSIZ)
                    if not data:
                        break
                    self.deal_data(tcpCliSock, data)
                except Exception as err:
                    print(err)
                    break
            tcpCliSock.close()
        tcpSerSock.close()

def start_server(use_gpu = False):
    wp = yolov3_wm(use_gpu)
    wp.start_socket_server()

if __name__ == "__main__":

    start_server()




