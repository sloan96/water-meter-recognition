# -*- coding: utf-8 -*-
# @Time    : 19-3-7 下午2:18
# @Author  : Sloan
# @Email   : 630298149@qq.com
# @File    : server_main.py
# @Software: PyCharm
import grpc
import time
import numpy as np
from ctypes import *
import time
import sys,os
from concurrent import futures
import data_pb2,data_pb2_grpc
import base64


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


# 实现一个派生类,重写rpc中的接口函数.自动生成的grpc文件中比proto中的服务名称多了一个Servicer
class FormatData(data_pb2_grpc.FormatDataServicer):
    def __init__(self,use_gpu = True):
        try:

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
            self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                               POINTER(c_int)]
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
            # python3 need trans
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

    # 重写接口函数.输入和输出都是proto中定义的Data类型
    def DoFormat(self, request, context):
        rec_data = request.data
        # print(src_data,type(src_data),len(src_data))
        with open('temp.jpg','wb') as fw:
            fw.write(base64.b64decode(rec_data))
        print("temp.jpg is saved.")
        predict = self.back_recog_result()
        print("predict:",predict)
        return data_pb2.actionresponse(text=predict)  # 返回一个类实例


def serve(use_gpu=True):

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    _HOST = '[::]'      #localhost
    _PORT = '21328'

    # 定义服务器并设置最大连接数,corcurrent.futures是一个并发库，类似于线程池的概念
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))   # 创建一个服务器
    data_pb2_grpc.add_FormatDataServicer_to_server(FormatData(use_gpu), grpcServer)  # 在服务器中添加派生的接口服务（自己实现了处理函数）
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)    # 添加监听端口
    grpcServer.start()    #  启动服务器

    print("waitting to connect......")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0) # 关闭服务器



if __name__ == '__main__':
    serve()
