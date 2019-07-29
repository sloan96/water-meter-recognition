# -*- coding: utf-8 -*-
import shutil
import cv2
import os
import random
import numpy as np
from socket import *
import struct
import shutil
from svm_train import svm_load
from hog_svm_detect_digit import digits_rect_detect, draw_rect


class MyException(Exception):
    def __init__(self, args):
        self.args = args


class LabelWatermeter:
    def __init__(self, src_path, dst_path=None):
        self._src_path = src_path
        if dst_path is not None:
            self._dst_path = dst_path
        else:
            self._dst_path = os.path.join(src_path, "bak")

        if self._src_path != self._dst_path and not os.path.exists(self._dst_path):
            os.mkdir(self._dst_path)

        self._img_title = 'Label Watermeter'

        self._total_imgs = 0
        self._label_imgs = 0

        self._label_list = []

        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.svm = svm_load(os.path.join(self.current_dir, "HOGSVM_Digits.bin"))
        self.target_shape = (48, 128)

        self.lbl_x = 80
        self.lbl_y = 90
        self.lbl_gap = 40

    @staticmethod
    def remote_predict(filename, ip, port=21327):
        tcp_client_sock = socket(AF_INET, SOCK_STREAM)
        # tcp_client_sock.connect(("127.0.0.1", 21327))
        tcp_client_sock.connect((ip, port))

        with open(filename, "rb") as f:
            file_bytes = f.read()
            # print(len(file_bytes))

            send_data = struct.pack('IIIII', 1, 2, 3, len(file_bytes), 5)

            tcp_client_sock.send(send_data)

            data = tcp_client_sock.recv(2)
            if data != b"OK":
                tcp_client_sock.close()
                # print(data)
                return "ERROR"

            tcp_client_sock.sendall(file_bytes)

            data = tcp_client_sock.recv(6)
            if data != b"FINISH":
                tcp_client_sock.close()
                # print(data)
                return "ERROR"

            data = tcp_client_sock.recv(5)
            # print(str(data, "ascii"))
            # tcp_client_sock.sendall(file_bytes)
        tcp_client_sock.close()
        return data

    @staticmethod
    def is_labeled(filename):
        if len(filename) < 7:
            return False

        label_txt = filename[:5]

        if label_txt.isdigit() and filename[5] == '_':
            return True

        return False

    def show_label(self, img, label_txt):
        img_canvas = np.copy(img)
        scale_factor = 2
        width = img_canvas.shape[1]
        height = img_canvas.shape[0]
        img_canvas = cv2.resize(img_canvas, (width * scale_factor, height * scale_factor))

        text = str(label_txt)

        correct_pos = [0, 0, 0, 0, 0, 0]
        for i in range(len(text)):
            cv2.putText(img_canvas, text[i], (self.lbl_x + i * self.lbl_gap + correct_pos[i], self.lbl_y),
                        cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0), 3)
        info = "({},{})".format(self._total_imgs, self._label_imgs + len(self._label_list))
        cv2.putText(img_canvas, info, (0, img_canvas.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.imshow(self._img_title, img_canvas)

    @staticmethod
    def calc_step(start_tag, start_label, list_x):
        step = 0
        for tag, label, filename, new_filename, in list_x:
            step += (label - start_label) / (tag - start_tag)
        step = step / len(list_x)
        print("new step:", step)
        return step

    def rename_file(self, filename, label):
        if self.is_labeled(filename):
            new_filename = '{:05}_{}'.format(label, filename[6:])
        else:
            new_filename = '{:05}_{}'.format(label, filename)
        old_file_path = os.path.join(self._src_path, filename)
        new_file_path = os.path.join(self._dst_path, new_filename)
        # os.rename(old_file_path, new_file_path)
        shutil.move(old_file_path, new_file_path)
        return new_filename

    def input_number(self, img, wait_time=-1):
        text = "_"
        tmp = ""
        self.show_label(img, text)
        ch = cv2.waitKey(wait_time)

        while True:
            if ord('9') >= ch >= ord('0') and 5 > len(tmp) >= 0:
                tmp += chr(ch)
                # print(tmp)
            elif ch == ord('\b') and 5 >= len(tmp) > 0:
                tmp = tmp[:-1]
                # print(tmp)
            elif ch == ord('\r') or ch == ord('\n'):
                break
            elif ch == ord('q'):
                raise MyException('Exit')

            text = tmp + "_"
            self.show_label(img, text)
            ch = cv2.waitKey(wait_time)
        return tmp

    @staticmethod
    def label_to_text(label):
        label = int(label)
        if label < 0:
            label = 99999
        elif label > 99999:
            label = 0
        text = "%05d" % label
        return text, label

    def label(self, remote=None):

        file_list = []

        dir_path = self._src_path

        for (root, dirs, files) in os.walk(dir_path, False):
            # dirs[:] = []
            for filename in files:
                if not filename.endswith('.bmp'):
                    continue
                if self.is_labeled(filename):
                    # self._label_imgs += 1
                    label = int(filename[:5])
                    # print(filename, label)
                    file_list.append((filename, int(label)))
                    continue

                fname, ext = os.path.splitext(filename)
                tag = fname.split('_')[1]
                # print(filename, tag)
                file_list.append((filename, float(tag)))

        self._total_imgs = len(file_list) + self._label_imgs

        # random.shuffle(file_list)

        rec_number = 0
        # if remote is not None:
        #     rec_number = 0
        # else:
        #     rec_number = 2
        #
        #     for filename, tag in file_list[:rec_number]:
        #         file_path = os.path.join(self._dst_path, filename)
        #         print(file_path, tag)
        #         img = cv2.imread(file_path)
        #         cv2.imshow(self._img_title, img)
        #         cv2.waitKey(10)
        #         # label = input('please input label:')
        #         label = self.input_number(img, wait_time=-1)
        #         # if label is None or len(label) <= 0:
        #         #     continue
        #         new_filename = self.rename_file(filename, int(label))
        #         self._label_list.append((tag, int(label), filename, new_filename))
        #
        #     # label_list.sort()
        #     print(self._label_list)
        #     start_tag, start_label = self._label_list[0][:2]
        #     step = self.calc_step(start_tag, start_label, self._label_list[1:rec_number])

        wait_time = -1
        index = rec_number
        while index < len(file_list):
            filename, tag = file_list[index]
            file_path = os.path.join(self._src_path, filename)

            img = cv2.imread(file_path)

            if img is None:
                index += 1
                self._label_imgs += 1
                continue

            digits_rect = digits_rect_detect(img, self.svm, self.target_shape)
            self.lbl_x, self.lbl_y = digits_rect[:2]
            self.lbl_x += 30
            self.lbl_y += 30
            # draw_rect(img, digits_rect, (0, 255, 0))

            print(file_path, tag)
            label = 0
            if type(tag) is int:
                label = tag
            elif remote is not None:
                label = self.remote_predict(file_path, remote)
                if label == "ERROR" or label == b'ERROR':
                    label = 0  # int(step * (tag - start_tag) + start_label)
                else:
                    label = int(label)
            # else:
            #   int(step * (tag - start_tag) + start_label)

            text = "%05d" % label
            print(text)

            self.show_label(img, text)

            while True:
                ch = cv2.waitKey(wait_time)
                changed = False
                if ch == ord('n'):  # rename
                    label = self.input_number(img, wait_time=wait_time)
                    label = int(label)
                    text = "%05d" % label
                    changed = True
                elif ch == ord('r') and remote is not None:
                    label = self.remote_predict(file_path, remote)
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('k'):  # add
                    label += 1
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('j'):  # sub
                    label -= 1
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('i'):  # add
                    label += 10
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('u'):  # sub
                    label -= 10
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('y'):  # add
                    label += 100
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('h'):  # sub
                    label -= 100
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('t'):  # add
                    label += 1000
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('g'):  # sub
                    label -= 1000
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('o'):  # add
                    label += 10000
                    text, label = self.label_to_text(label)
                    changed = True
                elif ch == ord('l'):  # sub
                    label -= 10000
                    text, label = self.label_to_text(label)
                    changed = True
                # elif ch == ord('s'): #skip
                #     wait_time = 1
                elif ch == ord('\r') or ch == ord('\n') or ch == ord('x') or ch == ord('q') or ch == ord('b'):
                    changed = True
                    break

                if wait_time == 1:
                    break
                if changed:
                    print(text)
                    self.show_label(img, text)

            if ch == ord('q'):  # quit
                raise MyException('Exit')
            elif ch == ord('x'):  # delete
                os.remove(file_path)
                del file_list[index]
                print("file [{}] has deleted!".format(file_path))
                self._total_imgs -= 1
                continue
            elif ch == ord('b'):  # recover last img
                if len(self._label_list) < 1:
                    print('No history!')
                    continue
                tag, label, filename, new_filename = self._label_list.pop()
                os.rename(os.path.join(self._dst_path, new_filename),
                          os.path.join(self._src_path, filename))
                index -= 1
                continue

            new_filename = self.rename_file(filename, int(label))
            self._label_list.append((tag, int(label), filename, new_filename))

            index += 1

        # self._label_list.sort()
        # with open(os.path.join(self._dst_path, 'label_list.txt'), 'w') as f:
        #     for tag, label, filename, new_filename in self._label_list:
        #         text = '{}, {}, {}, {}\n'.format(tag, label, filename, new_filename)
        #         f.writelines(text)

    def random_select(self, limits=1000):
        appendix = ".bmp"

        file_list = []

        for (root, dirs, files) in os.walk(self._src_path, False):
            for filename in files:
                if not filename.endswith(appendix):
                    continue
                file_path = os.path.join(root, filename)
                file_list.append(file_path)

        random.shuffle(file_list)
        if len(file_list) < limits:
            raise MyException('Total files less than random number {}'.format(limits))
        sub_list = file_list[:limits]
        for file_path in sub_list:
            shutil.move(file_path, self._dst_path)
            print('[{}] move to [{}]'.format(file_path, self._dst_path))
        print('{} random files selected!'.format(limits))


if __name__ == "__main__":
    # lwm = LabelWatermeter('D:/watermeter_datasets', 'm1', 'm1_output1')
    # lwm = LabelWatermeter('D:\\support\\watermeter\\back-13\\back')
    lwm = LabelWatermeter('D:\\support\\watermeter\\exception_samples\\dst\\')
    # lwm = LabelWatermeter('G:\\62xxx', '8', '8')
    try:
        # lwm.random_select(1000)
        # lwm.label()
        lwm.label(remote="172.16.64.79")
    except MyException as e:
        print(e)
