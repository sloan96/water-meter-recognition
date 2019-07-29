# -*- coding: utf-8 -*-
# @Time : 2019/3/25 0025 22:01
# @Author : sloan
# @Email : 630298149@qq.com
# @File : yolo_demo.py
# @Software: PyCharm
import numpy as np
import cv2
import time

weights_path = "configuration_file/yolov3-voc.weights"
config_path = "configuration_file/yolov3-voc.cfg"
labels_path = "configuration_file/voc.names"
labels = open(labels_path).read().strip().split('\n')
colors = np.random.randint(0,255,size=(len(labels),3),dtype="uint8")

def value_func(np_boxes_A,np_boxes_B, choose = 1):
    #4种评估方式
    if choose == 1:
        return (np_boxes_A[3]-np_boxes_A[1])>(np_boxes_B[3]-np_boxes_B[1])
    elif choose == 2:
        A_ratio = (np_boxes_A[3] - np_boxes_A[1])/((np_boxes_B[3] - np_boxes_B[1])+(np_boxes_A[3] - np_boxes_A[1]))
        return A_ratio+np_boxes_A[-1]>(1-A_ratio)+np_boxes_B[-1]
    elif choose == 3:
        height_ratio = 0.65
        A_ratio = (np_boxes_A[3] - np_boxes_A[1]) / ((np_boxes_B[3] - np_boxes_B[1]) + (np_boxes_A[3] - np_boxes_A[1]))
        return A_ratio*height_ratio + np_boxes_A[-1]*(1-height_ratio) > (1 - A_ratio)*height_ratio + np_boxes_B[-1]*(1-height_ratio)
    else:
        return np_boxes_A[-1]>np_boxes_B[-1]

def check_boxes(boxes,value_method = 3):
    if len(boxes)<=5:
        return boxes
    else:
        record = -1     #record up:1 or down:0
        #找到xmin差距最小的一组，移除得分较低的框
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

                if value_func(np_boxes[min_idx], np_boxes[min_idx + 1], value_method):
                    record = int(np_boxes[min_idx][1] < np_boxes[min_idx + 1][1])
                    del boxes[min_idx + 1]
                else:
                    record = int(np_boxes[min_idx][1] > np_boxes[min_idx + 1][1])
                    del boxes[min_idx]
            else:
                remove_idx = min_idx*int(int(np_boxes[min_idx][1] > np_boxes[min_idx + 1][1])==record)+\
                             (min_idx+1)*(1-int(int(np_boxes[min_idx][1] > np_boxes[min_idx + 1][1])==record))
                del boxes[remove_idx]

        return boxes



start_time = time.time()
#加载网络
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
end1_time = time.time()
print("load net:",end1_time-start_time)
def detect():
    is_check = True
    is_show = True
    value_method = 3

    boxes = []
    wm_box = []
    confidences = []
    classIDs = []
    image=cv2.imread("configuration_file/017007.jpg")
    (H, W) = image.shape[:2]
    # 得到 YOLO需要的输出层
    ln = net.getLayerNames()
    end2_time = time.time()
    print("getLayerNames:",end2_time-end1_time)
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    end3_time = time.time()
    print("layerOutputs:",end3_time-end2_time)
    #在每层输出上循环
    for output in layerOutputs:
        # 对每个检测进行循环
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #过滤掉那些置信度较小的检测结果
            if confidence > 0.5:
                #框后接框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                #边框的左上角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新检测出来的框
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)
    # 极大值抑制
    end4_time = time.time()
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
    nms_box = []
    if len(idxs)>0:
        for i in idxs.flatten():
            if labels[classIDs[i]] != "wm":
                nms_box.append([labels[classIDs[i]],boxes[i][0],boxes[i][1],boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3],confidences[i]])
            else:
                wm_box.append([boxes[i][0],boxes[i][1],boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3]])

    sorted_boxes = sorted(nms_box, key=lambda x: x[1])
    if is_check:
        sorted_boxes = check_boxes(sorted_boxes, value_method)
    detect_number = [nums[0] for nums in sorted_boxes[:]]
    print(detect_number)
    print("NMSBoxes:",time.time()-end4_time)
    print("total time:", time.time() - start_time)
    if is_show:
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # 在原图上绘制边框和类别
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                text = "{}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Image", image)
        c = cv2.waitKey(0) & 0xFF
        if c == ord('q'):
            exit(0)
if __name__ == "__main__":
    detect()