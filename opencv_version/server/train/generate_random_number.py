import cv2
import numpy as np
import math
import random
import string
from image_util import paste_img
from keras.utils import np_utils


# 生成随机噪点
def generate_random_noise(img):
    rows, cols, dims = img.shape
    pn = np.random.randint(0, 10) + 5
    for i in range(pn):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 255
    pn = np.random.randint(0, 10) + 5
    for i in range(pn):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 0

    # 左侧边框
    pn = np.random.randint(0, 5)
    if pn != 0:
        pn = np.random.randint(0, 25) + 50
        for i in range(pn):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, 3)
            img[x, y, :] = 0

    # 右侧边框
    pn = np.random.randint(0, 5)
    if pn != 0:
        pn = np.random.randint(0, 25) + 50
        for i in range(pn):
            x = np.random.randint(0, rows)
            y = np.random.randint(-3, 0)
            img[x, y, :] = 0
    return img


#  生成0-9数字列表，纵向实现模拟水表转轮
def get_number_collection_img():
    img = np.zeros((14*22, 24, 3), np.uint8)
    img.fill(255)

    str = "01234567890"
    for index in range(11):
        cv2.putText(img, str[index], (0, (index + 1) * 28), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    return img


#  生成0-9数字列表，纵向实现模拟水表转轮
def get_number_collection_img2():
    img = np.zeros((14*22, 28, 3), np.uint8)
    img.fill(255)

    str = "01234567890"
    for index in range(11):
        cv2.putText(img, str[index], (2, (index + 1) * 28), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    return img


# 随机左右平移, range单位px
def random_swift(img, x_range=1, y_range=1):
    rows, cols, dims = img.shape
    sx = np.random.randint(-x_range, x_range + 1)
    sy = np.random.randint(-y_range, y_range + 1)
    # print(sx)

    x = x_range
    y = y_range
    img_n = np.zeros((rows+y_range*2, cols+y_range*2, dims), np.uint8)
    img_n.fill(255)
    paste_img(img_n, img, x+sx, y+sy)

    return img_n[0:rows, x_range:cols, :]


# 随机旋转和缩放
def random_rotate_scale(img, angle_range=13, scale_range_min=-0.05,  scale_range_max=0.1):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)  # 11
    angle = np.random.randint(-angle_range * 2, angle_range * 2 + 1) / 2.0
    scale = np.random.randint(scale_range_min * 100, (scale_range_max*100) + 1) / 100.0 + 1.0
    # print(angle, scale)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated


# 判断标签，步长0.5，[0, 10]
def get_label_y_21(label):
    lbl = math.floor(label * 4) / 4
    # print(str(lbl), str(math.floor(lbl)), str(lbl - math.floor(lbl)))
    diff = math.floor(lbl*100) - math.floor(lbl)*100
    if diff == 25:
        return lbl - 0.25
    elif diff == 75:
        return lbl + 0.25
    else:
        return lbl


# 判断标签，步长0.5，[0, 9.5]
def get_label_y(label):
    lbl = get_label_y_21(label)
    if lbl == 10:
        return 0
    return lbl


# 生成随机数字, img是已经生成好的0-9数字列表，纵向随机截取窗口
def get_random_num(img, w, h):
    min = 0
    ih, iw, ic = img.shape
    max = ih - (h*2)

    x = random.randint(min, max)
    label = x/(h*2)
    # print(min, max, x, label)
    img_n = img.copy()[x: x+2*h, 0: w, :]
    img_n = random_swift(img_n, y_range=0)
    img_n = generate_random_noise(img_n)
    img_n = random_rotate_scale(img_n)
    return [img_n, get_label_y(label)]


def get_label_last_num(img, w, h):
    min = 0
    ih, iw, ic = img.shape
    max = ih - (h*2)

    x = random.randint(min, max)
    label = x/(h*2)
    # print(min, max, x, label)
    img_n = img.copy()[x: x+2*h, 0: w, :]
    img_n = generate_random_noise(img_n)
    img_n = random_rotate_scale(img_n, angle_range=2.5, scale_range_min=0, scale_range_max=0.1)
    return [img_n, get_label_y(label)]


# 生成随机数字9.5
def get_label_95(img, w, h):
    min = int(9.25 * h * 2)
    ih, iw, ic = img.shape
    max = int(9.75 * h * 2)

    x = random.randint(min, max)
    label = x / (h * 2)
    # print(min, max, x, label)
    img_n = img.copy()[x: x + 2 * h, 0: w, :]
    return img_n


# 生成随机数字, img是已经生成好的0-9数字列表，纵向随机截取窗口
def get_label_num(img, w, h, n):

    # img_n = random_swift(img_n, y_range=1)

    if n == 10:
        # print("n==10 get 9.5")
        img_n = get_label_95(img, w, h)
        n = 9
    else:
        x = n * h * 2
        img_n = img.copy()[x: x + 2 * h, 0: w, :]

    img_n = generate_random_noise(img_n)
    img_n = random_rotate_scale(img_n, angle_range=2.5, scale_range_min=0, scale_range_max=0.1)
    return [img_n, n]


def get_random_label(img, w, h, n):
    off_x = w * n

    img_lbl = np.zeros((int(h * 2), int(off_x), 3), np.uint8)

    label = []

    off_x -= w
    img_num, lbl_num = get_label_last_num(img, w, h)
    # label = str(lbl_num) + label
    label.insert(0, int(lbl_num))
    paste_img(img_lbl, img_num, off_x, 0)

    flag = True
    for x in range(n-1):
        off_x -= w
        if lbl_num >= 9 and flag:
            i = random.randint(0, 10)
        else:
            flag = False
            i = random.randint(0, 9)
        img_num, lbl_num = get_label_num(img, w, h, i)
        # label = str(lbl_num) + label
        label.insert(0, lbl_num)
        paste_img(img_lbl, img_num, off_x, 0)

    img_lbl = generate_random_noise(img_lbl)

    # img_lbl = random_swift(img_lbl)

    img_lbl = random_rotate_scale(img_lbl, angle_range=2.5, scale_range_min=-0.05, scale_range_max=0.1)

    return [img_lbl, label]


def get_label_regular(img, w, h, char_num):
    image, captcha_str = get_random_label(img, int(w), int(h), int(char_num))
    # print("label = " + str(captcha_str), image.shape)
    # cv2.imshow("image", image)

    classes = 10
    # characters = string.digits
    characters = range(0, 10)

    # print(str(Y.shape))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (160, 32))
    height, width = image.shape
    X = np.zeros([1, height, width, 1])
    Y = np.zeros([1, char_num, classes])

    # image = np.array(image.getdata())
    X[0] = np.reshape(image, [height, width, 1]) / 255.0
    for j, ch in enumerate(captcha_str):
        Y[0, j, ch] = 1
    # print(str(Y))
    # Y = np.reshape(Y, (1, char_num * classes))
    return X, Y


def keras_label_generator(batch_size=50):
    img = get_number_collection_img2()
    height = 32
    width = 160
    char_num = 5
    classes = 10
    w = 28
    h = 28 / 2
    x_train = np.zeros([batch_size, height, width, 1])
    y_train = np.zeros([batch_size, char_num, classes])

    while True:
        for index in range(batch_size):
            x, y = get_label_regular(img, w, h, char_num)
            x_train[index] = x
            y_train[index] = y
        y_train = np.reshape(y_train, (batch_size, char_num * classes))
        yield x_train, y_train


def get_random_num_minst_sample(img):
    img3, label = get_random_num(img, 24, 14)
    img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
    img3 = cv2.resize(img3, (28, 28))
    # lbl = np.zeros(21, np.uint8)
    # lbl[int(label*2)] = 1

    return [img3, label*2]


def keras_generate(batch_size=60000):
    img = get_number_collection_img()
    while True:
        x_train = np.zeros((batch_size, 28, 28), np.uint8)
        y_train = np.zeros(batch_size, np.uint8)
        print("generate train data")
        for index in range(batch_size):
            x, y = get_random_num_minst_sample(img)
            x_train[index] = x
            y_train[index] = y
            #if index % 1000 == 999:
            #    print("index:" + str(index))
        # (60000,28,28)->(60000,28,28,1)
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        # 换one hot格式
        y_train = np_utils.to_categorical(y_train, num_classes=21)
        yield (x_train, y_train)


def load_data_random(train_batch_size=60000, test_batch_size=10000):
    img = get_number_collection_img()
    x_train = np.zeros((train_batch_size, 28, 28), np.uint8)
    y_train = np.zeros(train_batch_size, np.uint8)
    print("generate train data")
    for index in range(train_batch_size):
        x, y = get_random_num_minst_sample(img)
        x_train[index] = x
        y_train[index] = y
        if index % 1000 == 999:
            print("index:" + str(index))

    print("generate test data")
    x_test = np.zeros((test_batch_size, 28, 28), np.uint8)
    y_test = np.zeros(test_batch_size, np.uint8)
    for index in range(test_batch_size):
        x, y = get_random_num_minst_sample(img)
        x_test[index] = x
        y_test[index] = y
        if index % 1000 == 999:
            print("index:" + str(index))
    return (x_train, y_train), (x_test, y_test)


def load_data_random(train_batch_size=60000, test_batch_size=10000):
    img = get_number_collection_img()
    x_train = np.zeros((train_batch_size, 28, 28), np.uint8)
    y_train = np.zeros(train_batch_size, np.uint8)
    print("generate train data")
    for index in range(train_batch_size):
        x, y = get_random_num_minst_sample(img)
        x_train[index] = x
        y_train[index] = y
        if index % 100 == 99:
            print("index:" + str(index))

    print("generate test data")
    x_test = np.zeros((test_batch_size, 28, 28), np.uint8)
    y_test = np.zeros(test_batch_size, np.uint8)
    for index in range(test_batch_size):
        x, y = get_random_num_minst_sample(img)
        x_test[index] = x
        y_test[index] = y
        if index % 1000 == 999:
            print("index:" + str(index))
    return (x_train, y_train), (x_test, y_test)


def save_data_random():
    times = 5
    (x_train, y_train), (x_test, y_test) = load_data_random(60000*times, 10000*times)
    print("save data")
    np.savez("result.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def load_data(path='result.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

