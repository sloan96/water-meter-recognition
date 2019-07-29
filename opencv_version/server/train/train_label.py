
import tensorflow as tf
import numpy as np
import os
import generate_random_number
import generate_watermeter_label
import captcha_model
import cv2
from tensorflow.python.tools import freeze_graph
from generate_watermeter_label import jpeg_compression, threshold_img, get_digits_area
from svm_train import svm_load
from hog_svm_detect_digit import hog_predict2

svm = svm_load("HOGSVM_Digits.bin")
target_shape = (48, 128)


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
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 55)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 65)
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


def pre_precess_img(image, width, height):
    theta_deg = detect_skew(image)
    image = jpeg_compression(image)
    # cv2.imshow('jpeg_compression', image)
    image = threshold_img(image)
    # cv2.imshow('threshold_img', image)
    image = rotate_img(image, theta_deg)

    image = threshold_img(image)
    # image2 = get_digits_area(image)
    # if image2 is None:
    #     image2 = hog_predict2(image, svm, box_shape=target_shape)
    # image = cv2.resize(image2, (width, height))
    image = hog_predict2(image, svm, box_shape=target_shape)
    image = cv2.resize(image, (width, height))
    return image


def test_model():
    # width = 160
    # height = 120
    width = 120
    height = 32
    char_num = 5
    characters = range(10)
    classes = 10

    # dirpath = "D:\\support\\watermeter\\image_enhance"
    dirpath = "D:\\support\\watermeter\\image3"
    dirpath = "G:\\0"

    x = tf.placeholder(tf.float32, [None, height, width, 1], name="x")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    model = captcha_model.captchaModel(width, height, char_num, classes)
    y_conv = model.create_model(x, keep_prob)
    predict = tf.argmax(tf.reshape(y_conv, [-1, char_num, classes]), 2, name="out")
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    # with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init_op)
        # ckpt = tf.train.get_checkpoint_state('./model-watermeter/')
        ckpt = tf.train.get_checkpoint_state('./model-watermeter/')

        save_path = str(ckpt.model_checkpoint_path)
        saver.restore(sess, save_path)

        
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        
        print(save_path)
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, save_path, 'out', 'save/restore_all',
                                  'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")

        for (root, dirs, files) in os.walk(dirpath, False):
            for filename in files:
                # if not filename.endswith(".jpg"):
                if not filename.endswith(".bmp"):
                    continue
                file_path = os.path.join(root, filename)
                image = cv2.imread(file_path)
                image_src = np.copy(image)
                image = pre_precess_img(image, width, height)
                if image is None:
                    continue
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # image = cv2.resize(image, (160, 120))
                # cv2.imshow("image", image)

                test_x = np.reshape(image, [height, width, 1]) / 255.0

                pre_list = sess.run(predict, feed_dict={x: [test_x], keep_prob: 1})
                for i in pre_list:
                    s = ''
                    for j in i:
                        s += str(characters[j])
                print(filename, s)
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                cv2.putText(image_src, s, (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.imshow("image", image_src)
                cv2.waitKey(-1)


def train(train_batch_size=64):
    captcha = generate_watermeter_label.WaterWaterData(dirpath="/home/test/WMImages")

    width = 120  #160
    height = 32  # 120
    char_num = 5
    characters = range(10)
    classes = 10


    # width,height,char_num,characters,classes = captcha.get_parameter()

    x = tf.placeholder(tf.float32, [None, height, width, 1])
    y_ = tf.placeholder(tf.float32, [None, char_num * classes])
    keep_prob = tf.placeholder(tf.float32)

    model = captcha_model.captchaModel(width, height, char_num, classes)
    y_conv = model.create_model(x, keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # tf.summary.histogram('train_step', train_step)
    predict = tf.reshape(y_conv, [-1, char_num, classes])
    real = tf.reshape(y_, [-1, char_num, classes])
    correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    # tf.summary.histogram('accuracy', accuracy)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        step = 1
        sess.run(tf.global_variables_initializer())
        # ckpt = tf.train.get_checkpoint_state('./model/')
        ckpt = tf.train.get_checkpoint_state('./model-watermeter/')
        if ckpt and ckpt.model_checkpoint_path:
            save_path = str(ckpt.model_checkpoint_path)
            saver.restore(sess, save_path)
            step = int(save_path[save_path.rfind("-") + 1:]) + 1
            print(save_path, str(step))

        # log_dir = 'summary/graph2'
        # train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

        while True:
            # batch_x, batch_y = next(generate_random_number.keras_label_generator(64))
            batch_x, batch_y = next(captcha.keras_label_generator(train_batch_size))
            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
            print('step:%d,loss:%f' % (step, loss))

            if step % 10 == 0:
                # saver.save(sess, "./model/capcha_model.ckpt-" + str(step))
                saver.save(sess, "./model-watermeter/capcha_model.ckpt-" + str(step))

                # batch_x_test, batch_y_test = next(generate_random_number.keras_label_generator(64))
                batch_x_test, batch_y_test = next(captcha.keras_label_generator(train_batch_size*8))
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                print('###############################################step:%d,accuracy:%f' % (step, acc))

                if acc > 0.999:
                    saver.save(sess, "./model-watermeter/capcha_model.ckpt-" + str(step))
                    break
            step += 1

        # save freeze graph file
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        print(save_path)
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, save_path, 'out', 'save/restore_all',
                                  'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")
 
if __name__ == '__main__':
    train(train_batch_size=64)
    # test_model()
