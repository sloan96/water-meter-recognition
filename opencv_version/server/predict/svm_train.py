# -*- coding: utf-8 -*-

import numpy as np
import cv2


# svm参数配置
def svm_config():
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)

    return svm


# svm训练
def svm_train(svm, features, labels):
    x = np.array(features)
    y = np.array(labels)
    print(x.shape)
    svm.train(x, cv2.ml.ROW_SAMPLE, y)


# svm参数保存
def svm_save(svm, name):
    svm.save(name)

# svm加载参数
def svm_load(name):
    svm = cv2.ml.SVM_load(name)

    return svm
