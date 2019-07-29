# 水表识别算法使用说明文档
## 目录
[TOC]

## 运行环境
### 开发软件环境

```
解释器：
Python3.5/3.6

Python库：
Tensorflow      1.0.8+
opencv-python   3.4.5.20
keras           2.0
numpy

编辑器推荐：
VI/PyCharm/Notepad

操作系统：
Windows/Ubuntu/CentOS/MacOS/Docker
```
### 开发硬件环境
```
Intel Core i5 4370
8G Memory
1T HDD
```

## 检测识别基本流程

### 数字识别总体流程


```flow
st=>start: 开始
e=>end: 结束
sub1=>subroutine: 图像预处理
sub2=>subroutine: 检测数字区域
sub3=>subroutine: 数字识别

st->sub1->sub2->sub3->e

```
### 图像预处理流程

流程中加入JPEG压缩是因为训练的图像都是jpeg训练，而部分原图是bmp的，实际场景是jpeg压缩后上报再进行识别。

```flow
st=>start: 开始
e=>end: 结束
input=>inputoutput: 获取输入图像
cond=>condition: 是否是BMP原图?
op_deskew=>operation: 倾斜校正度数计算
op_jpeg_compression=>operation: JPEG压缩图像
op_threshold=>operation: 高斯二值化
op_rotate=>operation: 倾斜校正图像旋转
output=>inputoutput: 输出处理后的图像

st->input->op_deskew->cond
cond(no)->op_jpeg_compression->op_threshold
cond(yes)->op_threshold
op_threshold->op_rotate->output->e


```



## 检测模型训练

### 数字区域检测模型训练

推荐数字区域最好范围是在**128px * 48px** 附近，即摄像头距离表盘玻璃面**40mm**最佳。

#### 训练说明

模型训练修改文件hog_svm_detect_digit.py

修改这里为样本图片的路径，使用标准的识别模型训练样本即可，会采用传统轮廓算法计算出样本然后用来训练

```python
def generate_hog_data(batch_size=1000):
	# 修改这里为样本图片的路径
    dirpath = "Z:\\work\\WMImages\\image3\\"
    features = np.zeros((batch_size, 75, 32), np.float32)
    labels = np.zeros(batch_size, np.float32)
    pos_count = 0
    ......
```

变更启动函数为训练

```python
if __name__ == "__main__":
    # main()
    hog_train()
    # hog_test()
    # hog_label()
```

之后会在执行目录下生成HOGSVM_Digits.bin即为数字区域识别模型

执行命令`python3 hog_svm_detect_digit.py`

#### 训练样本

数量大致要求在3000~5000为佳，正负样本程序会自动生成，需要手动将异常样本（即轮廓算法识别有问题的样本）去除。



#### 样本标注

##### 执行标注程序

标注一个文件夹，注意标注后图片的文件名为`XXXXX_源文件名`，其中的XXXXX为标注结果。

注意修改标注的文件夹和远程识别服务的IP

```python
if __name__ == "__main__":
    lwm = LabelWatermeter('D:\\support\\watermeter\\exception_samples\\dst\\')
    try:
        # lwm.random_select(1000)
        # lwm.label()
        lwm.label(remote="172.16.64.79")
    except MyException as e:
        print(e)
```
标注好的图像会存在bak文件夹内，可以关闭以后继续标注。

执行`python3 label_watermeter.py`即可运行标注程序

##### 快捷键说明

```
N 完全重新标注
R 直接使用远程服务器自动标注
K 标注数+1
J 标注数-1
I 标注数+10
U 标注数-10
Y 标注数+100
H 标注数-100
T 标注数+1000
G 标注数-1000
O 标注数+10000
L 标注数-10000
Q 退出程序
回车 提交当前标注结果
B 回滚，标记上一张图片
X 删除当前这张图片

N按钮下的完全重新标注模式下说明：
输入5位数以后按回车提交
按退格键删除一位
按Q退出整个程序
```






### 数字识别模型训练

#### 训练说明

注意修改train_label.py中的样本图像的路径，支持断点续训练

```python
def train(train_batch_size=64):
    captcha = generate_watermeter_label.WaterWaterData(dirpath="/WMImages/test001")
    ......
```

执行命令`python3 train_label.py`进行训练，模型保存在`model-watermeter/capcha_model.ckpt-*`

在准确率高于0.999或者每训练10 epoches， 模型会自动保存。



#### 生成pb文件

修改train_label.py，

```python
if __name__ == '__main__':
    # train(train_batch_size=64)
    test_model()
```

然后**将test_model函数中的dirpath指向一个空目录**

```python
def test_model():
    # width = 160
    # height = 120
    width = 120
    height = 32
    char_num = 5
    characters = range(10)
    classes = 10

    dirpath = "G:\\0"
```
然后执行`python3 train_label.py`
代码实际执行到freeze_graph.freeze_graph就会生成output_model/pb_model/frozen_model.pb
```python
tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
print(save_path)
freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, save_path, 'out', 'save/restore_all','save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")
```



#### 训练样本

推荐样本数字分布要平均，样本数量在30万~100万左右，光照条件和数字条件可以多种多样，提升模型鲁棒性。



## 表盘度数检测服务

注意正式部署的时候应当将大部分代码使用Cython转成pyd后执行加速

### 检测服务所需文件

需要的执行文件如下：

```
hog_svm_detect_digit.py
svm_train.py
verify_label.py
start.py
frozen_model.pb
HOGSVM_Digits.bin
```

### 部署运行环境

```
解释器：
Python3.5/3.6

Python库：
Tensorflow 1.0.8+
opencv-python
keras2.0
numpy

操作系统：
Windows/Ubuntu/CentOS/MacOS/Docker
```

### 运行服务

这里以Ubuntu 16.0.4为例，手动安装Python3.6.5的源码后。

```
# 直接执行
python3.6 start.py
# 后台运行
nohup python3.6 start.py
```



## 测试客户端

使用文件socket_client.py，注意修改文件名为要检测的文件/文件夹全路径以及修改IP为服务的IP

```python
if __name__ == "__main__":
	# 检测单个文件
    # remote_predict(filename, "127.0.0.1")
    # 检测一个文件夹内的文件，注意
    dirpath = "D:\\support\\watermeter\\image99"
    remote_label(dirpath, "172.16.61.13")
```

注意函数remote_label_verify内有后缀名判断，可以修改：

```python
def remote_label_verify(dir_path, ip):
    for (root, dirs, files) in os.walk(dir_path, False):
        for filename in files:
            # if filename != "60914_2018-11-23-9-19-58_003565861.9557.bmp":
            if not filename.endswith(".jpg") and not filename.endswith(".bmp"):
                continue
```

