## 目录
[TOC]

## 测试运行环境
### 开发软件环境

```
解释器：
Python 3.5

Python库：
opencv-python>=3.4.2
numpy

编辑器推荐：
PyCharm

操作系统：
Ubuntu 16.04

```
### 开发硬件环境
```
Intel Core i7 8700K
16G Memory
1T HDD
```
### CPU下耗时
```
通过opencv调用yolo使cpu模式下运行更快，注意opencv版本最好>=3.4.2
i7 4500  一次耗时1.2s
i7 8700k 一次耗时约450毫秒
```
### YOLO权重下载
```
因权重较大，且经常训练更新，放置网盘比较合理。
最新权重百度网盘下载：
链接: https://pan.baidu.com/s/1wBh5IAizSaQvzgO-RumwiA 提取码: 9q76
下载后将weights权重文件放置在
/water-meter-demo/yolo_opencv_version/configuration_file/ 文件夹下
```
### TODO
完善通信协议，考虑其他网络结构。
