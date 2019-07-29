## 目录
[TOC]

## 测试运行环境
### 开发软件环境

```
解释器：
Python 3.5

Python库：
opencv-python
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
### CPU/GPU 模式切换
```
/water-meter-demo/yolo_version/server/predict/ 目录下cmd命令
python3 main.py --use_gpu=True -->使用gpu,False则使用cpu
```
### YOLO权重下载
```
因权重较大，且经常训练更新，放置网盘比较合理。
最新权重百度网盘下载：
链接: https://pan.baidu.com/s/1wBh5IAizSaQvzgO-RumwiA 提取码: 9q76
下载后将weights权重文件放置在
/water-meter-demo/yolo_version/server/predict/configuration_file/ 文件夹下
```

## 关于服务端的启动

|版本|编译|运行|
|---|----|----|
|GPU版本|$ nvidia-docker build -t watermeter-server-predict.yolo.gpu -f Dockerfile.yolo.server.gpu . | $ nvidia-docker run -p 21328:21328 -it --rm watermeter-server-predict.yolo.gpu |
|CPU版本|$ docker build -t watermeter-server-predict.yolo.cpu -f Dockerfile.yolo.server.cpu . | $ docker run -p 21328:21328 -it --rm watermeter-server-predict.yolo.cpu |

## 关于客户端的启动

### Build Client
$ docker build -t watermeter-client-app.yolo -f Dockerfile.yolo.client .

### Run Client
|服务器   |命令|
|--------|---|
|阿里云   | $ docker run -e WATER_METER_SERVER=watermeter.houghlines.cn -it --rm --name watermeter-client-app.yolo watermeter-client-app.yolo |
|本地     | $ docker run -e WATER_METER_SERVER={HostPC_IPaddress} -it --rm --name watermeter-client-app.yolo watermeter-client-app.yolo |

以上。
