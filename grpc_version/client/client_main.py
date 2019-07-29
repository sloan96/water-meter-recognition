# -*- coding: utf-8 -*-
# @Time    : 19-3-7 下午2:22
# @Author  : Sloan
# @Email   : 630298149@qq.com
# @File    : client_main.py
# @Software: PyCharm
import grpc
import data_pb2,data_pb2_grpc
import os
import base64



def run(dir_path,_HOST ,_PORT = '21328'):

    # with grpc.insecure_channel('localhost:50051') as channel:
    #     stub = helloworld_pb2_grpc.GreeterStub(channel)
    #     response1 = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))

    # conn = grpc.insecure_channel(_HOST + ':' + _PORT)  # 监听频道
    # print(conn)
    # client = data_pb2_grpc.FormatDataStub(channel=conn)   # 客户端使用Stub类发送请求,参数为频道,为了绑定链接
    # print(client)

    with grpc.insecure_channel(_HOST + ':' + _PORT) as channel:
        stub = data_pb2_grpc.FormatDataStub(channel)
        for (root, dirs, files) in os.walk(dir_path, False):
            for filename in files:
                if not filename.endswith(".jpg") and not filename.endswith(".bmp"):
                    continue
                file_path = os.path.join(root, filename)
                with open(file_path, "rb") as f:
                    file_bytes = base64.b64encode(f.read())
                    # print("len(file_bytes)=", len(file_bytes),type(file_bytes))
                    # print(file_bytes)
                response = stub.DoFormat(data_pb2.actionrequest(data=file_bytes))  # 返回的结果就是proto中定义的类
                print(filename)
                print("received result: " + response.text)


if __name__ == '__main__':
    dir_path = './data/image1/'
    server_ip = '127.0.0.1'     #localhost
    if os.getenv('WATER_METER_SERVER', 0):
        server_ip = os.environ['WATER_METER_SERVER']
    print(server_ip)
    run(dir_path,server_ip)
