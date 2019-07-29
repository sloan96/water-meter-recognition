from socket import *
import struct

def remote_predict(filename, ip, port=21327):
    tcp_client_sock = socket(AF_INET, SOCK_STREAM)
    # tcp_client_sock.connect(("127.0.0.1", 21327))
    tcp_client_sock.connect((ip, port))

    with open(filename, "rb") as f:
        file_bytes = f.read()
        # print("len(file_bytes)=", len(file_bytes))

        send_data = struct.pack('IIIII', 1, 2, 3, len(file_bytes), 5)

        tcp_client_sock.send(send_data)

        data = tcp_client_sock.recv(2)
        if data != b"OK":
            tcp_client_sock.close()
            print(data)
            return "ERROR"

        tcp_client_sock.sendall(file_bytes)

        data = tcp_client_sock.recv(6)
        if data != b"FINISH":
            tcp_client_sock.close()
            print(data)
            return "ERROR"

        data = tcp_client_sock.recv(5)
        print(str(data, "ascii"))
        # tcp_client_sock.sendall(file_bytes)
    tcp_client_sock.close()
    return data


import os
import cv2
def remote_label(dir_path, ip):
    for (root, dirs, files) in os.walk(dir_path, False):
        for filename in files:
            # if filename != "60914_2018-11-23-9-19-58_003565861.9557.bmp":
            if not filename.endswith(".jpg") and not filename.endswith(".bmp"):
                continue
            file_path = os.path.join(root, filename)
            # print(file_path)
            data = remote_predict(file_path, ip)
            data = str(data, "ascii")
            print(file_path, data)
            # img = cv2.imread(file_path)
            # cv2.putText(img, data, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            # cv2.imshow("img", img)
            # cv2.waitKey()


def remote_label_verify(dir_path, ip):
    for (root, dirs, files) in os.walk(dir_path, False):
        for filename in files:
            # if filename != "60914_2018-11-23-9-19-58_003565861.9557.bmp":
            if not filename.endswith(".jpg") and not filename.endswith(".bmp"):
                continue
            file_path = os.path.join(root, filename)
            print(file_path)
            data = remote_predict(file_path, ip)
            data = str(data, "ascii")
            img = cv2.imread(file_path)
            cv2.putText(img, data, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow("img", img)
            cv2.waitKey()

if __name__ == "__main__":
    # dirpath = "./data/image-process-2-2"
    dirpath = "./data/zhiheng-19-1-2"
    socket_server = "127.0.0.1"
    if os.getenv('WATER_METER_SERVER', 0) :
        socket_server = os.environ['WATER_METER_SERVER']

    print(socket_server)
    # remote_predict("./data/zhiheng-19-1-2/2018-12-26-10-41-59_000000060.0034.jpg", socket_server)
    remote_label(dirpath, socket_server)

