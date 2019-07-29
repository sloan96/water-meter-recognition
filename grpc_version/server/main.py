# -*- coding: utf-8 -*-
# @Time    : 19-3-8 下午3:36
# @Author  : Sloan
# @Email   : 630298149@qq.com
# @File    : main.py
# @Software: PyCharm
import server_main as sm
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser(description='recog_wm')
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Whether to use GPU,Any of (yes,true,True,t,y,1) means use GPU,else (no,false,False,f,n,0)',
        default=True,
        type=str2bool
    )
    return parser.parse_args()


args = parse_args()
try:
    use_gpu = args.use_gpu
    sm.serve(use_gpu)
except:
    pass
print("exit")