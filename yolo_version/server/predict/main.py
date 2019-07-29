import yolov3_socket as yolo
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true','True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False','f', 'n', '0'):
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
    yolo.start_server(use_gpu)
except:
    pass
print("exit")
