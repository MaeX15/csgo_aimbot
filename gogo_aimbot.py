import mss
import numpy as np
import cv2
import torch.cuda
from models.common import DetectMultiBackend
import os
import sys
from pathlib import Path
from utils.torch_utils import select_device
import win32gui
import win32ui
import win32con
import win32api
import pynput
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from lock import lock
import time
#file location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#model parameters
weights=ROOT / 'gogo.pt'
device='0'
dnn=False  # use OpenCV DNN for ONNX inference
data=ROOT / 'data/gogodata_banana.yaml'
half=False  # use FP16 half-precision inference
augment=False
visualize=False
conf_thres=0.7
iou_thres=0.25
classes=None
agnostic_nms=False
max_det=10
imgsz=(640, 640)
#import model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
bs = 1  # batch_size
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#create mouse
mouse = pynput.mouse.Controller()
flag = 0
#grab screen
sct = mss.mss()
# width = 1280
# height = 720
width = 1280
height = 720
# width = 1920
# height = 1080
monitor = {
    'left': 0,
    'top': 0,
    'width': width,
    'height': height}
cv2.namedWindow('csgo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('csgo', 720, 405)
dx_last = 0
t0 = time.time()
while True:
    flag +=1
    img = sct.grab(monitor)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#input image pre-processing
    im = letterbox(img, imgsz, stride=int(model.stride), auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
#prediction
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # print('pred:    ', pred)
    # print('img.shape:', img.shape)
    aims = []
    for i, det in enumerate(pred):  # per image
        s = ''
        s += '%gx%g ' % im.shape[2:]
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results(bbox:(tag, x_center, y_center, x_width, y_width))
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                aim = ('%g ' * len(line)).rstrip() % line
                print(aim)
                aim = aim.split(' ')#str to list
                aims.append(aim)
#plot
        if len(aims):
            dx_last = lock(aims, mouse, float(width), float(height), flag, dx_last)
            for _, det in enumerate(aims):
                c, x_center, y_center, kuan, gao = det
                x_center, kuan = float(width) * float(x_center), float(width) * float(kuan)
                y_center, gao = float(height) * float(y_center), float(height) * float(gao)
                top_left = (int(x_center-kuan/2.), int(y_center-gao/2.))
                bottom_right = (int(x_center+kuan/2.), int(y_center+gao/2.))
                # color = [(255,255,0),(255,0,255)][int(c)]
                # color = [(255,255,0),(60,20,255)][int(c)]
                color = [(255,255,0),(255,0,255)][int(c)]
                thickness = [3, 3][int(c)]
                cv2.rectangle(img, top_left, bottom_right, color, thickness)
    cv2.imshow('csgo', img)
    hwnd = win32gui.FindWindow(None,'csgo')
    CVRECT = cv2.getWindowImageRect('csgo')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    # key = cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
print('fps:',flag/(time.time()-t0))











