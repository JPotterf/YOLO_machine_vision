from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from DNModel import net as Darknet
from img_process import inp_to_image, custom_resize
import pandas as pd
import random
import argparse


def prepare_input(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    Perform tranpose and return Tensor

    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (custom_resize(orig_im, (inp_dim, inp_dim)))

    # Opencv image format,[Channels * Width * Height]. The opencv channles are BGR,
    # img[:,:,::-1] is convert to RGB. transpose(2,0,1) is to change them to [C * W * H].
    # then creates a copy
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()

    # unsqueeze returns new tensor with a dimension of 0
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


# creates rectangles around objects
# colors of t
def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (0, 255, 0)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


if __name__ == '__main__':

    # point to the movie you want to test on here
    video = "sidewalk.mp4"

    # Object Confidence to filter predictions
    confidence = 0.5

    # Non-Maximum Suppression(NMS) Threshold. Anything below this value is ignored 
    nms_thesh = 0.4

    # YOLO Config file
    cfgfile = "cfg/yolov3.cfg"

    # Dataset on which the network has been trained
    dataset = "pascal"

    # weightsfile taken from:
    weightsfile = "yolov3.weights"

    # Input resolution of the network. Increase to increase accuracy. Decrease to increase speed
    reso = 128

    start = 0

    # CUDA tensor types, that implement the same function as CPU tensors, but utilizes GPU for computation
    # My laptop does not have GPU therefore I can't test this functionality.
    CUDA = torch.cuda.is_available()

    # 80 catagories have been trained in the model
    # see coco.names
    num_classes = 80

    print("Loading network")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network loaded")
    classes = load_classes('data/coco.names')

    model.DNInfo["height"] = reso
    inp_dim = int(model.DNInfo["height"])

    model.eval()

    videofile = video

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source: was the correct movie loaded?'
    print("YOLO test program running press escape to exit")
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            # prepares each video frame for YOLO network
            img, orig_im, dim = prepare_input(frame, inp_dim)

            # YOLO algo makes predictions in three scales therfore the input dimension has to
            # be repeated for a total of three times
            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(
                output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                cv2.imshow("frame", orig_im)
                if cv2.waitKey(1) == 27:
                    break

                continue

            # the original image was scaled by height to preserve aspect ratio
            # this rescales to original image size to coordinates for the bounding box to be
            # displayed on original frame size.
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  im_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  im_dim[:, 1].view(-1, 1))/2

            output[:, 1:5] /= scaling_factor

            # this lambda function writes all objects onto original image.
            list(map(lambda x: write(x, orig_im), output))

            # moves to next frame
            cv2.imshow("frame", orig_im)
            if cv2.waitKey(1) == 27:
                break

        # if next videoframe cant be rendered the end of video and program ends
        else:
            break
