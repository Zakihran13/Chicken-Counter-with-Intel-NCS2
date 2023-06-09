#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import shutil
import pathlib
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from moviepy.editor import VideoFileClip

import cv2

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self,  side):
        self.num = 3 #if 'num' not in param else int(param['num'])
        self.coords = 4 #if 'coords' not in param else int(param['coords'])
        self.classes = 1 #if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0, 373.0, 326.0] #if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


class Inference:
    def __init__(self):
        self.inframe_inferenced = None
    
    def get_inframe(self):
        return self.inframe_inferenced

    def letterbox(self, img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        w, h = size

        # Scale ratio (new / old)
        r = min(h / shape[0], w / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (w, h)
            ratio = w / shape[1], h / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        top2, bottom2, left2, right2 = 0, 0, 0, 0
        if img.shape[0] != h:
            top2 = (h - img.shape[0])//2
            bottom2 = top2
            img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
        elif img.shape[1] != w:
            left2 = (w - img.shape[1])//2
            right2 = left2
            img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
        return img


    def scale_bbox(self, x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h=640, resized_im_w=640):
        gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
        pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
        x = int((x - pad[0])/gain)
        y = int((y - pad[1])/gain)

        w = int(width/gain)
        h = int(height/gain)
    
        xmin = max(0, int(x - w / 2))
        ymin = max(0, int(y - h / 2))
        xmax = min(im_w, int(xmin + w))
        ymax = min(im_h, int(ymin + h))
        # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
        # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
        return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


    def entry_index(self, side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


    def parse_yolo_region(self, blob, resized_image_shape, original_im_shape, params, threshold):
        # ------------------------------------------ Validating output parameters ------------------------------------------    
        out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
        predictions = 1.0/(1.0+np.exp(-blob)) 
                    
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                        "be equal to width. Current height = {}, current width = {}" \
                                        "".format(out_blob_h, out_blob_w)

        # ------------------------------------------ Extracting layer parameters -------------------------------------------
        orig_im_h, orig_im_w = original_im_shape
        resized_image_h, resized_image_w = resized_image_shape
        objects = list()
    
        side_square = params.side * params.side

        # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
        bbox_size = int(out_blob_c/params.num) #4+1+num_classes

        for row, col, n in np.ndindex(params.side, params.side, params.num):
            bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
            
            x, y, width, height, object_probability = bbox[:5]
            class_probabilities = bbox[5:]
            if object_probability < threshold:
                continue
            x = (2*x - 0.5 + col)*(resized_image_w/out_blob_w)
            y = (2*y - 0.5 + row)*(resized_image_h/out_blob_h)
            if int(resized_image_w/out_blob_w) == 8 & int(resized_image_h/out_blob_h) == 8: #80x80, 
                idx = 0
            elif int(resized_image_w/out_blob_w) == 16 & int(resized_image_h/out_blob_h) == 16: #40x40
                idx = 1
            elif int(resized_image_w/out_blob_w) == 32 & int(resized_image_h/out_blob_h) == 32: # 20x20
                idx = 2

            width = (2*width)**2* params.anchors[idx * 6 + 2 * n]
            height = (2*height)**2 * params.anchors[idx * 6 + 2 * n + 1]
            class_id = np.argmax(class_probabilities)
            confidence = object_probability
            objects.append(self.scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                    im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h, resized_im_w=resized_image_w))
        return objects


    def intersection_over_union(self, box_1, box_2):
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union

    def capture_recorded(self, video_path, count_file):
        clip = VideoFileClip(video_path)
        starting_point = 25          # mulai dari detik ke 25
        end_point = 28               # berhenti setelah 3 detik
        subclip = clip.subclip(starting_point, end_point)
        subclip.write_videofile("./temp/history_{}_{}.mp4".format(str(datetime.now().strftime("%Y_%m_%d")), count_file))
        
        return cv2.VideoCapture('./temp/history_{}_{}.mp4'.format(str(datetime.now().strftime("%Y_%m_%d")), count_file))

