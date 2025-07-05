# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import math
import ctypes
from typing import Tuple, Dict
from collections import deque
from threading import Condition
import numpy as np
import torch
from pathlib import Path
import shutil
import cv2
from PIL import Image
from time import perf_counter
import pathlib
from ultralytics.utils.plotting import colors
import openvino.runtime as ov
from openvino.runtime import Core, AsyncInferQueue
from utils.model import Model

# Define labels
labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
          'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
          'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
          'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class YoloModelBase:
    """Base class for all YOLO models with common functionality"""
    
    def __init__(self):
        self.labels = labels
        self.num_masks = 32  # Default for YOLOv8
        self.conf_threshold = 0.5
        self.iou_threshold = 0.2
        self.input_width = None  # Will be set by derived classes or during inference
        self.input_height = None  # Will be set by derived classes or during inference

    def postprocess(self, pred_boxes, image):
        boxes, scores, class_ids = self.process_box_output(pred_boxes, image)
        return image, boxes, scores, class_ids

    def get_boxes(self, box_predictions, orig_img):
        img_height = orig_img.shape[0]
        img_width = orig_img.shape[1]

        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([img_width, img_height, img_width, img_height])

        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        xy = boxes.copy()
        boxes[..., 0] = xy[..., 0] - xy[..., 2] / 2
        boxes[..., 1] = xy[..., 1] - xy[..., 3] / 2
        boxes[..., 2] = xy[..., 0] + xy[..., 2] / 2
        boxes[..., 3] = xy[..., 1] + xy[..., 3] / 2

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)

        return boxes

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []

        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def process_box_output(self, box_output, orig_img):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        box_predictions = predictions[..., :num_classes+4]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.get_boxes(box_predictions, orig_img)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def draw_detections(self, image, boxes, scores, class_ids):
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0008
        text_thickness = int(min([img_height, img_width]) * 0.002)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            label = self.labels[class_id]
            color = colors(class_id)
            x1, y1, x2, y2 = box.astype(int)

            # Call plot_one_box
            image = self.plot_one_box(image, x1, y1, x2, y2, score, label, color)

        return image

    def plot_one_box(self, image, x1, y1, x2, y2, score, label, color):
        """Default implementation for drawing a single bounding box"""
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text = f'{label} {score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)

        # Draw text
        cv2.putText(image, text, (x1, y1 - 5), font, 0.5, (255, 255, 255), 1)

        return image


class YoloV8ModelBase(YoloModelBase):
    """YOLOv8 specific implementation"""
    
    def __init__(self):
        super().__init__()
        self.num_masks = 32


class YoloV11ModelBase(YoloModelBase):
    """YOLOv11 specific implementation"""
    
    def __init__(self):
        super().__init__()
        # YOLOv11 specific settings
        self.num_masks = 32  # Adjust if YOLOv11 uses a different number


class YoloV8Model(YoloV8ModelBase, Model):
    def __init__(self, model_path, device):
        YoloV8ModelBase.__init__(self)
        Model.__init__(self, model_path, device,  255.0)


class YoloV11Model(YoloV11ModelBase, Model):
    def __init__(self, model_path, device):
        YoloV11ModelBase.__init__(self)
        Model.__init__(self, model_path, device, 255.0)
