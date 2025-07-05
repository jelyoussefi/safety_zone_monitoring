# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os				
from collections import deque
import numpy as np
import cv2
from time import perf_counter
from openvino.runtime import Core, AsyncInferQueue


class Model():
	def __init__(self, model_path, device, scale_factor=1):
		
		self.model_path = model_path
		self.device = device
		self.scale_factor = scale_factor
		self.core = Core()

		self.ov_model = self.core.read_model(model_path)
		self.input_layer_ir = self.ov_model.input(0)
		self.input_layer_name = self.input_layer_ir.get_any_name()
		self.input_height = self.input_layer_ir.shape[2]
		self.input_width = self.input_layer_ir.shape[3]		
		self.ov_model.reshape({0: [1, 3, self.input_height, self.input_width]})
		self.model = self.core.compile_model(self.ov_model, self.device.upper())
		self.output_tensor = self.model.outputs[0]	
		self.running = True

	def predict(self, image:np.ndarray):
		frame = None
		try:
			if image is not None :
				
				resized_image = self.preprocess(image)
								
				result = self.model(resized_image)[self.output_tensor]
				return self.postprocess(result, image)
							
		except Exception as e:
			pass

		return frame

	def preprocess(self, image):
		input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		input_img = cv2.resize(input_img, (self.input_width, self.input_height))
		input_img = input_img / self.scale_factor
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

		return input_tensor


	def plot_one_box(self, image,  xmin, ymin, xmax, ymax, score, label, color):
		img_height, img_width = image.shape[:2]
		size = min([img_height, img_width]) * 0.0008
		text_thickness = int(min([img_height, img_width]) * 0.002)

		xmin = int(xmin)
		ymin = int(ymin)
		xmax = int(xmax)
		ymax = int(ymax)

		# Draw rectangle
		cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

		caption = f'{label} {int(score * 100)}%'
		
		(tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
									  fontScale=size, thickness=text_thickness)
		th = int(th * 1.2)

		cv2.rectangle(image, (xmin, ymin), (xmin + tw, ymin - th), color, -1)

		cv2.putText(image, caption, (xmin, ymin),
					cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
		return image

