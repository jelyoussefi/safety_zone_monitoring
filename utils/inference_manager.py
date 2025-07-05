import os, argparse
from pathlib import Path
import cv2
import numpy as np
from time import perf_counter
from collections import deque
import psutil
import pathlib
from utils.images_capture import VideoCapture
import utils.perf_visualizer as pv


class InferenceManager():
	def __init__(self, detect_model_adapter, input=None):
		self.adapter = detect_model_adapter
		self.input = input
		self.cap = VideoCapture(input, True) if input is not None else None
		self.async_mode = async_mode
		self.frames_number = 0
		self.start_time = None
		self.window_name = "Smart Parking"
		self.running = False
		
	def fps(self):
		return self.frames_number/(perf_counter() - self.start_time)

	def getFrame(self):
		return self.cap.read()

	def run(self):
		if self.cap is None:
			print("No input provided")
			return False

		self.frames_number = 0
		self.start_time = perf_counter()
		self.running = True
		while self.running:

			image = self.cap.read()
			if image is None:
				self.cv.acquire()
				break
			
			self.adapter.infer(image)
			
			image = self.adapter.result()

			cv2.imshow(self.window_name, image)
			cv2.waitKey(0)
			
			self.image = image

		return True