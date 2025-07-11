import os
import sys
import time
from pathlib import Path
import copy
import cv2
import numpy as np


class InvalidInput(Exception):

    def __init__(self, message):
        self.message = message


class OpenError(Exception):

    def __init__(self, message):
        self.message = message


class ImagesCapture:

    def read():
        raise NotImplementedError

    def get_distance(self, x, y):
        return None

    def release(self):
        return

    def fps():
        raise NotImplementedError

    def get_type():
        raise NotImplementedError


class ImreadWrapper(ImagesCapture):

    def __init__(self, input, loop):
        self.loop = loop
        if not os.path.isfile(input):
            raise InvalidInput("Can't find the image by {}".format(input))
        self.image = cv2.imread(input, cv2.IMREAD_COLOR)
        if self.image is None:
            raise OpenError("Can't open the image from {}".format(input))
        self.can_read = True

    def read(self):
        if self.loop:
            return copy.deepcopy(self.image)
        if self.can_read:
            self.can_read = False
            return copy.deepcopy(self.image)
        return None

    def fps(self):
        return 1.0

    def get_type(self):
        return 'IMAGE'


class DirReader(ImagesCapture):

    def __init__(self, input, loop):
        self.loop = loop
        self.dir = input
        if not os.path.isdir(self.dir):
            raise InvalidInput("Can't find the dir by {}".format(input))
        self.names = sorted(os.listdir(self.dir))
        if not self.names:
            raise OpenError("The dir {} is empty".format(input))
        self.file_id = 0
        for name in self.names:
            filename = os.path.join(self.dir, name)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is not None:
                return
        raise OpenError("Can't read the first image from {}".format(input))

    def read(self):
        while self.file_id < len(self.names):
            filename = os.path.join(self.dir, self.names[self.file_id])
            print(filename)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.file_id += 1
            if image is not None:
                return image
        if self.loop:
            self.file_id = 0
            while self.file_id < len(self.names):
                filename = os.path.join(self.dir, self.names[self.file_id])
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                self.file_id += 1
                if image is not None:
                    return image
        return None

    def fps(self):
        return 1.0

    def get_type(self):
        return 'DIR'


class VideoCapWrapper(ImagesCapture):

    def __init__(self, input, loop):
        self.loop = loop
        self.cap = cv2.VideoCapture()
        status = self.cap.open(input)
        if not status:
           raise InvalidInput("Can't open the video from {}".format(input))
            
    def read(self):
        status, image = self.cap.read()
        if not status:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status, image = self.cap.read()
            if not status:
                return None
        return image

    def release(self):
        self.cap.release()

    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_type(self):
        return 'VIDEO'


class CameraCapWrapper(ImagesCapture):

    def __init__(self, input, camera_resolution):
        
        self.cap = cv2.VideoCapture()
        try:
            status = self.cap.open(int(input))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            if not status:
                raise OpenError("Can't open the camera from {}".format(input))
        except ValueError:
            raise InvalidInput("Can't find the camera {}".format(input))

    def read(self):
        status, image = self.cap.read()
        if not status:
            return None
        return image

    def release(self):
        self.cap.release()

    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_type(self):
        return 'CAMERA'

class VideoCapture():
    def __init__(self, input, loop=True, camera_resolution=(1280, 720)):
       
        self.inputs = input.split(',')
        self.nb_inputs = len(self.inputs)
        self.input_index = -1
        self.loop = loop
        self.camera_resolution = camera_resolution 
        self.next()

    def read(self):
        if self.reader is not None:
            return self.reader.read()
        return None

    def release(self):
        if self.reader is not None:
            return self.reader.release()
        return None

    def next(self):

        self.reader = None

        while self.reader == None:
            errors = {InvalidInput: [], OpenError: []}

            self.input_index = (self.input_index+1) % self.nb_inputs
            input = self.inputs[self.input_index]
        
            for reader in (ImreadWrapper, DirReader, VideoCapWrapper):
                try:
                    self.reader = reader(input, self.loop)
                    if self.reader is not None:
                        return
                except (InvalidInput, OpenError) as e:
                    errors[type(e)].append(e.message)
            try:
                self.reader = CameraCapWrapper(input, self.camera_resolution)
                if self.reader is not None:
                    return

            except (InvalidInput, OpenError) as e:
                errors[type(e)].append(e.message)
            if not errors[OpenError]:
                print(*errors[InvalidInput], file=sys.stderr, sep='\n')
            else:
                print(*errors[OpenError], file=sys.stderr, sep='\n')
    
    def get_distance(self, x, y):
        if self.reader is not None:
            return self.reader.get_distance(x,y)
        return None
